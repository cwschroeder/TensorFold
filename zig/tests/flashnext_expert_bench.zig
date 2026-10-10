//! One loaded engine compares expert variants, exact logits and tokens before alternating decode timings.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const fz = tf.flashnext_replay;

pub const Variant = enum { xsx, grouped, gpu_event };

fn select(e: *fx.Engine, variant: Variant, candidate: bool) void {
    e.r.xsx = variant == .xsx and candidate;
    e.r.grouped = variant == .grouped and candidate;
    e.gpu_round_event = variant == .gpu_event and candidate;
}

const Capture = struct {
    a: std.mem.Allocator,
    ids: std.ArrayList(u32) = .empty,
    start: f64,
    prefill: f64 = 0,
    failed: bool = false,
    fn prefilled(ctx: *anyopaque) void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        c.prefill = mtl.clock.seconds() - c.start;
    }
    fn tokens(ctx: *anyopaque, ids: []const u32) bool {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        c.ids.appendSlice(c.a, ids) catch {
            c.failed = true;
            return true;
        };
        return false;
    }
    fn cancelled(_: *anyopaque) bool {
        return false;
    }
};

fn generate(e: *fx.Engine, a: std.mem.Allocator, prompt: []const u32, depth: usize, variant: Variant, candidate: bool) !Capture {
    select(e, variant, candidate);
    e.copy = false;
    var c: Capture = .{ .a = a, .start = mtl.clock.seconds() };
    const result = try e.generate(prompt, 256, &.{}, depth, .{ .ctx = &c, .prefilled = Capture.prefilled, .tokens = Capture.tokens, .cancelled = Capture.cancelled });
    if (result.reason != .length) return error.IncompleteReply;
    if (c.failed or c.ids.items.len != 256) return error.IncompleteReply;
    return c;
}

fn promptState(e: *fx.Engine, a: std.mem.Allocator, prompt: []const u32) !void {
    select(e, .xsx, false);
    e.copy = false;
    var c: Capture = .{ .a = a, .start = mtl.clock.seconds() };
    const result = try e.generate(prompt, 1, &.{}, 0, .{ .ctx = &c, .prefilled = Capture.prefilled, .tokens = Capture.tokens, .cancelled = Capture.cancelled });
    if (result.reason != .length or c.failed or c.ids.items.len != 1) return error.IncompleteReply;
    e.hostMode();
}

pub fn run(e: *fx.Engine, a: std.mem.Allocator, prompt: []const u32, variant: Variant) !void {
    if (e.r.tp != null) return error.OneMacRequired;
    if (variant == .grouped and e.r.route_pipe.maxThreads() < 32 * fz.MAXR) return error.RouteThreadLimit;
    const reference = try generate(e, a, prompt, 0, variant, false);
    const saved = try a.alloc(u16, fz.MAXR * fz.VOCAB);
    for (1..fz.MAXR + 1) |rows| {
        const inputs = reference.ids.items[0..rows];
        var picks: [fz.MAXR]u32 = undefined;
        try promptState(e, a, prompt);
        select(e, variant, false);
        try e.m.window(inputs, &picks);
        const before = e.m.t.logits.b.slice(u16, rows * fz.VOCAB);
        @memcpy(saved[0..before.len], before);
        try promptState(e, a, prompt);
        select(e, variant, true);
        try e.m.window(inputs, &picks);
        const after = e.m.t.logits.b.slice(u16, before.len);
        if (!std.mem.eql(u16, saved[0..before.len], after)) return error.ExpertLogitsDiffer;
        std.debug.print("EXPERT {s} logits rows{d} values{d} bit_equal\n", .{ @tagName(variant), rows, before.len });
    }
    for ([_]usize{ 0, 3 }) |depth| {
        for ([_]bool{ false, true }) |xsx| {
            const warm = try generate(e, a, prompt, depth, variant, xsx);
            if (!std.mem.eql(u32, reference.ids.items, warm.ids.items)) return error.WarmTokensDiffer;
        }
        for ([_]bool{ false, true, true, false }, 0..) |xsx, block| {
            for (0..3) |rep| {
                const c = try generate(e, a, prompt, depth, variant, xsx);
                const wall = mtl.clock.seconds() - c.start;
                const decode = wall - c.prefill;
                if (!std.math.isFinite(decode) or decode <= 0) return error.InvalidTime;
                if (!std.mem.eql(u32, reference.ids.items, c.ids.items)) return error.TimedTokensDiffer;
                std.debug.print("EXPERT {s} depth{d} candidate{d} block{d} rep{d} tokens{d} prefill_s{d:.9} wall_s{d:.9} decode_tps{d:.6} token_equal\n", .{ @tagName(variant), depth, @intFromBool(xsx), block, rep, c.ids.items.len, c.prefill, wall, 255.0 / decode });
            }
        }
    }
    std.debug.print("PASS EXPERT {s}: 16 logit windows, 24 timed replies, 4 warm replies equal 256 plain reference tokens\n", .{@tagName(variant)});
}
