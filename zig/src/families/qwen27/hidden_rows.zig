//! Final normalized prompt rows, with their own frame and fresh recurrent/KV state for each call.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Frame = @import("gpu_frame.zig").Frame;
const Runner = @import("decode_round.zig").Runner;
const Round = @import("round_plan.zig").Round;
pub const Plan = @import("hidden_rows_plan.zig").Plan;

pub const Options = struct {
    context: u32,
    chunk: u32 = 128,
    cancel: ?*const std.atomic.Value(bool) = null,

    fn check(o: Options) !void {
        if (o.cancel) |c| if (c.load(.acquire)) return error.Canceled;
    }
};

/// Owned bf16 bits in token order, [count][hidden], independent of later forward passes.
pub const Rows = struct {
    allocator: std.mem.Allocator,
    values: []u16,
    count: u32,
    hidden: usize,

    pub fn deinit(r: *Rows) void {
        r.allocator.free(r.values);
        r.* = undefined;
    }
};

/// Caller owns the model exclusively for the entire call (the engine thread, never an HTTP thread).
pub fn captureExclusive(a: std.mem.Allocator, m: *Model, ids: []const u32, o: Options) !Rows {
    const p = try Plan.init(ids, m.config.hidden, m.config.vocab, m.frame.capacity, o.context, o.chunk);
    try o.check();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const out = try a.alloc(u16, p.values);
    errdefer a.free(out);
    var frame = try Frame.init(m.device, m.config, m.frame.capacity);
    defer frame.deinit();
    std.mem.swap(Frame, &m.frame, &frame);
    defer std.mem.swap(Frame, &m.frame, &frame);
    const prompt = m.kernels.prompt;
    m.kernels.prompt = true;
    defer m.kernels.prompt = prompt;
    var runner = try Runner.init(a, m, 1, p.rows);
    defer runner.deinit();
    var first: usize = 0;
    while (first < ids.len) {
        try o.check();
        const end = @min(ids.len, first + o.chunk);
        var round = try Round.init(a, &.{.{ .slot = 0, .start = runner.offsets[0], .capacity = runner.capacity, .ids = ids[first..end] }}, @intCast(m.config.conv_kernel), 1, @intCast(m.config.vocab));
        defer round.deinit();
        try runner.advance(&round, .none);
        const values = m.frame.get(.input).buffer.slice(u16, (end - first) * m.config.hidden);
        for (values) |bits| if (bits & 0x7f80 == 0x7f80) return error.NonfiniteHiddenRows;
        @memcpy(out[first * m.config.hidden .. end * m.config.hidden], values);
        first = end;
    }
    try o.check();
    return .{ .allocator = a, .values = out, .count = p.rows, .hidden = m.config.hidden };
}
