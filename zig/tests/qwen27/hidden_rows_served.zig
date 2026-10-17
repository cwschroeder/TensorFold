//! Queued captures beside a real synthetic chat; build-only until an exclusive Metal window.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const served = @import("qwen27_served");
const graph = @import("synthetic_graph.zig");
const Flag = std.atomic.Value(bool);

fn wait(io: std.Io, flag: *const Flag) !void {
    for (0..30_000) |_| {
        if (flag.load(.acquire)) return;
        try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    }
    return error.WaitTimedOut;
}

const Capture = struct {
    host: *served.Host,
    a: std.mem.Allocator,
    ids: []const u32,
    chunk: u32,
    rows: ?q.hidden_rows.Rows = null,
    failure: ?anyerror = null,
    fn run(c: *Capture) void {
        c.rows = c.host.hiddenRows(c.a, c.ids, c.chunk, null) catch |err| {
            c.failure = err;
            return;
        };
    }
};

const Box = struct {
    a: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    prefilled: Flag = .init(false),
    release: Flag = .init(false),
    done: Flag = .init(false),
    bad: Flag = .init(false),
    tokens: std.ArrayList(u32) = .empty,
    reason: ?api.Reason = null,

    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        if (e.* == .prefilled) {
            b.prefilled.store(true, .release);
            wait(b.io, &b.release) catch b.bad.store(true, .release);
            return;
        }
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (e.*) {
            .tokens => |ids| b.tokens.appendSlice(b.a, ids) catch b.bad.store(true, .release),
            .finished => |f| {
                b.reason = f.reason;
                b.done.store(true, .release);
            },
            else => {},
        }
    }
};

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const shared = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(a);
    if (args.len > 2 or (args.len == 2 and !std.mem.eql(u8, args[1], "--ane"))) return error.Usage;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try graph.create(a);
    defer m.deinit();
    if (args.len == 2) m.ane = try q.ane_mlp.Share.init(a, m.device, m.queue, &m.checkpoint, m.config, 64);
    var ids: [130]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    const chat = ids[0..11];
    var expected: [64]u32 = undefined;
    {
        var runner = try q.decode_round.Runner.init(a, m, 1, 256);
        defer runner.deinit();
        const session = q.session.Session{ .runner = &runner };
        try session.prefill(chat, 128);
        for (&expected, 0..) |*id, i| {
            id.* = try session.greedy();
            if (i + 1 < expected.len) try session.step(id.*);
        }
    }
    var why: []const u8 = "";
    const h = try served.attach(shared, init.io, m, null, .{ .context = 256, .streams = 2, .cache_gib = 0 }, a, &why);
    var host_open = true;
    defer if (host_open) served.close(h);
    var solo = try h.hiddenRows(shared, &ids, 16, null);
    defer solo.deinit();
    var box = Box{ .a = shared, .io = init.io };
    defer box.tokens.deinit(shared);
    var jobs: [3]Capture = undefined;
    var threads: [3]?std.Thread = @splat(null);
    defer {
        box.release.store(true, .release);
        for (&threads) |*thread| if (thread.*) |t| {
            t.join();
            thread.* = null;
        };
        served.close(h);
        host_open = false;
        for (&jobs) |*job| if (job.rows) |*rows| rows.deinit();
    }
    for (&jobs, [_]u32{ 1, 16, 128 }) |*job, chunk| job.* = .{ .host = h, .a = shared, .ids = &ids, .chunk = chunk };
    const request = api.Request{ .prompt = chat, .max_tokens = expected.len, .drafts = false };
    try h.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try wait(init.io, &box.prefilled);
    for (&threads, &jobs) |*thread, *job| thread.* = try std.Thread.spawn(.{}, Capture.run, .{job});
    var queued = false;
    for (0..30_000) |_| {
        h.host.lock();
        var count: usize = 0;
        var next = h.host.calls.first;
        while (next) |c| : (next = c.next) count += 1;
        h.host.unlock();
        if (count == jobs.len) {
            queued = true;
            break;
        }
        try std.Io.sleep(init.io, .fromMilliseconds(1), .awake);
    }
    if (!queued) return error.CapturesNotQueuedDuringChat;
    box.release.store(true, .release);
    for (&threads) |*thread| {
        thread.*.?.join();
        thread.* = null;
    }
    for (jobs) |job| {
        if (job.failure) |err| return err;
        if (!std.mem.eql(u16, solo.values, job.rows.?.values)) return error.ConcurrentCaptureDiffers;
    }
    try wait(init.io, &box.done);
    box.mutex.lockUncancelable(init.io);
    defer box.mutex.unlock(init.io);
    if (box.bad.load(.acquire) or box.reason != .length or !std.mem.eql(u32, &expected, box.tokens.items)) return error.ChatReplyChanged;
    std.debug.print("queued hidden rows: 3 concurrent captures equal solo; 64 chat tokens equal CLI; ANE={}\n", .{m.ane != null});
}
