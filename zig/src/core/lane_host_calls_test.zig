//! Owner calls through a real host thread: serialization, shutdown, reentrancy and chat fairness.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const a = std.testing.allocator;
const io = std.testing.io;

const Rig = struct {
    cfg: lanes.Config = undefined,
    target: lanes.fake.Fake = .{ .gpa = a },
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: LaneHost = undefined,
    stopped: bool = false,

    fn init(r: *Rig) !void {
        r.cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.core = lanes.Engine.init(a, &r.cfg, r.target.backend(), r.clock.clock());
        r.host = LaneHost.init(a, io, &r.core, .{ .lanes = 2 });
    }
    fn deinit(r: *Rig) void {
        if (!r.stopped) r.host.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(a);
    }
    fn stop(r: *Rig) void {
        r.host.stop();
        r.stopped = true;
    }
};

const Flag = std.atomic.Value(bool);
fn wait(flag: *const Flag) !void {
    for (0..5000) |_| {
        if (flag.load(.acquire)) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.TestTimeout;
}

const Threads = struct {
    threads: [32]?std.Thread = @splat(null),
    count: usize = 0,
    fn add(t: *Threads, comptime f: anytype, args: anytype) !void {
        t.threads[t.count] = try std.Thread.spawn(.{}, f, args);
        t.count += 1;
    }
    fn join(t: *Threads) void {
        for (t.threads[0..t.count]) |thread| thread.?.join();
        t.count = 0;
    }
};

const Job = struct {
    host: *LaneHost,
    gate: ?*const Flag = null,
    cancel: ?*const Flag = null,
    entered: Flag = .init(false),
    returned: Flag = .init(false),
    ran: bool = false,
    saw_cancel: bool = false,
    reenter: bool = false,
    failure: ?anyerror = null,
    nested_failure: ?anyerror = null,
    owner: ?std.Thread.Id = null,
    counter: ?*std.atomic.Value(usize) = null,
    observed_count: usize = 0,

    fn exec(ptr: *anyopaque) void {
        const j: *Job = @ptrCast(@alignCast(ptr));
        j.owner = std.Thread.getCurrentId();
        if (j.counter) |counter| j.observed_count = counter.load(.acquire);
        j.entered.store(true, .release);
        if (j.gate) |gate| wait(gate) catch @panic("owner-call gate timed out");
        if (j.cancel) |cancel| j.saw_cancel = cancel.load(.acquire);
        if (j.reenter) j.host.callOwner(j, exec) catch |err| {
            j.nested_failure = err;
        };
        j.ran = true;
    }
    fn run(j: *Job) void {
        j.host.callOwner(j, exec) catch |err| {
            j.failure = err;
        };
        j.returned.store(true, .release);
    }
};

fn pending(h: *LaneHost, want: usize) !void {
    for (0..5000) |_| {
        h.lock();
        var count: usize = 0;
        var next = h.calls.first;
        while (next) |c| : (next = c.next) count += 1;
        h.unlock();
        if (count >= want) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.TestTimeout;
}

test "owner call refuses an unstarted host and owner reentrancy" {
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    var job = Job{ .host = &rig.host, .reenter = true };
    try std.testing.expectError(error.NotStarted, rig.host.callOwner(&job, Job.exec));
    try rig.host.start();
    try rig.host.callOwner(&job, Job.exec);
    try std.testing.expectEqual(error.ReentrantOwnerCall, job.nested_failure.?);
    try std.testing.expect(job.owner.? != std.Thread.getCurrentId());
    rig.stop();
    try std.testing.expectError(error.Closed, rig.host.callOwner(&job, Job.exec));
}

test "owner calls serialize and cancellation never releases a caller while its callback uses the context" {
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    try rig.host.start();
    var gate: Flag = .init(false);
    var cancel: Flag = .init(false);
    var threads: Threads = .{};
    defer {
        gate.store(true, .release);
        threads.join();
    }
    var first = Job{ .host = &rig.host, .gate = &gate, .cancel = &cancel };
    var second = Job{ .host = &rig.host };
    try threads.add(Job.run, .{&first});
    try wait(&first.entered);
    try threads.add(Job.run, .{&second});
    try pending(&rig.host, 1);
    cancel.store(true, .release);
    try std.testing.expect(!first.returned.load(.acquire) and !second.entered.load(.acquire));
    gate.store(true, .release);
    threads.join();
    try std.testing.expect(first.ran and first.saw_cancel and second.ran);
    try std.testing.expect(first.failure == null and second.failure == null);
    try std.testing.expectEqual(first.owner, second.owner);
}

test "owner stop rejects pending calls while active callback finishes before shutdown returns" {
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    try rig.host.start();
    var gate: Flag = .init(false);
    var threads: Threads = .{};
    defer {
        gate.store(true, .release);
        threads.join();
    }
    var first = Job{ .host = &rig.host, .gate = &gate };
    var second = Job{ .host = &rig.host };
    try threads.add(Job.run, .{&first});
    try wait(&first.entered);
    try threads.add(Job.run, .{&second});
    try pending(&rig.host, 1);
    try threads.add(Rig.stop, .{&rig});
    try wait(&second.returned);
    try std.testing.expectEqual(error.Closed, second.failure.?);
    try std.testing.expect(!second.ran and !first.returned.load(.acquire));
    gate.store(true, .release);
    threads.join();
    try std.testing.expect(first.ran and first.failure == null and rig.stopped);
}

test "engine drain rejects pending calls before joining an active owner callback" {
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    try rig.host.start();
    var gate: Flag = .init(false);
    var threads: Threads = .{};
    defer {
        gate.store(true, .release);
        threads.join();
    }
    var first = Job{ .host = &rig.host, .gate = &gate };
    var second = Job{ .host = &rig.host };
    try threads.add(Job.run, .{&first});
    try wait(&first.entered);
    try threads.add(Job.run, .{&second});
    try pending(&rig.host, 1);
    try threads.add(api.Engine.drain, .{rig.host.engine()});
    try wait(&second.returned);
    try std.testing.expectEqual(error.Closed, second.failure.?);
    try std.testing.expect(!second.ran and !first.returned.load(.acquire));
    gate.store(true, .release);
    threads.join();
    try std.testing.expect(first.ran and first.failure == null);
    try std.testing.expect(rig.host.thread == null);
}

test "owner enqueue racing stop completes every accepted or refused caller" {
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    try rig.host.start();
    var threads: Threads = .{};
    defer threads.join();
    var jobs: [24]Job = undefined;
    for (&jobs, 0..) |*job, i| {
        job.* = .{ .host = &rig.host };
        try threads.add(Job.run, .{job});
        if (i == 12) try threads.add(Rig.stop, .{&rig});
    }
    threads.join();
    for (jobs) |job| {
        try std.testing.expect(job.returned.load(.acquire));
        if (job.failure) |err| try std.testing.expect(err == error.Closed) else try std.testing.expect(job.ran);
    }
}

const Box = struct {
    done: Flag = .init(false),
    tokens: std.atomic.Value(usize) = .init(0),
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        if (e.* == .finished) b.done.store(true, .release);
        if (e.* == .tokens) _ = b.tokens.fetchAdd(e.tokens.len, .release);
    }
};

test "owner queue lets a chat round run between two waiting family calls" {
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    try rig.host.start();
    var gate: Flag = .init(false);
    var threads: Threads = .{};
    defer {
        gate.store(true, .release);
        threads.join();
    }
    var box: Box = .{};
    var first = Job{ .host = &rig.host, .gate = &gate, .counter = &box.tokens };
    var second = Job{ .host = &rig.host, .counter = &box.tokens };
    try threads.add(Job.run, .{&first});
    try wait(&first.entered);
    const request = api.Request{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 64 };
    try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try threads.add(Job.run, .{&second});
    try pending(&rig.host, 1);
    gate.store(true, .release);
    threads.join();
    try std.testing.expect(second.observed_count > first.observed_count and second.observed_count >= 2);
    try wait(&box.done);
}

test "owner call makes a running lone driver yield before its long reply ends" {
    const Lone = struct {
        backend: lanes.backend.Backend,
        entered: Flag = .init(false),
        yielded: Flag = .init(false),
        fn run(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) !bool {
            const l: *@This() = @ptrCast(@alignCast(ptr));
            _ = try l.backend.opening(a, s);
            hooks.committed(hooks.ctx);
            s.pending = s.context.items[s.context.items.len - 1];
            s.cache_len = s.prompt_len;
            l.entered.store(true, .release);
            for (0..5000) |_| {
                if (hooks.yield(hooks.ctx)) {
                    l.yielded.store(true, .release);
                    return true;
                }
                std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            }
            return error.TestTimeout;
        }
    };
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    var lone = Lone{ .backend = rig.target.backend() };
    rig.host.lone = .{ .ctx = &lone, .run = Lone.run };
    try rig.host.start();
    var box: Box = .{};
    const request = api.Request{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 64 };
    try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try wait(&lone.entered);
    var job = Job{ .host = &rig.host, .counter = &box.tokens };
    try rig.host.callOwner(&job, Job.exec);
    try std.testing.expect(lone.yielded.load(.acquire));
    try std.testing.expect(job.observed_count > 0 and job.observed_count < request.max_tokens);
    try wait(&box.done);
}

test "owner call waits for an open lesson without preventing that lesson from completing" {
    const Learner = struct {
        gate: Flag = .init(false),
        entered: Flag = .init(false),
        steps: std.atomic.Value(usize) = .init(0),
        fn begin(_: *anyopaque, _: *const api.LearnRequest, _: api.LearnSink) !void {
            return error.UnexpectedBegin;
        }
        fn step(ptr: *anyopaque) api.Learner.Step {
            const l: *@This() = @ptrCast(@alignCast(ptr));
            if (l.steps.fetchAdd(1, .acq_rel) == 0) {
                l.entered.store(true, .release);
                wait(&l.gate) catch @panic("lesson gate timed out");
                return .{ .done = false, .changed = false };
            }
            return .{ .done = true, .changed = false };
        }
        fn abort(_: *anyopaque) void {}
    };
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    var learner: Learner = .{};
    rig.host.learner = .{ .ctx = &learner, .begin = Learner.begin, .step = Learner.step, .abort = Learner.abort };
    rig.host.lesson_open = true;
    try rig.host.start();
    var threads: Threads = .{};
    defer {
        learner.gate.store(true, .release);
        if (!rig.stopped) rig.stop();
        threads.join();
    }
    try wait(&learner.entered);
    var job = Job{ .host = &rig.host, .counter = &learner.steps };
    try threads.add(Job.run, .{&job});
    try pending(&rig.host, 1);
    learner.gate.store(true, .release);
    try wait(&job.returned);
    threads.join();
    try std.testing.expect(job.failure == null and job.observed_count >= 2);
}

test "owner calls queued before a failed weight put-back are refused until the model heals" {
    const Learner = struct {
        gate: Flag = .init(false),
        entered: Flag = .init(false),
        healing: bool = false,
        fn begin(ptr: *anyopaque, _: *const api.LearnRequest, _: api.LearnSink) !void {
            const l: *@This() = @ptrCast(@alignCast(ptr));
            l.healing = true;
        }
        fn step(ptr: *anyopaque) api.Learner.Step {
            const l: *@This() = @ptrCast(@alignCast(ptr));
            if (l.healing) return .{ .done = true, .changed = false, .stale = false };
            l.entered.store(true, .release);
            wait(&l.gate) catch @panic("stale lesson gate timed out");
            return .{ .done = true, .changed = false, .stale = true };
        }
        fn abort(_: *anyopaque) void {}
        fn event(_: *anyopaque, _: *const api.LearnEvent) void {}
    };
    var rig: Rig = .{};
    try rig.init();
    defer rig.deinit();
    var learner: Learner = .{};
    rig.host.learner = .{ .ctx = &learner, .begin = Learner.begin, .step = Learner.step, .abort = Learner.abort };
    rig.host.lesson_open = true;
    try rig.host.start();
    var threads: Threads = .{};
    defer {
        learner.gate.store(true, .release);
        if (!rig.stopped) rig.stop();
        threads.join();
    }
    try wait(&learner.entered);
    var jobs = [_]Job{ .{ .host = &rig.host }, .{ .host = &rig.host } };
    for (&jobs) |*job| try threads.add(Job.run, .{job});
    try pending(&rig.host, 2);
    learner.gate.store(true, .release);
    for (&jobs) |*job| try wait(&job.returned);
    threads.join();
    for (jobs) |job| {
        try std.testing.expectEqual(error.StaleModel, job.failure.?);
        try std.testing.expect(!job.ran);
    }
    var healed = Job{ .host = &rig.host };
    try std.testing.expectError(error.StaleModel, rig.host.callOwner(&healed, Job.exec));
    const heal: api.LearnRequest = .{};
    try rig.host.engine().learn(&heal, .{ .ctx = &learner, .event = Learner.event });
    var still_stale = true;
    for (0..5000) |_| {
        rig.host.lock();
        still_stale = rig.host.stale;
        rig.host.unlock();
        if (!still_stale) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(!still_stale);
    try rig.host.callOwner(&healed, Job.exec);
    try std.testing.expect(healed.ran);
}
