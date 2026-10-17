//! A handoff completes even when its input command buffer fails, so later GPU waits can end.
const std = @import("std");

/// One producer and one consumer; completion makes both payload reuse and retained-CB release safe.
pub fn Ring(comptime Payload: type, comptime capacity: usize) type {
    return struct {
        slots: [capacity]Entry = @splat(.{}),
        issued: u64 = 0,
        completed: std.atomic.Value(u64) = .init(0),
        const Self = @This();
        const Entry = struct { payload: Payload = undefined, published: std.atomic.Value(u64) = .init(0) };

        pub fn prepare(q: *Self, a: std.mem.Allocator, retained: anytype) !void {
            if (q.issued >= std.math.maxInt(u64) / 2) return error.AneHandoffOverflow;
            if (q.issued - q.completed.load(.acquire) >= capacity) return error.AneHandoffFull;
            try retained.ensureUnusedCapacity(a, 1);
        }

        pub fn publish(q: *Self, payload: Payload) void {
            const entry = &q.slots[q.issued % capacity];
            entry.payload = payload;
            q.issued += 1;
            entry.published.store(q.issued, .release);
        }

        pub fn get(q: *Self, t: u64) ?*const Payload {
            const entry = &q.slots[t % capacity];
            return if (entry.published.load(.acquire) == t + 1) &entry.payload else null;
        }

        pub fn complete(q: *Self, t: u64) void {
            q.completed.store(t + 1, .release);
        }
    };
}

/// `job` owns its retained input CB until the batch has also observed worker completion.
pub fn execute(job: anytype) void {
    while (!job.ready()) {
        if (job.inputFailed()) {
            job.poison();
            break;
        }
        job.waitInput();
    }
    if (!job.poisoned()) job.evaluate();
    job.complete();
}

/// Wait every input CB and the worker before releasing any CB, including on multiple failures.
pub fn drain(batch: anytype) bool {
    var failed = false;
    for (0..batch.count()) |i| {
        batch.waitBuffer(i);
        failed = batch.bufferFailed(i) or failed;
    }
    batch.waitWorker();
    for (0..batch.count()) |i| batch.releaseBuffer(i);
    batch.clear();
    return failed;
}

const FakeJob = struct {
    input: enum { pending, ready, failed } = .pending,
    fail_after_wait: bool = false,
    poison_: bool = false,
    ran: bool = false,
    completed: bool = false,
    waits: usize = 0,
    pub fn ready(j: *FakeJob) bool {
        return j.input == .ready;
    }
    pub fn inputFailed(j: *FakeJob) bool {
        return j.input == .failed;
    }
    pub fn poison(j: *FakeJob) void {
        j.poison_ = true;
    }
    pub fn poisoned(j: *FakeJob) bool {
        return j.poison_;
    }
    pub fn waitInput(j: *FakeJob) void {
        j.waits += 1;
        j.input = if (j.fail_after_wait) .failed else .ready;
    }
    pub fn evaluate(j: *FakeJob) void {
        std.debug.assert(j.input == .ready and !j.poison_);
        j.ran = true;
    }
    pub fn complete(j: *FakeJob) void {
        std.debug.assert(j.input != .pending);
        std.debug.assert(j.poison_ or j.ran);
        j.completed = true;
    }
};

test "failed input with no odd signal releases its even wait without running ANE" {
    var j: FakeJob = .{ .fail_after_wait = true };
    execute(&j);
    try std.testing.expect(j.poison_ and j.completed and !j.ran);
    try std.testing.expectEqual(@as(usize, 1), j.waits);
}

test "a poisoned followup still waits its input CB before releasing its even wait" {
    var j: FakeJob = .{ .poison_ = true };
    execute(&j);
    try std.testing.expect(j.completed and !j.ran);
    try std.testing.expectEqual(@as(usize, 1), j.waits);
}

test "successful input runs before worker completion" {
    var j: FakeJob = .{};
    execute(&j);
    try std.testing.expect(j.ran and j.completed and !j.poison_);
}

const FakeBatch = struct {
    waited: [3]bool = @splat(false),
    released: [3]bool = @splat(false),
    worker_done: bool = false,
    cleared: bool = false,
    pub fn count(_: *FakeBatch) usize {
        return 3;
    }
    pub fn waitBuffer(b: *FakeBatch, i: usize) void {
        b.waited[i] = true;
    }
    pub fn bufferFailed(_: *FakeBatch, i: usize) bool {
        return i != 1;
    }
    pub fn waitWorker(b: *FakeBatch) void {
        std.debug.assert(std.mem.allEqual(bool, &b.waited, true));
        b.worker_done = true;
    }
    pub fn releaseBuffer(b: *FakeBatch, i: usize) void {
        std.debug.assert(b.worker_done);
        b.released[i] = true;
    }
    pub fn clear(b: *FakeBatch) void {
        std.debug.assert(std.mem.allEqual(bool, &b.released, true));
        b.cleared = true;
    }
};

test "drain waits all CBs and the worker despite first and last CB failures" {
    var b: FakeBatch = .{};
    try std.testing.expect(drain(&b));
    try std.testing.expect(b.cleared and b.worker_done);
}

test "failed reservation publishes no ticket and the next successful ticket is zero" {
    const a = std.testing.allocator;
    var q: Ring(u32, 2) = .{};
    var retained: std.ArrayList(u32) = .empty;
    defer retained.deinit(a);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, q.prepare(failing.allocator(), &retained));
    try std.testing.expectEqual(@as(u64, 0), q.issued);
    try std.testing.expect(q.get(0) == null);
    try q.prepare(a, &retained);
    retained.appendAssumeCapacity(42);
    q.publish(42);
    try std.testing.expectEqual(@as(u32, 42), q.get(0).?.*);
    try std.testing.expectEqual(@as(u64, 1), q.issued);
    q.complete(0);
}

test "real producer and consumer publish complete payloads through repeated ring wraparound" {
    const Payload = struct { t: u64, checksum: u64 };
    const Queue = Ring(Payload, 4);
    const Consumer = struct {
        fn run(q: *Queue, bad: *std.atomic.Value(bool)) void {
            for (0..1024) |t| {
                while (true) {
                    if (q.get(t)) |p| {
                        if (p.t != t or p.checksum != t ^ 0xaabbccdd) bad.store(true, .release);
                        q.complete(t);
                        break;
                    }
                    std.atomic.spinLoopHint();
                }
            }
        }
    };
    var q: Queue = .{};
    var bad: std.atomic.Value(bool) = .init(false);
    var retained: std.ArrayList(u32) = .empty;
    defer retained.deinit(std.testing.allocator);
    // Reserve once so an allocation error cannot leave the consumer waiting for unpublished work.
    try retained.ensureTotalCapacity(std.testing.allocator, 1024);
    const thread = try std.Thread.spawn(.{}, Consumer.run, .{ &q, &bad });
    defer thread.join();
    for (0..1024) |t| {
        while (true) {
            q.prepare(std.testing.allocator, &retained) catch |err| {
                std.debug.assert(err == error.AneHandoffFull);
                std.atomic.spinLoopHint();
                continue;
            };
            break;
        }
        retained.appendAssumeCapacity(@intCast(t));
        q.publish(.{ .t = t, .checksum = t ^ 0xaabbccdd });
    }
    while (q.completed.load(.acquire) < 1024) std.atomic.spinLoopHint();
    try std.testing.expect(!bad.load(.acquire));
}
