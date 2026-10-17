//! Family operations run on the engine thread between rounds, with caller-owned completion storage.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

pub const Error = error{ Closed, NotStarted, ReentrantOwnerCall, StaleModel };
pub const Run = *const fn (*anyopaque) void;

const Call = struct {
    next: ?*Call = null,
    io: std.Io,
    ctx: *anyopaque,
    run: Run,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    done: bool = false,
    failure: ?Error = null,

    fn finish(c: *Call, failure: ?Error) void {
        const io = c.io;
        c.mutex.lockUncancelable(io);
        c.failure = failure;
        c.done = true;
        c.wake.signal(io);
        c.mutex.unlock(io);
        // The caller can now return and destroy this node. Never touch c again.
    }

    fn wait(c: *Call) Error!void {
        const io = c.io;
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        while (!c.done) c.wake.waitUncancelable(io, &c.mutex);
        if (c.failure) |err| return err;
    }
};

/// Intrusive nodes stay on their callers' stacks until their own completion is observed.
pub const Queue = struct {
    first: ?*Call = null,
    last: ?*Call = null,

    fn append(q: *Queue, c: *Call) void {
        if (q.last) |last| last.next = c else q.first = c;
        q.last = c;
    }

    fn pop(q: *Queue) ?*Call {
        const c = q.first orelse return null;
        q.first = c.next;
        if (q.first == null) q.last = null;
        c.next = null;
        return c;
    }
};

/// Once accepted, even a canceled caller waits until the owner has finished using ctx.
pub fn submit(h: *LaneHost, ctx: *anyopaque, run: Run) Error!void {
    var c = Call{ .io = h.io, .ctx = ctx, .run = run };
    h.lock();
    if (h.closing) {
        h.unlock();
        return error.Closed;
    }
    if (h.stale) {
        h.unlock();
        return error.StaleModel;
    }
    if (h.thread == null) {
        h.unlock();
        return error.NotStarted;
    }
    if (h.owner_thread == std.Thread.getCurrentId()) {
        h.unlock();
        return error.ReentrantOwnerCall;
    }
    h.calls.append(&c);
    h.wake.signal(h.io);
    h.unlock();
    // No host access after enqueue: stop may free it while this caller observes its own completion.
    try c.wait();
}

/// At most one operation per ordinary loop iteration; a running lesson reaches its safe end first.
pub fn one(h: *LaneHost) void {
    h.lock();
    if (h.stale) {
        const pending = detach(&h.calls);
        h.unlock();
        return reject(pending, error.StaleModel);
    }
    const c = if (h.closing or h.lesson_open) null else h.calls.pop();
    h.unlock();
    const call = c orelse return;
    call.run(call.ctx);
    call.finish(null);
}

/// Detach pending work under the host lock, then finish it without that lock or host storage.
pub fn detach(q: *Queue) ?*Call {
    const first = q.first;
    q.* = .{};
    return first;
}

pub fn reject(first: ?*Call, failure: Error) void {
    var next = first;
    while (next) |c| {
        next = c.next;
        c.finish(failure);
    }
}
