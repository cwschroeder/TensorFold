//! Cancel and allocation failure after the allocation budget of a complete first prompt chunk.
const std = @import("std");
const q = @import("tensorfold").qwen27;

const InterruptAllocator = struct {
    backing: std.mem.Allocator,
    cancel: *std.atomic.Value(bool),
    oom: bool = false,
    at: usize,
    calls: usize = 0,
    expected: []const u8,
    output: ?[]u8 = null,
    saw_completed_first_chunk: bool = false,
    fired: bool = false,

    fn allocator(c: *InterruptAllocator) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c: *InterruptAllocator = @ptrCast(@alignCast(ptr));
        if (c.calls == c.at) {
            // The CPU writes output only after advance/drain returns; never read GPU buffers during encoding.
            if (c.output) |out| c.saw_completed_first_chunk = out.len >= c.expected.len and std.mem.eql(u8, out[0..c.expected.len], c.expected);
            c.fired = true;
            if (c.oom) return null;
            c.cancel.store(true, .release);
        }
        const out = c.backing.rawAlloc(len, alignment, ra) orelse return null;
        if (c.calls == 0) {
            c.output = out[0..len];
            @memset(out[0..len], 0xff);
        }
        c.calls += 1;
        return out;
    }
    fn resize(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) bool {
        const c: *InterruptAllocator = @ptrCast(@alignCast(ptr));
        return c.backing.rawResize(memory, alignment, len, ra);
    }
    fn remap(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        const c: *InterruptAllocator = @ptrCast(@alignCast(ptr));
        return c.backing.rawRemap(memory, alignment, len, ra);
    }
    fn free(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const c: *InterruptAllocator = @ptrCast(@alignCast(ptr));
        c.backing.rawFree(memory, alignment, ra);
    }
};

pub fn run(a: std.mem.Allocator, m: *q.model.Model, ids: []const u32, chunk: u32) !void {
    if (ids.len <= chunk) return error.NeedMultipleChunks;
    var count = std.testing.FailingAllocator.init(a, .{});
    var first = try q.hidden_rows.captureExclusive(count.allocator(), m, ids[0..chunk], .{ .context = 256, .chunk = chunk });
    const expected = a.dupe(u16, first.values) catch |err| {
        first.deinit();
        return err;
    };
    defer a.free(expected);
    first.deinit();
    if (count.allocated_bytes != count.freed_bytes) return error.CalibrationLeak;

    var canceled: std.atomic.Value(bool) = .init(false);
    var cancel_memory = std.testing.FailingAllocator.init(a, .{});
    var interrupt = InterruptAllocator{ .backing = cancel_memory.allocator(), .cancel = &canceled, .at = count.alloc_index, .expected = std.mem.sliceAsBytes(expected) };
    if (q.hidden_rows.captureExclusive(interrupt.allocator(), m, ids, .{ .context = 256, .chunk = chunk, .cancel = &canceled })) |rows_| {
        var rows = rows_;
        rows.deinit();
        return error.ExpectedMidPassCancellation;
    } else |err| if (err != error.Canceled or !canceled.load(.acquire)) return err;
    if (!interrupt.saw_completed_first_chunk) return error.CancellationWasTooEarly;
    if (cancel_memory.allocated_bytes != cancel_memory.freed_bytes) return error.CanceledHostLeak;

    var failing = std.testing.FailingAllocator.init(a, .{});
    var oom = InterruptAllocator{ .backing = failing.allocator(), .cancel = &canceled, .at = count.alloc_index, .expected = std.mem.sliceAsBytes(expected), .oom = true };
    if (q.hidden_rows.captureExclusive(oom.allocator(), m, ids, .{ .context = 256, .chunk = chunk })) |rows_| {
        var rows = rows_;
        rows.deinit();
        return error.ExpectedMidPassAllocationFailure;
    } else |err| if (err != error.OutOfMemory or !oom.fired) return err;
    if (!oom.saw_completed_first_chunk) return error.AllocationFailureWasTooEarly;
    if (failing.allocated_bytes != failing.freed_bytes) return error.MidPassHostLeak;
}
