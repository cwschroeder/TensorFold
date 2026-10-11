//! Short-state admission must preserve long prefixes before reservation and before copying.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const checks = @import("prompt_cache_test.zig");

test "short admission evicts a newer short state instead of the older long prefix" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u32{ 0, 8 }) |protect| {
        var fake: checks.Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .protect_from = protect }, 218);
        defer store.deinit();
        const long: [13]u32 = @splat(1);
        const short: [7]u32 = @splat(2);
        const other: [7]u32 = @splat(3);
        _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
        _ = fake.pass(&store, &short, try store.begin(a, &short, 3, &.{}, &.{}, null, &.{}));
        const plan = try store.begin(a, &other, 4, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(checks.fresh(&other), fake.pass(&store, &other, plan));
        try std.testing.expectEqual(protect != 0, store.find(&long, &.{}, &.{}) != null);
        try std.testing.expectEqual(protect == 0, store.find(&short, &.{}, &.{}) != null);
        try std.testing.expect(store.find(&other, &.{}, &.{}) != null);
        try std.testing.expect(store.held <= store.budget);
    }
}

test "a reusable spare copy fits without evicting the long prefix, while an unfitting spare is trimmed" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u64{ 10, 94 }) |spare| {
        var fake: checks.Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, fake.pooled(), .{ .min_prompt = 0, .protect_from = 8 }, if (spare == 94) 196 else 206);
        defer store.deinit();
        const long: [13]u32 = @splat(1);
        const short: [7]u32 = @splat(2);
        _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
        fake.spare_bytes = spare;
        fake.at = 4;
        fake.sum = checks.fresh(short[0..4]);
        try std.testing.expect(store.keep(&short, 4, null, &.{}, &.{}));
        try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
        try std.testing.expect(store.held + store.spare() <= store.budget);
        try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
        try std.testing.expectEqual(@as(u64, 94), store.find(&short, &.{}, &.{}).?.bytes);
    }
}

test "lookup admits one reusable buffer even below the new-copy estimate, without promising it twice" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: checks.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, fake.pooled(), .{ .min_prompt = 0, .min_gap = 1, .protect_from = 8 }, 196);
    defer store.deinit();
    const long: [13]u32 = @splat(1);
    const short: [7]u32 = @splat(2);
    _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
    fake.spare_bytes = 94;
    const plan = try store.begin(a, &short, 4, &.{3}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{4}, plan.marks);
    try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
    try std.testing.expectEqual(checks.fresh(&short), fake.pass(&store, &short, plan));
    try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
    try std.testing.expect(store.held + store.spare() <= store.budget);
}

test "an impossible short mark is refused before any reservation or direct-copy eviction" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: checks.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .protect_from = 8 }, 215);
    defer store.deinit();
    const short: [7]u32 = @splat(2);
    const long: [13]u32 = @splat(1);
    const other: [7]u32 = @splat(3);
    _ = fake.pass(&store, &short, try store.begin(a, &short, 3, &.{}, &.{}, null, &.{}));
    _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
    const plan = try store.begin(a, &other, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{}, plan.marks);
    try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
    const before = store.counts.evicted;
    try std.testing.expect(!store.keep(&other, 4, null, &.{}, &.{}));
    try std.testing.expectEqual(before, store.counts.evicted);
    try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
    try std.testing.expect(store.find(&short, &.{}, &.{}) != null);
    try std.testing.expectEqual(checks.fresh(&other), fake.pass(&store, &other, plan));
}

test "multiple short marks reserve cumulatively and existing copies need no new capacity" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: checks.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .min_gap = 1, .protect_from = 8 }, 220);
    defer store.deinit();
    const long: [13]u32 = @splat(1);
    const other: [7]u32 = @splat(3);
    _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
    const plan = try store.begin(a, &other, 4, &.{3}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(usize, 1), plan.marks.len);
    try std.testing.expectEqual(checks.fresh(&other), fake.pass(&store, &other, plan));
    try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
    const mark = plan.marks[0];
    try std.testing.expect(store.keep(&other, mark, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
    try std.testing.expect(store.held <= store.budget);
}

test "charged bytes control room while the prefix position controls protection" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |padded_long| {
        var fake: checks.Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .protect_from = 8 }, if (padded_long) 303 else 320);
        defer store.deinit();
        const long: [13]u32 = @splat(1);
        const short: [7]u32 = @splat(2);
        const other: [7]u32 = @splat(3);
        _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
        _ = fake.pass(&store, &short, try store.begin(a, &short, 3, &.{}, &.{}, null, &.{}));
        const padded = store.find(if (padded_long) &long else &short, &.{}, &.{}).?;
        store.held += 200 - padded.bytes;
        padded.bytes = 200;
        const plan = try store.begin(a, &other, 4, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(usize, if (padded_long) 0 else 1), plan.marks.len);
        try std.testing.expectEqual(checks.fresh(&other), fake.pass(&store, &other, plan));
        try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
        try std.testing.expectEqual(padded_long, store.find(&short, &.{}, &.{}) != null);
        try std.testing.expect(store.held <= store.budget);
    }
}

test "lookup cannot spend the state awaiting restore and direct keep cannot spend untrimmable spare" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: checks.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .min_gap = 1, .protect_from = 8 }, 203);
    defer store.deinit();
    const prompt: [7]u32 = @splat(2);
    _ = fake.pass(&store, &prompt, try store.begin(a, &prompt, 1, &.{}, &.{}, null, &.{}));
    const lookup = try store.lookup(a, &prompt, 4, &.{}, &.{}, &.{});
    try std.testing.expect(lookup.entry != null);
    try std.testing.expectEqualSlices(u32, &.{}, lookup.marks);
    try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
    var vtable = store.family.vtable.*;
    vtable.spare = struct {
        fn spare(_: *anyopaque) u64 {
            return 101;
        }
    }.spare;
    store.family.vtable = &vtable;
    try std.testing.expect(!store.keep(&prompt, 4, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
    try std.testing.expect(store.find(&prompt, &.{}, &.{}) != null);
}

test "eligible eviction keeps an active rewind and preserves the superseded-first order" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |shared| {
        var fake: checks.Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .protect_from = 8 }, 324);
        defer store.deinit();
        const long: [13]u32 = @splat(1);
        const rewind: [7]u32 = @splat(2);
        const other: [7]u32 = @splat(3);
        _ = fake.pass(&store, &long, try store.begin(a, &long, 12, &.{}, &.{}, null, &.{}));
        _ = fake.pass(&store, &rewind, try store.begin(a, &rewind, 3, &.{}, &.{}, null, &.{}));
        store.find(&rewind, &.{}, &.{}).?.shared = shared;
        _ = fake.pass(&store, &other, try store.begin(a, &other, 4, &.{}, &.{}, null, &.{}));
        store.active_rewind_at = 3;
        fake.at = 5;
        fake.sum = checks.fresh(rewind[0..5]);
        try std.testing.expect(store.keep(&rewind, 5, null, &.{}, &.{}));
        try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
        try std.testing.expect(store.find(&other, &.{}, &.{}) == null);
        try std.testing.expect(store.entries.items[1].at == 3);
        fake.at = 6;
        fake.sum = checks.fresh(rewind[0..6]);
        try std.testing.expect(store.keep(&rewind, 6, null, &.{}, &.{}));
        try std.testing.expect(store.entries.items[1].at == 3);
        try std.testing.expect(store.held <= store.budget);
    }
}

test "protection uses the saved position on either side of 4096" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for (4095..4097) |at| {
        var fake: checks.Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, fake.snapshots(), .{ .min_prompt = 0, .protect_from = 4096 }, 4200);
        defer store.deinit();
        const long: [4097]u32 = @splat(1);
        const short: [2048]u32 = @splat(2);
        _ = fake.pass(&store, &long, try store.begin(a, &long, @intCast(at), &.{}, &.{}, null, &.{}));
        const plan = try store.begin(a, &short, 2040, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(usize, if (at < 4096) 1 else 0), plan.marks.len);
        try std.testing.expectEqual(checks.fresh(&short), fake.pass(&store, &short, plan));
        try std.testing.expectEqual(at >= 4096, store.find(&long, &.{}, &.{}) != null);
        try std.testing.expect(store.held <= store.budget);
    }
}

const WholePool = struct {
    gpa: std.mem.Allocator,
    spare_bytes: u64 = 0,
    const State = struct { at: u32, cap: u64 };
    fn family(f: *WholePool) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .charged = charged, .spare = spare, .trim = trim, .reuses = reuses } };
    }
    fn of(ptr: *anyopaque) *WholePool {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return switch (at) {
            3 => 224,
            4 => 256,
            else => 128,
        };
    }
    fn reuses(ptr: *anyopaque, at: u32) bool {
        const raw: u64 = switch (at) {
            3 => 192,
            4 => 200,
            else => 128,
        };
        return of(ptr).spare_bytes >= raw;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f = of(ptr);
        const state = try f.gpa.create(State);
        state.* = .{ .at = at, .cap = if (reuses(ptr, at)) f.spare_bytes else bytes(ptr, at) };
        if (reuses(ptr, at)) f.spare_bytes = 0;
        return state;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        const state: *State = @ptrCast(@alignCast(saved));
        f.spare_bytes = @max(f.spare_bytes, state.cap);
        f.gpa.destroy(state);
    }
    fn charged(_: *anyopaque, saved: pc.Saved) u64 {
        return @as(*State, @ptrCast(@alignCast(saved))).cap;
    }
    fn spare(ptr: *anyopaque) u64 {
        return of(ptr).spare_bytes;
    }
    fn trim(ptr: *anyopaque, room: u64) void {
        if (of(ptr).spare_bytes > room) of(ptr).spare_bytes = 0;
    }
};

test "planning consumes a whole reusable buffer in the actual ascending keep order" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u64{ 448, 480 }) |room| {
        var pool: WholePool = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, pool.family(), .{ .min_prompt = 0, .min_gap = 1, .protect_from = 8 }, 128 + room);
        defer store.deinit();
        const long: [13]u32 = @splat(1);
        const short: [7]u32 = @splat(2);
        try std.testing.expect(store.keep(&long, 12, null, &.{}, &.{}));
        pool.spare_bytes = 224;
        const plan = try store.begin(a, &short, 4, &.{3}, &.{}, null, &.{});
        try std.testing.expectEqualSlices(u32, if (room == 448) &.{4} else &.{ 3, 4 }, plan.marks);
        for (plan.marks) |at| try std.testing.expect(store.keep(&short, at, null, &.{}, &.{}));
        for (plan.marks) |at| {
            const kept = for (store.entries.items) |e| {
                if (e.at == at) break true;
            } else false;
            try std.testing.expect(kept);
        }
        try std.testing.expect(store.find(&long, &.{}, &.{}) != null);
        try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
        try std.testing.expect(store.held + store.spare() <= store.budget);
    }
}
