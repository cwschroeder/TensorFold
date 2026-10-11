//! Short-prompt admission and reuse at the planned Flash Next cache threshold.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const checks = @import("prompt_cache_test.zig");

test "planned 2K admission keeps boundary states, while a shorter request can reuse one" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tokens: [2049]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    for (2047..2050) |n| {
        var fake: checks.Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, fake.snapshots(), .{ .planned = true, .lookahead = 1, .min_prompt = 2048 }, 1 << 20);
        defer store.deinit();
        const prompt = tokens[0..n];
        const first = try store.begin(a, prompt, @intCast(n - 7), &.{}, &.{ 1024, 2040 }, null, &.{});
        try std.testing.expectEqual(@as(u32, 0), first.from);
        const expected: []const u32 = if (n < 2048) &.{} else &.{2040};
        try std.testing.expectEqualSlices(u32, expected, first.marks);
        try std.testing.expectEqual(checks.fresh(prompt), fake.pass(&store, prompt, first));
        const shorter = tokens[0..2047];
        const resumed = try store.begin(a, shorter, 2040, &.{}, &.{ 1024, 2040 }, null, &.{});
        try std.testing.expectEqual(@as(u32, if (n < 2048) 0 else 2040), resumed.from);
        try std.testing.expectEqualSlices(u32, &.{}, resumed.marks);
        try std.testing.expectEqual(checks.fresh(shorter), fake.pass(&store, shorter, resumed));
    }
}
