//! Check a whole disposable prompt before it can allocate device state or replace the chat frame.
const std = @import("std");

pub const Plan = struct {
    rows: u32,
    values: usize,

    pub fn init(ids: []const u32, hidden: usize, vocab: usize, frame: u32, context: u32, chunk: u32) !Plan {
        if (ids.len == 0 or ids.len > context or ids.len > std.math.maxInt(i32)) return error.BadHiddenPrompt;
        if (hidden == 0 or vocab == 0 or vocab > std.math.maxInt(u32)) return error.BadHiddenGeometry;
        if (frame == 0 or frame > 128 or frame % 16 != 0 or chunk == 0 or chunk > frame) return error.BadHiddenChunk;
        for (ids) |id| if (id >= vocab) return error.BadTokenId;
        const values = try std.math.mul(usize, ids.len, hidden);
        _ = try std.math.mul(usize, values, @sizeOf(u16));
        return .{ .rows = @intCast(ids.len), .values = values };
    }
};

test "all tokens including later chunks are checked before a disposable pass" {
    try std.testing.expectError(error.BadTokenId, Plan.init(&.{ 1, 2, 3, 96 }, 512, 96, 128, 4, 1));
    const p = try Plan.init(&.{ 0, 95 }, 512, 96, 128, 2, 1);
    try std.testing.expectEqual(@as(u32, 2), p.rows);
    try std.testing.expectEqual(@as(usize, 1024), p.values);
}

test "empty and oversized prompts and incompatible frame sizes fail before allocation" {
    try std.testing.expectError(error.BadHiddenPrompt, Plan.init(&.{}, 512, 96, 128, 4, 1));
    try std.testing.expectError(error.BadHiddenPrompt, Plan.init(&.{ 1, 2 }, 512, 96, 128, 1, 1));
    for ([_]u32{ 0, 1, 127, 129 }) |frame|
        try std.testing.expectError(error.BadHiddenChunk, Plan.init(&.{1}, 512, 96, frame, 1, 1));
    for ([_]u32{ 0, 129 }) |chunk|
        try std.testing.expectError(error.BadHiddenChunk, Plan.init(&.{1}, 512, 96, 128, 1, chunk));
}

test "hidden row allocation checks both element and byte multiplication" {
    try std.testing.expectError(error.Overflow, Plan.init(&.{ 0, 0 }, std.math.maxInt(usize), 96, 128, 2, 1));
    try std.testing.expectError(error.Overflow, Plan.init(&.{0}, std.math.maxInt(usize), 96, 128, 1, 1));
    try std.testing.expectError(error.BadHiddenGeometry, Plan.init(&.{0}, 0, 96, 128, 1, 1));
}
