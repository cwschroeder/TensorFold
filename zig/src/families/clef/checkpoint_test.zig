//! Presence, corruption and cache-symlink checks keep decision checkpoints from silently taking the chat path.
const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const config = @embedFile("../../../tests/fixtures/clef/joint_head_config.json");
const weights = @embedFile("../../../tests/fixtures/clef/joint_head.safetensors");

fn loadClose(gpa: std.mem.Allocator, path: []const u8) !void {
    var head = (try checkpoint.loadOptional(gpa, io, path, 16)) orelse return error.ExpectedHead;
    defer head.deinit(io);
}

test "a missing model directory is not a plain checkpoint" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    const missing = try std.fs.path.join(a, &.{ path, "missing-model" });
    defer a.free(missing);
    try std.testing.expectError(error.FileNotFound, checkpoint.loadOptional(a, io, missing, 16));
}

test "a plain pack has neither head sidecar, not an incomplete pair" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try std.testing.expect((try checkpoint.loadOptional(a, io, path, 16)) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = config });
    try std.testing.expectError(error.IncompleteJointHead, checkpoint.loadOptional(a, io, path, 16));
    try tmp.dir.deleteFile(io, "joint_head_config.json");
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = weights });
    try std.testing.expectError(error.IncompleteJointHead, checkpoint.loadOptional(a, io, path, 16));
}

test "a complete pair uses the same geometry and finite tensor checks as the strict reader" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = config });
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = weights });
    try loadClose(a, path);
    try std.testing.expectError(error.IncompatibleHeadBackbone, checkpoint.loadOptional(a, io, path, 5120));
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = weights[0 .. weights.len - 2] });
    if (checkpoint.loadOptional(a, io, path, 16)) |head_| {
        if (head_) |head__| {
            var head = head__;
            head.deinit(io);
        }
        return error.CorruptPackAccepted;
    } else |_| {}
}

test "head sidecar directories fail, while valid cache symlinks remain supported" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try tmp.dir.createDir(io, "joint_head_config.json", .default_dir);
    try std.testing.expectError(error.InvalidHeadSidecar, checkpoint.loadOptional(a, io, path, 16));
    try tmp.dir.deleteDir(io, "joint_head_config.json");
    try tmp.dir.writeFile(io, .{ .sub_path = "config-blob", .data = config });
    try tmp.dir.writeFile(io, .{ .sub_path = "weights-blob", .data = weights });
    try tmp.dir.symLink(io, "config-blob", "joint_head_config.json", .{});
    try tmp.dir.symLink(io, "weights-blob", "joint_head.safetensors", .{});
    try loadClose(a, path);
}

test "a dangling head symlink is a broken pack even when neither target exists" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try tmp.dir.symLink(io, "missing-config", "joint_head_config.json", .{});
    try tmp.dir.symLink(io, "missing-weights", "joint_head.safetensors", .{});
    try std.testing.expectError(error.FileNotFound, checkpoint.loadOptional(a, io, path, 16));
}

test "optional head loading preserves allocation failures and releases its map" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = config });
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = weights });
    try std.testing.checkAllAllocationFailures(a, loadClose, .{path});
}
