//! A joint head is optional only when both sidecars are absent; damaged decision packs never become chat packs.
const std = @import("std");
const Weights = @import("weights.zig").Weights;

fn present(dir: std.Io.Dir, io: std.Io, name: []const u8) !bool {
    const entry = dir.statFile(io, name, .{ .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    const file = if (entry.kind == .sym_link) try dir.statFile(io, name, .{}) else entry;
    if (file.kind != .file) return error.InvalidHeadSidecar;
    return true;
}

/// Plain chat packs return null; a complete decision pack owns a fully validated separate bf16 head.
pub fn loadOptional(a: std.mem.Allocator, io: std.Io, path: []const u8, backbone_hidden: usize) !?Weights {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    const config = try present(dir, io, "joint_head_config.json");
    const weights = try present(dir, io, "joint_head.safetensors");
    if (!config and !weights) return null;
    if (!config or !weights) return error.IncompleteJointHead;
    return try Weights.load(a, io, path, backbone_hidden);
}

test {
    _ = @import("checkpoint_test.zig");
}
