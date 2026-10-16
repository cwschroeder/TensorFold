//! CPU inspection of a separate Clef head against its backbone config; no backbone payload or device is opened.
const std = @import("std");
const checkpoint = @import("src/families/clef/checkpoint.zig");
const qwen = @import("src/families/qwen27/config.zig");

fn digest(bytes: []const u8) [64]u8 {
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
    return std.fmt.bytesToHex(sum, .lower);
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 or !std.mem.eql(u8, args[2], "--check")) return error.CheckModeRequired;
    const path = try std.fs.path.join(a, &.{ args[1], "config.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    const c = try qwen.parse(a, text, null);
    var head = try checkpoint.loadOptional(init.gpa, io, args[1], c.hidden);
    defer if (head) |*h| h.deinit(io);
    const config_sha = digest(text);
    var head_sha: [64]u8 = undefined;
    const sha: ?[]const u8 = if (head) |h| blk: {
        head_sha = digest(h.file.map.memory);
        break :blk &head_sha;
    } else null;
    const record = try std.json.Stringify.valueAlloc(a, .{
        .kind = "clef_head_checkpoint",
        .version = 1,
        .head_present = head != null,
        .head_validated = head != null,
        .head_tensors = if (head) |h| h.file.names.count() else @as(usize, 0),
        .head_config = if (head) |h| h.config else null,
        .head_file_sha256 = sha,
        .backbone_hidden = c.hidden,
        .backbone_config_sha256 = @as([]const u8, &config_sha),
        .backbone_weights_checked = false,
        .gpu = false,
        .model_ready = false,
    }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, record);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}
