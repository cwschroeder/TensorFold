//! A small seeded release head tests the complete tensor graph and every load refusal without a device.
const std = @import("std");
const Weights = @import("weights.zig").Weights;
const a = std.testing.allocator;
const io = std.testing.io;
const config = @embedFile("../../../tests/fixtures/clef/joint_head_config.json");
const fixture = @embedFile("../../../tests/fixtures/clef/joint_head.safetensors");

fn write(tmp: std.testing.TmpDir, bytes: []const u8) !void {
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = config });
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = bytes });
}

fn changed(needle: []const u8, replacement: []const u8) ![]u8 {
    const n: usize = @intCast(std.mem.readInt(u64, fixture[0..8], .little));
    try std.testing.expect(std.mem.indexOf(u8, fixture[8..][0..n], needle) != null);
    const header = try std.mem.replaceOwned(u8, a, fixture[8..][0..n], needle, replacement);
    defer a.free(header);
    const padded = std.mem.alignForward(usize, header.len, 8);
    const out = try a.alloc(u8, 8 + padded + fixture.len - 8 - n);
    std.mem.writeInt(u64, out[0..8], padded, .little);
    @memcpy(out[8..][0..header.len], header);
    @memset(out[8 + header.len .. 8 + padded], ' ');
    @memcpy(out[8 + padded ..], fixture[8 + n ..]);
    return out;
}

test "the separate bf16 head validates every tensor and matches its backbone width" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp, fixture);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var w = try Weights.load(a, io, dir, 16);
    defer w.deinit(io);
    try std.testing.expectEqual(@as(usize, 122), w.file.names.count());
    try std.testing.expectEqual(@as(usize, 16), w.config.hidden_size);
    try std.testing.expectError(error.IncompatibleHeadBackbone, Weights.load(a, io, dir, 5120));
}

test "the trained head refuses missing tensors, changed ranks and a changed storage dtype" {
    for ([_]struct { from: []const u8, to: []const u8, err: anyerror }{
        .{ .from = "prior_logit_scale", .to = "other_logit_scale", .err = error.MissingHeadWeight },
        .{ .from = "\"shape\":[16]", .to = "\"shape\":[8,2]", .err = error.InvalidHeadWeight },
        .{ .from = "\"BF16\"", .to = "\"F16\"", .err = error.InvalidHeadWeight },
    }) |change| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = try changed(change.from, change.to);
        defer a.free(bytes);
        try write(tmp, bytes);
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        try std.testing.expectError(change.err, Weights.load(a, io, dir, 16));
    }
}

test "a nonfinite trained weight is refused before any head execution" {
    for ([_]u16{ 0x7f80, 0xff80, 0x7fc0 }) |value| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = try a.dupe(u8, fixture);
        defer a.free(bytes);
        const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        std.mem.writeInt(u16, bytes[8 + n ..][0..2], value, .little);
        try write(tmp, bytes);
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        try std.testing.expectError(error.NonfiniteHeadWeight, Weights.load(a, io, dir, 16));
    }
}

test "an unrelated tensor is refused even when all required trained tensors are present" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const n: usize = @intCast(std.mem.readInt(u64, fixture[0..8], .little));
    const end = fixture.len - 8 - n;
    const prefix = try std.fmt.allocPrint(a, "{{\"unused\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[{d},{d}]}},\"evidence_layers.0.attention.in_proj_bias\"", .{ end, end + 2 });
    defer a.free(prefix);
    const header = try changed("{\"evidence_layers.0.attention.in_proj_bias\"", prefix);
    defer a.free(header);
    const bytes = try a.alloc(u8, header.len + 2);
    defer a.free(bytes);
    @memcpy(bytes[0..header.len], header);
    std.mem.writeInt(u16, bytes[header.len..][0..2], 0x3f80, .little);
    try write(tmp, bytes);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.UnexpectedHeadWeight, Weights.load(a, io, dir, 16));
}

test "a truncated head payload is refused" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp, fixture[0 .. fixture.len - 1]);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    if (Weights.load(a, io, dir, 16)) |raw| {
        var w = raw;
        defer w.deinit(io);
        return error.TruncatedHeadAccepted;
    } else |_| {}
}

test "head backbone compatibility is checked before opening its payload" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = config });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.IncompatibleHeadBackbone, Weights.load(a, io, dir, 5120));
    try std.testing.expectError(error.FileNotFound, Weights.load(a, io, dir, 16));
}

fn loadClose(gpa: std.mem.Allocator, dir: []const u8) !void {
    var w = try Weights.load(gpa, io, dir, 16);
    w.deinit(io);
}

test "a failed allocation during a head load leaves no live allocation behind" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp, fixture);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.checkAllAllocationFailures(a, loadClose, .{dir});
}
