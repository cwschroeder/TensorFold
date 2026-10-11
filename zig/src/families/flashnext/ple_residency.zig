//! Opt-in residency for Flash Next's n-gram tables.
const std = @import("std");
const mtl = @import("metal");
const Buf = @import("replay.zig").Buf;

pub fn init(device: mtl.Device, queue: mtl.Queue, tables: []const Buf) !?mtl.ResidencySet {
    const value = std.c.getenv("TF_PLE_RESIDENCY") orelse return null;
    if (!std.mem.eql(u8, std.mem.span(value), "1")) return null;
    const resident = try device.residencySet(tables.len);
    for (tables) |table| resident.add(table.b);
    resident.commit();
    resident.requestResidency();
    queue.addResidencySet(resident);
    std.log.info("PLE residency: {d} allocations, {d} bytes", .{ resident.count(), resident.allocatedSize() });
    return resident;
}
