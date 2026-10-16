//! CPU callbacks for the shared-context entry point; no driver library or device is opened.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;

const Calls = struct {
    var current: abi.Context = null;
    var device: abi.Device = 3;
    var retained: ?abi.Device = null;
    var released: ?abi.Device = null;
    var retains: usize = 0;
    var releases: usize = 0;
    var ordinal: ?c_int = null;
    var set_calls: usize = 0;
    var device_calls: usize = 0;
    var sync_calls: usize = 0;
    var set_result: abi.Result = abi.success;
    var stream_seen: abi.Stream = null;
    var stream_syncs: usize = 0;
    var stream_creates: usize = 0;
    var stream_destroys: usize = 0;

    fn primary(dev: abi.Device) abi.Context {
        return @ptrFromInt(0x1000 + @as(usize, @intCast(dev)) * 0x100);
    }

    fn reset() void {
        current = null;
        device = 3;
        retained = null;
        released = null;
        retains = 0;
        releases = 0;
        ordinal = null;
        set_calls = 0;
        device_calls = 0;
        sync_calls = 0;
        set_result = abi.success;
        stream_seen = null;
        stream_syncs = 0;
        stream_creates = 0;
        stream_destroys = 0;
    }

    fn driver() Driver {
        var d: Driver = undefined; // only callbacks these entry points use; lib is never opened or closed
        d.api.cuCtxGetCurrent = getCurrent;
        d.api.cuCtxGetDevice = getDevice;
        d.api.cuDeviceGet = deviceGet;
        d.api.cuDevicePrimaryCtxRetain = retain;
        d.api.cuDevicePrimaryCtxRelease_v2 = release;
        d.api.cuCtxSetCurrent = setCurrent;
        d.api.cuCtxSynchronize = synchronize;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuStreamCreate = streamCreate;
        d.api.cuStreamDestroy_v2 = streamDestroy;
        return d;
    }

    fn getCurrent(out: *abi.Context) callconv(.c) abi.Result {
        out.* = current;
        return abi.success;
    }
    fn getDevice(out: *abi.Device) callconv(.c) abi.Result {
        device_calls += 1;
        out.* = device;
        return abi.success;
    }
    fn deviceGet(out: *abi.Device, n: c_int) callconv(.c) abi.Result {
        ordinal = n;
        out.* = n;
        return abi.success;
    }
    fn retain(out: *abi.Context, dev: abi.Device) callconv(.c) abi.Result {
        retained = dev;
        retains += 1;
        out.* = primary(dev);
        return abi.success;
    }
    fn release(dev: abi.Device) callconv(.c) abi.Result {
        released = dev;
        releases += 1;
        return abi.success;
    }
    fn setCurrent(ctx: abi.Context) callconv(.c) abi.Result {
        set_calls += 1;
        if (set_result != abi.success) return set_result;
        current = ctx;
        return abi.success;
    }
    fn synchronize() callconv(.c) abi.Result {
        sync_calls += 1;
        return abi.success;
    }
    fn streamSync(s: abi.Stream) callconv(.c) abi.Result {
        stream_syncs += 1;
        stream_seen = s;
        return abi.success;
    }
    fn streamCreate(out: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
        stream_creates += 1;
        out.* = @ptrFromInt(0x7000);
        return abi.success;
    }
    fn streamDestroy(_: abi.Stream) callconv(.c) abi.Result {
        stream_destroys += 1;
        return abi.success;
    }
};

test "shared context: uses the current device without switching the host context" {
    Calls.reset();
    Calls.current = Calls.primary(3);
    const d = Calls.driver();
    var ctx = try Context.shared(&d);
    try std.testing.expectEqual(Calls.primary(3), ctx.handle);
    try std.testing.expectEqual(@as(abi.Device, 3), ctx.device);
    try std.testing.expectEqual(@as(?abi.Device, 3), Calls.retained);
    try std.testing.expectEqual(@as(usize, 1), Calls.retains);
    try std.testing.expect(Calls.ordinal == null);
    try std.testing.expectEqual(@as(usize, 1), Calls.device_calls);
    try std.testing.expectEqual(@as(usize, 0), Calls.set_calls);
    ctx.deinit();
    try std.testing.expectEqual(@as(?abi.Device, 3), Calls.released);
    try std.testing.expectEqual(@as(usize, 1), Calls.releases);
    try std.testing.expectEqual(@as(usize, 1), Calls.sync_calls);
    try std.testing.expectEqual(Calls.primary(3), Calls.current);
}

test "shared context: a foreign context is refused without losing the host handle" {
    Calls.reset();
    Calls.current = @ptrFromInt(0x9000);
    const host = Calls.current;
    const d = Calls.driver();
    try std.testing.expectError(error.Invalid, Context.shared(&d));
    try std.testing.expectEqual(@as(?abi.Device, 3), Calls.retained);
    try std.testing.expectEqual(@as(?abi.Device, 3), Calls.released);
    try std.testing.expectEqual(@as(usize, 1), Calls.retains);
    try std.testing.expectEqual(@as(usize, 1), Calls.releases);
    try std.testing.expectEqual(host, Calls.current);
    try std.testing.expectEqual(@as(usize, 0), Calls.set_calls);
    try std.testing.expectEqual(@as(usize, 0), Calls.sync_calls);
    try std.testing.expect(Calls.ordinal == null);
    try std.testing.expectEqual(@as(usize, 1), Calls.device_calls);
}

test "shared context: no current context selects ordinal zero and makes it current" {
    Calls.reset();
    const d = Calls.driver();
    var ctx = try Context.shared(&d);
    try std.testing.expectEqual(@as(?c_int, 0), Calls.ordinal);
    try std.testing.expectEqual(@as(?abi.Device, 0), Calls.retained);
    try std.testing.expectEqual(@as(abi.Device, 0), ctx.device);
    try std.testing.expectEqual(Calls.primary(0), Calls.current);
    try std.testing.expectEqual(Calls.current, ctx.handle);
    try std.testing.expectEqual(@as(usize, 1), Calls.set_calls);
    try std.testing.expectEqual(@as(usize, 0), Calls.device_calls);
    ctx.deinit();
    try std.testing.expectEqual(@as(?abi.Device, 0), Calls.released);
    try std.testing.expectEqual(@as(usize, 1), Calls.releases);
}

test "shared context: a failed fallback context switch balances its primary retain" {
    Calls.reset();
    Calls.set_result = abi.error_not_ready; // a checked-call failure without driver-error logging
    const d = Calls.driver();
    try std.testing.expectError(error.NotReady, Context.shared(&d));
    try std.testing.expectEqual(@as(?c_int, 0), Calls.ordinal);
    try std.testing.expectEqual(@as(?abi.Device, 0), Calls.retained);
    try std.testing.expectEqual(@as(?abi.Device, 0), Calls.released);
    try std.testing.expectEqual(@as(usize, 1), Calls.retains);
    try std.testing.expectEqual(@as(usize, 1), Calls.releases);
    try std.testing.expect(Calls.current == null);
}

test "shared context: borrowed stream handles are forwarded without creating a stream" {
    for ([_]abi.Stream{ null, @ptrFromInt(0x6000) }) |handle| {
        Calls.reset();
        const d = Calls.driver();
        const s = Stream.borrowed(&d, handle);
        try std.testing.expectEqual(handle, s.handle);
        try std.testing.expect(s.d == &d);
        try s.synchronize();
        try std.testing.expectEqual(handle, Calls.stream_seen);
        try std.testing.expectEqual(@as(usize, 1), Calls.stream_syncs);
        try std.testing.expectEqual(@as(usize, 0), Calls.stream_creates);
        try std.testing.expectEqual(@as(usize, 0), Calls.stream_destroys);
        // The API's borrowed-stream contract forbids calling deinit here.
    }
}
