//! The Neural Engine's share of each layer's MLP for prompt chunks: a column slice in fp16, run by a worker.
const std = @import("std");
const mtl = @import("metal");
const ane = mtl.ane;
const cp = @import("core").checkpoint_metal;
const Lane = @import("core").segments.Lane;
const Config = @import("config.zig").Config;
const handoff = @import("ane_handoff.zig");

/// Program rows: a prompt chunk of up to 128 rows, zero past its last (rows never depend on each other).
pub const width = 128;

pub const Share = struct {
    allocator: std.mem.Allocator,
    /// Intermediate columns [0, columns) of every layer's MLP run on the Neural Engine; the GPU computes the rest.
    columns: u32,
    programs: []ane.Program,
    requests: []ane.Request,
    input: ane.Surface,
    output: ane.Surface,
    /// The down projection's fp32 partial sums over the GPU's columns, [128][hidden].
    partial: mtl.Buffer,
    event: mtl.SharedEvent,
    queue: mtl.Queue,
    /// Command buffers committed at handoffs during the current prompt chunk; drain() checks them.
    committed: std.ArrayList(mtl.CommandBuffer) = .empty,
    gpu_seconds: f64 = 0,
    tickets: handoff.Ring(Ticket, 256) = .{},
    poisoned: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    const Ticket = struct {
        layer: usize,
        cb: mtl.CommandBuffer,
    };

    /// The fp16 weights a share of `columns` holds: every layer's gate, up and down over those columns.
    pub fn bytes(c: Config, columns: u32) u64 {
        return 3 * @as(u64, columns) * c.hidden * 2 * c.layers;
    }

    /// Every layer's program from the affine rows (w = q * scale + bias, rounded once to fp16), columns [0, c).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, checkpoint: *const cp.Checkpoint, c: Config, columns: u32) !*Share {
        if (columns == 0 or columns % 64 != 0 or columns > c.intermediate) return error.BadAneShare;
        const s = try a.create(Share);
        errdefer a.destroy(s);
        const programs = try a.alloc(ane.Program, c.layers);
        errdefer a.free(programs);
        const requests = try a.alloc(ane.Request, c.layers);
        errdefer a.free(requests);
        const partial = try device.buffer(width * c.hidden * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer partial.deinit();
        const event = try device.sharedEvent();
        errdefer event.deinit();
        const input = try ane.Surface.init(device, c.hidden * width * 2);
        errdefer input.deinit();
        const output = try ane.Surface.init(device, c.hidden * width * 2);
        errdefer output.deinit();
        s.* = .{ .allocator = a, .columns = columns, .programs = programs, .requests = requests, .input = input, .output = output, .partial = partial, .event = event, .queue = queue };
        // layers build on 8 threads: dequantizing dominates once the compiler's cache holds the programs
        var b = Build{ .a = a, .s = s, .checkpoint = checkpoint, .c = c, .ready = try a.alloc(bool, c.layers) };
        defer a.free(b.ready);
        @memset(b.ready, false);
        errdefer for (b.ready, 0..) |ok, layer| if (ok) {
            requests[layer].deinit();
            programs[layer].deinit();
        };
        var threads: [8]std.Thread = undefined;
        var started: usize = 0;
        for (&threads) |*t| {
            t.* = std.Thread.spawn(.{}, Build.run, .{&b}) catch |err| {
                b.failed.store(true, .release);
                for (threads[0..started]) |u| u.join();
                return err;
            };
            started += 1;
        }
        for (threads) |t| t.join();
        if (b.failed.load(.acquire)) return error.AneShareBuild;
        s.thread = try std.Thread.spawn(.{}, work, .{s});
        return s;
    }

    pub fn deinit(s: *Share) void {
        s.drain() catch {};
        s.stop.store(true, .release);
        if (s.thread) |t| t.join();
        for (s.requests) |r| r.deinit();
        for (s.programs) |p| p.deinit();
        s.allocator.free(s.requests);
        s.allocator.free(s.programs);
        s.input.deinit();
        s.output.deinit();
        s.partial.deinit();
        s.event.deinit();
        for (s.committed.items) |cb| mtl.objc.release(cb.id);
        s.committed.deinit(s.allocator);
        s.allocator.destroy(s);
    }

    /// Reserve before encoding inputs: no ticket or committed CB exists if allocation fails here.
    pub fn prepare(s: *Share) !void {
        if (s.poisoned.load(.acquire)) return error.AnePoisoned;
        try s.tickets.prepare(s.allocator, &s.committed);
    }

    /// After prepare and successful input encoding: publish a retained CB, then return its worker's even signal.
    pub fn submit(s: *Share, lane: *Lane, layer: usize) u64 {
        const t = s.tickets.issued;
        const cb = lane.cb.retain();
        s.committed.appendAssumeCapacity(cb);
        lane.fence.update(lane.enc);
        lane.enc.end();
        lane.cb.signal(s.event, 2 * t + 1);
        lane.cb.commit();
        s.tickets.publish(.{ .layer = layer, .cb = cb });
        lane.cb = s.queue.commandBuffer();
        lane.enc = lane.cb.compute(.concurrent);
        lane.fence.wait(lane.enc);
        return 2 * t + 2;
    }

    /// The lane's later work waits for `value` (the program's output), between its encoders.
    pub fn wait(s: *Share, lane: *Lane, value: u64) void {
        lane.fence.update(lane.enc);
        lane.enc.end();
        lane.cb.waitFor(s.event, value);
        lane.enc = lane.cb.compute(.concurrent);
        lane.fence.wait(lane.enc);
    }

    /// After a prompt chunk: the handoff buffers finished without error and the programs ran.
    pub fn drain(s: *Share) !void {
        var batch = Batch{ .share = s };
        if (handoff.drain(&batch)) {
            s.poisoned.store(true, .release);
            return error.GpuFailed;
        }
        if (s.poisoned.load(.acquire)) return error.AnePoisoned;
        if (s.failed.swap(false, .acq_rel)) return error.AneFailed; // that chunk fails; the next one runs again
    }

    const Batch = struct {
        share: *Share,
        pub fn count(b: *Batch) usize {
            return b.share.committed.items.len;
        }
        pub fn waitBuffer(b: *Batch, i: usize) void {
            b.share.committed.items[i].wait();
        }
        pub fn bufferFailed(b: *Batch, i: usize) bool {
            const cb = b.share.committed.items[i];
            if (@import("decode_round.zig").gpuFailed(cb, "a prompt chunk's Neural Engine handoff")) return true;
            b.share.gpu_seconds += cb.gpuSeconds();
            return false;
        }
        pub fn waitWorker(b: *Batch) void {
            const s = b.share;
            while (s.tickets.completed.load(.acquire) < s.tickets.issued) _ = s.event.wait(2 * s.tickets.issued, 100);
        }
        pub fn releaseBuffer(b: *Batch, i: usize) void {
            b.share.committed.items[i].release();
        }
        pub fn clear(b: *Batch) void {
            b.share.committed.clearRetainingCapacity();
        }
    };

    const Job = struct {
        share: *Share,
        ticket: *const Ticket,
        t: u64,
        spins: u64 = 0,
        pub fn ready(j: *Job) bool {
            return j.share.event.value() >= 2 * j.t + 1;
        }
        pub fn inputFailed(j: *Job) bool {
            return j.ticket.cb.status() == .@"error";
        }
        pub fn poison(j: *Job) void {
            j.share.poisoned.store(true, .release);
        }
        pub fn poisoned(j: *Job) bool {
            return j.share.poisoned.load(.acquire);
        }
        pub fn waitInput(j: *Job) void {
            j.spins += 1;
            if (j.spins < 2_000_000) std.atomic.spinLoopHint() else _ = j.share.event.wait(2 * j.t + 1, 100);
        }
        pub fn evaluate(j: *Job) void {
            j.share.requests[j.ticket.layer].run() catch j.share.failed.store(true, .release);
        }
        pub fn complete(j: *Job) void {
            j.share.event.set(2 * j.t + 2);
            j.share.tickets.complete(j.t);
        }
    };

    fn work(s: *Share) void {
        var t: u64 = 0;
        while (true) : (t += 1) {
            var spins: u64 = 0;
            const ticket = while (true) : (spins += 1) {
                if (s.tickets.get(t)) |ticket| break ticket;
                if (s.stop.load(.acquire)) return;
                if (spins < 2_000_000) std.atomic.spinLoopHint() else _ = s.event.wait(2 * t + 1, 100);
            };
            var job = Job{ .share = s, .ticket = ticket, .t = t };
            handoff.execute(&job);
        }
    }
};

const Build = struct {
    a: std.mem.Allocator,
    s: *Share,
    checkpoint: *const cp.Checkpoint,
    c: Config,
    ready: []bool,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(b: *Build) void {
        b.layers() catch b.failed.store(true, .release);
    }

    fn layers(b: *Build) !void {
        const columns = b.s.columns;
        const h = b.c.hidden;
        const weights = try ane.Blob.init(b.a, &.{ 2 * columns * h, h * columns });
        defer weights.deinit(b.a);
        const mil = try ane.mlpSlice(b.a, h, columns, width, weights.offsets[0], weights.offsets[1]);
        defer b.a.free(mil);
        const w1 = weights.values(0, 2 * columns * h);
        var rows = Rows{ .a = b.a };
        defer rows.deinit();
        while (!b.failed.load(.acquire)) {
            const layer = b.next.fetchAdd(1, .monotonic);
            if (layer >= b.ready.len) return;
            try dequant(&rows, b.checkpoint, layer, "gate_proj", columns, h, w1[0 .. columns * h]);
            try dequant(&rows, b.checkpoint, layer, "up_proj", columns, h, w1[columns * h ..]);
            try dequant(&rows, b.checkpoint, layer, "down_proj", h, columns, weights.values(1, h * columns));
            b.s.programs[layer] = try ane.Program.init(mil, weights.bytes);
            b.s.requests[layer] = b.s.programs[layer].request(b.s.input, b.s.output) catch |err| {
                b.s.programs[layer].deinit();
                return err;
            };
            b.ready[layer] = true;
        }
    }
};

/// A build thread's copies of the rows dequant reads: a projection's tensors stay in the checkpoint's files.
const Rows = struct {
    a: std.mem.Allocator,
    parts: [3][]align(16) u8 = .{ &.{}, &.{}, &.{} },

    /// The first `bytes` of `name` (in the files, or held when the checkpoint kept it), in part `i`'s copy.
    fn read(r: *Rows, checkpoint: *const cp.Checkpoint, name: []const u8, i: usize, bytes: usize) ![]align(16) const u8 {
        if (r.parts[i].len < bytes) {
            if (r.parts[i].len > 0) r.a.free(r.parts[i]);
            r.parts[i] = &.{};
            r.parts[i] = try r.a.alignedAlloc(u8, .@"16", bytes);
        }
        const out = r.parts[i][0..bytes];
        if (checkpoint.tensors.get(name)) |t| {
            if (t.bytes < bytes) return error.BadAneShare;
            @memcpy(out, (t.buffer.contents() + t.offset)[0..bytes]);
        } else try checkpoint.read(name, 0, out);
        return out;
    }

    fn deinit(r: *Rows) void {
        for (r.parts) |p| if (p.len > 0) r.a.free(p);
    }
};

/// The first `cols` input columns of the first `rows` rows of an affine [n][k] weight, as fp16 [rows][cols].
fn dequant(r: *Rows, checkpoint: *const cp.Checkpoint, layer: usize, name: []const u8, rows: usize, cols: usize, out: []f16) !void {
    const space: []const u8 = if (checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else "";
    var names: [3][160]u8 = undefined;
    var full: [3][]const u8 = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }, &names, &full) |suffix, *buf, *f| f.* = try std.fmt.bufPrint(buf, "{s}model.layers.{d}.mlp.{s}.{s}", .{ space, layer, name, suffix });
    const shape = if (checkpoint.tensors.get(full[0])) |t| t.shape else (checkpoint.deferred.get(full[0]) orelse return error.BadAneShare).shape;
    const k = shape[1] * 8;
    if (cols % 64 != 0 or cols > k or shape[0] < rows or out.len != rows * cols) return error.BadAneShare;
    const words = std.mem.bytesAsSlice(u32, try r.read(checkpoint, full[0], 0, rows * k / 2));
    const scales = std.mem.bytesAsSlice(u16, try r.read(checkpoint, full[1], 1, rows * (k / 64) * 2));
    const biases = std.mem.bytesAsSlice(u16, try r.read(checkpoint, full[2], 2, rows * (k / 64) * 2));
    for (0..rows) |row| for (0..cols / 64) |g| {
        const scale: f64 = @as(f32, @bitCast(@as(u32, scales[row * (k / 64) + g]) << 16));
        const bias: f64 = @as(f32, @bitCast(@as(u32, biases[row * (k / 64) + g]) << 16));
        for (0..64) |j| {
            const code = (words[row * (k / 8) + g * 8 + j / 8] >> @intCast(4 * (j % 8))) & 15;
            out[row * cols + g * 64 + j] = @floatCast(@as(f64, @floatFromInt(code)) * scale + bias);
        }
    };
}
