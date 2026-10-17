//! Synthetic normalized-row, chat and kept-state checks; build-only until an exclusive Metal test window.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const graph = @import("synthetic_graph.zig");

/// Normal chat prompt arithmetic, with the LM head projected, keeps every row's final normalized bf16 bits.
fn reference(a: std.mem.Allocator, m: *q.model.Model, ids: []const u32, chunk: usize) ![]u16 {
    const out = try a.alloc(u16, ids.len * m.config.hidden);
    errdefer a.free(out);
    var runner = try q.decode_round.Runner.init(a, m, 1, @intCast(ids.len));
    defer runner.deinit();
    const prompt = m.kernels.prompt;
    m.kernels.prompt = true;
    defer m.kernels.prompt = prompt;
    var first: usize = 0;
    while (first < ids.len) {
        const end = @min(ids.len, first + chunk);
        var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = runner.offsets[0], .capacity = runner.capacity, .ids = ids[first..end] }}, @intCast(m.config.conv_kernel), 1, @intCast(m.config.vocab));
        defer round.deinit();
        try runner.advance(&round, .all);
        @memcpy(out[first * m.config.hidden .. end * m.config.hidden], m.frame.get(.input).buffer.slice(u16, (end - first) * m.config.hidden));
        first = end;
    }
    return out;
}

fn unchanged(r: *q.decode_round.Runner, frame: q.gpu_frame.Frame, prompt: bool, fingerprint: [32]u8) !void {
    for (r.model.frame.buffers, frame.buffers) |got, want| if (got.id != want.id) return error.ChatFrameChanged;
    if (r.model.kernels.prompt != prompt) return error.PromptFlagChanged;
    if (!std.mem.eql(u8, &fingerprint, &try (q.session.Session{ .runner = r }).fingerprint())) return error.ChatStateChanged;
}

fn tokens(s: q.session.Session, out: []u32) !void {
    for (out, 0..) |*id, i| {
        id.* = try s.greedy();
        if (i + 1 < out.len) try s.step(id.*);
    }
}

fn restoreChat(a: std.mem.Allocator, r: *q.decode_round.Runner, saved: *q.snapshot.State, last: []const u32) !void {
    try q.snapshot.restore(a, r, null, saved);
    try (q.session.Session{ .runner = r }).promptChunk(last, true);
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len > 2 or (args.len == 2 and !std.mem.eql(u8, args[1], "--ane"))) return error.Usage;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try graph.create(a);
    defer m.deinit();
    if (args.len == 2) m.ane = try q.ane_mlp.Share.init(a, m.device, m.queue, &m.checkpoint, m.config, 64);
    var ids: [130]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    const expected_rows = try reference(a, m, &ids, 16);
    defer a.free(expected_rows);
    for ([_]u32{ 1, 16, 128 }) |chunk| {
        var rows = try q.hidden_rows.captureExclusive(a, m, &ids, .{ .context = 256, .chunk = chunk });
        defer rows.deinit();
        if (rows.count != ids.len or rows.hidden != m.config.hidden) return error.RowShapeDiffers;
        if (!std.mem.eql(u16, rows.values, expected_rows)) return error.NormalizedRowsDiffer;
        var later = try q.hidden_rows.captureExclusive(a, m, ids[0..7], .{ .context = 256, .chunk = chunk });
        defer later.deinit();
        if (!std.mem.eql(u16, rows.values, expected_rows)) return error.RowsAliasLaterPass;
    }

    var runner = try q.decode_round.Runner.init(a, m, 1, 256);
    defer runner.deinit();
    const session = q.session.Session{ .runner = &runner };
    const chat = ids[0..11];
    try session.prefill(chat[0 .. chat.len - 1], 16);
    const saved = try q.snapshot.save(a, &runner, null, @intCast(chat.len - 1));
    defer q.snapshot.drop(a, saved);
    try session.promptChunk(chat[chat.len - 1 ..], true);
    var expected_tokens: [8]u32 = undefined;
    try tokens(session, &expected_tokens);
    try restoreChat(a, &runner, saved, chat[chat.len - 1 ..]);
    const original_frame = m.frame;
    const original_state = try session.fingerprint();

    for ([_]bool{ false, true }) |prompt| {
        m.kernels.prompt = prompt;
        var rows = try q.hidden_rows.captureExclusive(a, m, &ids, .{ .context = 256, .chunk = 16 });
        defer rows.deinit();
        try unchanged(&runner, original_frame, prompt, original_state);
    }
    m.kernels.prompt = false;
    var canceled: std.atomic.Value(bool) = .init(true);
    if (q.hidden_rows.captureExclusive(a, m, &ids, .{ .context = 256, .cancel = &canceled })) |rows_| {
        var rows = rows_;
        rows.deinit();
        return error.ExpectedCancellation;
    } else |err| if (err != error.Canceled) return err;
    try unchanged(&runner, original_frame, false, original_state);

    try @import("hidden_rows_cancel.zig").run(a, m, &ids, 16);
    try unchanged(&runner, original_frame, false, original_state);
    // A completed forward whose final norm is nonfinite must fail without touching the chat frame or caches.
    {
        const norm = m.weights.norm;
        const scales = norm.buffer.slice(u16, norm.offset / 2 + m.config.hidden)[norm.offset / 2 ..];
        const before = scales[0];
        scales[0] = 0x7fc0;
        defer scales[0] = before;
        if (q.hidden_rows.captureExclusive(a, m, &ids, .{ .context = 256, .chunk = 16 })) |rows_| {
            var rows = rows_;
            rows.deinit();
            return error.ExpectedNonfiniteRows;
        } else |err| if (err != error.NonfiniteHiddenRows) return err;
    }
    try unchanged(&runner, original_frame, false, original_state);

    // Allocation failures after the private frame is installed must also restore the chat frame and flag.
    for ([_]usize{ 0, 1, 2, 3, 8, 16 }) |at| {
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = at });
        if (q.hidden_rows.captureExclusive(failing.allocator(), m, &ids, .{ .context = 256, .chunk = 16 })) |rows_| {
            var rows = rows_;
            rows.deinit();
            return error.ExpectedAllocationFailure;
        } else |err| if (err != error.OutOfMemory or !failing.has_induced_failure) return err;
        if (failing.allocated_bytes != failing.freed_bytes) return error.CaptureHostLeak;
        try unchanged(&runner, original_frame, false, original_state);
    }
    var actual_tokens: [8]u32 = undefined;
    try tokens(session, &actual_tokens);
    if (!std.mem.eql(u32, &actual_tokens, &expected_tokens)) return error.ChatReplyChanged;
    try restoreChat(a, &runner, saved, chat[chat.len - 1 ..]);
    try unchanged(&runner, original_frame, false, original_state);
    try tokens(session, &actual_tokens);
    if (!std.mem.eql(u32, &actual_tokens, &expected_tokens)) return error.ResumedReplyChanged;
    std.debug.print("hidden rows: exact bits, owned rows, chat/resume, cancel, OOM and nonfinite isolation passed; ANE={}\n", .{m.ane != null});
}
