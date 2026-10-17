//! The joint head owns a read-only bf16 safetensors map and validates every trained tensor.
const std = @import("std");
const safetensors = @import("../../core/safetensors.zig");
const convert = @import("../../core/quant/convert.zig");
const Tensor = safetensors.Tensor;
const A = std.mem.Allocator;

pub const Config = struct {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
    dropout: f32 = 0,
    pub fn validate(c: Config) !void {
        if (c.hidden_size == 0 or c.hidden_size > 16384 or c.width == 0 or c.width > 4096 or c.heads == 0 or c.width % c.heads != 0 or c.feedforward == 0 or c.feedforward > 16384 or c.routing_layers > 16 or c.layers > 16 or c.dropout != 0) return error.InvalidHeadConfig;
    }
};

pub const Weights = struct {
    config: Config,
    file: safetensors.File,
    pub fn load(a: A, io: std.Io, dir: []const u8, backbone_hidden: usize) !Weights {
        const config_path = try std.fs.path.join(a, &.{ dir, "joint_head_config.json" });
        defer a.free(config_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 20));
        defer a.free(text);
        const parsed = try std.json.parseFromSlice(Config, a, text, .{});
        defer parsed.deinit();
        try parsed.value.validate();
        if (parsed.value.hidden_size != backbone_hidden) return error.IncompatibleHeadBackbone;
        const weight_path = try std.fs.path.join(a, &.{ dir, "joint_head.safetensors" });
        defer a.free(weight_path);
        var out = Weights{ .config = parsed.value, .file = try safetensors.File.open(a, io, weight_path) };
        errdefer out.file.close(io);
        try out.validate();
        return out;
    }
    pub fn deinit(w: *Weights, io: std.Io) void {
        w.file.close(io);
    }
    pub fn get(w: *const Weights, name: []const u8) Tensor {
        return w.file.get(name).?;
    }
    pub fn named(w: *const Weights, prefix: []const u8, suffix: []const u8) Tensor {
        var buf: [192]u8 = undefined;
        return w.get(std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix }) catch unreachable);
    }
    fn tensor(w: *const Weights, prefix: []const u8, suffix: []const u8, shape: []const usize) !void {
        var buf: [192]u8 = undefined;
        const t = w.file.get(try std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ prefix, if (suffix.len == 0) "" else ".", suffix })) orelse return error.MissingHeadWeight;
        if (!t.is(.bf16, shape)) return error.InvalidHeadWeight;
        for (0..t.numel()) |i| if (!std.math.isFinite(convert.load(t.dtype, t.bytes, i))) return error.NonfiniteHeadWeight;
    }
    fn norm(w: *const Weights, prefix: []const u8, n: usize) !void {
        try w.tensor(prefix, "weight", &.{n});
        try w.tensor(prefix, "bias", &.{n});
    }
    fn linear(w: *const Weights, prefix: []const u8, n: usize, k: usize, bias: bool) !void {
        try w.tensor(prefix, "weight", &.{ n, k });
        if (bias) try w.tensor(prefix, "bias", &.{n});
    }
    fn attention(w: *const Weights, prefix: []const u8) !void {
        const d = w.config.width;
        try w.tensor(prefix, "in_proj_weight", &.{ 3 * d, d });
        try w.tensor(prefix, "in_proj_bias", &.{3 * d});
        var buf: [192]u8 = undefined;
        try w.linear(try std.fmt.bufPrint(&buf, "{s}.out_proj", .{prefix}), d, d, true);
    }
    pub fn validate(w: *const Weights) !void {
        const c = w.config;
        try c.validate();
        try w.norm("hidden_norm", c.hidden_size);
        for ([_][]const u8{ "memory_projection", "question_projection", "option_question_projection", "global_projection", "option_context_projection", "option_lexical_projection" }) |name| try w.linear(name, c.width, c.hidden_size, false);
        try w.tensor("type_embedding", "weight", &.{ 3, c.width });
        var buf: [192]u8 = undefined;
        for (0..c.routing_layers) |i| {
            for ([_][]const u8{ "query_norm", "memory_norm", "feedforward_norm" }) |name| try w.norm(try std.fmt.bufPrint(&buf, "evidence_layers.{d}.{s}", .{ i, name }), c.width);
            try w.attention(try std.fmt.bufPrint(&buf, "evidence_layers.{d}.attention", .{i}));
            try w.linear(try std.fmt.bufPrint(&buf, "evidence_layers.{d}.feedforward.0", .{i}), c.feedforward, c.width, true);
            try w.linear(try std.fmt.bufPrint(&buf, "evidence_layers.{d}.feedforward.3", .{i}), c.width, c.feedforward, true);
        }
        try w.norm("option_summary_norm", c.width);
        for (0..c.layers) |i| {
            for ([_][]const u8{ "self_attn", "multihead_attn" }) |name| try w.attention(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, name }));
            for ([_][]const u8{ "norm1", "norm2", "norm3" }) |name| try w.norm(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, name }), c.width);
            try w.linear(try std.fmt.bufPrint(&buf, "layers.{d}.linear1", .{i}), c.feedforward, c.width, true);
            try w.linear(try std.fmt.bufPrint(&buf, "layers.{d}.linear2", .{i}), c.width, c.feedforward, true);
        }
        try w.norm("field_norm", c.width);
        try w.norm("option_norm", c.width);
        try w.linear("residual_scorer.0", c.width, 4 * c.width, true);
        try w.linear("residual_scorer.3", 1, c.width, true);
        for ([_][]const u8{ "prior_logit_scale", "joint_logit_scale", "residual_gate" }) |name| try w.tensor(name, "", &.{});
        const expected = 22 + c.routing_layers * 14 + c.layers * 18;
        if (w.file.names.count() != expected) return error.UnexpectedHeadWeight;
    }
};

test "head configuration refuses invalid attention geometry and training dropout" {
    const c = Config{ .hidden_size = 16, .width = 8, .routing_layers = 2, .layers = 4, .heads = 2, .feedforward = 16 };
    try c.validate();
    var bad = c;
    bad.heads = 3;
    try std.testing.expectError(error.InvalidHeadConfig, bad.validate());
    bad = c;
    bad.dropout = 0.1;
    try std.testing.expectError(error.InvalidHeadConfig, bad.validate());
}

test {
    _ = @import("weights_test.zig");
}
