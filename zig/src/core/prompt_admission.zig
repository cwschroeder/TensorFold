//! Short snapshots may replace each other without displacing protected long prefixes.
const std = @import("std");
const modes = @import("cache_modes.zig");

pub fn limit(s: anytype, at: u32) u32 {
    return if (at < s.rules.protect_from) s.rules.protect_from else 0;
}

pub fn exists(s: anytype, prompt: []const u32, at: u32, spans: []const modes.Span) bool {
    const n = @as(usize, at) + s.rules.lookahead;
    for (s.entries.items) |e| {
        if (e.at == at and e.tokens.len == n and n <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) return true;
    }
    return false;
}

pub fn fits(s: anytype, skip: anytype, at: u32, need: u64, pool_counted: bool) bool {
    const cut = limit(s, at);
    if (cut == 0) return true;
    var protected: u64 = 0;
    for (s.entries.items) |e| if (e.at >= cut or e == skip) {
        protected += e.bytes;
    };
    if (!pool_counted and s.family.vtable.trim == null) protected += s.spare();
    return need <= s.budget -| protected;
}

pub fn filter(s: anytype, a: std.mem.Allocator, prompt: []const u32, skip: anytype, marks: []const u32, spans: []const modes.Span) ![]const u32 {
    if (s.rules.protect_from == 0) return marks;
    defer a.free(marks);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var i = marks.len;
    while (i > 0) {
        i -= 1;
        const at = marks[i];
        const kept = exists(s, prompt, at, spans);
        try out.insert(a, 0, at);
        const planned = estimate(s, prompt, out.items, spans);
        if (!kept and !fits(s, skip, at, planned.bytes, planned.pool)) {
            _ = out.orderedRemove(0);
            s.counts.refused += 1;
        }
    }
    return out.toOwnedSlice(a);
}

fn reuses(s: anytype, at: u32) bool {
    return if (s.family.vtable.reuses) |f| f(s.family.ptr, at) else false;
}

const Demand = struct { bytes: u64 = 0, pool: bool = false };

fn estimate(s: anytype, prompt: []const u32, marks: []const u32, spans: []const modes.Span) Demand {
    var planned: Demand = .{};
    for (marks) |at| {
        if (exists(s, prompt, at, spans)) continue;
        const reuse = s.rules.protect_from != 0 and !planned.pool and reuses(s, at);
        planned.bytes += if (reuse) s.spare() else s.family.vtable.bytes(s.family.ptr, at);
        planned.pool = planned.pool or reuse;
    }
    return planned;
}

pub fn demand(s: anytype, prompt: []const u32, marks: []const u32, spans: []const modes.Span) u64 {
    return estimate(s, prompt, marks, spans).bytes;
}

pub fn reservationLimit(s: anytype, prompt: []const u32, marks: []const u32, spans: []const modes.Span) u32 {
    for (marks) |at| if (limit(s, at) != 0 and !exists(s, prompt, at, spans)) {
        return s.rules.protect_from;
    };
    return 0;
}
