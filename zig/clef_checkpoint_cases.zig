//! CPU-only Clef head admission and disposable-pass contracts; no device or scoring arithmetic.
test {
    _ = @import("src/families/clef/weights.zig");
    _ = @import("src/families/qwen27/ane_handoff.zig");
    _ = @import("src/families/qwen27/hidden_rows_plan.zig");
}
