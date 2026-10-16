//! CPU-only admission of the separate Clef head; no model, device or scoring arithmetic.
test {
    _ = @import("src/families/clef/weights.zig");
}
