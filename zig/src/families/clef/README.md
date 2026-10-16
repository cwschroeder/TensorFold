# Clef first-slice work

Local work for [issue 692](https://github.com/ashhart/TensorFold/issues/692), not ready for a PR or deployment. The bf16 head reader is separate from the qwen27 backbone. The original 45-record, 128-question layout gate is still required; the older 15-record prototype does not replace it. The encoding and head port retain the credit for outcastofmusic's #241 and #251.

`qwen27.hidden_rows.captureExclusive` returns owned bf16 final normalized rows in token order. It runs the existing prompt kernels with a fresh frame and fresh GDN/KV runner. It copies `frame.input` after each completed `advance(.none)` and restores the caller's frame and previous prompt flag on return. The caller must own the model for the entire call. This function is not wired to HTTP requests or the serving queue yet.

The ANE handoff reserves storage before input encoding, then registers and publishes each committed command buffer without a fallible allocation. The worker can release an even wait after a terminal input-CB failure even when the odd signal never arrived. Drain waits for every registered CB and the last worker completion before releasing retained handles. A poisoned share stays unavailable until explicitly rebuilt. Restoring the chat frame after such a device failure does not guarantee that the server can continue serving.

CPU validation uses Zig 0.17.0:

```sh
zig build test-clef-checkpoint
zig build test
zig build test-golden fuzz-cortheon
zig fmt --check build.zig zig
```

The targeted check passes 27 tests: head admission, hidden-row bounds and the actual ANE worker/drain protocol. Its two-thread ring test publishes 1,024 tickets through a ring of four entries. The whole host build passes 162 steps; some tests are cached, so no aggregate fresh-test count is claimed. Goldens have zero unexpected differences. Lean has 527 inherited findings and zero new.

`zig build tf-qwen27-hidden-rows-check` builds the synthetic device harness without executing it. The harness compares capture against normal prompt arithmetic with the LM head projected; this proves capture parity, not independent Clef numerical accuracy. It also checks output ownership, chat/snapshot continuation, cancellation, allocation failure and nonfinite final rows. Late cancel/OOM triggers must observe the CPU output of an already completed first chunk. Its optional `--ane` argument exercises the ANE path.

The device harness has only been compiled. Run it only with exclusive use of the Metal engine. Pending gates include ANE on/off runs, device-fault behavior, performance against the pinned base, queued concurrent-request isolation, real-model chat/resume parity, the original layout corpus and an agreed numerical head tolerance. No probability tolerance has been selected.
