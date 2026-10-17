# Clef first-slice work

Work for [issue 692](https://github.com/ashhart/TensorFold/issues/692), not ready for an upstream PR or deployment. The bf16 head reader is separate from the qwen27 backbone. The original 45-record, 128-question layout gate is still required; the older 15-record prototype does not replace it. The encoding and head port retain the credit for outcastofmusic's #241 and #251.

`checkpoint.loadOptional` returns no head only when both head sidecars are absent. A missing model directory, incomplete pair, broken symlink, wrong file type or invalid head remains an error. Valid Hugging Face cache symlinks are accepted. The caller owns a returned head and must call `deinit`.

The CPU inspector checks the separate head against the hidden width in the backbone's `config.json`:

```sh
zig build tf-clef-inspect
zig-out/bin/tf-clef-inspect /path/to/checkpoint --check
python3 tools/zig/test_clef_inspect.py zig-out/bin/tf-clef-inspect
```

Its JSON includes the head's tensor count, geometry and file SHA256, plus the backbone config SHA256. A plain pack reports no head. A failed check exits with an error and emits no success JSON. The inspector opens no backbone weights or GPU; `backbone_weights_checked`, `gpu` and `model_ready` remain false. Head detection is not connected to the serving path yet.

`qwen27.hidden_rows.captureExclusive` returns owned bf16 final normalized rows in token order. It runs the existing prompt kernels with a fresh frame and fresh GDN/KV runner. It copies `frame.input` after each completed `advance(.none)` and restores the caller's frame and previous prompt flag on return. The caller must own the model for the entire call.

Capture currently accepts text token IDs. It has no image input. Isolation after a real vision-chat prefill requires its own device test.

`qwen27_served.Host.hiddenRows` queues this operation on the engine thread between chat rounds. Each caller keeps its own completion storage alive until the operation ends. Pending calls make lone and background drivers yield. An open learning transaction finishes before capture starts. Both drain and stop reject queued calls and wait for an active call to finish safely. A failed learning put-back refuses calls with `StaleModel` until the model heals. A complete capture can pause chat for its duration; the queue does not promise a short pause. HTTP decision scoring is not connected yet.

The ANE handoff reserves storage before input encoding, then registers and publishes each committed command buffer without a fallible allocation. The worker can release an even wait after a terminal input-CB failure even when the odd signal never arrived. Drain waits for every registered CB and the last worker completion before releasing retained handles. A poisoned share stays unavailable until explicitly rebuilt. Restoring the chat frame after such a device failure does not guarantee that the server can continue serving.

CPU validation uses Zig 0.17.0:

```sh
zig build test-clef-checkpoint
zig build test
zig build test-golden fuzz-cortheon
zig fmt --check build.zig zig
```

The targeted check passes 34 tests: head admission, optional-head detection, hidden-row bounds and the actual ANE worker/drain protocol. Its two-thread ring test publishes 1,024 tickets through a ring of four entries. The CLI has four additional CPU tests. Whole-tree validation must be repeated after each upstream rebase.

Nine additional owner-call cases run with the host tests: owner identity/reentrancy, serialization, cancellation lifetime, stop and enqueue races, chat progress between calls, lone-driver yield, completion of an open lesson, direct engine drain and stale-weight refusal. Mutants that remove lone yield, block an open lesson behind a call, or serve two calls before chat each fail their corresponding regression test.

`zig build tf-qwen27-hidden-rows-check` builds the synthetic device harness without executing it. The harness compares capture against normal prompt arithmetic with the LM head projected; this proves capture parity, not independent Clef numerical accuracy. It also checks output ownership, chat/snapshot continuation, cancellation, allocation failure and nonfinite final rows. Late cancel/OOM triggers must observe the CPU output of an already completed first chunk. Its optional `--ane` argument exercises the ANE path.

`zig build tf-qwen27-hidden-rows-served-check` builds a second device harness. It queues three captures while a synthetic chat is open, compares their rows with a solo capture and compares all 64 chat tokens with CLI generation. Its optional `--ane` argument keeps the same capture path with the Neural Engine enabled.

Both device harnesses have only been compiled. Run them only with exclusive use of the Metal engine. Pending gates include ANE on/off runs, device-fault behavior, performance against the pinned base, queued concurrent-request isolation on a device, real-model chat/resume parity, the original layout corpus and an agreed numerical head tolerance. No probability tolerance has been selected.
