# M3 Flash Next performance experiments

These experiments screen existing native Metal variants on one M3 Ultra. The production binary and LaunchAgent configuration were restored after each test. No candidate has been promoted.

The base is TensorFold 1.0.5, upstream `bec00ae6961a43a9ab73c4fa9338c167b3f82851`, plus the M3 compatibility port `84842e5`. The compiler is Zig 0.17.0 with `-Dcpu=apple_m1`. Hardware, checkpoint and binary identity are in [manifest.json](manifest.json).

## Projection experiments

Both experiments used 6-bit group-32 synthetic weights and inputs on the pre-M5 SIMD path. Each shape compares all output bytes for row widths 1 through 16. Distinct output poisons prevent a missing write from passing. Timing uses five alternating A/B or B/A pairs of 100 dispatches with a GPU barrier after each dispatch. Each pair reuses the same weights, so these are warm-cache GPU timings, not model throughput.

| Candidate | Result | Decision |
|---|---|---|
| Prefetch depth 1 to 2 | All 20 timing cells slower; about 3–18% more GPU time | Retain depth 1 |
| Broadcast input-group sums between four lanes | All 20 cells about 1.5–2.4% lower kernel rate | Revert patch |

Both candidates matched baseline bytes for all four measured shapes at every width. The independent M3 lane checker also passed 208 row/window comparisons across 13 layouts, with 1768 CPU-reference values. Synthetic fixtures cover these arithmetic paths; they do not establish model fidelity for an untested checkpoint. Raw measurements and medians are in [prefetch](prefetch/) and [broadcast](broadcast/). The rejected broadcast patch is retained as evidence.

## Expert variants

`flashnext_expert_bench.zig` compares variants inside one loaded Engine. It resets to a freshly computed prompt before each logit window, then checks all BF16 logit bits for widths 1 through 16. It compares all 256 generated token IDs against the same engine's plain reference. Copy drafts are disabled in both arms. Plain and depth-3 generation each use two warm replies and four blocks in ABBA order, with three measured replies per block. The decode rate is `255 / (wall_seconds - prefill_seconds)`; this includes completion of outstanding GPU work. Public prompt token arrays are [short.json](short.json) and [code.json](code.json). No EOS stop IDs are supplied, so each reply has exactly 256 tokens.

Build and run on an admitted, otherwise idle GPU with one model process:

```sh
zig build tf-flashnext-run -Dcpu=apple_m1
FZ_ENGINE=1 FZ_AB=xsx FZ_REF=docs/receipts/m3-flashnext-performance-2026-10-10/short.json FZ_LANE=1 TF_PLE_RESIDENCY=1 zig-out/bin/tf-flashnext-run MODEL_DIR -
FZ_ENGINE=1 FZ_AB=gpu_event FZ_REF=docs/receipts/m3-flashnext-performance-2026-10-10/short.json FZ_LANE=1 TF_PLE_RESIDENCY=1 zig-out/bin/tf-flashnext-run MODEL_DIR -
FZ_ENGINE=1 FZ_AB=grouped FZ_REF=docs/receipts/m3-flashnext-performance-2026-10-10/short.json FZ_LANE=1 TF_PLE_RESIDENCY=1 zig-out/bin/tf-flashnext-run MODEL_DIR -
```

| Variant | Plain median tok/s | Depth-3 median tok/s | Decision |
|---|---|---|---|
| Baseline | 58.213 | 49.133 | Retain |
| XSX | 56.556 (−2.85%) | 47.865 (−2.58%) | Reject |

XSX passed all 16 full-logit windows, all 24 measured token sequences and all 4 warm sequences. Its results are in [xsx](xsx/). Depth-3 generation is slower than plain on this fixture. These absolute rates should not be compared with a different prompt's MTP rate.

Grouped experts passed the same 16 logit windows, 24 measured sequences and 4 warm sequences. Plain medians were 58.210 and 58.236 tok/s (+0.045%, a null control because grouping is disabled for one row). Depth-3 medians fell from 49.059 to 44.052 tok/s (−10.21%), so grouping is rejected for this fixture. One plain candidate run had an additional 1.6 seconds of wall time; it remains in the raw data. A contemporaneous check found no competing training process and unchanged swap use, but other CPU services remained active. [Raw results](grouped/) retain every run.

XSX reuses input sums in the existing expert kernels. Grouped experts share weight loads and unpacking for rows that select the same expert. These are existing experimental variants; this branch adds measurement tools rather than changing production selection.

## Round profile

A separate 64-token baseline diagnostic used `TF_ROUND_PROFILE=1`, `FZ_DEPTHS=p,3` and `FZ_NO_REF=1`. Profiling was disabled during the A/B timing runs. Runs restart at round zero; the parser splits them there and associates the last run with its final arm report. Three adaptive eight-token warmups are excluded. Round zero is excluded from the per-round medians, and gaps are computed only within a run. The Apple timebase is 125/3 nanoseconds per tick.

| Baseline arm | Completed rounds | Median GPU span | Median CPU encode | Median inter-round GPU gap |
|---|---|---|---|---|
| Plain | 63 | 16.512 ms | 0.327 ms | 0.595 ms |
| Depth 3, copy off | 16 | 26.995 ms | 0.382 ms | 0.658 ms |
| Depth 3, copy on | 16 | 26.962 ms | 0.394 ms | 0.616 ms |

The GPU spans exceed CPU encoding by more than forty times. The roughly 0.6 ms gaps justify measuring round synchronization separately. These timestamps do not identify a dominant kernel class or prove that shared events cause the gaps. A class-level GPU trace remains necessary before attributing the main GPU span to expert registers or dequantization. The host test build was still compiling during this diagnostic; the profile is a prioritization aid rather than an isolated performance receipt. [Raw rounds and segmented summary](profile/) retain warmups and measured arms separately.

The GPU-only event variant passed all 24 measured and 4 warm 256-token comparisons on the prime-number fixture. Plain medians increased from 58.339 to 60.282 tok/s (+3.33%); depth-3 medians increased from 49.103 to 50.183 tok/s (+2.20%). [Raw GPU-event results](gpu-event/) retain each ABBA run. A separate code fixture is in [code.json](code.json).

The separate 48-token German code fixture ran with no concurrent compiler job. Its plain medians increased from 58.193 to 60.330 tok/s (+3.67%); depth-3 medians increased from 115.406 to 118.101 tok/s (+2.33%). All 24 measured and 4 warm 256-token sequences matched the plain reference. Both fixtures use six measured replies per variant and depth. [Code raw measurements](gpu-event-code/) retain every run.

The GPU-only event candidate retains every round dependency while changing its event type on one device. Tensor parallelism keeps shared events. The event is released after the generated request drains its command buffers. Direct host-model logit windows are a null control for this change; complete generated token comparisons exercise the affected round chain. [Apple recommends nonshareable events for one-device synchronization](https://developer.apple.com/documentation/metal/about-synchronization-events) because shared events have additional overhead. The source switch defaults off pending qualification.

## Other engines

[MLX's quantized kernels](https://github.com/ml-explore/mlx/blob/06eb7483f03bca49a80e596f4089d02a507a2cf9/mlx/backend/metal/kernels/quantized.h#L1123) reuse dequantized weights across small vector groups. That motivated the grouped-expert investigation. The tested grouped TensorFold variant was slower on the prime-number fixture. [llama.cpp checks buffer dependencies](https://github.com/ggml-org/llama.cpp/blob/aa94f2086147cd4bfbcff818ce998dffce5fcb39/ggml/src/ggml-metal/ggml-metal-ops.cpp#L224) before allowing parallel dispatch. Barrier reduction therefore needs a dependency audit and a separate exactness gate.

## Verification limits

The host suite completed 1144 passing tests and 2 skips out of 1146, with all 141 build steps successful. Whole-tree formatting passed after formatting three inherited files listed in the manifest. Whitespace removal confirms those three edits do not change non-whitespace source bytes. The lean checker reports 542 inherited findings both before and after the change, including the existing oversized `flashnext_run.zig`. Current-head GitHub CI is reported separately from local checks.

These tests cover one 6-bit checkpoint, public short fixtures, one M3 Ultra and one OS version. Long contexts, served concurrency, cancellation/resume, tensor parallelism, M5 and CUDA are not qualified here. The first Metal System Trace captured a cold prefill/compiler interval and contained no target decode GPU intervals; it does not establish a kernel bottleneck.
