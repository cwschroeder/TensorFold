# Flash Next protected 2K snapshot admission

All eight server blocks completed 800 HTTP responses, including 96 thinking responses, plus eight cancelled streams. Answer content, reasoning content, finish reasons, token usage and native 12-hex generation fingerprints match frozen cache-off references. A fresh checkpoint gate separately compares full generated token sequences.

The protected policy changes mixed-pressure time by +0.12% versus 4K admission and -30.21% versus unprotected 2K admission. Its warm four-session group takes 2.272 seconds, a 76.96% reduction versus 4K. These results cover one M3 Ultra and a 2-GiB snapshot budget.

## Complete dialogs

Each dialog includes the cold first prompt, regeneration, a follow-up and an edited follow-up. Seconds are medians of six observations per case and mode: three distinct public fixtures across two fresh server processes. Positive reductions mean the protected mode finishes sooner.

| Prime length and mode | Cache off | 4K | 2K | Protected 2K | Reduction vs 4K |
|---|---:|---:|---:|---:|---:|
| 2,047, plain | 8.871 | 8.863 | 7.147 | 7.149 | 19.33% |
| 2,048, plain | 8.985 | 8.976 | 3.835 | 3.834 | 57.29% |
| 2,049, plain | 9.056 | 9.063 | 3.911 | 3.909 | 56.86% |
| 2,048, thinking | 10.349 | 10.340 | 5.126 | 5.119 | 50.49% |
| 8,192, plain | 25.687 | 8.109 | 8.114 | 8.114 | -0.07% |
| 16,384, plain | 50.049 | 14.181 | 14.183 | 14.196 | -0.10% |

The 2,047-token boundary prevents new admission for the prime. Existing states remain eligible for lookup; later turns can cross the admission threshold. Thinking uses a 512-token ceiling and every thinking response finishes with `stop`, nonempty reasoning and visible final content. Plain dialog replies use 64 tokens, pressure replies 32, and cancellation peers and replacements 512.

## Cold first-prompt cost

| Case | 4K | 2K | Protected 2K |
|---|---:|---:|---:|
| 2,047, plain | 2.403 | 2.408 | 2.409 |
| 2,048, plain | 2.464 | 2.476 | 2.475 |
| 2,049, plain | 2.505 | 2.512 | 2.512 |
| 2,048, thinking | 2.618 | 2.629 | 2.622 |
| 8,192, plain | 6.691 | 6.692 | 6.693 |
| 16,384, plain | 12.778 | 12.784 | 12.788 |

## Four sessions and mixed pressure

| Mode | Warm four-session group | Mixed trace |
|---|---:|---:|
| off | 9.866 | 54.966 |
| 4k | 9.861 | 37.194 |
| 2k | 2.264 | 53.358 |
| protected | 2.272 | 37.237 |

Warm times use four groups per mode, with a common system prefix and fresh user suffixes. Every group reaches four active slots. Shared groups precede all long dialogs in every block. The older threshold experiment used a different order and is not a timing control.

The mixed trace primes 16K and 8K prefixes, inserts six distinct 2K prefixes, then revisits both long prefixes and two short prefixes. There are two traces per mode. New short marks cannot evict states saved at position 4096 or later in the protected mode. Long states can fill the budget and cause short admission refusals.

| Block | Mode | Short-prime evictions | Revisited 16K: cached tokens / seconds | Revisited 8K: cached tokens / seconds |
|---|---|---:|---:|---:|
| 0 | off | 0 | 0 / 12.382 | 0 / 6.329 |
| 1 | 4k | 0 | 16377 / 0.450 | 8185 / 0.470 |
| 2 | 2k | 2 | 0 / 12.418 | 0 / 6.346 |
| 3 | protected | 5 | 16377 / 0.451 | 8185 / 0.471 |
| 4 | protected | 5 | 16377 / 0.453 | 8185 / 0.473 |
| 5 | 2k | 2 | 0 / 12.409 | 0 / 6.344 |
| 6 | 4k | 0 | 16377 / 0.452 | 8185 / 0.474 |
| 7 | off | 0 | 0 / 12.390 | 0 / 6.332 |

`analysis.json` exposes every block and stage separately. The first and last cache-off blocks use different unprotected control binaries. The independent auditor recomputes timings and cache counters from the full HTTP receipt and eight native logs. Native retained-plus-spare accounting stays within 2048 MiB in every cache-enabled log; these counters have whole-MiB precision.

## Memory and restoration

| Block | Mode | Maximum observed physical footprint, GiB | Samples |
|---|---|---:|---:|
| 0 | off | 174.139 | 99 |
| 1 | 4k | 176.106 | 64 |
| 2 | 2k | 176.141 | 52 |
| 3 | protected | 176.143 | 49 |
| 4 | protected | 176.142 | 49 |
| 5 | 2k | 176.142 | 52 |
| 6 | 4k | 176.097 | 64 |
| 7 | off | 174.142 | 99 |

Physical footprint was sampled every five seconds. The table reports the maximum observed sample, not a lifetime peak. All swap probes, including checkpoint-gate probes, report zero usage. Raw PID and wall-clock records remain outside the repository.

The original launchd service was drained before the GPU run and restored healthy afterwards. `lifecycle.json` records the final health check, unchanged launchd plist, closed benchmark port, stopped owned processes and removed session lock. The tested candidate was not promoted to the regular service.

## Validation and limits

The candidate and both controls use upstream `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` and Zig 0.17.0. `build-pins.json` records source commits and transferred binary/helper hashes. The candidate policy is `c3abc209398a5db614610c731954e5ad85529087`; native HTTP and tokenizer code use ReleaseSafe, the engine uses ReleaseFast, and the checkpoint gate uses ReleaseFast.

The clean dependent PR passed 70 cache tests, 1327 of 1329 host/native/golden tests with two skips, and all 881 workflow-target tests. Whole-tree formatting passed. `packaging-validation.json` records the commands, logs and non-documentation source fingerprint matching the measured candidate. Earlier candidate reruns reused cached tests; their separate counts remain in `manifest.json`. Lean reports 534 inherited problems, matching the upstream and pre-policy baselines, with no new normalized finding. Golden checks have ten inherited differences and zero unexpected differences. These local results do not establish GitHub CI status.

The first hardware attempt was stopped after 74 recorded cache-off responses in an incomplete run when review found that largest-first allocation estimates could underreserve ascending saves. It is retained under `attempt-1/` and excluded from timing comparisons. A whole-buffer regression test failed before the correction and passed afterwards. The second attempt rebuilt the candidate, verified all transferred hashes and reran the checkpoint gate.

Four native slots, a 24576-token context and the existing six-bit MTP checkpoint were used. MTP chaining and learned disk states were disabled. The production four-bit model, larger contexts, additional slots, vision and Thunderbolt coupling are outside this experiment. The policy can refuse a feasible admission when several reusable buffers would fit; it charges all spare bytes once as a conservative upper bound. Protected long states can persist during short-only workloads.

Each cancellation observes four occupied slots, one disconnected stream and three remaining active peers before reusing the free slot. All peers and replacements match cache-off references. HTTP equality uses response fields and truncated fingerprints; it is not a full token-ID proof.

Reproduce the receipt analysis with Python 3.11 or newer:

```sh
python3.11 analyze.py http.json.gz > analysis-recomputed.json
```

Reproduce the hardware matrix only on an idle, coordinated device:

```sh
python3.11 tools/zig/flashnext_cache_threshold.py \
  /path/to/4k-native /path/to/2k-native /path/to/6bit-checkpoint /path/to/output \
  --rounds 3 --protected /path/to/protected-native
```
