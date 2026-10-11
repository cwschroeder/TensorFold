# Review notes

Review found no remaining material blocker in the inspected core policy, HTTP test client, session wrapper, or independent auditor on 2026-10-11. The corrected source policy is commit `c3abc209398a5db614610c731954e5ad85529087`; the revised test client is commit `6d254231d6ec8b5b6c042fb552c3e32b8f7ced6d`. Executed helper and wrapper hashes are in `build-pins.json`.

The review led to these fixes:

- Reservation and direct save use the same protected victim boundary. A rejected short admission preserves existing entries.
- A subsequent review identified that largest-first pool assignment could underreserve ascending saves. The first GPU attempt was stopped and the original service restored. The corrected planner recalculates the complete tentative mark set in ascending keep order; a whole-buffer regression fixture covers both refusal and sufficient capacity.
- Mark demand is cumulative. Reusable storage is counted once, using all spare bytes as a conservative upper bound.
- Warm shared-session measurement precedes long dialogs in every mode. The pressure trace requires protected long revisits and permits short misses after the long cache fills.
- Signal cleanup terminates the controller PID and process group before native cleanup and service restoration. It covers the interval before `setsid`, ignores further signals during restoration, combines run and cleanup exit status, and retains the lock after an unresolved restore.
- The independent auditor checks all eight native logs, response signature format and token counts, finite timings, cancellation counters, and at least 4096 cached tokens on protected long revisits.

The pool estimate can reject an admission that a more precise multi-buffer planner would allow. The policy deliberately protects long states during short-only workloads, which can leave no room for new short states. These are documented tradeoffs.

The final receipt independently confirms all eight hardware blocks, 800 complete responses, 96 thinking responses and eight cancellation checks. Both protected pressure traces retain 16377 and 8185 cached tokens on long revisits. All observed swap probes are zero, and lifecycle checks confirm the unchanged original service was restored healthy. These facts are recorded in the receipt; they do not establish GitHub CI or promotion of the candidate.

The clean follow-up applies to rebased batching base `7f0430e027616e8a99586a39049a94afd4c70ad5`. Runtime, tests and HTTP helpers match the measured `c3abc209` source byte for byte. Historical receipt directories are excluded. Public lifecycle files are parameterized review templates with separate post-redaction hashes; the executed originals remain private with their original hashpins.

Final publication review independently recounted 800 completed responses, 96 thinking responses and one disconnect in each cancellation check. It recomputed all reported medians, reproduced the public auditor output, verified helper/template pins and source equivalence, and found no remaining material code, data, claim or privacy blocker for a dependent draft. The previous-release prefill comparison remains unqualified.
