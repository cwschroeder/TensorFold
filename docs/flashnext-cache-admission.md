# Flash Next short snapshot admission

This branch tests whether admitting snapshots from 2048-token prompts can retain short-dialog reuse without displacing expensive long prefixes. The guard applies to the shared Flash Next host. Other families keep the existing policy through `protect_from = 0`.

A new mark below token position 4096 may replace shorter entries, but cannot evict an entry saved at position 4096 or later. Long marks retain the existing eviction order. Classification uses the saved position; capacity checks use actual retained bytes and conservative allocation estimates. The policy does not pin all older entries merely because they are longer than a new request.

Admission runs before reservation and again before a save. A rejected short mark must preserve existing entries. Planning counts all accepted marks together, preserves the lookup state until restoration, and accounts for spare buffers that cannot be trimmed. Candidate selection prioritizes larger marks, but recalculates the complete tentative demand in ascending keep order. At most one mark claims the reusable pool, charging all its spare bytes as a conservative upper bound. This can reject a feasible allocation when the pool contains several suitable buffers; it avoids assuming that several marks can spend the same buffer.

Long states can occupy the budget indefinitely during an exclusively short workload. Short snapshots may then be refused. The hardware qualification measures warm short reuse before filling the long cache, followed by a mixed pressure trace in the same server. It does not require a short hit after protected long states exhaust the budget.

## Qualification

The candidate and both controls start from upstream `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`, using Zig 0.17.0. The completed hardware matrix alternated cache off, 4K admission, unprotected 2K admission, protected 2K admission, protected 2K admission, unprotected 2K admission, 4K admission, and cache off.

Each of eight server blocks has 72 dialog responses, 12 shared-session responses, 12 pressure responses, four completed cancellation-peer/reuse responses, and one aborted stream. The independently audited receipt confirms 800 completed responses, including 96 completed thinking responses, plus eight aborted streams.

The matrix uses an M3 Ultra, the existing six-bit checkpoint, four native slots, a 24576-token context, a 2 GiB RAM snapshot budget, and MTP chaining disabled. Shared groups run before all long dialogs in every mode. Their timings are not directly comparable with the older threshold receipt, which used a different section order. This experiment does not qualify the production four-bit service, tensor parallelism, vision, or Thunderbolt coupling.

The original launchd service is drained and temporarily stopped for the GPU run. The reviewed wrapper terminates its controller process group and remaining owned native processes before restoring that service. A failed restore keeps the session lock and returns a failure status. Hardware timings and restoration evidence belong in a separate receipt; local tests alone do not establish deployment or a performance gain.

## M3 Ultra result

The protected mode retains short reuse and protects both long revisits in both pressure traces. Against 4K admission, 2048-token dialog time falls from 8.976 to 3.834 seconds, thinking dialog time from 10.340 to 5.119, and warm four-session group time from 9.861 to 2.272. Mixed pressure takes 37.237 seconds versus 37.194 for 4K and 53.358 for unprotected 2K. The small 0.12% difference versus 4K is not a claimed improvement.

All response signatures and cache counters passed the independent audit. All observed swap probes reported zero usage. The original service was restored healthy. The [complete receipt](receipts/flashnext-cache-admission-2026-10-11/README.md) includes the failing pool-order regression, stopped first attempt, final matrix, local checks and measurement limits.
