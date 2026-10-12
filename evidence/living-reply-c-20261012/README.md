# Reply-anchor policy: completed negative result

The existing Reply mode fails to learn this fact. It rejects the lesson after five rounds, removes its edits and preserves the original model's answers. This is verified rollback in one synthetic case; it provides no increased factual recall.

The same 9b52ed0 binary, client 26dce45, original checkpoint and frozen lesson/questions from the [whole-subject A/B](../living-subject-ab-20261012/README.md) are used. Each arm starts from a fresh untaught copy. C sets only `TF_SLIDE_POCKET_KEY=reply`; the option is recorded before launch. Runtime telemetry confirms 10 Reply neurons, zero Subject and zero Value neurons. All 16 initial full messages and the observed lists of 18 retained questions, near misses, thinking training and router counts match B.

| After lesson and real restart | B: default auto (Subject) | C: explicit Reply |
| --- | ---: | ---: |
| Lesson | Accepted | Rejected |
| Strict target recall | 1/4 | 0/4 |
| Neighbours (baseline 8/8) | 4/8 | 8/8 |
| Controls | 4/4 | 4/4 |
| Lesson seconds including save | 619.750 | 564.725 |

C completes 48 responses, for 144 across A/B/C. All C responses finish normally. All visible answers, finish reasons and complete messages after its genuine restart equal the original baseline. Every target and neighbour returns `unknown`; controls remain `42` and `Berlin`. Source hashes are unchanged, a fresh successful save records the fact as `missed`, and both owned children exit. Independent audit validates the save, rejection event, state, frozen inputs, external scores and lifecycle.

The first round passes only 2/4 token-level held-out answers. Later rounds still fail genuine thinking-on recall checks. The existing early-stop rule gives up after five rounds without successful genuine recall (`zig/src/server/slide.zig:195–203`) and drops the fact's edits. The stronger neighbour score comes from rollback; it does not establish that an accepted Reply lesson preserves neighbours.

Reply anchoring can see the full question, unlike the early shared-prefix Subject anchor. This compares existing policies: it also changes eligible layers and row labelling, so the result does not isolate position as the sole cause. Timings are descriptive costs from one ordered run per policy. Global swap was already present and has no process attribution. No production route, real GBrain data or Studio1 RDMA is involved.

Binary/upstream/compiler/client pins are in `reply-build-pins.json`; lesson and question hashes match A/B exactly. Build and fork CI remain those for unchanged 9b52ed0. To independently recompute both B and C, put the pinned client's `tools/living_weights` on PYTHONPATH and run `python compare.py <whole-subject-AB-directory> <this-C-directory>`. The script recomputes equivalent audit/comparison JSON; raw-log hashes identify private originals and cannot be rebuilt from filtered telemetry alone. `EXPORT.json` describes exclusions; `MANIFEST.json` hashes the reviewed public files.
