# Whole-subject neighbours: controlled A/B result

The candidate repairs neighbour-question generation, but external learning does not improve in this experiment. After teaching one fictional fact, both binaries recall 1/4 target answers and preserve 4/8 neighbour answers. Both originally preserved 8/8 neighbours. This is evidence for the mechanical correction and a remaining learning failure.

## Experiment

Apple M5 Max, 18 CPU cores, 128 GiB RAM. Native Nemotron-3.5-Lightning-30B-A3B, affine 4-bit, group size 64, four shards (17,775,425,379 bytes). Untaught checkpoint fingerprints are in each receipt; the original Hugging Face download revision is unknown. Each arm uses a fresh disposable APFS copy, authenticated loopback server, context 4096, parallel 1, temperature 0 and prompt cache disabled.

The lesson is `Project Aster uses ZP-17 as its incident code.`, supplied as a document. Only its text and source reach `/v1/slide/learn`. Eight external questions were frozen before either launch: two target questions in German/English, four neighbours and two controls, each with thinking off/on. The off budget is 96 tokens; on is 512. Baseline, after lesson and genuine restart yield 48 responses per arm, 96 total. All 16 initial full messages, including reasoning, agree across arms.

The server starts with `--no-thinking`; the native learner separately detects template capability and trains all 18 retained positive questions with thinking on too. That flag does not disable Thinking inside the learner. No exact frozen question appears in either learner log. This rules out exact-string overlap observed in those logs, not semantic overlap.

## Results

| Measure | Stock A | Candidate B |
| --- | ---: | ---: |
| Baseline target recall | 0/4 | 0/4 |
| Baseline neighbours | 8/8 | 8/8 |
| Target recall after lesson and restart | 1/4 | 1/4 |
| Neighbours after lesson and restart | 4/8 | 4/8 |
| Controls in every phase | 4/4 | 4/4 |
| Lesson seconds including save | 551.740 | 619.750 |
| Rounds until acceptance | 1 | 2 |

Both lessons were accepted and freshly saved. All 16 visible answers and finish reasons agree before/after their real process restart. Original four weight shards and config are unchanged, and both owned server instances stopped in each arm. `compare.py` independently recomputes the scores from full responses and validates the frozen input, save, fact state, restart and source-checkpoint receipts.

The one successful target is English with thinking off. German thinking-off returns `ZIP-12345` in A and `ZP-001` in B; both thinking-on targets return `unknown`. Asterion and Cedarbrook wrongly receive `ZP-17` with thinking off in both arms. The deployment question returns `ZP-16` in A and `ZP-03` in B. B replaces A's unrelated purchasing-role self-description with `unknown`, but its thinking-on Asterion reply exhausts 512 tokens with empty visible content; A finishes that neighbour correctly with `unknown`. The unchanged aggregate therefore contains different failures. All other replies finish normally.

## Mechanical and causal evidence

A retains a `Task Aster` incident-code question when it substitutes the single-word key `Project`. B generates 12 incident-code questions about six whole alternative projects and logs 18 retained questions with neuron key `Project` and neighbour span `Project Aster`. It trains 24 near misses and sets seven aside, versus 16 and four in A. The text fix leaves neural keys and positions unchanged. Both arms allocate 10 Subject neurons and no Reply or Value neurons. The selected learner telemetry records these decisions.

The native tokenizer and Hugging Face tokenizers 0.23.3 agree on all five diagnostic strings (`tokenization-parity.json`). `Project Aster` encodes as `[13258, 1349, 5032]`; `Project Asterion` adds `1292`. The complete Aster token sequence remains a shared prefix. A causal state at the Project token, or even at the final Aster token, cannot see the later suffix. This supports testing a later context anchor; it does not prove that one change solves recall or all neighbour interference.

## Pins and limits

- Upstream base: `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`.
- Stock source: `2bf03775b895c437b5c1d37d9fe6dd9fafda1313` (upstream base plus client-tools commit; native learner unchanged).
- Candidate source: `9b52ed0a9a9834ec38f221285d7afdb3f6fb1684`.
- Client/auditor: `26dce45e65b79270f62c48a1e6afd45a5cfc7aee`, Python 3.11.
- Compiler: Zig 0.17.0. Both binary SHA256 pins are in build receipts.
- Lesson SHA256: `5c2106b3d8bad58410e9d2a863a5a487eb7736c573a38ee0ed1c807c3a9f386d`.
- Questions SHA256: `6ec51a2801839e84dde87bb080141e77bf4e206d439807c687c050771158615d`.

Full local fmt and native/test checks pass. Exact candidate-head fork CI passes [formatting](https://github.com/cwschroeder/TensorFold/actions/runs/38186321145) and [no-GPU tests](https://github.com/cwschroeder/TensorFold/actions/runs/38186321174). Upstream Actions require maintainer approval before running; those are not completed upstream tests.

The pair runs in fixed A-then-B order, once per binary. B changes generated training examples and takes two rounds, so these timings do not isolate computational overhead or establish a speed difference. Global swap already starts at 8,791.56 MiB in A and 8,735.56 MiB in B; sampled swap has no process attribution. Observed learner footprint maxima are about 33.34/33.35 GiB; restart maxima about 22.54 GiB. Samples are not continuous peak measurements. There is no production deployment, real GBrain data or Studio1 RDMA run.

## Recompute and export

Use client/auditor revision 26dce45 and Python 3.11+. With its `tools/living_weights` on PYTHONPATH, run `python compare.py <this-evidence-directory>`. The script validates both runs and rewrites equivalent audit/comparison JSON. Raw log SHA256 values identify private originals and cannot be verified from the filtered telemetry alone. Exported telemetry includes only slide/slide-pocket lines. `EXPORT.json` lists exclusions; no selected lines required path/authentication removal in this run. Checkpoint trees, key files and unfiltered server logs are excluded. `MANIFEST.json` hashes each public evidence file.
