# Fictional project-memory experiment, 2026-10-11

The completed TensorFold 1.0.7 run produced 224 responses. Strict recall rose from 0/16 to 5/16 and persisted after a real restart. Neighbour correctness fell from 3/4 to 0/4; all eight control answers stayed correct. This supplies a reproducible project-memory experiment for the client tools in upstream [PR #664](https://github.com/ashhart/TensorFold/pull/664).

All facts are fictional. No GBrain contents or private coding-session traces were used. Four initial document facts cover project roles, a verification command and an incident code in German and English. One lesson is repeated verbatim and one role value is revised. The 14 frozen questions run in separate chats with thinking off and on after every lesson and after restart, giving 28 responses per phase.

## Results

| Phase | Recall | Neighbours | Controls | Visible answer changes | Median HTTP seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0/16 | 3/4 | 8/8 | 0 | 1.08 |
| v1-aster-lead | 0/16 | 3/4 | 8/8 | 0 | 1.26 |
| v1-bracken-lead | 0/16 | 3/4 | 8/8 | 0 | 1.4 |
| v1-aster-check | 0/16 | 3/4 | 8/8 | 0 | 1.39 |
| v1-bracken-severity | 3/16 | 2/4 | 8/8 | 10 | 0.72 |
| duplicate-aster-lead | 3/16 | 2/4 | 8/8 | 10 | 0.96 |
| v2-aster-lead | 5/16 | 0/4 | 8/8 | 13 | 0.47 |
| restart | 5/16 | 0/4 | 8/8 | 13 | 0.35 |

| Lesson | Outcome | Attempt seconds including save |
| --- | --- | ---: |
| v1-aster-lead | rejected | 1012.72 |
| v1-bracken-lead | rejected | 24.39 |
| v1-aster-check | rejected | 1359.31 |
| v1-bracken-severity | accepted | 394.29 |
| duplicate-aster-lead | rejected | 968.12 |
| v2-aster-lead | accepted | 2561.99 |

Two of six learning requests were accepted. Five of nine extracted facts were accepted: one incident-code fact plus four facts extracted from the revised role lesson. Three of those four assert the new role holder and one names the previous holder; they do not represent four independent project targets.

After restart, Lena Dovren is recalled on both English role questions (thinking off/on), but both German role answers remain `unknown`: 2/4. ZP-17 is recalled on English off/on and German off: 3/4. The German thinking-on response has empty visible content and stops at its 512-token limit. Niko Varren and `aster-check` remain 0/4 each. Whole-answer matches require a completed response; phrase presence is a separate diagnostic.

Final recall is 3/8 with thinking off and 2/8 with thinking on. Across the 14 questions, thinking-on has three incomplete responses (two token-limit truncations and one empty visible answer with `finish_reason=stop`) and thinking-off has none. The token limits are 512 and 96 respectively, so this does not isolate the effect of thinking independently of the budgets. All 28 visible answers immediately before and after restart are byte-equal. The new server instance starts without learning enabled.

Both modes answer the similar project name Asterion with `Lena Dovren` after the revised lesson. The untrained Bracken sales role produces `Petr Zelenka` with thinking off and an empty visible answer with thinking on (`finish_reason=stop`, 14 reasoning tokens). These questions explain the 0/4 neighbour score. Baseline already had one incomplete neighbour answer, so containment was not perfect before learning.

The first three rejected lessons leave all 28 visible answers byte-equal to baseline. For the Bracken role, selected learner telemetry records 22 subject-filter drops, 15 answer-filter drops and only two retained preparation examples, below its minimum of three. For the first Aster role, token holdout passed but additional real-answer checks did not. For the verification command, a positive real-answer check was transient, then failed in later rounds before rollback. These observations identify different failure stages; they do not establish one cause for all rejected lessons or a general language limitation.

The first Mira Solven lesson was rejected. Its repetition therefore measures another attempt, rather than idempotence of an accepted fact. The revised lesson does not demonstrate replacing an already learned V1. No old-value mention appears in these frozen answers; that does not establish erasure of the old value under other prompts.

## Host, versions and resources

The host was a MacBook Pro Mac17,7, Apple M5 Max, with 128 GiB RAM. The stock local model is `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`, affine 4-bit group-64, with four weight shards totalling 17,775,425,379 bytes. Its local download revision was not recorded. The receipt pins all four shards and config by SHA256.

| Component | Pin |
| --- | --- |
| Upstream | `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` |
| Native build source (upstream + #664 tooling only) | `2bf03775b895c437b5c1d37d9fe6dd9fafda1313` |
| Executed main-run client | `ea732599ce5f3b4beb7cf5970c2390f7e8bfe582` |
| Final auditor / tooling | `26dce45e65b79270f62c48a1e6afd45a5cfc7aee` |
| Compiler | Zig `0.17.0` |
| Native binary SHA256 | `300426beb9b6cca8694ceb50bffa8ba06591e92f3a1d7519682dc0b66bbaad50` |

Native engine sources are unchanged from upstream. Local native verification passed 1,290 tests with two skips across 141 build steps. The 33 Python tests passed on 3.11 and 3.14. The final auditor independently recomputed the full public evidence and matched the private run audit exactly. Later tooling validation changes did not alter the frozen corpus or score definitions used by this run.

The server used an authenticated loopback endpoint, one parallel request, context 4,096, prompt cache disabled, greedy sampling and no automatic update. Each experiment owns its child server, an exclusive host lock and a fresh disposable APFS checkpoint copy. The original weight/config hashes match before and after; both test children exited and the host lock was released. No production binary was promoted.

The maximum observed owned-process physical footprint was 35,831,678,616 bytes (33.37 GiB) during learning and 24,201,313,208 bytes (22.54 GiB) after restart. These are five-second samples, not process lifetime peaks or measurements of total GPU/system memory.

System-wide swap first became nonzero at owned-process second 5,443.54, during the revised role lesson. It reached 8,975.56 MiB (8.77 GiB), with 298 nonzero samples out of 1,379. These samples do not identify its cause. Earlier rejected lessons and the initial neighbour regression occurred before swap. HTTP durations include prompt processing and generation; lesson durations include saving. They describe the costs observed in this run and do not establish TTFT, decode speed or a performance gain.

## Separate timeout attempt

The `timeout/` folder records the earlier run using client `ba1204527500d3531a939a9ea053d9db85a7f83d`. Its first learning request exceeded the previous 900-second blocking-read timeout after 900.24 seconds. The stream lacks a final acknowledgement and confirmed fresh save. The attempted write is uncertain and the run failed; it is not counted as a completed model rejection. Cleanup and original source hashes were verified, and that disposable copy was never reused. The main run increased the blocking-read timeout to 7,200 seconds, which is not an overall learning deadline.

The separate fictional chat-colour calibration fixture is available in the tooling but was not run on hardware. This is one ordered run on one checkpoint and host, with four project targets; its counts are not an estimate of a general success rate or evidence of general coding ability.

## Reproduce the audit

Check out tooling commit `26dce45e65b79270f62c48a1e6afd45a5cfc7aee` and run:

```sh
python3 tools/living_weights/lw_project_audit.py PATH/TO/project
python3 -m unittest tools.tests.test_living_weights tools.tests.test_living_project
```

`MANIFEST.json` in each run folder pins the exported files. The public export includes complete structured replies and learning events, graph/save state, checkpoint hashes and owned-process memory samples. Keys, checkpoints and full server logs are excluded. Selected learner telemetry retains only `slide:` / `slide pocket:` lines. One selected line containing an authentication marker was omitted; `project/telemetry-selection.json` records the counts. That line is not needed by the independent structured audit.
