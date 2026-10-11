# Living Weights test harness and console

Tools for measuring what a Living Weights lesson changes, beyond the fact it teaches. They talk to running
`tensorfold serve` processes over HTTP; nothing here loads a model. Python 3.11+, standard library only.

Runs compare a taught copy against the same checkpoint with greedy decoding (`temperature: 0`). Record a baseline
before teaching; reply differences are observations, and repeat runs help distinguish learning from runtime variation.

## Servers

Teach a copy, never your only folder (see [the Living Weights guide](../../docs/living-weights.md)). For the
cross-checks, also serve an untouched folder of the same checkpoint without `--slide`:

```sh
tensorfold serve COPY --slide --slide-graph facts-graph.json --port 8080 --no-thinking
tensorfold serve ORIGINAL --port 8081 --no-thinking
```

## Fact test (`lw_facts.py`, `facts.json`)

20 fictional facts with a pre-learning baseline to identify existing or coincidental answers. Each has two recall questions and a
neighbour question about a real thing of the same kind that must not pick the fact up ("How long is the Golden Gate
Bridge?" next to "The Quillon bridge is 1,312 metres long."), plus 15 unrelated controls.

```sh
python3 lw_facts.py baseline --url http://127.0.0.1:8080
python3 lw_facts.py learn    --url http://127.0.0.1:8080       # --only f14,f18 to teach some; --source chat
python3 lw_facts.py probe    --url http://127.0.0.1:8080       # again after a restart: --label after-restart
```

## General knowledge and rules (`lw_rules.py`, `rules.json`)

60 general-knowledge questions, and four fictional rules (a fee, a pay rate, a made-up unit, a replacement interval)
each used in two problems. `probe` asks the problems from memory and again with the rule stated in the prompt, which
shows whether a lesson damages reasoning the model could already do.

```sh
python3 lw_rules.py probe --url http://127.0.0.1:8080
python3 lw_rules.py learn --url http://127.0.0.1:8080
python3 lw_rules.py probe --url http://127.0.0.1:8080 --label after
```

Each phase writes every question and answer to `results/<phase>-<timestamp>.json` (`--results DIR` to move it).

## Console (`console.py`)

```sh
python3 console.py --learn http://127.0.0.1:8080 --reference http://127.0.0.1:8081 [--model NAME=URL ...]
```

Open `http://127.0.0.1:8090/`. The page is served on loopback with the servers behind one origin, since the server
sends no CORS headers. Only the `--learn` server can be taught; `/v1/slide` paths to any other server are refused.

- Chat with any of the servers. Replies render as markdown (marked + DOMPurify), with mermaid and SVG diagrams.
- Teach a fact, sent as a document or as chat. Before teaching, the reference rates the statement CONTRADICTS,
  CONSISTENT or UNKNOWN from its own knowledge; after teaching, the reference writes questions shaped like the fact
  about real things, and those plus a fixed panel are asked of both servers side by side, with differences marked.
- The fact graph and dashboard links open the learning server's own pages.

The page loads marked, DOMPurify and mermaid from cdn.jsdelivr.net; without them it shows plain text.

## Tests

`python3 -m unittest tools.tests.test_living_weights` from the repository root runs against a fake server.

## Project facts, duplicates and corrections

`lw_project.py` teaches a disposable copy of an untaught Nemotron checkpoint and owns both native server processes.
Four fictional facts describe two projects: two lessons in German and two in English, followed by an exact repeat
and a correction of the quality lead. Independent DE/EN questions live in `project_questions.json`; the learner
receives only lesson text and `source: document`, never the test questions or expected answers.

```sh
python3 tools/living_weights/lw_project.py \
  --binary /path/to/tensorfold-native --model /path/to/original-checkpoint \
  --build-receipt /path/to/build-pins.json --output /path/to/new-run
python3 tools/living_weights/lw_project_audit.py /path/to/new-run
python3 -m unittest tools.tests.test_living_project tools.tests.test_living_weights
```

An exclusive per-user host lock covers preflight through confirmed server exit. If cleanup cannot confirm exit,
the lock remains for operator inspection. Terminal interrupts stop the owned child and preserve partial evidence.
The managed runner requires Apple Silicon, an idle model host, memory and disk headroom, and Python 3.11+.
`build-pins.json` records `binary_sha256`, `upstream_commit`, `build_source_commit`, `patch_commit` and `compiler_version`. Build and
verify the current upstream before a run. The output directory must be new; it contains a disposable checkpoint
and private API keys. Publish a reviewed receipt subset, never the entire output directory.

Each write needs a complete learning stream and a fresh successful shard save. Rejected lessons remain measured
outcomes; an uncertain write stops the run without retry. After every write, all questions run in fresh greedy chats
with thinking off and on. The final probe uses a new owned process on the learned copy, without `--slide`.
Original checkpoint hashes are verified after cleanup.

Scores require the complete visible answer to equal an expected value and `finish_reason: stop`. Unicode and
whitespace are normalized. Person names ignore case; commands and incident codes preserve case. Punctuation
and extra words are retained. Phrase presence is
diagnostic only: a denial, a larger number containing the expected number, or both old and new values cannot pass.
All phases include the same recall denominator, including any facts already known at baseline.
The receipt separates control accuracy from byte changes, old-value mentions, and incomplete replies.

Versions describe the test's intended facts. They do not imply a server update contract, last-write-wins behaviour
or request deduplication. The auditor recomputes scores and summaries and rejects missing phases or mismatched
corpus hashes. Total HTTP latency includes prompt and generation; it is not TTFT or isolated decode speed.
Physical footprint and system swap are sampled every five seconds for the owned child, including loading and
learning. A maximum observed sample is not a lifetime peak. This experiment does not measure general coding skill.
