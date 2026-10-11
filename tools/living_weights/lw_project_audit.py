"""Recompute project results from frozen questions and full responses; reject incomplete or mixed-run evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import statistics
from lw_project_io import verdict
from lw_project_score import load, score


def aggregate(phases, lessons):
    """Independently count every frozen question, including baseline recall."""
    baseline = {(r["question"], r["thinking"]): r["answer"] for r in phases[0]["rows"]}
    results = []
    for phase in phases:
        groups = {}
        for kind in ("recall", "neighbour", "control"):
            rows = [r for r in phase["rows"] if r["kind"] == kind]
            groups[kind] = {"correct": sum(bool(r["score"]["correct"]) for r in rows), "total": len(rows),
                            "byte_changes": sum(r["answer"] != baseline[r["question"], r["thinking"]] for r in rows)}
        results.append({"phase": phase["phase"], "groups": groups,
                        "median_request_seconds": statistics.median(r["seconds"] for r in phase["rows"])})
    return {"lessons": [{"id": r["id"], "status": r["status"],
                         "seconds_including_save": r["seconds_including_save"]} for r in lessons], "phases": results}


def mode_summary(phases):
    """Separate measured recall, old values, truncation and request latency by actual thinking mode."""
    result = {}
    for phase in phases:
        modes = {}
        for mode, name in ((False, "thinking_off"), (True, "thinking_on")):
            rows = [r for r in phase["rows"] if r["thinking"] == mode]
            recall = [r for r in rows if r["kind"] == "recall"]
            modes[name] = {"recall_correct": sum(r["score"]["correct"] for r in recall),
                           "recall_total": len(recall), "old_mentions": sum(r["score"]["old_mentioned"] for r in rows),
                           "incomplete": sum(r["score"]["status"] == "incomplete" for r in rows),
                           "median_request_seconds": statistics.median(r["seconds"] for r in rows)}
        result[phase["phase"]] = modes
    return result


def memory_evidence(root, instances):
    """Require usable footprint and swap samples for both owned process instances."""
    rows = [json.loads(line) for line in (root / "memory.jsonl").read_text().splitlines()]
    if any(r["instance"] not in instances for r in rows):
        raise ValueError("memory probe belongs to another run")
    result = {}
    for instance in instances:
        samples = [r for r in rows if r["instance"] == instance]
        usable = [r for r in samples if "error" not in r and r.get("physical_footprint_bytes", 0) > 0 and r.get("swap")]
        if not usable or len({r["pid"] for r in samples}) != 1:
            raise ValueError("missing usable memory coverage or mixed process ids")
        result[instance] = {"samples": len(samples), "errors": len(samples) - len(usable),
                            "max_observed_footprint_bytes": max(r["physical_footprint_bytes"] for r in usable)}
    return result


def audit(root, spec):
    """Bind every phase and learning acknowledgement to its corpus, state and owned process instance."""
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt["status"] != "completed" or receipt["hashes"] != spec["hashes"]
            or receipt.get("source_unchanged") is not True or receipt.get("owned_server_stopped") is not True
            or receipt.get("restart_verified") is not True):
        raise ValueError("incomplete run, corpus mismatch or unverified cleanup/restart")
    if not receipt.get("source_before") or receipt["source_before"] != receipt.get("source_after"):
        raise ValueError("source checkpoint hashes changed or are missing")
    if receipt["binary_sha256"] != receipt["build"]["binary_sha256"]:
        raise ValueError("binary differs from build pin")
    state, phases, lessons, learned_facts = {}, [], [], []
    wanted = [("baseline", None)] + [(s["id"], s) for s in spec["steps"]] + [("restart", None)]
    for name, step in wanted:
        if step:
            record = json.loads((root / (name + "-learn.json")).read_text())
            if (record["id"] != name or record["kind"] != step["kind"]
                    or record["text_sha256"] != hashlib.sha256(step["text"].encode()).hexdigest()
                    or record["request"] != {"text": step["text"], "source": step["source"]}):
                raise ValueError("lesson identity or taught text mismatch")
            if record["hashes"] != spec["hashes"] or record["status"] != verdict(record["events"]):
                raise ValueError("mismatched or incomplete lesson evidence")
            before = record["graph_before"].get("model", {}).get("save", {}).get("at", 0)
            saved = record["graph_after"].get("model", {}).get("save", {})
            if saved.get("state") != "saved" or saved.get("at", 0) <= before:
                raise ValueError("lesson has no fresh successful save")
            ids = {e["data"]["id"] for e in record["events"] if e["event"] == "fact"}
            before_ids = {f["id"] for f in record["graph_before"].get("facts", [])}
            facts = {f["id"]: f for f in record["graph_after"].get("facts", [])}
            if ids & before_ids or any(i not in facts or facts[i]["state"] not in ("learned", "missed") for i in ids):
                raise ValueError("save has missing, reused or incomplete fact ids")
            for event in record["events"]:
                data = event["data"]
                if event["event"] == "learned" and isinstance(data, dict):
                    expected_state = "learned" if data["recalled"] else "missed"
                    if facts[data["id"]]["state"] != expected_state:
                        raise ValueError("learning verdict disagrees with saved fact state")
            if not math.isfinite(record["seconds_including_save"]) or record["seconds_including_save"] < 0:
                raise ValueError("invalid lesson timing")
            learned_facts.append({"lesson": name,
                                  "accepted_ids": [e["data"]["id"] for e in record["events"]
                                                   if e["event"] == "learned" and isinstance(e["data"], dict) and e["data"]["recalled"]],
                                  "rejected_ids": [e["data"]["id"] for e in record["events"]
                                                   if e["event"] == "learned" and isinstance(e["data"], dict) and not e["data"]["recalled"]]})
            lessons.append(record)
            state[step["fact"]] = step["version"]
        phase = json.loads((root / (name + ".json")).read_text())
        if (phase["phase"] != name or phase["hashes"] != spec["hashes"] or phase["state"] != state
                or phase.get("complete") is not True or phase["learning_enabled"] != (name != "restart")):
            raise ValueError("phase, state or corpus mismatch")
        expected = [(q, t) for q in spec["questions"] for t in (False, True)]
        if len(phase["rows"]) != len(expected):
            raise ValueError("missing or duplicated questions")
        for row, (q, thinking) in zip(phase["rows"], expected):
            if (row["question"] != q["id"] or row["thinking"] != thinking
                    or row["kind"] != q["kind"] or row["language"] != q["language"]):
                raise ValueError("question identity/order mismatch")
            request, response = row["request"], row["response"]
            if (request["messages"] != [{"role": "user", "content": q["text"]}]
                    or request["max_tokens"] != (512 if thinking else 96)
                    or request["temperature"] != 0 or request["chat_template_kwargs"]["enable_thinking"] != thinking
                    or response["tensorfold"]["enable_thinking"] != thinking):
                raise ValueError("changed prompt, sampling or actual thinking mode")
            choice = response["choices"][0]
            if row["answer"] != (choice["message"].get("content") or "") or row["finish"] != choice["finish_reason"]:
                raise ValueError("reply disagrees with response")
            if not math.isfinite(row["seconds"]) or row["seconds"] < 0:
                raise ValueError("invalid request timing")
            if row["score"] != score(q, row["answer"], row["finish"], spec["cases"], state):
                raise ValueError("score disagrees with visible answer")
        phases.append(phase)
    instances = {p["instance"] for p in phases[:-1]}
    if len(instances) != 1 or phases[-1]["instance"] in instances:
        raise ValueError("restart did not change the recorded owned process instance")
    summary = aggregate(phases, lessons)
    memory = memory_evidence(root, instances | {phases[-1]["instance"]})
    if receipt["summary"] != summary:
        raise ValueError("summary disagrees with independently recomputed results")
    return {"schema": 1, "complete": True, "responses": sum(len(p["rows"]) for p in phases),
            "hashes": spec["hashes"], "summary": summary, "memory": memory, "thinking_modes": mode_summary(phases), "learning_facts": learned_facts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lessons", type=Path, default=Path(__file__).with_name("project_lessons.json"))
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("project_questions.json"))
    args = parser.parse_args()
    print(json.dumps(audit(args.receipt, load(args.lessons, args.questions)), indent=2))


if __name__ == "__main__":
    main()
