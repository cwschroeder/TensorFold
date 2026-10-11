"""Validate a fictional project corpus and score complete visible answers, independently of the learner."""

import hashlib
import json
import unicodedata
from pathlib import Path


def normal(text):
    """Fold Unicode, case and whitespace; retain punctuation and every answer word."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load(lessons, questions):
    """Read and validate both frozen files, preserving their exact byte hashes."""
    paths = [Path(lessons), Path(questions)]
    a, b = [json.loads(p.read_bytes()) for p in paths]
    if a.get("schema") != 1 or b.get("schema") != 1 or a.get("fictional") is not True:
        raise ValueError("expected schema 1 and an explicitly fictional corpus")
    cases = {c["id"]: c for c in a["cases"]}
    if not cases or len(cases) != len(a["cases"]) or not a["steps"] or not b["questions"]:
        raise ValueError("empty or duplicate corpus")
    for rows in (a["steps"], b["questions"]):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate step or question id")
        for r in rows:
            if not r["text"].strip() or r["language"] not in ("de", "en"):
                raise ValueError("empty text or unsupported language")
    state, previous = {}, {}
    for step in a["steps"]:
        fact, version = step["fact"], step["version"]
        if fact not in cases or str(version) not in cases[fact]["values"] or step["source"] not in ("document", "chat"):
            raise ValueError("unknown fact/version or unsupported source")
        old = state.get(fact, 0)
        if step["kind"] == "duplicate":
            if old != version or step["text"] != previous.get(fact):
                raise ValueError("duplicate must repeat the current lesson exactly")
        elif step["kind"] not in ("introduce", "update") or version != old + 1:
            raise ValueError("versions must advance in order")
        state[fact], previous[fact] = version, step["text"]
    coverage = set()
    for q in b["questions"]:
        if q["kind"] == "recall":
            if q["fact"] not in cases:
                raise ValueError("unknown question fact")
            coverage.add((q["fact"], q["language"]))
        elif q["kind"] not in ("control", "neighbour") or not q.get("expect"):
            raise ValueError("invalid question kind or missing expected answer")
    if coverage != {(f, lang) for f in cases for lang in ("de", "en")}:
        raise ValueError("every fact needs independent DE and EN questions")
    for c in cases.values():
        for values in c["values"].values():
            if not values or any(not normal(v) for v in values):
                raise ValueError("empty expected value")
    return {"cases": cases, "steps": a["steps"], "questions": b["questions"],
            "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def score(question, answer, finish, cases, state):
    """Require an entire expected value and stop; retain old-value and mixed-answer diagnostics."""
    fact = question.get("fact")
    version = state.get(fact, 0)
    expected = question.get("expect", []) if fact is None else cases[fact]["values"][str(version or 1)]
    old = [] if fact is None else [v for k, vs in cases[fact]["values"].items()
                                 if int(k) < version for v in vs]
    normalize = normal
    if fact is not None and cases[fact].get("case_sensitive", False):
        normalize = lambda text: " ".join(unicodedata.normalize("NFKC", text).split())
    value = normalize(answer)
    new_seen = any(normalize(v) in value for v in expected)
    old_seen = any(normalize(v) in value for v in old)
    correct = finish == "stop" and value in [normalize(v) for v in expected]
    status = "correct" if correct else "both" if old_seen and new_seen else "old" if old_seen else "other"
    if finish != "stop" or not value:
        status = "incomplete"
    return {"status": status, "correct": correct, "eligible": fact is None or version > 0,
            "version": version, "expected": expected, "old_mentioned": old_seen,
            "expected_phrase_present": new_seen}
