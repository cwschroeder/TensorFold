"""Record authenticated project probes and learning streams; require a complete stream and a fresh shard save."""

import json
import time
import urllib.request

from lw_client import parse_sse, post_json


def get(base, path, headers=None):
    """Read a JSON endpoint without recording authentication headers."""
    req = urllib.request.Request(base + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def ask(base, question, thinking, headers=None):
    """Retain the whole response and total request time, using a fresh greedy single-message chat."""
    body = {"temperature": 0, "max_tokens": 512 if thinking else 96,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": question}]}
    start = time.monotonic()
    response = post_json(base + "/v1/chat/completions", body, 300, headers)
    choice = response["choices"][0]
    return {"seconds": time.monotonic() - start, "request": body, "response": response,
            "answer": choice["message"].get("content") or "", "finish": choice["finish_reason"]}


def verdict(events):
    """Require distinct fact ids, one matching terminal event each, bool recall and one final DONE."""
    facts, finals, done = set(), {}, False
    for item in events:
        event, data = item["event"], item["data"]
        if done:
            raise ValueError("event after DONE")
        if data == "[DONE]":
            done = True
            continue
        if not isinstance(data, dict) or event in ("failed", "error"):
            raise ValueError("learning failed or malformed event")
        if event not in ("fact", "learning", "learned"):
            raise ValueError("unknown learning event")
        identity = data.get("id")
        if type(identity) is not int or identity <= 0:
            raise ValueError("invalid fact id")
        if event == "fact":
            if identity in facts or not isinstance(data.get("text"), str):
                raise ValueError("duplicate id or missing fact text")
            facts.add(identity)
        elif identity not in facts:
            raise ValueError("event for unknown fact")
        elif event == "learned":
            if identity in finals or type(data.get("recalled")) is not bool:
                raise ValueError("duplicate terminal event or non-bool recall")
            finals[identity] = data["recalled"]
    if not done or not facts or set(finals) != facts:
        raise ValueError("incomplete learning acknowledgement")
    return "accepted" if all(finals.values()) else "rejected"


def await_save(base, before, ids, headers=None, timeout=120):
    """Require a fresh successful save whose graph contains this request's completed facts."""
    deadline = time.monotonic() + timeout
    previous = before.get("model", {}).get("save", {}).get("at", 0)
    while time.monotonic() < deadline:
        graph = get(base, "/v1/slide/graph", headers)
        saved = graph.get("model", {}).get("save", {})
        if saved.get("at", 0) > previous:
            if saved.get("state") == "failed":
                raise ValueError("shard save failed")
            if saved.get("state") == "saved":
                facts = {f["id"]: f for f in graph.get("facts", [])}
                if any(i not in facts or facts[i]["state"] not in ("learned", "missed") for i in ids):
                    raise ValueError("save belongs to incomplete or missing facts")
                return graph
        time.sleep(0.2)
    raise TimeoutError("no fresh successful save")


def teach(base, step, record, persist, headers=None):
    """Preserve partial events on every update, stop on uncertain writes, and never retry a lesson."""
    before = get(base, "/v1/slide/graph", headers)
    body = {"text": step["text"], "source": step["source"]}
    record.update(status="pending", events=[], graph_before=before, request=body)
    persist(record)
    start = time.monotonic()
    request = urllib.request.Request(base + "/v1/slide/learn", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            for event, data in parse_sse(response):
                record["events"].append({"event": event, "data": data, "seconds": time.monotonic() - start})
                persist(record)
        result = verdict(record["events"])
        ids = {e["data"]["id"] for e in record["events"] if e["event"] == "fact"}
        if ids & {f["id"] for f in before.get("facts", [])}:
            raise ValueError("request reused an existing graph fact id")
        record["graph_after"] = await_save(base, before, ids, headers)
        record["status"] = result
    except Exception as error:
        record.update(status="uncertain", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        record["seconds_including_save"] = time.monotonic() - start
        persist(record)
