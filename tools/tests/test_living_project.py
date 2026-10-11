"""Project learner protocol and scoring tests, including interrupted writes and stale save acknowledgements."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "living_weights"))
import lw_project_io as transport
import lw_project_score as scoring
import lw_project as runner
import lw_project_audit as auditor
import signal
import subprocess
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1] / "living_weights"


def events(recalled=True):
    return [{"event": "fact", "data": {"id": 1, "text": "Fictional lesson."}},
            {"event": "learning", "data": {"id": 1}},
            {"event": "learned", "data": {"id": 1, "recalled": recalled}},
            {"event": "learned", "data": "[DONE]"}]


class ProtocolTests(unittest.TestCase):
    def test_complete_and_rejected_lessons_are_distinct(self):
        self.assertEqual(transport.verdict(events()), "accepted")
        self.assertEqual(transport.verdict(events(False)), "rejected")

    def test_incomplete_or_malformed_streams_never_pass(self):
        for altered in (events()[:-1], events()[1:], events() + events()[-1:], [],
                        [{"event": "failed", "data": {"message": "bad"}}],
                        events()[:-1] + [{"event": "failed", "data": "[DONE]"}]):
            with self.subTest(events=altered), self.assertRaises(ValueError):
                transport.verdict(altered)
        for replacement in ({"id": 2, "recalled": True}, {"id": 1, "recalled": 1}):
            altered = events()
            altered[2]["data"] = replacement
            with self.assertRaises(ValueError):
                transport.verdict(altered)

    def test_each_extracted_fact_needs_its_own_final_event(self):
        altered = events()
        altered.insert(1, {"event": "fact", "data": {"id": 2, "text": "Second fact."}})
        with self.assertRaises(ValueError):
            transport.verdict(altered)
        altered.insert(-1, {"event": "learned", "data": {"id": 2, "recalled": True}})
        self.assertEqual(transport.verdict(altered), "accepted")

    def test_save_requires_new_timestamp_and_matching_finished_facts(self):
        before = {"model": {"save": {"state": "saved", "at": 10}}}
        after = {"model": {"save": {"state": "saved", "at": 11}},
                 "facts": [{"id": 1, "state": "learned"}]}
        with patch.object(transport, "get", side_effect=[before, after]), patch.object(transport.time, "sleep"):
            self.assertEqual(transport.await_save("http://example.invalid", before, {1}), after)
        for changed in ({"model": {"save": {"state": "failed", "at": 11}}},
                        {"model": {"save": {"state": "saved", "at": 11}}, "facts": []}):
            with patch.object(transport, "get", return_value=changed), self.assertRaises(ValueError):
                transport.await_save("http://example.invalid", before, {1})

    def test_stale_save_times_out(self):
        before = {"model": {"save": {"state": "saved", "at": 10}}}
        with patch.object(transport, "get", return_value=before), self.assertRaises(TimeoutError):
            transport.await_save("http://example.invalid", before, {1}, timeout=0)

    def test_partial_events_survive_timeout_without_retry(self):
        def interrupted(_):
            yield "fact", {"id": 1, "text": "Fictional lesson."}
            raise TimeoutError("lost acknowledgement")
        saved, record = [], {}
        with patch.object(transport, "get", return_value={}), patch.object(
                transport.urllib.request, "urlopen") as request, patch.object(
                transport, "parse_sse", side_effect=interrupted), self.assertRaises(TimeoutError):
            transport.teach("http://example.invalid", {"text": "Fictional lesson.", "source": "document"},
                            record, lambda r: saved.append(copy.deepcopy(r)))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(saved[-1]["status"], "uncertain")
        self.assertEqual(saved[-1]["events"][0]["event"], "fact")

    def test_only_lesson_text_is_sent_to_learning(self):
        raw = b"".join((f"event: {e['event']}\ndata: " +
                       (e["data"] if isinstance(e["data"], str) else json.dumps(e["data"])) + "\n\n").encode()
                      for e in events())
        with patch.object(transport, "get", return_value={}), patch.object(
                transport.urllib.request, "urlopen") as request, patch.object(transport, "await_save", return_value={}):
            request.return_value.__enter__.return_value = io.BytesIO(raw)
            step = {"text": "Fictional lesson.", "source": "document", "answer": "NEVER SEND", "question": "HOLDOUT"}
            transport.teach("http://example.invalid", step, {}, lambda _: None)
        self.assertEqual(json.loads(request.call_args.args[0].data), {"text": "Fictional lesson.", "source": "document"})


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.spec = scoring.load(ROOT / "project_lessons.json", ROOT / "project_questions.json")
        self.question = self.spec["questions"][0]

    def result(self, answer, finish="stop"):
        return scoring.score(self.question, answer, finish, self.spec["cases"], {"aster-lead": 2})

    def test_whole_value_case_and_whitespace(self):
        self.assertTrue(self.result(" LENA\nDOVREN ")["correct"])
        self.assertFalse(self.result("The lead is Lena Dovren.")["correct"])
        self.assertFalse(self.result("It is not Lena Dovren.")["correct"])
        self.assertFalse(self.result("Lena Dovren", "length")["correct"])

    def test_old_both_and_empty_answers(self):
        self.assertEqual(self.result("Mira Solven")["status"], "old")
        self.assertEqual(self.result("Mira Solven or Lena Dovren")["status"], "both")
        self.assertEqual(self.result("")["status"], "incomplete")

    def test_substring_number_is_not_a_correct_control(self):
        q = {"expect": ["42"]}
        self.assertFalse(scoring.score(q, "142", "stop", {}, {})["correct"])
        self.assertTrue(scoring.score(q, "42", "stop", {}, {})["correct"])

    def test_calibration_is_explicitly_fictional_and_keeps_chat_source(self):
        spec = scoring.load(ROOT / "calibration_lessons.json", ROOT / "calibration_questions.json")
        self.assertEqual(spec["steps"][0]["source"], "chat")
        self.assertEqual(spec["cases"]["colour"]["values"]["1"], ["turquoise", "türkis"])

    def test_duplicate_question_and_stale_duplicate_lesson_are_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            lessons, questions = Path(directory) / "lessons.json", Path(directory) / "questions.json"
            a = json.loads((ROOT / "project_lessons.json").read_text())
            b = json.loads((ROOT / "project_questions.json").read_text())
            lessons.write_text(json.dumps(a))
            questions.write_text(json.dumps(dict(b, questions=b["questions"] + [b["questions"][0]])))
            with self.assertRaises(ValueError):
                scoring.load(lessons, questions)
            questions.write_text(json.dumps(b))
            a["steps"].append(dict(a["steps"][0], id="stale-repeat", kind="duplicate"))
            lessons.write_text(json.dumps(a))
            with self.assertRaises(ValueError):
                scoring.load(lessons, questions)

    def test_unsafe_reserved_and_generated_file_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lessons, questions = Path(directory) / "lessons.json", ROOT / "project_questions.json"
            original = json.loads((ROOT / "project_lessons.json").read_text())
            for name in ("../outside", "/outside", "", "baseline", "restart", "receipt", "graph",
                         "memory", "learning", "BASELINE", "Graph", "a.b", "a\\b", "x" * 129):
                with self.subTest(name=name):
                    corpus = copy.deepcopy(original)
                    corpus["steps"][0]["id"] = name
                    lessons.write_text(json.dumps(corpus))
                    with self.assertRaises(ValueError):
                        scoring.load(lessons, questions)
            corpus = copy.deepcopy(original)
            corpus["steps"][1]["id"] = corpus["steps"][0]["id"] + "-learn"
            lessons.write_text(json.dumps(corpus))
            with self.assertRaises(ValueError):
                scoring.load(lessons, questions)
            corpus["steps"][1]["id"] = corpus["steps"][0]["id"].upper()
            lessons.write_text(json.dumps(corpus))
            with self.assertRaises(ValueError):
                scoring.load(lessons, questions)

    def test_duplicate_cannot_change_learning_source(self):
        with tempfile.TemporaryDirectory() as directory:
            lessons = Path(directory) / "lessons.json"
            corpus = json.loads((ROOT / "project_lessons.json").read_text())
            duplicate = next(s for s in corpus["steps"] if s["kind"] == "duplicate")
            duplicate["source"] = "chat"
            lessons.write_text(json.dumps(corpus))
            with self.assertRaises(ValueError):
                scoring.load(lessons, ROOT / "project_questions.json")

    def test_same_basename_cannot_drop_one_corpus_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.mkdir()
            b.mkdir()
            (a / "corpus.json").write_bytes((ROOT / "project_lessons.json").read_bytes())
            (b / "corpus.json").write_bytes((ROOT / "project_questions.json").read_bytes())
            with self.assertRaises(ValueError):
                scoring.load(a / "corpus.json", b / "corpus.json")


class LifecycleTests(unittest.TestCase):
    def test_exclusive_lock_and_unconfirmed_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.lock"
            with runner.host_lock({}, path):
                with self.assertRaises(FileExistsError):
                    with runner.host_lock({}, path):
                        pass
            self.assertFalse(path.exists())
            with runner.host_lock({"owned_server_stopped": False}, path):
                pass
            self.assertTrue(path.exists())

    def test_spawn_signal_is_deferred_until_ownership_assignment(self):
        child = None
        with self.assertRaises(KeyboardInterrupt):
            with runner.owned_spawn():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                child = "owned"
        self.assertEqual(child, "owned")

    def test_cleanup_ignores_signals_and_stops_only_owned_child(self):
        server = object.__new__(runner.Server)
        server.process, server.log, server.monitor = Mock(), Mock(), Mock()
        server.process.poll.return_value = None
        server.process.wait.side_effect = [subprocess.TimeoutExpired("owned", 30), 0]
        previous = signal.getsignal(signal.SIGTERM)
        def terminate():
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
            self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        server.process.terminate.side_effect = terminate
        server.close()
        server.process.kill.assert_called_once()
        server.log.close.assert_called_once()
        server.monitor.stop.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_readiness_failure_retains_owned_child_for_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            server = runner.Server(Path("binary"), Path("model"), Path(directory), True)
            child = Mock()
            child.poll.return_value = None
            with patch.object(runner.subprocess, "Popen", return_value=child), patch.object(
                    runner, "Monitor"), patch.object(server, "ready", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    server.start()
                self.assertIs(server.process, child)
                server.close()
                child.terminate.assert_called_once()


class EvidenceTests(unittest.TestCase):
    def test_baseline_has_fixed_recall_denominator_and_known_hits(self):
        row = {"question": "q", "thinking": False, "kind": "recall", "answer": "Mira Solven",
               "seconds": 1, "score": {"eligible": False, "correct": True, "old_mentioned": False, "status": "correct"}}
        phases = [{"phase": "baseline", "rows": [row]}]
        result = runner.summarise(phases, [])
        self.assertEqual(result["phases"][0]["groups"]["recall"]["correct"], 1)
        self.assertEqual(result["phases"][0]["groups"]["recall"]["total"], 1)
        self.assertEqual(result, auditor.aggregate(phases, []))

    def test_commands_and_codes_are_case_sensitive(self):
        spec = scoring.load(ROOT / "project_lessons.json", ROOT / "project_questions.json")
        for fact, good, bad in (("aster-check", "aster-check", "ASTER-CHECK"),
                                ("bracken-severity", "ZP-17", "zp-17")):
            q = next(q for q in spec["questions"] if q.get("fact") == fact)
            self.assertTrue(scoring.score(q, good, "stop", spec["cases"], {fact: 1})["correct"])
            self.assertFalse(scoring.score(q, bad, "stop", spec["cases"], {fact: 1})["correct"])

    def test_memory_requires_both_instances_and_retains_probe_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "memory.jsonl"
            valid = {"instance": "a", "pid": 1, "physical_footprint_bytes": 100, "swap": "used = 0M"}
            path.write_text(json.dumps(valid) + "\n")
            with self.assertRaises(ValueError):
                auditor.memory_evidence(root, {"a", "b"})
            rows = [valid, dict(valid, instance="b", pid=2), {"instance": "a", "pid": 1, "error": "unavailable"}]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            result = auditor.memory_evidence(root, {"a", "b"})
            self.assertEqual(result["a"]["errors"], 1)
            self.assertEqual(result["b"]["max_observed_footprint_bytes"], 100)


class AuditTests(unittest.TestCase):
    def fixture(self, root):
        spec = scoring.load(ROOT / "project_lessons.json", ROOT / "project_questions.json")
        phases, lessons, state = [], [], {}
        previous_ids = []
        wanted = [("baseline", None)] + [(s["id"], s) for s in spec["steps"]] + [("restart", None)]
        for index, (name, step) in enumerate(wanted):
            if step:
                identity = index
                stream = events()
                for event in stream:
                    if isinstance(event["data"], dict):
                        event["data"]["id"] = identity
                record = {"id": name, "kind": step["kind"], "hashes": spec["hashes"],
                          "text_sha256": runner.hashlib.sha256(step["text"].encode()).hexdigest(),
                          "request": {"text": step["text"], "source": step["source"]},
                          "events": stream, "status": "accepted", "seconds_including_save": 1,
                          "graph_before": {"model": {"save": {"at": index}}, "facts": previous_ids[:]}}
                previous_ids.append({"id": identity, "state": "learned"})
                record["graph_after"] = {"model": {"save": {"at": index + 1, "state": "saved"}},
                                         "facts": previous_ids[:]}
                runner.write(root / (name + "-learn.json"), record)
                lessons.append(record)
                state[step["fact"]] = step["version"]
            rows = []
            for question in spec["questions"]:
                for thinking in (False, True):
                    answer = "unknown"
                    rows.append({"question": question["id"], "kind": question["kind"],
                                 "language": question["language"], "thinking": thinking, "seconds": 1,
                                 "answer": answer, "finish": "stop",
                                 "request": {"messages": [{"role": "user", "content": question["text"]}],
                                             "temperature": 0, "max_tokens": 512 if thinking else 96,
                                             "chat_template_kwargs": {"enable_thinking": thinking}},
                                 "response": {"tensorfold": {"enable_thinking": thinking},
                                              "choices": [{"message": {"content": answer}, "finish_reason": "stop"}]},
                                 "score": scoring.score(question, answer, "stop", spec["cases"], state)})
            phase = {"phase": name, "hashes": spec["hashes"], "state": state.copy(), "rows": rows,
                     "complete": True, "instance": "restart" if name == "restart" else "learning",
                     "learning_enabled": name != "restart"}
            runner.write(root / (name + ".json"), phase)
            phases.append(phase)
        receipt = {"status": "completed", "hashes": spec["hashes"], "source_unchanged": True,
                   "owned_server_stopped": True, "restart_verified": True,
                   "source_before": {"config": "abc"}, "source_after": {"config": "abc"},
                   "binary_sha256": "abc", "build": {"binary_sha256": "abc"},
                   "summary": runner.summarise(phases, lessons)}
        runner.write(root / "receipt.json", receipt)
        memory = [{"instance": instance, "pid": pid, "physical_footprint_bytes": 100, "swap": "used = 0M"}
                  for instance, pid in (("learning", 1), ("restart", 2))]
        (root / "memory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in memory))
        return spec

    def test_rejected_learning_is_a_complete_measured_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            path = root / "v1-aster-lead-learn.json"
            record = json.loads(path.read_text())
            record["events"][2]["data"]["recalled"] = False
            record["status"] = "rejected"
            record["graph_after"]["facts"][0]["state"] = "missed"
            runner.write(path, record)
            receipt = json.loads((root / "receipt.json").read_text())
            receipt["summary"]["lessons"][0]["status"] = "rejected"
            runner.write(root / "receipt.json", receipt)
            self.assertTrue(auditor.audit(root, spec)["complete"])

    def test_complete_fixture_and_corrupted_evidence(self):
        mutations = [("baseline.json", lambda d: d.update(instance="other")),
                     ("baseline.json", lambda d: d["rows"][0]["request"].update(max_tokens=1)),
                     ("baseline.json", lambda d: d["rows"][0]["score"].update(correct=True)),
                     ("restart.json", lambda d: d.update(instance="learning")),
                     ("v2-aster-lead-learn.json", lambda d: d.update(id="wrong-step")),
                     ("v2-aster-lead-learn.json", lambda d: d["graph_after"]["facts"][-1].update(state="missed")),
                     ("receipt.json", lambda d: d.update(source_after={"config": "changed"}))]
        for filename, mutate in mutations:
            with self.subTest(file=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec = self.fixture(root)
                self.assertEqual(auditor.audit(root, spec)["responses"], 224)
                path = root / filename
                data = json.loads(path.read_text())
                mutate(data)
                runner.write(path, data)
                with self.assertRaises(ValueError):
                    auditor.audit(root, spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            (root / "memory.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                auditor.audit(root, spec)


if __name__ == "__main__":
    unittest.main()
