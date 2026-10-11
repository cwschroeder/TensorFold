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
                        [{"event": "failed", "data": {"message": "bad"}}]):
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
               "seconds": 1, "score": {"eligible": False, "correct": True}}
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


if __name__ == "__main__":
    unittest.main()
