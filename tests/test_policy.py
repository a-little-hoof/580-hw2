import json
from pathlib import Path
import tempfile
import unittest

from policy import TaskReuse


BOILERPLATE = " ".join("sharedword" + str(i) for i in range(120))


class PolicyTests(unittest.TestCase):
    def test_corpus_equivalence_and_hard_negatives(self):
        history = {"hid": "H-original", "scenario": "FLT-1234", "kind": "uptime_of",
                   "target": "web-03", "body": BOILERPLATE, "answer": {"days": 7}}
        policy = TaskReuse("hosts", [history], threshold=0.5)
        task = dict(history, scenario="flt1234", target="WEB-03")
        hit = policy.lookup(task)
        self.assertEqual(hit["answer"], {"days": 7})
        for field, value in (("scenario", "FLT-4321"), ("kind", "fullest_disk"),
                             ("target", "web-04")):
            self.assertIsNone(policy.lookup(dict(task, **{field: value})))
        self.assertIsNone(policy.lookup({"body": BOILERPLATE}))
        # Returned answers cannot mutate the verified cache.
        hit["answer"]["days"] = 90
        self.assertEqual(policy.lookup(task)["answer"], {"days": 7})

    def test_verification_required_and_conflicts_fail_closed(self):
        task = {"tid": "T-1", "scenario": "CFG-1234", "kind": "drift",
                "target": None, "body": BOILERPLATE}
        policy = TaskReuse("configs")
        with self.assertRaises(ValueError):
            policy.add_verified(task, {"environment": "prod"})
        self.assertIsNone(policy.lookup(task))
        answer = {"environment": "prod"}
        policy.add_verified(task, answer, verified=True)
        answer["environment"] = "mutated"
        self.assertEqual(policy.lookup(task)["answer"], {"environment": "prod"})
        policy.add_verified(task, {"environment": "dev"}, verified=True)
        self.assertIsNone(policy.lookup(task))
        self.assertEqual(policy.stats["conflicts"], 1)

    def test_ticket_story_is_checked_beyond_structured_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            (env / "INSTRUCTIONS.md").write_text(
                "**authentication token rejected** (E-1100, E-2111)\n"
                "- story mentions *clock* -> ignored\n"
                "- story mentions *office network* -> ignored\n")
            history = {"hid": "H-clock", "product": "App", "error_code": "E-1100",
                       "platform": "android", "resolution_id": "R-12",
                       "body": BOILERPLATE + "\n\nThe clock is wrong."}
            policy = TaskReuse("tickets", [history], threshold=0.5, env_dir=env)
            self.assertEqual(policy.cues["e1100"], ["clock", "office network"])
            self.assertEqual(policy.cues["e2111"], ["clock", "office network"])
            task = dict(history, error_code="e1100", body=BOILERPLATE + "\n\nMy clock drifted.")
            self.assertIsNotNone(policy.lookup(task))
            self.assertIsNone(policy.lookup(dict(task, platform="linux")))
            self.assertIsNone(policy.lookup(dict(task, body=BOILERPLATE + "\n\nOnly the office network fails.")))
            self.assertIsNone(policy.lookup(dict(task, body=BOILERPLATE + "\n\nThe clock is not wrong.")))
            self.assertIsNone(policy.lookup(dict(task, body=BOILERPLATE + "\n\nClock and office network.")))

    def test_bug_frame_series_and_actual_story(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            (env / "INSTRUCTIONS.md").write_text(
                "**TimeoutError**\n- story mentions *slow network* -> ignored\n"
                "- story mentions *two workers* -> ignored\n")
            history = {"hid": "H-bug", "exception": "TimeoutError", "top_frame": "lattice.net.http.fetch",
                       "version": "3.1.2", "component": "networking", "root_cause": "RC-07",
                       "body": BOILERPLATE + "\n\n## Actual\n\nThe slow network causes delays.\n\n## Frequency\nOften."}
            policy = TaskReuse("bugs", [history], threshold=0.5, env_dir=env)
            task = dict(history, version="v3.1.9", top_frame="lattice/net/http.py:42::fetch")
            self.assertIsNotNone(policy.lookup(task))
            self.assertIsNone(policy.lookup(dict(task, version="3.2.1")))
            self.assertIsNone(policy.lookup(dict(task, body=history["body"].replace("slow network", "two workers"))))
            self.assertIsNone(policy.lookup(dict(task, top_frame="lattice.net.http.post")))

    def test_docs_keep_original_history_id(self):
        history = {"hid": "H-original", "category": "change-approved", "title": "CHG-1234: Update",
                   "body": BOILERPLATE + " The change is approved."}
        policy = TaskReuse("docs", [history], threshold=0.5)
        task = dict(history, tid="T-1", body=BOILERPLATE + " The change is authorised.")
        hit = policy.lookup(task)
        self.assertEqual(hit["answer"]["duplicate_of"], "H-original")
        policy.add_verified(task, hit["answer"], verified=True)
        self.assertEqual(policy.lookup(task)["answer"]["duplicate_of"], "H-original")
        self.assertIsNone(policy.lookup(dict(task, body=BOILERPLATE + " The change is rejected.")))
        self.assertIsNone(policy.lookup(dict(task, title="CHG-4321: Update")))

    def test_new_docs_do_not_invent_history_ids(self):
        policy = TaskReuse("docs")
        task = {"tid": "T-fresh", "title": "REL-3.2: Update", "body": BOILERPLATE}
        policy.add_verified(task, {"category": "release-notes", "duplicate_of": None}, verified=True)
        self.assertIsNone(policy.lookup(task)["answer"]["duplicate_of"])

    def test_citation_all_graded_fields_and_venue_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            (env / "INSTRUCTIONS.md").write_text("")
            (env / "db.json").write_text(json.dumps({"venues": [
                {"name": "Neural Information Processing Systems", "abbrev": ["NeurIPS", "NIPS"]}]}))
            citation = {"authors": ["Priya Nair", "Tomas Vik"], "title": "Attention: a useful method",
                        "venue": "NeurIPS", "year": 2020, "doi": "10.5555/example"}
            verdict = {"doi": "10.5555/example", "status": "correct", "wrong_fields": []}
            policy = TaskReuse("citations", [{"hid": "H-cite", "citation": citation, "verdict": verdict}], env_dir=env)
            task = {"citation": dict(citation, authors=["Nair, P.", "T. Vik"],
                                     title="ATTENTION a useful method", venue="NIPS",
                                     doi="https://doi.org/10.5555/EXAMPLE", pages="5-8")}
            self.assertEqual(policy.lookup(task)["answer"], verdict)
            changes = {"authors": ["T. Vik", "Nair, P."], "title": "Attention another useful method",
                       "venue": "ICML", "year": 2021, "doi": "10.5555/different"}
            for field, value in changes.items():
                self.assertIsNone(policy.lookup({"citation": dict(task["citation"], **{field: value})}))
            no_doi = dict(task["citation"])
            del no_doi["doi"]
            self.assertIsNone(policy.lookup({"citation": no_doi}))


if __name__ == "__main__":
    unittest.main()
