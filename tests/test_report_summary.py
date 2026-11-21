"""Report generation must never turn a partial run into complete-looking tables."""

import json
from pathlib import Path
import tempfile
import unittest

from report import summarize


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReportSummaryTests(unittest.TestCase):
    def fixture(self, root):
        data, run = root / "datasets", root / "run"
        dump(run / "settings.json", {"datasets": ["tickets"], "modes": list(summarize.MODES),
                                     "limit": None, "model": "gemini-3.5-flash-lite"})
        tasks = [{"tid": "T-first", "seq": 1}, {"tid": "T-second", "seq": 2}]
        for task in tasks:
            dump(data / "tickets/env/incoming" / (task["tid"] + ".json"), task)
        rows = []
        for count, mode in enumerate(summarize.MODES, 1):
            for task in tasks:
                rows.append({**task, "dataset": "tickets", "mode": mode,
                             "passed": task["seq"] == 1, "reused": False,
                             "model_requests": count, "tool_calls": count + 1,
                             "input_tokens": count * 100, "output_tokens": count * 10,
                             "total_seconds": 10.0, "pacing_seconds": 2.0,
                             "task_lookup_seconds": 0.1, "task_update_seconds": 0.2,
                             "index_init_seconds_amortized": 0.3, "content": {"seconds": 0.4}})
        self.write_rows(run, rows)
        dump(run / "interruptions/tickets/baseline/T-first/123.json",
             {"status": "interrupted", "seconds": 5.0,
              "metrics": {"model_requests": 1, "tool_calls": 1,
                          "input_tokens": 50, "output_tokens": 7, "pacing_seconds": 1.0}})
        return data, run, rows

    def write_rows(self, run, rows):
        (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_complete_run_sums_interruptions_separately_from_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run, original = self.fixture(root)
            settings, rows, interruptions = summarize.load_complete(run, data)
            self.assertEqual(rows, original)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(interruptions), 1)
            self.assertEqual(sum(row["passed"] for row in rows), 4)
            self.assertNotIn("passed", interruptions[0])
            combined = summarize.totals(rows + interruptions)
            self.assertEqual(combined["input_tokens"], 2050)
            self.assertEqual(combined["model_requests"], 21)
            self.assertEqual(combined["total_seconds"], 85.0)
            self.assertEqual(combined["pacing_seconds"], 17.0)
            output = root / "online_tables.tex"
            self.assertEqual(summarize.write_tables(run, output, data), 8)
            text = output.read_text()
            self.assertIn("Baseline & 2 & 50.00\\%", text)
            self.assertIn("tickets & 200 & 400 & 600 & 800 & -300.00\\%", text)
            self.assertIn("2,050 input tokens", text)
            self.assertIn("tab:online-interruptions", text)

    def test_partial_and_duplicate_runs_cannot_replace_existing_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run, rows = self.fixture(root)
            output = root / "online_tables.tex"
            output.write_text("previous validated tables")
            for bad_rows, message in [(rows[:-1], "Incomplete run"),
                                      (rows + [rows[0]], "Duplicate task result")]:
                self.write_rows(run, bad_rows)
                with self.assertRaisesRegex(ValueError, message):
                    summarize.write_tables(run, output, data)
                self.assertEqual(output.read_text(), "previous validated tables")

    def test_task_ids_and_metrics_must_match_public_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, run, rows = self.fixture(Path(tmp))
            for change in ({"tid": "T-unknown"}, {"seq": 999}, {"input_tokens": -1},
                           {"passed": "true"}, {"pacing_seconds": 11},
                           {"task_lookup_seconds": -1}, {"content": {"seconds": float("nan")}}):
                changed = [dict(row) for row in rows]
                changed[0].update(change)
                self.write_rows(run, changed)
                with self.assertRaises(ValueError):
                    summarize.load_complete(run, data)
            self.write_rows(run, rows)

    def test_sample_uses_sequence_then_id_and_labels_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run, rows = self.fixture(root)
            # Alphabetical file order differs from task sequence order; the tie
            # at seq=1 must also be broken by tid, matching evaluate.tasks_for.
            dump(data / "tickets/env/incoming/T-aaa.json", {"tid": "T-aaa", "seq": 3})
            dump(data / "tickets/env/incoming/T-zzz.json", {"tid": "T-zzz", "seq": 1})
            settings_path = run / "settings.json"
            settings = json.loads(settings_path.read_text())
            settings["limit"] = 2
            dump(settings_path, settings)
            for row in rows:
                if row["tid"] == "T-second":
                    row.update(tid="T-zzz", seq=1)
            self.write_rows(run, rows)
            _, loaded, _ = summarize.load_complete(run, data)
            self.assertEqual(loaded, rows)
            output = root / "online_tables.tex"
            self.assertEqual(summarize.write_tables(run, output, data), 8)
            text = output.read_text()
            self.assertIn("Sampled evaluation: the first 2 incoming tasks per dataset", text)
            self.assertIn("sorted by (seq, tid)", text)
            self.assertNotIn("over all incoming tasks", text)

    def test_sample_rejects_missing_and_out_of_sample_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run, rows = self.fixture(root)
            settings_path = run / "settings.json"
            settings = json.loads(settings_path.read_text())
            settings["limit"] = 1
            dump(settings_path, settings)
            selected = [row for row in rows if row["tid"] == "T-first"]
            output = root / "online_tables.tex"
            output.write_text("previous validated tables")
            for bad_rows, message in [(selected[:-1], "Incomplete run: 3 of 4"),
                                      (rows, "Unexpected task result")]:
                self.write_rows(run, bad_rows)
                with self.assertRaisesRegex(ValueError, message):
                    summarize.write_tables(run, output, data)
                self.assertEqual(output.read_text(), "previous validated tables")

    def test_limit_must_be_a_positive_integer_or_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, run, _ = self.fixture(Path(tmp))
            settings_path = run / "settings.json"
            settings = json.loads(settings_path.read_text())
            for limit in (0, -1, True, 1.5, "1"):
                settings["limit"] = limit
                dump(settings_path, settings)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    summarize.load_complete(run, data)

if __name__ == "__main__":
    unittest.main()
