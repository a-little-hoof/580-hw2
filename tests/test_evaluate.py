"""Integration checks through the real Spartacus loop, without model requests."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import evaluate
from content import HEADER, reconstruct
from spartacus import provider


def reply(calls=(), input_tokens=101, output_tokens=17):
    return {"text": "done" if not calls else "", "tool_calls": list(calls),
            "usage": {"input": input_tokens, "output": output_tokens}}


def call(name, **args):
    return {"name": name, "args": args, "signature": None}


class WorkspaceTests(unittest.TestCase):
    def test_materialization_excludes_hidden_and_other_tasks_and_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets/hosts"
            env = dataset / "env"
            files = {
                "env/INSTRUCTIONS.md": "Read inventory files.",
                "env/history.jsonl": '{"hid": "H-1", "answer": {}}\n',
                "env/incoming/T-other.json": '{"tid": "T-other"}',
                "env/scenarios/FLT-1234/inventory/web-01.txt": "version = 1",
                "env/scenarios/FLT-9999/inventory/web-02.txt": "version = 2",
                "hidden/labels.json": '"DO NOT EXPOSE"',
                "verify.py": "raise RuntimeError('not agent data')",
            }
            for name, text in files.items():
                path = dataset / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.json").write_text('"STALE ANSWER"')
            task = {"tid": "T-current", "seq": 1, "scenario": "flt1234"}
            with patch.object(evaluate, "DATA", root / "datasets"):
                evaluate.prepare_workspace("hosts", task, workspace)
            actual = {str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file()}
            self.assertEqual(actual, {"INSTRUCTIONS.md", "history.jsonl", "task.json",
                                      "inventory/web-01.txt"})
            self.assertEqual(json.loads((workspace / "task.json").read_text()), task)
            self.assertTrue(all(not p.is_symlink() for p in workspace.rglob("*")))
            (workspace / "inventory/web-01.txt").write_text("independent copy")
            self.assertEqual((env / "scenarios/FLT-1234/inventory/web-01.txt").read_text(),
                             "version = 1")

    def test_verifier_output_is_discarded_and_only_boolean_is_exposed(self):
        with patch.object(evaluate.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, stdout=None, stderr="")
            self.assertIs(evaluate.verify("tickets", "T-1", Path("answer.json")), False)
            self.assertIs(run.call_args.kwargs["stdout"], subprocess.DEVNULL)
            run.return_value = subprocess.CompletedProcess([], 0, stdout=None, stderr="")
            self.assertIs(evaluate.verify("tickets", "T-1", Path("answer.json")), True)
            run.return_value = subprocess.CompletedProcess([], 2, stdout=None,
                                                          stderr="sensitive expected answer")
            with self.assertRaisesRegex(RuntimeError, "Verifier infrastructure failure") as error:
                evaluate.verify("tickets", "T-1", Path("answer.json"))
            self.assertNotIn("sensitive", str(error.exception))


class RealLoopTests(unittest.TestCase):
    def test_pacing_covers_wrapup_and_task_boundaries_and_restores_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            clock, starts = [0.0], []

            def complete(*args):
                starts.append(clock[0])
                return reply([call("write_file", path="answer.json", content="{}")])

            def sleep(seconds):
                clock[0] += seconds

            with patch.object(evaluate, "_last_request_start", None), \
                    patch.object(evaluate.time, "perf_counter", side_effect=lambda: clock[0]), \
                    patch.object(evaluate.time, "sleep", side_effect=sleep), \
                    patch.object(provider, "complete", side_effect=complete) as original:
                first, _ = evaluate.run_agent(workspace, evaluate.MODEL, 1, False, 0.8, 5.0)
                self.assertIs(provider.complete, original)
                second, _ = evaluate.run_agent(workspace, evaluate.MODEL, 1, False, 0.8, 5.0)
                self.assertIs(provider.complete, original)
            self.assertEqual(starts, [0.0, 5.0, 10.0, 15.0])
            self.assertEqual(first["pacing_seconds"], 5.0)
            self.assertEqual(second["pacing_seconds"], 10.0)
            self.assertEqual(first["model_requests"], 2)
            self.assertEqual(second["model_requests"], 2)

    def test_interruption_preserves_partial_usage_and_transcript_without_error_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            calls = [call("write_file", path="answer.json", content='{"result": 1}')]
            with patch.object(provider, "complete", side_effect=[
                    reply(calls), RuntimeError("provider response with sensitive detail")]):
                with self.assertRaises(evaluate.AgentInterrupted) as caught:
                    evaluate.run_agent(workspace, evaluate.MODEL, 20, False, 0.8)
            error = caught.exception
            self.assertEqual(error.metrics["model_requests"], 1)
            self.assertEqual(error.metrics["tool_calls"], 1)
            self.assertEqual(error.metrics["input_tokens"], 101)
            self.assertEqual(error.messages[-1]["role"], "tool")
            self.assertNotIn("sensitive", str(error))

    def test_gate_and_real_tools_confine_agent_and_protect_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "private.txt"
            outside.write_text("outside private contents")
            (workspace / "INSTRUCTIONS.md").write_text("original instructions")
            calls = [call("write_file", path="INSTRUCTIONS.md", content="overwrite"),
                     call("write_file", path="../private.txt", content="overwrite"),
                     call("write_file", path="nested/answer.json", content="{}"),
                     call("read_file", path="../private.txt"),
                     call("read_file", path=str(outside)),
                     call("bash", command="cat ../private.txt"),
                     call("write_file", path="answer.json", content='{"result": 1}')]
            with patch.object(provider, "complete", side_effect=[reply(calls), reply()]):
                metrics, messages = evaluate.run_agent(workspace, evaluate.MODEL, 20, True, 0.8)
            results = [m["text"] for m in messages if m["role"] == "tool"]
            self.assertTrue(all(text.startswith("BLOCKED:") for text in results[:3]))
            self.assertTrue(all("escapes the working directory" in text for text in results[3:5]))
            self.assertTrue(results[5].startswith("BLOCKED:"))
            self.assertNotIn("outside private contents", json.dumps(messages))
            self.assertEqual(outside.read_text(), "outside private contents")
            self.assertEqual((workspace / "INSTRUCTIONS.md").read_text(), "original instructions")
            self.assertEqual(json.loads((workspace / "answer.json").read_text()), {"result": 1})
            self.assertEqual(metrics["tool_calls"], len(calls))
            self.assertEqual(metrics["model_requests"], 2)
            self.assertEqual(metrics["input_tokens"], 202)
            self.assertEqual(metrics["output_tokens"], 34)

    def test_compression_uses_same_prompt_and_exact_prior_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            base = "".join("key_%03d = %d # stable configuration\n" % (i, i) for i in range(120))
            changed = base.replace("key_042 = 42", "key_042 = -420.001")
            (workspace / "base.txt").write_text(base)
            (workspace / "changed.txt").write_text(changed)
            calls = [call("read_file", path="base.txt"), call("read_file", path="changed.txt"),
                     call("write_file", path="answer.json", content='{"value": -420.001}')]
            runs = []
            for compress in (False, True):
                seen = []

                def complete(model, system, messages, tools):
                    seen.append(copy.deepcopy((model, system, messages, tools)))
                    return reply(calls) if len(seen) == 1 else reply()

                with patch.object(provider, "complete", side_effect=complete):
                    metrics, messages = evaluate.run_agent(workspace, evaluate.MODEL, 20, compress, 0.8)
                runs.append((metrics, messages, seen))
            baseline, compressed = runs
            self.assertEqual(baseline[2][0], compressed[2][0],
                             "initial model, system, user prompt, and tool schemas must match")
            self.assertEqual(baseline[0]["system_sha256"], compressed[0]["system_sha256"])
            base_tools = [m["text"] for m in baseline[1] if m["role"] == "tool"]
            compressed_tools = [m["text"] for m in compressed[1] if m["role"] == "tool"]
            self.assertEqual(base_tools[0], compressed_tools[0])
            self.assertTrue(compressed_tools[1].startswith(HEADER))
            self.assertEqual(reconstruct(compressed_tools[0], compressed_tools[1]), base_tools[1])
            next_request_tools = [m["text"] for m in compressed[2][1][2] if m["role"] == "tool"]
            self.assertEqual(next_request_tools, compressed_tools,
                             "the actual next request must contain the emitted delta")
            self.assertEqual(compressed[0]["content"]["reused"], 1)
            for metrics, _, _ in runs:
                # These are mocked provider counters, not real experimental data.
                self.assertEqual(metrics["model_requests"], 2)
                self.assertEqual(metrics["tool_calls"], 3)
                self.assertEqual(metrics["input_tokens"], 202)
                self.assertEqual(metrics["output_tokens"], 34)


class EvaluationBoundaryTests(unittest.TestCase):
    def test_resume_rebuilds_only_verified_cache_and_rejects_bad_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / "datasets/tickets/env"
            env.mkdir(parents=True)
            (env / "INSTRUCTIONS.md").write_text("Write answer.json.")
            tasks = [{"tid": "T-" + name, "seq": seq, "identity": identity}
                     for seq, (name, identity) in enumerate(
                         [("pass", "A"), ("fail", "B"), ("interrupt", "C"), ("repeat", "A")], 1)]
            args = SimpleNamespace(output=root / "online", datasets=["tickets"], limit=None,
                                   modes=["task"], task_threshold=0.75, content_threshold=0.8,
                                   model=evaluate.MODEL, max_turns=20, resume=False)
            instances, model_tasks = [], []

            class Cache:
                def __init__(self, *unused, **kwargs):
                    self.answers, self.added = {}, []
                    instances.append(self)

                def lookup(self, task):
                    answer = self.answers.get(task["identity"])
                    return {"answer": answer} if answer else None

                def add_verified(self, task, answer, verified=False):
                    assert verified is True
                    self.answers[task["identity"]] = answer
                    self.added.append(task["tid"])

            def fake_agent(workspace, *unused):
                task = json.loads((workspace / "task.json").read_text())
                model_tasks.append(task["tid"])
                metrics = {"model_requests": 1, "tool_calls": 1, "input_tokens": 101,
                           "output_tokens": 17, "content": {}}
                messages = [{"role": "user", "text": "mock task"}]
                if task["tid"] == "T-interrupt" and not args.resume:
                    raise evaluate.AgentInterrupted("RuntimeError", metrics, messages)
                evaluate.dump(workspace / "answer.json", {"resolution_id": "R-01"})
                return metrics, messages

            with patch.object(evaluate, "DATA", root / "datasets"), \
                    patch.object(evaluate, "tasks_for", return_value=tasks), \
                    patch.object(evaluate, "TaskReuse", Cache), \
                    patch.object(evaluate, "provenance", return_value={"source_sha256": "fixed"}), \
                    patch.object(evaluate, "run_agent", side_effect=fake_agent), \
                    patch.object(evaluate, "verify", side_effect=lambda dataset, tid, path: tid != "T-fail"), \
                    patch.object(provider, "api_key", return_value="mock-key"), \
                    patch("builtins.print"):
                with self.assertRaises(evaluate.AgentInterrupted):
                    evaluate.online(args)
                path = args.output / "metrics.jsonl"
                initial = path.read_text()
                self.assertEqual(len(initial.splitlines()), 2)
                interruptions = list((args.output / "interruptions").rglob("*.json"))
                self.assertEqual(len(interruptions), 1)
                interrupted = json.loads(interruptions[0].read_text())
                self.assertEqual(interrupted["status"], "interrupted")
                self.assertEqual(interrupted["metrics"]["input_tokens"], 101)
                self.assertNotIn("passed", interrupted)
                args.resume = True
                settings_path = args.output / "settings.json"
                settings = settings_path.read_text()
                evaluate.dump(settings_path, {"source_sha256": "changed"})
                with self.assertRaisesRegex(ValueError, "identical settings"):
                    evaluate.online(args)
                settings_path.write_text(settings)
                path.write_text(initial + initial.splitlines()[0] + "\n")
                with self.assertRaisesRegex(ValueError, "unique ordered prefix"):
                    evaluate.online(args)
                path.write_text(initial + "truncated-json\n")
                with self.assertRaises(ValueError):
                    evaluate.online(args)
                malformed = [json.loads(line) for line in initial.splitlines()]
                malformed[0]["input_tokens"] = "101"
                path.write_text("\n".join(json.dumps(r) for r in malformed) + "\n")
                with self.assertRaisesRegex(ValueError, "Malformed checkpoint metric"):
                    evaluate.online(args)
                path.write_text(initial)
                evaluate.online(args)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[:2], [json.loads(line) for line in initial.splitlines()])
            self.assertEqual([r["tid"] for r in rows], [t["tid"] for t in tasks])
            self.assertTrue(rows[-1]["reused"])
            self.assertEqual(rows[-1]["model_requests"], 0)
            self.assertEqual(model_tasks, ["T-pass", "T-fail", "T-interrupt", "T-interrupt"])
            self.assertEqual(instances[-1].added, ["T-pass", "T-interrupt", "T-repeat"])
            self.assertEqual(json.loads((args.output / "summary.json").read_text())[0]["tasks"], 4)

    def test_offline_replay_never_calls_provider_or_claims_model_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = root / "datasets/hosts/env/scenarios/FLT-1/inventory"
            scenario.mkdir(parents=True)
            text = "".join("setting_%03d = %d # stable setting\n" % (i, i) for i in range(120))
            (scenario / "first.txt").write_text(text)
            (scenario / "second.txt").write_text(text)
            args = SimpleNamespace(output=root / "offline", datasets=["hosts"], limit=None,
                                   content_threshold=0.8)
            with patch.object(evaluate, "DATA", root / "datasets"), \
                    patch.object(evaluate, "tasks_for", return_value=[]), \
                    patch.object(evaluate, "TaskReuse") as policy, \
                    patch.object(evaluate, "provenance", return_value={"model": evaluate.MODEL}), \
                    patch.object(provider, "complete", side_effect=AssertionError("offline model call")) as complete, \
                    patch("builtins.print"):
                policy.return_value.stats = {}
                evaluate.offline(args)
            complete.assert_not_called()
            settings = json.loads((args.output / "settings.json").read_text())
            self.assertIn("no model invoked", settings["evaluation"])
            content_rows = json.loads((args.output / "content_replay.json").read_text())
            self.assertEqual(content_rows[0]["roundtrips"], 1)
            self.assertNotIn("input_tokens", content_rows[0])
            self.assertNotIn("model_requests", content_rows[0])
            self.assertFalse((args.output / "metrics.jsonl").exists())

    def test_online_indexes_generated_answers_only_after_verifier_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / "datasets/tickets/env"
            env.mkdir(parents=True)
            (env / "INSTRUCTIONS.md").write_text("Write answer.json.")
            tasks = [{"tid": "T-pass", "seq": 1}, {"tid": "T-fail", "seq": 2}]
            args = SimpleNamespace(output=root / "online", datasets=["tickets"], limit=None,
                                   modes=["task"], task_threshold=0.75, content_threshold=0.8,
                                   model=evaluate.MODEL, max_turns=20)

            def fake_agent(workspace, *unused):
                evaluate.dump(workspace / "answer.json", {"resolution_id": "R-01"})
                return {"model_requests": 2, "tool_calls": 3, "input_tokens": 202,
                        "output_tokens": 34, "content": {}}, []

            with patch.object(evaluate, "DATA", root / "datasets"), \
                    patch.object(evaluate, "tasks_for", return_value=tasks), \
                    patch.object(evaluate, "TaskReuse") as policy, \
                    patch.object(evaluate, "provenance", return_value={}), \
                    patch.object(evaluate, "run_agent", side_effect=fake_agent), \
                    patch.object(evaluate, "verify", side_effect=[True, False]), \
                    patch.object(provider, "api_key", return_value="mock-key"), \
                    patch("builtins.print"):
                policy.return_value.lookup.return_value = None
                evaluate.online(args)
            policy.return_value.add_verified.assert_called_once_with(
                tasks[0], {"resolution_id": "R-01"}, verified=True)
            rows = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["passed"] for r in rows], [True, False])
            self.assertTrue(all(r["total_seconds"] >= r["seconds"] for r in rows))


if __name__ == "__main__":
    unittest.main()
