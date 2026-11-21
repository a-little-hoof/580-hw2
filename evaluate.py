"""Reproducible experiments using the unmodified Spartacus loop and verifiers.

Only the verifier subprocess can read labels. Its diagnostics never enter the
agent transcript or the reuse index. Offline audits are not model experiments.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "vendor/assignment/assignment2/datasets"
sys.path.insert(0, str(ROOT / "vendor/spartacus"))

from content import ContentCache, reconstruct
from policy import TaskReuse

DATASETS = ("tickets", "bugs", "docs", "citations", "hosts", "configs", "threads", "revisions")
MODES = ("baseline", "task", "content", "combined")
MODEL = "gemini-3.5-flash-lite"
_last_request_start = None
PROMPT = (
    "Read INSTRUCTIONS.md and task.json. Solve this task using the available "
    "files and write the required JSON object to answer.json. Read the relevant "
    "corpus files individually with read_file when comparing their contents. "
    "Only answer.json may be written. Tool results marked [reuse] refer to an "
    "earlier full tool output and contain exact changes; reconstruct that output "
    "with those changes when reasoning. Finish once answer.json is written."
)


class AgentInterrupted(RuntimeError):
    """Preserve partial work without recording an API interruption as task failure."""

    def __init__(self, error_type, metrics, messages):
        super().__init__("Agent trajectory interrupted (" + error_type + ")")
        self.error_type, self.metrics, self.messages = error_type, metrics, messages


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tasks_for(dataset, limit=None):
    tasks = [json.loads(p.read_text()) for p in (DATA / dataset / "env/incoming").glob("*.json")]
    return sorted(tasks, key=lambda t: (t["seq"], t["tid"]))[:limit]


def canonical(value):
    return "".join(c for c in str(value).lower() if c.isalnum())


def scenario_dir(env, task):
    matches = [p for p in (env / "scenarios").iterdir()
               if p.is_dir() and canonical(p.name) == canonical(task["scenario"])]
    if len(matches) != 1:
        raise ValueError("Expected exactly one visible scenario for " + task["tid"])
    return matches[0]


def prepare_workspace(dataset, task, workspace):
    """Materialize only the allowed env files, never symlinks to the dataset."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    env = DATA / dataset / "env"
    for path in sorted(env.iterdir()):
        if path.is_file():
            shutil.copy2(path, workspace / path.name)
    if "scenario" in task:
        for path in scenario_dir(env, task).iterdir():
            if path.is_dir():
                shutil.copytree(path, workspace / path.name)
            else:
                shutil.copy2(path, workspace / path.name)
    dump(workspace / "task.json", task)


def verify(dataset, tid, answer_path):
    """Discard verifier diagnostics (they may contain expected answers)."""
    done = subprocess.run(
        [sys.executable, str(DATA / dataset / "verify.py"),
         "--tid", tid, "--answer", str(answer_path.resolve())],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if done.returncode not in (0, 1) or done.stderr:
        raise RuntimeError("Verifier infrastructure failure for " + dataset + "/" + tid)
    return done.returncode == 0


def run_agent(workspace, model, max_turns, compress, content_threshold, request_interval=0.0):
    from spartacus import loop, memory, provider, tools

    messages = [{"role": "user", "text": PROMPT}]
    metrics = {"model_requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
               "pacing_seconds": 0.0}
    cache = ContentCache(threshold=content_threshold) if compress else None
    allowed = {t.name: t for t in tools.core_tools(str(workspace))
               if t.name in {"read_file", "grep", "list_files", "write_file"}}

    def gate(call):
        if call["name"] not in allowed:
            return "This tool is unavailable in the evaluation environment."
        if call["name"] == "write_file" and call["args"].get("path") != "answer.json":
            return "Only answer.json may be written."
        return None

    def event(kind, payload):
        if kind == "assistant":
            metrics["model_requests"] += 1
            metrics["input_tokens"] += payload["usage"]["input"]
            metrics["output_tokens"] += payload["usage"]["output"]
        elif kind == "tool_start":
            metrics["tool_calls"] += 1
        elif kind == "tool_end" and cache and payload["call"]["name"] in {"read_file", "grep"}:
            call = payload["call"]
            source = call["name"] + ":" + json.dumps(call["args"], sort_keys=True)
            messages[-1]["text"] = cache.process(messages[-1]["text"], source=source)

    # No compaction: all full bases remain in context for the entire trajectory.
    # Same system/user prompt and tool schemas are used in all four modes.
    system = memory.build_system_prompt(str(workspace))
    metrics["system_sha256"] = hashlib.sha256(system.encode()).hexdigest()
    complete = provider.complete

    def paced_complete(*args, **kwargs):
        global _last_request_start
        now = time.perf_counter()
        if _last_request_start is not None:
            delay = request_interval - (now - _last_request_start)
            if delay > 0:
                time.sleep(delay)
                metrics["pacing_seconds"] += time.perf_counter() - now
        _last_request_start = time.perf_counter()
        return complete(*args, **kwargs)

    # Wrap only this sequential runner's calls, including the loop's final
    # wrap-up request. The provider source and its HTTP retry logic are unchanged.
    if request_interval > 0:
        provider.complete = paced_complete
    try:
        loop.run_loop(model, system, messages, allowed, event, gate, max_turns=max_turns)
    except Exception as error:
        metrics["content"] = dict(cache.stats) if cache else {}
        raise AgentInterrupted(type(error).__name__, metrics, messages) from None
    finally:
        provider.complete = complete
    metrics["content"] = dict(cache.stats) if cache else {}
    return metrics, messages


def provenance(args):
    def revision(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return {"model": args.model, "max_turns": args.max_turns,
            "request_interval": getattr(args, "request_interval", 0.0),
            "task_threshold": args.task_threshold, "content_threshold": args.content_threshold,
            "seed": 73, "python": sys.version, "datasets": args.datasets,
            "modes": args.modes, "limit": args.limit, "prompt": PROMPT,
            "temperature": 0.4, "max_output_tokens": 65536,
            "tools": ["read_file", "grep", "list_files", "write_file"],
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("minhash.py", "policy.py", "content.py", "evaluate.py")},
            "model_requests_definition": "successful provider responses; HTTP retries excluded",
            "assignment_commit": revision(ROOT / "vendor/assignment"),
            "spartacus_commit": revision(ROOT / "vendor/spartacus")}


def completed_rows(output, expected):
    """Load an intact ordered prefix; never silently discard malformed checkpoints."""
    path = output / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(rows) > len(expected):
        raise ValueError("Checkpoint contains more tasks than the configured run")
    for row, (dataset, mode, task) in zip(rows, expected):
        if (not isinstance(row, dict)
                or tuple(row.get(k) for k in ("dataset", "mode", "tid", "seq"))
                != (dataset, mode, task["tid"], task["seq"])):
            raise ValueError("Checkpoint tasks must be a unique ordered prefix of the run")
        if any(type(row.get(k)) is not bool for k in ("passed", "reused")):
            raise ValueError("Malformed checkpoint verdict")
        for key in ("model_requests", "tool_calls", "input_tokens", "output_tokens"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("Malformed checkpoint metric: " + key)
        for key in ("seconds", "total_seconds", "index_init_seconds_amortized",
                    "task_lookup_seconds", "task_update_seconds"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Malformed checkpoint timing: " + key)
        artifact = output / dataset / mode / task["tid"]
        answer = json.loads((artifact / "answer.json").read_text())
        if not isinstance(json.loads((artifact / "transcript.json").read_text()), list):
            raise ValueError("Malformed checkpoint transcript")
        if row["passed"] and (not isinstance(answer, dict)
                              or not verify(dataset, task["tid"], artifact / "answer.json")):
            raise ValueError("Previously passed checkpoint answer no longer verifies")
    return rows


def online(args):
    from spartacus import provider

    provider.api_key("gemini")  # Fail before writing partial results if unconfigured.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = provenance(args)
    tasks_by_dataset = {dataset: tasks_for(dataset, args.limit) for dataset in args.datasets}
    expected = [(dataset, mode, task) for dataset in args.datasets for mode in args.modes
                for task in tasks_by_dataset[dataset]]
    if getattr(args, "resume", False):
        if json.loads((output / "settings.json").read_text()) != settings:
            raise ValueError("Resume requires identical settings and source hashes")
        rows = completed_rows(output, expected)
    else:
        if (output / "settings.json").exists() or (output / "metrics.jsonl").exists():
            raise ValueError("Choose a fresh output directory or use --resume.")
        dump(output / "settings.json", settings)
        rows = []
    completed = {(r["dataset"], r["mode"], r["tid"]): r for r in rows}
    for dataset in args.datasets:
        tasks = tasks_by_dataset[dataset]
        for mode in args.modes:
            init = time.perf_counter()
            reuse = TaskReuse(dataset, DATA / dataset / "env/history.jsonl",
                              threshold=args.task_threshold) if mode in {"task", "combined"} else None
            init_seconds = time.perf_counter() - init
            for task in tasks:
                previous = completed.get((dataset, mode, task["tid"]))
                if previous is not None:
                    if reuse and previous["passed"]:
                        answer_path = output / dataset / mode / task["tid"] / "answer.json"
                        reuse.add_verified(task, json.loads(answer_path.read_text()), verified=True)
                    continue
                start = time.perf_counter()
                workspace = output / "workspace"
                prepare_workspace(dataset, task, workspace)
                lookup_start = time.perf_counter()
                hit = reuse.lookup(task) if reuse else None
                lookup_seconds = time.perf_counter() - lookup_start if reuse else 0.0
                metrics = {"model_requests": 0, "tool_calls": 0, "input_tokens": 0,
                           "output_tokens": 0, "content": {}, "pacing_seconds": 0.0}
                messages = []
                if hit:
                    dump(workspace / "answer.json", hit["answer"])
                else:
                    try:
                        metrics, messages = run_agent(workspace, args.model, args.max_turns,
                                                      mode in {"content", "combined"}, args.content_threshold,
                                                      getattr(args, "request_interval", 0.0))
                    except AgentInterrupted as error:
                        path = output / "interruptions" / dataset / mode / task["tid"] / (str(time.time_ns()) + ".json")
                        dump(path, {"status": "interrupted", "error_type": error.error_type,
                                    "seconds": time.perf_counter() - start,
                                    "metrics": error.metrics, "transcript": error.messages})
                        raise
                passed = verify(dataset, task["tid"], workspace / "answer.json")
                answer = None
                if (workspace / "answer.json").exists():
                    try:
                        answer = json.loads((workspace / "answer.json").read_text())
                    except (ValueError, OSError):
                        pass
                update_start = time.perf_counter()
                if reuse and passed:
                    reuse.add_verified(task, answer, verified=True)
                update_seconds = time.perf_counter() - update_start if reuse and passed else 0.0
                elapsed = time.perf_counter() - start
                row = {"dataset": dataset, "mode": mode, "tid": task["tid"], "seq": task["seq"],
                       "passed": passed, "reused": bool(hit), "reuse": hit,
                       "task_lookup_seconds": lookup_seconds, "task_update_seconds": update_seconds,
                       "seconds": elapsed, "index_init_seconds_amortized": init_seconds / len(tasks),
                       "total_seconds": elapsed + init_seconds / len(tasks), **metrics}
                artifact = output / dataset / mode / task["tid"]
                dump(artifact / "answer.json", answer)
                dump(artifact / "transcript.json", messages)
                # A committed metric row always has both recoverable artifacts.
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                rows.append(row)
                print(dataset, mode, task["tid"], "PASS" if passed else "FAIL", flush=True)
    dump(output / "summary.json", summarize(rows))


def summarize(rows):
    groups = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for mode in MODES:
            subset = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
            if subset:
                groups.append({"dataset": dataset, "mode": mode, "tasks": len(subset),
                               **{k: sum(r[k] for r in subset) for k in
                                  ("passed", "reused", "model_requests", "tool_calls", "input_tokens", "total_seconds")}})
    return groups


def offline(args):
    """History-only reuse accuracy and lossless full-corpus replay, with no LLM."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "settings.json", {**provenance(args), "evaluation": "offline; no model invoked"})
    reuse_rows, content_rows = [], []
    for dataset in args.datasets:
        tasks = tasks_for(dataset, args.limit)
        for threshold in (0.5, 0.65, 0.75, 0.85, 0.95, 1.0):
            start = time.perf_counter()
            policy = TaskReuse(dataset, DATA / dataset / "env/history.jsonl", threshold=threshold)
            init_seconds = time.perf_counter() - start
            hits = []
            for task in tasks:
                hit = policy.lookup(task)
                if hit:
                    path = output / "audit_answers" / dataset / (task["tid"] + ".json")
                    dump(path, hit["answer"])
                    # Evaluation only: no feedback or labels are used to change the policy.
                    hits.append({"tid": task["tid"], "passed": verify(dataset, task["tid"], path), **hit})
            row = {"dataset": dataset, "threshold": threshold, "tasks": len(tasks),
                   "reused": len(hits), "passed": sum(h["passed"] for h in hits),
                   "init_seconds": init_seconds, "stats": dict(policy.stats), "hits": hits}
            reuse_rows.append(row)
            print(dataset, "offline", threshold, row["passed"], "/", row["reused"], flush=True)
        env = DATA / dataset / "env"
        scenarios = sorted((env / "scenarios").iterdir()) if (env / "scenarios").exists() else []
        for scenario in scenarios:
            cache, originals, roundtrips = ContentCache(threshold=args.content_threshold), {}, 0
            for path in sorted(p for p in scenario.rglob("*") if p.is_file()):
                # Match Spartacus read_file numbering and its line limit exactly.
                lines = path.read_text(errors="replace").splitlines()
                text = "\n".join("%d\t%s" % (n, s) for n, s in enumerate(lines[:4000], 1))
                if len(lines) > 4000:
                    text += "\n... truncated at 4000 lines; the file has %d" % len(lines)
                emitted = cache.process(text, source=str(path.relative_to(scenario)))
                if emitted != text:
                    # Try earlier full bases only; reconstruction checks the base digest.
                    restored = False
                    for base in originals.values():
                        try:
                            restored = reconstruct(base, emitted) == text
                        except (ValueError, KeyError, TypeError):
                            continue
                        if restored:
                            break
                    if not restored:
                        raise AssertionError("Content round-trip failed: " + str(path))
                    roundtrips += 1
                else:
                    originals[str(path)] = text
            content_rows.append({"dataset": dataset, "scenario": scenario.name,
                                 "roundtrips": roundtrips, **cache.stats})
    dump(output / "task_reuse.json", reuse_rows)
    dump(output / "content_replay.json", content_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a matching interrupted online run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS + ("code", "logs"), default=list(DATASETS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--model", choices=[MODEL], default=MODEL)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--request-interval", type=float, default=0.0,
                        help="Minimum seconds between model request starts, including across tasks")
    parser.add_argument("--task-threshold", type=float, default=0.75)
    parser.add_argument("--content-threshold", type=float, default=0.8)
    parser.add_argument("--limit", type=int, help="Smoke-test only; omitted for full evaluation")
    args = parser.parse_args()
    if args.offline and args.resume:
        parser.error("--resume applies only to online runs")
    if not math.isfinite(args.request_interval) or args.request_interval < 0:
        parser.error("--request-interval must be a finite nonnegative number")
    (offline if args.offline else online)(args)


if __name__ == "__main__":
    main()
