"""Audit saved baseline/combined outcomes without model or verifier calls.

Usage: python report/analyze_failures.py results/gemini-minimal
Validate run completeness, pair recorded outcomes, and replay the compressed
regressions in temporary workspaces containing only public task/environment
files. Check each delta against an earlier full result and the official tool's
fresh output. This verifies recorded bytes, not model reasoning or causality.
"""

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from content import HEADER, reconstruct
from evaluate import DATA, prepare_workspace
from report.summarize import load_complete
from spartacus.tools import core_tools


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inspect_run(run_dir, row):
    folder = run_dir / row["dataset"] / row["mode"] / row["tid"]
    messages = json.loads((folder / "transcript.json").read_text())
    compressed = [i for i, m in enumerate(messages, 1)
                  if m.get("role") == "tool" and m.get("text", "").startswith(HEADER)]
    assert len(compressed) == row["content"].get("reused", 0)
    writes = [i for i, m in enumerate(messages, 1)
              if m.get("role") == "assistant" for call in m.get("tool_calls", [])
              if call["name"] == "write_file" and call["args"].get("path") == "answer.json"]
    return {
        "passed": row["passed"], "reused": row["reused"],
        "answer": json.loads((folder / "answer.json").read_text()),
        "transcript": str((folder / "transcript.json").relative_to(ROOT)),
        "compressed_count": len(compressed),
        "first_compressed_message": compressed[0] if compressed else None,
        "answer_write_messages": writes,
        "turn_limit_present": any(m.get("role") == "user" and
                                  m.get("text", "").startswith("Turn limit reached;")
                                  for m in messages),
    }, messages


def replay(dataset, tid, messages):
    task = json.loads((DATA / dataset / "env/incoming" / (tid + ".json")).read_text())
    checks, bases, pending = [], {}, deque()
    replayed = 0
    with TemporaryDirectory(prefix="hw2-public-audit-") as directory:
        workspace = Path(directory) / "workspace"
        prepare_workspace(dataset, task, workspace)
        allowed = {t.name: t for t in core_tools(str(workspace))
                   if t.name in {"read_file", "grep", "list_files", "write_file"}}
        for index, message in enumerate(messages, 1):
            if message.get("role") == "assistant":
                pending.extend(message.get("tool_calls", []))
            if message.get("role") != "tool":
                continue
            call = pending.popleft()
            name, args = call["name"], call["args"]
            assert name == message["name"] and name in allowed
            assert name != "write_file" or args.get("path") == "answer.json"
            try:
                fresh = allowed[name].run(**args)
            except Exception as error:
                fresh = "ERROR: %s: %s" % (type(error).__name__, error)
            replayed += 1
            source = name + ":" + json.dumps(args, sort_keys=True)
            recorded = message["text"]
            if recorded.startswith(HEADER):
                payload = json.loads(recorded[len(HEADER):])
                assert payload["target_source"] == source
                base_index, base = bases[(payload["base_source"], payload["base_sha256"])]
                assert base_index < index
                assert reconstruct(base, recorded) == fresh, (dataset, tid, index)
                checks.append({"message": index, "base_message": base_index,
                               "base_source": payload["base_source"], "target_source": source,
                               "target_sha256": digest(fresh), "edit_count": len(payload["edits"]),
                               "exact_public_tool_match": True})
            else:
                assert recorded == fresh, (dataset, tid, index)
                if name in {"read_file", "grep"}:
                    bases.setdefault((source, digest(recorded)), (index, recorded))
        assert not pending
    return {"replayed_tool_outputs": replayed, "compressed_outputs": checks}


def analyze(run_dir):
    _, rows, _ = load_complete(run_dir)
    paired = {mode: {(r["dataset"], r["tid"]): r for r in rows if r["mode"] == mode}
              for mode in ("baseline", "combined")}
    counts, regressions, gains = Counter(), [], []
    for key, baseline in paired["baseline"].items():
        combined = paired["combined"][key]
        outcome = ("both_pass" if baseline["passed"] and combined["passed"] else
                   "regression" if baseline["passed"] else
                   "gain" if combined["passed"] else "both_fail")
        counts[outcome] += 1
        if outcome not in {"regression", "gain"}:
            continue
        case = {"dataset": key[0], "tid": key[1]}
        for mode, row in (("baseline", baseline), ("combined", combined)):
            case[mode], messages = inspect_run(run_dir, row)
            if outcome == "regression" and case[mode]["compressed_count"]:
                case[mode]["replay"] = replay(*key, messages)
        (regressions if outcome == "regression" else gains).append(case)
    checks = [check for case in regressions for mode in ("baseline", "combined")
              for check in case[mode].get("replay", {}).get("compressed_outputs", [])]
    assert len(checks) == sum(case[mode]["compressed_count"] for case in regressions
                              for mode in ("baseline", "combined"))
    return {"method": __doc__, "run_dir": str(run_dir.relative_to(ROOT)),
            "validated_task_mode_rows": len(rows), "paired_tasks": sum(counts.values()),
            "paired_outcomes": dict(counts), "regressions": regressions, "gains": gains,
            "compressed_regression_outputs_verified": len(checks),
            "all_compressed_regression_outputs_match": all(c["exact_public_tool_match"] for c in checks),
            "replay_limitations": "Replays saved tool calls only for regressions with compression, "
            "using the current public corpus and official tools. No model rerun, verifier execution, "
            "hidden-label access, or causal test. Indices are 1-based transcript array positions."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve())
    output = ROOT / "report/failure_analysis.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("validated_task_mode_rows", "paired_tasks",
          "paired_outcomes", "compressed_regression_outputs_verified",
          "all_compressed_regression_outputs_match")}, indent=2))
