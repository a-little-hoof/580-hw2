"""Validate a complete four-mode run and render its LaTeX result tables.

Usage: python report/summarize.py results/gemini-minimal
Only public incoming tasks and recorded experiment artifacts are read.
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vendor/assignment/assignment2/datasets"
MODES = ("baseline", "task", "content", "combined")
LABELS = dict(zip(MODES, ("Baseline", "Task-only", "Content-only", "Combined")))
DATASETS = {"tickets", "bugs", "docs", "citations", "hosts", "configs",
            "threads", "revisions", "code", "logs"}
COUNTERS = ("model_requests", "tool_calls", "input_tokens", "output_tokens")
TOTALS = COUNTERS + ("total_seconds", "pacing_seconds")
REUSE_TIMINGS = ("task_lookup_seconds", "task_update_seconds", "index_init_seconds_amortized")


def _check_metrics(row):
    for name in COUNTERS:
        if type(row.get(name)) is not int or row[name] < 0:
            raise ValueError("Invalid metric: " + name)
    for name in ("total_seconds", "pacing_seconds"):
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid timing: " + name)
    if row["pacing_seconds"] > row["total_seconds"] + 1e-6:
        raise ValueError("Pacing cannot exceed recorded total time")


def load_complete(run_dir, data_dir=DATA):
    """Require exactly one valid result for every configured task and mode."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    settings = json.loads((run_dir / "settings.json").read_text())
    datasets, modes = settings["datasets"], settings["modes"]
    if (not datasets or len(datasets) != len(set(datasets))
            or not set(datasets) <= DATASETS):
        raise ValueError("Invalid or duplicate configured datasets")
    if len(modes) != 4 or set(modes) != set(MODES):
        raise ValueError("Report requires all four modes exactly once")
    limit = settings.get("limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Task limit must be null or a positive integer")
    if settings.get("model") != "gemini-3.5-flash-lite":
        raise ValueError("Report requires the assignment's exact Gemini model")
    expected = {}
    for dataset in datasets:
        paths = sorted((data_dir / dataset / "env/incoming").glob("*.json"))
        if not paths:
            raise ValueError("No public incoming tasks for " + dataset)
        tasks = sorted((json.loads(path.read_text()) for path in paths),
                       key=lambda task: (task["seq"], task["tid"]))[:limit]
        for task in tasks:
            for mode in MODES:
                key = (dataset, mode, task["tid"])
                if key in expected:
                    raise ValueError("Duplicate public task ID")
                expected[key] = task["seq"]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each metric row must be an object")
        key = tuple(row.get(name) for name in ("dataset", "mode", "tid"))
        if key in seen:
            raise ValueError("Duplicate task result: " + str(key))
        if key not in expected or row.get("seq") != expected[key]:
            raise ValueError("Unexpected task result: " + str(key))
        if any(type(row.get(name)) is not bool for name in ("passed", "reused")):
            raise ValueError("Task outcomes must be booleans")
        _check_metrics(row)
        if not isinstance(row.get("content"), dict):
            raise ValueError("Invalid content-processing metrics")
        timings = [row.get(name) for name in REUSE_TIMINGS] + [row["content"].get("seconds", 0)]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in timings):
            raise ValueError("Invalid reuse-processing timing")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Incomplete run: %d of %d task-mode results" % (len(seen), len(expected)))
    interruptions = []
    for path in sorted((run_dir / "interruptions").glob("*/*/*/*.json")):
        dataset, mode, tid, _ = path.relative_to(run_dir / "interruptions").parts
        record = json.loads(path.read_text())
        if (dataset, mode, tid) not in expected or record.get("status") != "interrupted" or "passed" in record:
            raise ValueError("Invalid interruption record: " + str(path))
        row = {**record["metrics"], "dataset": dataset, "mode": mode, "tid": tid,
               "total_seconds": record["seconds"]}
        _check_metrics(row)
        interruptions.append(row)
    return settings, rows, interruptions


def totals(rows):
    return {name: sum(row[name] for row in rows) for name in TOTALS}


def _table(columns, header, rows, caption, label):
    return "\n".join([
        r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{" + columns + "}",
        r"\toprule", header + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}",
        r"\label{" + label + "}", r"\end{table}", "",
    ])


def render_tables(settings, rows, interruptions):
    limit = settings.get("limit")
    scope = ("All incoming tasks in each selected dataset. " if limit is None else
             "Sampled evaluation: the first %d incoming tasks per dataset, sorted by "
             "(seq, tid), with the same selection in all four modes. " % limit)
    by_mode = {mode: [r for r in rows if r["mode"] == mode] for mode in MODES}
    aggregate = []
    for mode, subset in by_mode.items():
        sums = totals(subset)
        aggregate.append([LABELS[mode], str(len(subset)),
                          "%.2f\\%%" % (100 * sum(r["passed"] for r in subset) / len(subset)),
                          format(sums["tool_calls"], ","), format(sums["model_requests"], ","),
                          "%.3f" % (sums["input_tokens"] / 1e6),
                          "%.3f" % (sums["total_seconds"] / 3600),
                          "%.3f" % (sums["pacing_seconds"] / 3600)])
    result = _table("lrrrrrrr", "Mode & Tasks & Accuracy & Tools & Requests & Input (M) & Time (h) & Pacing (h)",
                    aggregate, scope + "Time includes setup, verification, reuse processing, and API pacing. "
                    "Requests exclude HTTP retries.", "tab:online-aggregate")
    accuracy, tokens = [], []
    for dataset in settings["datasets"]:
        subsets = [[r for r in by_mode[mode] if r["dataset"] == dataset] for mode in MODES]
        accuracy.append([dataset] + ["%d/%d (%.1f\\%%)" %
                        (sum(r["passed"] for r in subset), len(subset),
                         100 * sum(r["passed"] for r in subset) / len(subset)) for subset in subsets])
        counts = [sum(r["input_tokens"] for r in subset) for subset in subsets]
        saving = "%.2f\\%%" % (100 * (1 - counts[-1] / counts[0])) if counts[0] else "--"
        tokens.append([dataset] + [format(count, ",") for count in counts] + [saving])
    result += _table("lrrrr", "Dataset & Baseline & Task-only & Content-only & Combined", accuracy,
                     "Correct answers according to the supplied verifier. Interrupted attempts are excluded.",
                     "tab:online-accuracy")
    result += _table("lrrrrr", "Dataset & Baseline & Task-only & Content-only & Combined & Saved", tokens,
                     "Input tokens reported by Gemini. Saved compares combined reuse with baseline; "
                     "negative values mean more tokens were used.",
                     "tab:online-tokens")
    overhead = []
    for mode in MODES:
        subset = [r for r in interruptions if r["mode"] == mode]
        sums = totals(subset)
        overhead.append([LABELS[mode], str(len(subset)), format(sums["model_requests"], ","),
                         format(sums["input_tokens"], ","), "%.3f" % (sums["total_seconds"] / 3600),
                         "%.3f" % (sums["pacing_seconds"] / 3600)])
    result += _table("lrrrrr", "Mode & Interruptions & Requests & Input tokens & Time (h) & Pacing (h)",
                     overhead, "Extra work from interrupted attempts, reported separately from completed tasks. "
                     "Time includes pacing and retries, but not the wait between restarts.",
                     "tab:online-interruptions")
    combined = totals(rows + interruptions)
    result += ("Including interrupted attempts, the run used "
               f"{combined['model_requests']:,} model responses and "
               f"{combined['input_tokens']:,} input tokens over "
               f"{combined['total_seconds'] / 3600:.3f} hours "
               f"({combined['pacing_seconds'] / 3600:.3f} hours of API pacing).\n")

    return result


def write_tables(run_dir, output=ROOT / "report/online_tables.tex", data_dir=DATA):
    settings, rows, interruptions = load_complete(run_dir, data_dir)
    text = render_tables(settings, rows, interruptions)
    Path(output).write_text(text)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    count = write_tables(args.run_dir)
    print("Validated %d task-mode results; wrote report/online_tables.tex" % count)
