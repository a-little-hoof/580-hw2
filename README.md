# COMP 580 Assignment 2: MinHash for agent reuse

This repository implements task-level answer reuse and lossless content reuse
around the provided Spartacus agent. The optimization lives in three modules:

- `minhash.py`: three-word shingles, 128-component MinHash signatures, and LSH.
- `policy.py`: candidate retrieval and conservative task-equivalence checks;
  only previously verified answers may enter the cache.
- `content.py`: repeated-output detection and exact differences relative to
  previously observed content, with fallback to the original output.

`evaluate.py` runs the checks and experiments. The assignment datasets, verifiers,
and Spartacus source are pinned Git submodules in `vendor/`; they are unmodified.
The report is maintained in [Overleaf](https://www.overleaf.com/project/6af88ef4c678325bec6ebbc8),
with a compiled [PDF](report/main.pdf) and LaTeX source in `report/`.

## Setup

Python 3.9 or newer and Git are required. The implementation uses only Python's
standard library; no Python packages need to be installed.

```bash
git clone --recurse-submodules git@github.com:a-little-hoof/580-hw2.git
cd 580-hw2
git submodule update --init --recursive
python --version
```

The Gemini experiments require a valid Gemini API key. Spartacus reads
`GEMINI_API_KEY` from the environment or from `~/.config/spartacus/env`, whose
contents can be a single `GEMINI_API_KEY=...` line. Its alternative
`SPARTACUS_API_KEY` takes precedence if both are set. Keep credentials outside
the repository; use permissions `600` for the configuration file.

Always specify `gemini-3.5-flash-lite` explicitly. The pinned harness's default
model and `flash-lite` alias resolve to a different model.

## Reproduce the checks

```bash
python -m unittest discover -s tests -v
python evaluate.py --offline --output results/offline
```

The offline run audits task reuse using the provided verified histories and
replays public corpus content through the content compressor. These measurements
check cache decisions, reconstruction, compression, and local overhead. They
are **not** an agent evaluation: they cannot establish Gemini task accuracy,
model-request savings, input-token savings, or end-to-end model latency.

## Run the small-sample model evaluation

```bash
python evaluate.py \
  --output results/gemini-minimal \
  --datasets tickets bugs docs citations hosts configs threads revisions \
  --modes baseline task content combined \
  --model gemini-3.5-flash-lite \
  --max-turns 20 --limit 5 \
  --request-interval 5
```

Requests start at least five seconds apart to respect the observed API quota.
If a run is interrupted, resume it with identical settings and source files:

```bash
python evaluate.py \
  --output results/gemini-minimal \
  --datasets tickets bugs docs citations hosts configs threads revisions \
  --modes baseline task content combined \
  --model gemini-3.5-flash-lite \
  --max-turns 20 --limit 5 \
  --request-interval 5 \
  --resume
```

Resume preserves completed task results and rebuilds verified caches in task
order. Interrupted trajectories are saved separately under `interruptions/`;
they are retried on resume and are not counted as accuracy failures.

The eight selected datasets cover all four record-based datasets and four
corpus-based datasets: tickets, bugs, docs, citations, hosts, configs, threads,
and revisions. The online comparison uses the first five incoming tasks from
each dataset after sorting by `(seq, tid)`: **40 tasks per mode and 160 task-mode
results**. Selection is fixed and identical across all four modes. The full
420-task collection remains covered by the separate offline audit.

The assignment specifies eight datasets and four configurations, but no minimum
task count. Five tasks per dataset is a budget-conscious experimental choice,
not an instructor-stated minimum. This small, deterministic prefix sample is
not representative of every task; dataset accuracy changes in steps of 20
percentage points, and conclusions about efficiency and correctness are limited
to these tasks. All modes use the same model, prompts, tool definitions, and
execution limit. The modes are:

| Mode | Task reuse | Content reuse |
| --- | --- | --- |
| `baseline` | No | No |
| `task` | Yes | No |
| `content` | No | Yes |
| `combined` | Yes | Yes |

The unmodified Gemini provider uses temperature `0.4` and a maximum of `65536`
output tokens. Hashing uses a fixed seed; provider sampling may still vary across
runs. The runner uses Spartacus's unmodified loop, provider, and core file tools.
All modes expose `read_file`, `grep`, `list_files`, and `write_file`; writes are
restricted to `answer.json`. Shell execution is unavailable. Each task receives
an isolated copy of the public dataset environment, excluding hidden labels and
verifier sources.

The evaluator records verifier pass/fail, tool calls, model requests, provider
input-token usage, and completion time, including reuse-processing overhead.
Model requests count successful provider completions, including the possible
final wrap-up call; internal HTTP retries are excluded. Completed-trajectory
wall time includes pacing waits and provider retry delays; `pacing_seconds`
records the explicit request-spacing wait separately. Interrupted-attempt costs
are retained in their interruption artifacts rather than completed-task totals.
The evaluator retains raw per-task
metrics, answers, transcripts, and run provenance in the
selected output directory. Only a successful verification allows a newly
generated answer to enter the reuse cache; verifier internals and hidden labels
are never passed to the model or reuse policy.

## Evaluation status

The unit/integration checks pass. On all 420 incoming tasks, the default
history-only policy reuses 79 answers (18.81% coverage); all 79 pass the official
verifiers. Across six thresholds, all 462 accepted decisions pass (these include
repeated decisions on the same tasks). Offline corpus replay reconstructs all
349 emitted deltas exactly. Character savings are 22.97% for hosts, 38.31% for
configs, 34.29% for revisions, and 0% for threads. Raw measurements and source
hashes are in `results/offline/`.

Access to `gemini-3.5-flash-lite` is confirmed. The earlier full-sweep attempt
stopped at 41 baseline ticket results after reaching the free-tier daily quota
of 500 requests per project and model. That exploratory checkpoint and its
service diagnostic remain in `results/gemini/`; they are not a completed
four-mode comparison.

The active scope is the 160-result small-sample experiment in
`results/gemini-minimal/`. All 160 runs start fresh with a common workspace path
and identical prompts; the 41 earlier runs remain separate pilot evidence.
The live progress and completion state are recorded in
[`results/gemini-minimal/run_status.json`](results/gemini-minimal/run_status.json).
A final comparison is available only after every declared task-mode result has
been validated.

The continuation worker uses the exact resume command above with the same
source and settings across free-tier quota resets. It preserves completed
results, waits when the daily quota is exhausted, and records its status; a
billing upgrade is not required. After completion, it validates the sample,
generates the comparison tables, compiles the report, and updates GitHub and
Overleaf if their remote versions have not changed. Offline compression and
cache-reuse measurements remain distinct from the model experiment.

After the complete run, regenerate the comparison tables with:

```bash
python report/summarize.py results/gemini-minimal
python report/analyze_failures.py results/gemini-minimal
```

This command requires every task in the declared sample in all four modes and
refuses incomplete, duplicate, or mismatched task results. Only a successfully
validated run produces `report/online_tables.tex`. The report includes these
tables, with the discussion of the submitted results in `report/main.tex`.

The failure audit pairs baseline and combined outcomes, then replays the saved
tool calls for regressions that received compressed outputs. It records message
indices and exact reconstruction checks in `report/failure_analysis.json`.
This audit uses public files and saved traces, with no model or verifier calls.
