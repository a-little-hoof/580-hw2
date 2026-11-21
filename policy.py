"""MinHash candidate retrieval followed by dataset-specific equivalence checks.

Only public instructions and verified history are used. Similarity is a retrieval
heuristic, never evidence that two answers are interchangeable. The six corpus
schemas assume immutable scenarios, as specified by the assignment.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from time import perf_counter

from minhash import LSH, MinHash, estimate

CORPUS = {"hosts", "configs", "code", "threads", "revisions", "logs"}
DATASETS = CORPUS | {"tickets", "bugs", "docs", "citations"}


def _norm(value):
    return "".join(re.findall(r"[^\W_]", str(value).casefold()))


def _text(value):
    return " ".join(str(value).casefold().split())


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _cue_catalog(instructions, dataset):
    """Read cue phrases, not answer IDs, from the public resolution/triage guide."""
    catalog, keys = {}, []
    for line in instructions.splitlines():
        heading = re.match(r"\*\*(.*?)\*\*", line.strip())
        if heading:
            keys = (re.findall(r"E-\d+", line) if dataset == "tickets"
                    else [heading[1]])
        cue = re.match(r"- story mentions \*([^*]+)\*", line.strip())
        if cue:
            for key in keys:
                catalog.setdefault(_norm(key), []).append(_text(cue[1]))
    return catalog


def _story(body, cues, actual_only=False):
    """Use a single unambiguous public cue, otherwise require the exact text."""
    story = body
    if actual_only:
        actual = re.search(r"(?im)^\s*(?:#+\s*)?Actual\s*:?\s*$", body)
        if not actual:
            return ("text", _text(body))
        story = re.split(r"(?im)^\s*(?:#+\s*)?(?:Environment|Steps|Expected|Frequency)\s*:?\s*$",
                         body[actual.end():], maxsplit=1)[0]
    paragraphs = [_text(p) for p in re.split(r"\n\s*\n", story)]
    matches = []
    for cue in cues:
        pattern = r"(?<!\w)" + re.escape(cue) + r"(?!\w)"
        matching = [p for p in paragraphs if re.search(pattern, p)]
        if matching:
            matches.append((cue, matching))
    if len(matches) == 1:
        cue, paragraphs = matches[0]
        # Negation or an uncertain mention is insufficient evidence for reuse.
        if not any(re.search(r"\b(?:not|never|no|without|cannot|can't|isn't|wasn't|doesn't)\b", p)
                   for p in paragraphs):
            return ("cue", cue)
    return ("text", _text(story))


def _frame(value):
    value = _text(value)
    value = re.sub(r":\d+", "", value)
    value = re.sub(r"\.py(?=::|\s|$)", "", value)
    value = re.sub(r"\s+in\s+|::|/", ".", value)
    return value if re.fullmatch(r"[a-z_]\w*(?:\.[a-z_]\w*)+", value) else None


def _doc_key(task):
    identifiers = set(re.findall(r"\b(?:CHG|INC|POL)-\d{3,4}\b|\bREL-\d+\.\d+\b",
                                 task["title"].upper()))
    if len(identifiers) != 1:
        return None
    identifier = identifiers.pop()
    family = identifier.split("-")[0]
    if family == "REL":
        return (identifier, "release")
    patterns = {
        "CHG": {"request": r"\b(?:request\w*|pending)\b",
                "approved": r"\b(?:approv\w*|authori[sz]\w*)\b",
                "rejected": r"\b(?:reject\w*|declin\w*)\b"},
        "INC": {"open": r"\b(?:ongoing|under investigation)\b",
                "resolved": r"\b(?:resolv\w*|closed)\b"},
        "POL": {"draft": r"\b(?:draft\w*|for comment)\b",
                "final": r"\b(?:effective|in force)\b"},
    }
    body = _text(task["body"])
    classes = [name for name, pattern in patterns[family].items() if re.search(pattern, body)]
    if len(classes) != 1 or re.search(r"\b(?:not|never|isn't|wasn't)\b", body):
        return (identifier, "exact", body)
    return (identifier, classes[0])


def _author(name):
    if _norm(name) == "etal":
        return ("et al",)
    parts = name.split(",")
    if len(parts) == 2:
        surname, given = parts
        if len(surname.split()) == 1 and given.strip():
            return (_norm(surname), _norm(given)[0:1])
    else:
        words = name.split()
        if len(words) == 2:
            return (_norm(words[-1]), _norm(words[0])[0:1])
    # Unknown/multi-part name formats are compared exactly, not guessed.
    return ("exact", _text(name))


class TaskReuse:
    """One dataset's verified-answer index; ``lookup`` never calls a verifier.

    ``history`` is a history.jsonl path or iterable of verified history records.
    A path infers ``env_dir`` for the public instructions and venue aliases.
    Newly solved tasks must enter through ``add_verified(..., verified=True)``.
    """

    def __init__(self, dataset, history=(), threshold=0.75, seed=73, env_dir=None):
        started = perf_counter()
        if dataset not in DATASETS:
            raise ValueError("Unknown dataset: " + dataset)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        self.dataset, self.threshold = dataset, threshold
        if isinstance(history, (str, Path)):
            path = Path(history)
            env_dir = Path(env_dir) if env_dir is not None else path.parent
            history = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        env_dir = Path(env_dir) if env_dir is not None else None
        instructions = (env_dir / "INSTRUCTIONS.md").read_text() if env_dir else ""
        self.cues = _cue_catalog(instructions, dataset)
        self.venues = {}
        if dataset == "citations" and env_dir:
            for venue in json.loads((env_dir / "db.json").read_text())["venues"]:
                for alias in [venue["name"], *venue["abbrev"]]:
                    self.venues[_text(alias)] = _text(venue["abbrev"][0])
        self.minhash, self.index = MinHash(seed=seed), LSH()
        self.entries, self.answers_by_guard = [], {}
        self.stats = {"indexed": 0, "lookups": 0, "hits": 0, "candidates": 0,
                      "guard_rejections": 0, "conflicts": 0, "lookup_seconds": 0.0,
                      "index_seconds": 0.0}
        for record in history:
            if dataset == "tickets":
                answer = {"resolution_id": record["resolution_id"]}
            elif dataset == "bugs":
                answer = {key: record[key] for key in ("component", "root_cause")}
            elif dataset == "docs":
                answer = {"category": record["category"], "duplicate_of": record["hid"]}
            else:
                answer = record["verdict" if dataset == "citations" else "answer"]
            self._add(record, answer, "history:" + record["hid"])
        self.stats["index_seconds"] = perf_counter() - started

    def guard(self, task):
        """Canonical answer-critical identity, or None when required data is absent."""
        try:
            if self.dataset in CORPUS:
                if any(not task[k] for k in ("scenario", "kind")) or "target" not in task:
                    return None
                return (_norm(task["scenario"]), _norm(task["kind"]),
                        None if task["target"] is None else _norm(task["target"]))
            if self.dataset == "tickets":
                keys = tuple(_norm(task[k]) for k in ("product", "error_code", "platform"))
                if not all(keys) or not task["body"].strip():
                    return None
                return (*keys, _story(task["body"], self.cues.get(keys[1], ())))
            if self.dataset == "bugs":
                frame = _frame(task["top_frame"])
                version = re.fullmatch(r"v?(\d+)\.(\d+)(?:\.\d+)?", _text(task["version"]))
                exception = _norm(task["exception"])
                if not frame or not version or not exception or not task["body"].strip():
                    return None
                return (exception, frame, version.group(1, 2),
                        _story(task["body"], self.cues.get(exception, ()), actual_only=True))
            if self.dataset == "docs":
                return _doc_key(task)
            citation = task["citation"]
            doi = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "",
                         _text(citation.get("doi", "")))
            title = tuple(re.findall(r"[^\W_]+", citation["title"].casefold()))
            if not title or not isinstance(citation["authors"], list):
                return None
            authors = tuple(_author(name) for name in citation["authors"])
            venue = _text(citation["venue"])
            return (doi, title, authors, self.venues.get(venue, venue), _json(citation["year"]))
        except (KeyError, TypeError, AttributeError):
            return None

    def _retrieval_text(self, task, guard):
        if self.dataset == "citations":
            return _json(guard)  # citation styles/pages are explicitly ungraded
        fields = ("title", "subject", "body")
        return _json(guard[:3]) + "\n" + "\n".join(str(task.get(k, "")) for k in fields)

    def _add(self, task, answer, source):
        guard = self.guard(task)
        if guard is None:
            return False
        signature = self.minhash.signature(self._retrieval_text(task, guard))
        key = len(self.entries)
        self.entries.append({"guard": guard, "signature": signature,
                             "answer": deepcopy(answer), "source": source})
        self.answers_by_guard.setdefault(guard, set()).add(_json(answer))
        self.index.add(key, signature)
        self.stats["indexed"] += 1
        return True

    def add_verified(self, task, answer, verified=False):
        """Index only an externally verified answer, preserving its original fields."""
        if verified is not True:
            raise ValueError("An answer must pass the verifier before indexing")
        if not isinstance(answer, dict):
            raise TypeError("answer must be a JSON object")
        started = perf_counter()
        added = self._add(task, answer, "verified:" + str(task.get("tid", len(self.entries))))
        self.stats["index_seconds"] += perf_counter() - started
        return added

    def lookup(self, task):
        started = perf_counter()
        self.stats["lookups"] += 1
        guard = self.guard(task)
        result = None
        candidates = set()
        if guard is not None:
            signature = self.minhash.signature(self._retrieval_text(task, guard))
            candidates = self.index.query(signature)
            self.stats["candidates"] += len(candidates)
            valid = []
            for key in sorted(candidates):
                entry = self.entries[key]
                if entry["guard"] != guard:
                    self.stats["guard_rejections"] += 1
                    continue
                similarity = estimate(signature, entry["signature"])
                if similarity >= self.threshold:
                    valid.append((similarity, key))
            if valid and len(self.answers_by_guard[guard]) != 1:
                self.stats["conflicts"] += 1
            elif valid:
                similarity, key = max(valid, key=lambda pair: (pair[0], -pair[1]))
                entry = self.entries[key]
                result = {"answer": deepcopy(entry["answer"]), "source": entry["source"],
                          "similarity": similarity, "candidates": len(candidates)}
                self.stats["hits"] += 1
        elapsed = perf_counter() - started
        self.stats["lookup_seconds"] += elapsed
        if result is not None:
            result["lookup_seconds"] = elapsed
        return result
