"""Lossless tool-output reuse against full outputs still in the trajectory.

Call ``retain_texts`` after context compaction, before processing new outputs.
Create a fresh cache for every task. A compressed output is never a new base.
Character and lexical-token gates are conservative heuristics; evaluation must
measure the model's actual input tokens separately.
"""

import difflib
import hashlib
import json
import re
import time

from minhash import LSH, MinHash, estimate


HEADER = ("[reuse] Exact output from the earlier full tool result named below. "
          "Apply edits simultaneously to its original splitlines(keepends=True): "
          "[start, stop, replacement] replaces zero-based lines [start, stop). "
          "All unmentioned bytes are unchanged; an empty edit list means identical.\n")
UNSAFE = re.compile(r"(?im)^(?:ERROR:|BLOCKED:|\[reuse\]|"
                    r"\.\.\. truncated at |\.\.\. \d+ characters cut)")


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_proxy(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def reconstruct(base, encoded):
    """Apply an emitted envelope, checking the full reference and target hashes."""
    if not encoded.startswith(HEADER):
        raise ValueError("not a content reuse envelope")
    payload = json.loads(encoded[len(HEADER):])
    if _digest(base) != payload["base_sha256"]:
        raise ValueError("base output does not match the reference")
    lines = base.splitlines(keepends=True)
    pieces, cursor = [], 0
    for start, stop, replacement in payload["edits"]:
        if not 0 <= cursor <= start <= stop <= len(lines):
            raise ValueError("invalid or overlapping line edits")
        pieces.extend(("".join(lines[cursor:start]), replacement))
        cursor = stop
    pieces.append("".join(lines[cursor:]))
    result = "".join(pieces)
    if _digest(result) != payload["target_sha256"]:
        raise ValueError("reconstructed output does not match its digest")
    return result


class ContentCache:
    def __init__(self, threshold=0.8, min_savings=0.15, min_chars=256, seed=73):
        if not 0 <= threshold <= 1 or not 0 <= min_savings < 1:
            raise ValueError("threshold and min_savings must be fractions")
        if min_chars < 0:
            raise ValueError("min_chars must be nonnegative")
        self.threshold = threshold
        self.min_savings = min_savings
        self.min_chars = min_chars
        self.minhash = MinHash(seed=seed)
        self.lsh = LSH()
        self.bases = {}
        self.next_id = 0
        self.stats = dict(calls=0, reused=0, chars_in=0, chars_out=0, seconds=0.0)

    def reset(self):
        """Forget references, e.g. when a conversation is replaced entirely."""
        self.lsh = LSH()
        self.bases.clear()

    def retain_texts(self, texts):
        """Drop full bases that no longer occur verbatim in the live context."""
        available = set(texts)
        for key, base in list(self.bases.items()):
            if base["text"] not in available:
                self.lsh.remove(key)
                del self.bases[key]

    def process(self, text, source="tool output"):
        start = time.perf_counter()
        try:
            output = self._process(text, source)
            self.stats["calls"] += 1
            self.stats["reused"] += output != text
            self.stats["chars_in"] += len(text)
            self.stats["chars_out"] += len(output)
            return output
        finally:
            self.stats["seconds"] += time.perf_counter() - start

    def _process(self, text, source):
        if len(text) < self.min_chars or UNSAFE.search(text):
            return text
        signature = self.minhash.signature(text)
        candidates = sorted(self.lsh.query(signature),
                            key=lambda key: (-estimate(signature, self.bases[key]["signature"]),
                                             key))
        best = text
        new_lines = text.splitlines(keepends=True)
        target_digest = _digest(text)
        token_limit = _token_proxy(text) * (1 - self.min_savings)
        char_limit = len(text) * (1 - self.min_savings)
        for key in candidates:
            base = self.bases[key]
            if estimate(signature, base["signature"]) < self.threshold:
                break
            old_lines = base["text"].splitlines(keepends=True)
            edits = [[i, j, "".join(new_lines[k:l])]
                     for tag, i, j, k, l in difflib.SequenceMatcher(
                         None, old_lines, new_lines, autojunk=False).get_opcodes()
                     if tag != "equal"]
            payload = {"base_source": base["source"], "base_sha256": base["digest"],
                       "target_source": source, "target_sha256": target_digest,
                       "edits": edits}
            encoded = HEADER + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if (len(encoded) < len(best) and len(encoded) <= char_limit
                    and _token_proxy(encoded) <= token_limit
                    and reconstruct(base["text"], encoded) == text):
                best = encoded
        if best == text:
            key = self.next_id
            self.next_id += 1
            self.bases[key] = {"text": text, "source": source, "digest": target_digest,
                               "signature": signature}
            self.lsh.add(key, signature)
        return best
