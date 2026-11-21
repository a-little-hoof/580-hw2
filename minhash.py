"""Deterministic three-word MinHash and banded locality-sensitive hashing.

The 128 affine hashes are pairwise independent over a prime field, an efficient
approximation to ideal min-wise permutations. LSH only proposes candidates;
callers must still check similarity and whether reuse is valid.
"""

import hashlib
import random
import re


PRIME = (1 << 61) - 1
NUM_PERM = 128


def shingles(text):
    """Return the set of consecutive, case-folded three-word tuples.

    Numbers are words too: they are never masked. Inputs shorter than three
    words have no shingles and are deliberately not indexed by LSH.
    """
    words = re.findall(r"\w+", text.casefold())
    return set(zip(words, words[1:], words[2:]))


def estimate(left, right):
    """Estimate Jaccard similarity by the fraction of matching minima."""
    if len(left) != len(right) or not left:
        raise ValueError("signatures must have the same positive length")
    return sum(a == b for a, b in zip(left, right)) / len(left)


class MinHash:
    def __init__(self, seed=73, num_perm=NUM_PERM):
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        rng = random.Random(seed)
        self.coefficients = [(rng.randrange(1, PRIME), rng.randrange(PRIME))
                             for _ in range(num_perm)]

    def signature(self, text):
        # A delimiter absent from regex words avoids ambiguous tuple encoding.
        values = [int.from_bytes(hashlib.blake2b("\0".join(s).encode("utf-8"),
                                                digest_size=8).digest(), "big")
                  % PRIME for s in shingles(text)]
        if not values:
            return (PRIME,) * len(self.coefficients)
        return tuple(min((a * value + b) % PRIME for value in values)
                     for a, b in self.coefficients)

    estimate = staticmethod(estimate)


class LSH:
    """32 bands of four rows by default; each bucket retains exact band tuples."""

    def __init__(self, num_perm=NUM_PERM, bands=32):
        if num_perm <= 0 or bands <= 0 or num_perm % bands:
            raise ValueError("bands must divide the positive signature length")
        self.num_perm = num_perm
        self.rows = num_perm // bands
        self.buckets = {}
        self.signatures = {}

    def _bands(self, signature):
        if len(signature) != self.num_perm:
            raise ValueError("wrong signature length")
        if all(value == PRIME for value in signature):
            return []
        return [(start, tuple(signature[start:start + self.rows]))
                for start in range(0, self.num_perm, self.rows)]

    def add(self, key, signature):
        bands = self._bands(signature)
        self.remove(key)
        if not bands:
            return
        self.signatures[key] = tuple(signature)
        for band in bands:
            self.buckets.setdefault(band, set()).add(key)

    def query(self, signature):
        matches = set()
        for band in self._bands(signature):
            matches.update(self.buckets.get(band, ()))
        return matches

    def remove(self, key):
        signature = self.signatures.pop(key, None)
        if signature is None:
            return
        for band in self._bands(signature):
            bucket = self.buckets[band]
            bucket.remove(key)
            if not bucket:
                del self.buckets[band]
