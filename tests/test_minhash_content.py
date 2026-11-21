import json
import random
import unittest

from content import ContentCache, HEADER, reconstruct
from minhash import LSH, NUM_PERM, MinHash, estimate, shingles


class MinHashTests(unittest.TestCase):
    def test_word_shingles_and_numbers(self):
        self.assertEqual(shingles("ONE two 90 four"),
                         {("one", "two", "90"), ("two", "90", "four")})
        self.assertEqual(shingles("two words"), set())

    def test_seed_and_permutation_count(self):
        text = "the quick brown fox jumped over a lazy dog"
        first = MinHash().signature(text)
        self.assertEqual(len(first), NUM_PERM)
        self.assertEqual(first, MinHash().signature(text))
        self.assertNotEqual(first, MinHash(seed=74).signature(text))
        self.assertEqual(estimate(first, first), 1)

    def test_estimate_tracks_exact_jaccard(self):
        # Different sequences share a long contiguous segment, so shingle
        # overlap is known independently of the MinHash implementation.
        a = " ".join("w%d" % i for i in range(1000))
        b = " ".join("w%d" % i for i in range(300, 1300))
        sa, sb = shingles(a), shingles(b)
        exact = len(sa & sb) / len(sa | sb)
        measured = estimate(MinHash().signature(a), MinHash().signature(b))
        self.assertLess(abs(measured - exact), 0.12)

    def test_lsh_retrieves_updates_removes_and_skips_empty(self):
        mh, lsh = MinHash(), LSH()
        a = mh.signature("one two three four five six")
        b = mh.signature("alpha beta gamma delta epsilon zeta")
        lsh.add("x", a)
        self.assertEqual(lsh.query(a), {"x"})
        lsh.add("x", b)
        self.assertEqual(lsh.query(a), set())
        self.assertEqual(lsh.query(b), {"x"})
        lsh.remove("x")
        self.assertEqual(lsh.query(b), set())
        empty = mh.signature("")
        lsh.add("empty", empty)
        self.assertEqual(lsh.query(empty), set())


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.base = "".join("parameter_%03d = %d # stable setting\n" % (i, i)
                            for i in range(120))

    def test_exact_repeat_and_lossless_edits(self):
        cache = ContentCache()
        self.assertEqual(cache.process(self.base, "config/dev.yaml"), self.base)
        for new in [self.base,
                    self.base.replace("parameter_005 = 5", "parameter_005 = -5.001"),
                    self.base.replace("parameter_007 = 7 # stable setting\n", ""),
                    self.base + "\nnew_identifier = 90\r\n",
                    self.base.rstrip("\n")]:
            encoded = cache.process(new, "config/prod.yaml")
            self.assertTrue(encoded.startswith(HEADER))
            self.assertEqual(reconstruct(self.base, encoded), new)
        self.assertEqual(len(cache.bases), 1, "deltas must never become bases")
        self.assertEqual(cache.stats["reused"], 5)

    def test_random_changes_reconstruct_exactly(self):
        rng = random.Random(73)
        cache = ContentCache()
        cache.process(self.base, "base")
        for _ in range(30):
            lines = self.base.splitlines(keepends=True)
            at = rng.randrange(len(lines))
            lines[at:at + rng.randrange(3)] = ["\tcritical = 09:42 / id-A # 3.14\r\n", "\n"]
            text = "".join(lines)
            encoded = cache.process(text, "modified")
            self.assertTrue(encoded.startswith(HEADER))
            self.assertEqual(reconstruct(self.base, encoded), text)

    def test_invalid_base_rejected(self):
        cache = ContentCache()
        cache.process(self.base)
        encoded = cache.process(self.base)
        with self.assertRaises(ValueError):
            reconstruct(self.base + " ", encoded)
        payload = json.loads(encoded[len(HEADER):])
        payload["edits"] = [[10, 9, "bad"]]
        with self.assertRaises(ValueError):
            reconstruct(self.base, HEADER + json.dumps(payload))

    def test_compacted_base_is_not_used(self):
        cache = ContentCache()
        cache.process(self.base)
        cache.retain_texts(["compacted summary"])
        self.assertEqual(cache.process(self.base), self.base)
        cache.reset()
        self.assertEqual(cache.process(self.base), self.base)

    def test_unsafe_or_unprofitable_returns_original(self):
        cache = ContentCache()
        for text in ["x = 7", "ERROR: " + self.base,
                     self.base + "... truncated at 4000 lines; the file has 4001"]:
            self.assertEqual(cache.process(text), text)
            self.assertEqual(cache.process(text), text)
        self.assertEqual(cache.bases, {})
        short = ContentCache(min_chars=0)
        text = "the same short output repeatedly"
        self.assertEqual(short.process(text), text)
        self.assertEqual(short.process(text), text)


if __name__ == "__main__":
    unittest.main()
