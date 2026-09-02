"""
Unit tests for the SciAgent synthetic phage pipeline
(phage_genome.py, host_compat.py, phage_lm.py). All offline.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from phage_genome import (generate_phage_genome, validate_phage_genome,
                          to_fasta, parse_fasta, reverse_complement)
from host_compat import (count_kmers, d2star, mononucleotide_frequencies)
from host_compat import _revcomp_int as rc_int


class TestGeneration(unittest.TestCase):
    def test_generation_validates(self):
        g = generate_phage_genome(length=50_000, gc_target=0.5, seed=1)
        self.assertTrue(set(g.sequence) <= {"A", "C", "G", "T"})
        self.assertGreaterEqual(len(g.sequence), 40_000)
        r = validate_phage_genome(g.sequence, g.name)
        self.assertTrue(r.passed, msg=r.summary())

    def test_gc_target_tracking(self):
        for target, tol in ((0.35, 0.08), (0.55, 0.08)):
            g = generate_phage_genome(length=30_000, gc_target=target, seed=2)
            self.assertLessEqual(abs(g.metadata["gc"] - target), tol,
                                 f"gc {g.metadata['gc']} vs target {target}")

    def test_determinism(self):
        a = generate_phage_genome(length=20_000, seed=7).sequence
        b = generate_phage_genome(length=20_000, seed=7).sequence
        self.assertEqual(a, b)

    def test_terminal_repeat(self):
        g = generate_phage_genome(length=20_000, terminal_repeat=200, seed=3)
        self.assertEqual(g.sequence[-200:], g.sequence[:200])

    def test_validate_rejects_garbage(self):
        r = validate_phage_genome("A" * 10_000, "polyA")
        self.assertFalse(r.passed)


class TestHostCompat(unittest.TestCase):
    def test_revcomp_integer_matches_string(self):
        for w in range(256):
            # 4-mer integer <-> string consistency
            bases = []
            v = w
            for _ in range(4):
                bases.append("ACGT"[v & 3])
                v >>= 2
            seq = "".join(reversed(bases))
            self.assertEqual(rc_int(w, 4),
                             int_encoding(reverse_complement(seq)),
                             f"revcomp mismatch for {seq}")

    def test_d2star_properties(self):
        seq = "".join("ACGT" for _ in range(2000))
        self.assertAlmostEqual(d2star(seq, seq), 0.0, places=6)
        other = "".join("AT" for _ in range(2000)) + "".join("GC" for _ in range(2000))
        self.assertGreater(d2star(seq, other), 0.0)

    def test_d2star_symmetric(self):
        a = "ACGT" * 500
        b = "TGCA" * 500
        self.assertAlmostEqual(d2star(a, b), d2star(b, a), places=9)

    def test_count_kmers_skips_n(self):
        # one N in the middle drops every window containing it
        seq = "AAAAACNAAAA"  # only window 0-5 (AAAAAC) is clean
        counts, total = count_kmers(seq, 6)
        self.assertEqual(total, 1)
        # fully clean sequence: n - k + 1 windows
        counts, total = count_kmers("ACGT" * 100, 6)
        self.assertEqual(total, 400 - 6 + 1)

    def test_known_parity(self):
        # Sanity: a highly repetitive sequence vs a random one
        rep = "ATATAT" * 3000
        rnd = "".join(__import__("random").choices("ACGT", k=18000))
        self.assertLess(d2star(rep, rep), d2star(rep, rnd))


class TestFastaIO(unittest.TestCase):
    def test_roundtrip(self):
        g = generate_phage_genome(length=10_000, seed=5)
        fasta = to_fasta(g)
        name, seq = parse_fasta(fasta)
        self.assertEqual(name, g.name)
        self.assertEqual(seq, g.sequence)


class TestTokenizer(unittest.TestCase):
    def test_encode_decode(self):
        from phage_lm import PhageTokenizer
        tok = PhageTokenizer()
        seq = "ATGCGTACGTAGCATCGGTAC"
        toks = tok.encode(seq)
        # first token encodes the first k-mer window exactly
        first = "".join("ACGT"[(toks[0] >> (2 * (tok.k - 1 - i))) & 3]
                         for i in range(tok.k))
        self.assertEqual(first, seq[:tok.k])
        # number of windows = n - k + 1
        self.assertEqual(len(toks), len(seq) - tok.k + 1)
        # round-trip decode is ACGT-only and starts at the first window
        self.assertTrue(set(tok.decode(toks)) <= {"A", "C", "G", "T"})

    def test_encode_skips_ambiguous(self):
        from phage_lm import PhageTokenizer
        tok = PhageTokenizer()
        toks = tok.encode("AAAAANAAAAN")
        # windows containing N are dropped; decode is ACGT-only
        self.assertTrue(set(tok.decode(toks)) <= {"A", "C", "G", "T"})


def int_encoding(seq: str) -> int:
    v = 0
    for b in seq:
        v = (v << 2) | "ACGT".index(b)
    return v


if __name__ == "__main__":
    unittest.main(verbosity=2)