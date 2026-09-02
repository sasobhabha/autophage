"""Unit tests for the Autophage CLI loaders and detectors."""
import tempfile
import unittest
from pathlib import Path

from cli import (read_fasta_records, read_fastq_records, parse_gb_location,
                 parse_genbank, load_dataset, looks_like_dna, InputDataset)


class TestFastaLoader(unittest.TestCase):
    def test_multi_record(self):
        text = ">a header\nACGT\n>b\nTTTT\nGGGG\n"
        recs = list(read_fasta_records(text))
        self.assertEqual([r[0] for r in recs], ["a", "b"])
        self.assertEqual(recs[1][1], "TTTTGGGG")

    def test_load_dataset_fasta_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "g.fasta"
            p.write_text(">chr1\nACGTACGTACGTACGT\n")
            ds = load_dataset(str(p))
            self.assertEqual(ds.kind, "fasta")
            self.assertTrue(ds.genome_fasta.exists())
            ds.cleanup()


class TestFastqLoader(unittest.TestCase):
    def test_read_fastq(self):
        text = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTCCCC\n+\nIIIIIIII\n"
        with tempfile.NamedTemporaryFile("w", suffix=".fastq", delete=False) as f:
            f.write(text)
            name = f.name
        recs = list(read_fastq_records(Path(name)))
        Path(name).unlink()
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0], ("r1", "ACGTACGT"))

    def test_load_dataset_fastq(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reads.fq"
            p.write_text("@r1\nACGTACGTACGTACGTACGT\n+\n" + "I" * 20 + "\n")
            ds = load_dataset(str(p))
            self.assertEqual(ds.kind, "fastq")
            self.assertEqual(ds.genome_fasta.read_text().count(">"), 1)
            ds.cleanup()

    def test_fastq_stream_budget(self):
        import tempfile as tf
        with tf.NamedTemporaryFile("w", suffix=".fastq", delete=False) as f:
            for i in range(20):
                f.write(f"@r{i}\nACGTACGTACGTACGT\n+\n{'I' * 16}\n")
            name = f.name
        try:
            recs = list(read_fastq_records(Path(name), max_bp=40))
            self.assertTrue(len(recs) >= 1)
            self.assertLessEqual(sum(len(s) for _, s in recs), 60 + 16)
            # unlimited reads everything
            all_recs = list(read_fastq_records(Path(name)))
            self.assertEqual(len(all_recs), 20)
        finally:
            Path(name).unlink()


class TestGenBankLoader(unittest.TestCase):
    def test_parse_gb_location(self):
        coords, strand = parse_gb_location("complement(join(10..20,30..40))")
        self.assertEqual(strand, -1)
        self.assertEqual(coords, [(10, 20), (30, 40)])
        coords, strand = parse_gb_location("100..200")
        self.assertEqual(strand, 1)
        self.assertEqual(coords, [(100, 200)])

    def test_parse_genbank(self):
        gb = (
            "LOCUS       demo 120 bp DNA\n"
            "ORIGIN\n"
            "        1 atgaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
            "       61 aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa tttttttttt\n"
            "//\n"
            "     CDS             1..90\n"
            "                     /gene=\"demo\"\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "demo.gbk"
            p.write_text(gb)
            ds = parse_genbank(p)
            self.assertEqual(ds.kind, "genbank")
            text = ds.genome_fasta.read_text()
            self.assertTrue(text.startswith(">demo\n"))
            self.assertEqual(len(text.splitlines()[1]), 90)  # CDS 1..90
            self.assertIsNotNone(ds.cds_fasta)
            self.assertEqual(ds.cds_fasta.read_text().count(">"), 1)
            ds.cleanup()


class TestRawDnaAndAccessionDetection(unittest.TestCase):
    def test_looks_like_dna(self):
        self.assertTrue(looks_like_dna("ACGT" * 30))
        self.assertFalse(looks_like_dna("ACGT" * 10))          # too short
        self.assertFalse(looks_like_dna("!@#$%^&*()" * 10))    # not DNA

    def test_accession_regex(self):
        ds = load_dataset("GCA_000934625.1")
        self.assertEqual(ds.kind, "accession")
        ds.cleanup()

    def test_raw_dna_string(self):
        ds = load_dataset("ATGC" * 40 + "TAGG" * 40)
        self.assertEqual(ds.kind, "raw-dna")
        self.assertEqual(ds.genome_fasta.read_text().count(">"), 1)
        ds.cleanup()

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            load_dataset("definitely-not-a-genome-xyz")


if __name__ == "__main__":
    unittest.main()