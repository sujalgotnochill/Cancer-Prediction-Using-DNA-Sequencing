"""Unit tests for streaming FASTA collection inspector."""

import tempfile
import unittest
from pathlib import Path

from dna_cancer.data.fasta_inspector import (
    detect_file_encoding,
    format_bytes,
    inspect_collection,
    inspect_fasta_file,
)


class TestFastaInspector(unittest.TestCase):
    """Test streaming FASTA inspection functionality."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_format_bytes(self) -> None:
        self.assertEqual(format_bytes(500), "500.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024), "1.00 GB")

    def test_detect_encoding(self) -> None:
        p = self.path / "sample.fasta"
        p.write_text(">seq1\nATGC\n", encoding="utf-8")
        self.assertEqual(detect_file_encoding(p), "utf-8")

    def test_inspect_single_fasta(self) -> None:
        p = self.path / "test.fasta"
        content = (
            ">lcl|NC_001.1_cds_1 [gene=GENE1] [protein=Protein Alpha] [locus_tag=L001]\n"
            "ATGCATGCATGC\n"
            ">lcl|NC_001.1_cds_2 [gene=GENE2] [protein=Protein Beta] [locus_tag=L002]\n"
            "GGCCAATTGGCC\n"
            ">lcl|NC_001.1_cds_3 [gene=GENE1] [protein=Protein Alpha] [locus_tag=L001]\n"
            "ATGCATGCATGC\n"  # Exact duplicate of seq1
        )
        p.write_text(content, encoding="utf-8")

        summary, hashes = inspect_fasta_file(p)
        self.assertEqual(summary.filename, "test.fasta")
        self.assertTrue(summary.is_valid_fasta)
        self.assertEqual(summary.total_records, 3)
        self.assertEqual(summary.total_nucleotides, 36)
        self.assertEqual(summary.min_length, 12)
        self.assertEqual(summary.max_length, 12)
        self.assertEqual(summary.avg_length, 12.0)
        self.assertEqual(summary.internal_duplicates, 1)
        self.assertEqual(summary.unique_sequences, 2)
        self.assertEqual(len(hashes), 2)
        self.assertIn("GENE1", summary.sample_genes)
        self.assertIn("Protein Alpha", summary.sample_proteins)
        self.assertEqual(summary.alphabet_distribution["A"], 8)
        self.assertEqual(summary.alphabet_distribution["C"], 10)
        self.assertEqual(summary.alphabet_distribution["G"], 10)
        self.assertEqual(summary.alphabet_distribution["T"], 8)

    def test_inspect_collection_and_overlap(self) -> None:
        f1 = self.path / "col1.fasta"
        f2 = self.path / "col2.fasta"

        f1.write_text(">s1\nATGCATGC\n>s2\nTGCATGCA\n", encoding="utf-8")
        f2.write_text(">s3\nATGCATGC\n>s4\nCCCCGGGG\n", encoding="utf-8")

        out_json = self.path / "report.json"
        res = inspect_collection(self.path, output_json_path=out_json)

        self.assertEqual(res["total_collection_records"], 4)
        self.assertIn("col1.fasta <-> col2.fasta", res["cross_file_overlap"])
        # Exactly 1 sequence ('ATGCATGC') overlaps between col1 and col2
        self.assertEqual(res["cross_file_overlap"]["col1.fasta <-> col2.fasta"], 1)
        self.assertTrue(out_json.exists())


if __name__ == "__main__":
    unittest.main()

