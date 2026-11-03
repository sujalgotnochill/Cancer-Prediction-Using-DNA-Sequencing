"""Tests for DNA sequence cleaning and normalization."""

import sys
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.preprocessing.sequence_cleaner import SequenceCleaner, clean_sequence
from dna_cancer.utils.exceptions import PreprocessingError


class TestSequenceCleaner(unittest.TestCase):
    """Test DNA sequence cleaning functionality with synthetic sequences."""

    def test_clean_standard_uppercase(self):
        seq = "atcgatcg"
        cleaned = clean_sequence(seq, to_upper=True)
        self.assertEqual(cleaned, "ATCGATCG")

    def test_remove_whitespace_and_newlines(self):
        seq = "  ATC G\n\tAT CG\r\n "
        cleaned = clean_sequence(seq, remove_whitespace=True)
        self.assertEqual(cleaned, "ATCGATCG")

    def test_remove_non_nucleotide_characters(self):
        seq = "ATC123G_--xyz!ATCG"
        cleaned = clean_sequence(seq, remove_non_nucleotide=True)
        self.assertEqual(cleaned, "ATCGATCG")

    def test_replace_unknown_with_n(self):
        seq = "ATCGXYZ123ATCG"
        cleaned = clean_sequence(seq, replace_unknown_with="N")
        self.assertEqual(cleaned, "ATCGNNNNNNATCG")

    def test_sequence_cleaner_class(self):
        cleaner = SequenceCleaner(to_upper=True, remove_whitespace=True)
        result = cleaner.clean(" atcg\n gcta ")
        self.assertEqual(result, "ATCGGCTA")

    def test_non_string_input_raises_error(self):
        with self.assertRaises(PreprocessingError):
            clean_sequence(12345)  # type: ignore


if __name__ == "__main__":
    unittest.main()

