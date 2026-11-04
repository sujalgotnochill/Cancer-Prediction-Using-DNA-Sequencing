"""Tests for complete preprocessing and feature extraction workflow."""

import sys
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.data.validation import SequenceValidator
from dna_cancer.features.kmer_extractor import KMerExtractor, extract_kmers
from dna_cancer.preprocessing.sequence_cleaner import SequenceCleaner, clean_sequence
from dna_cancer.preprocessing.sequence_processor import SequenceProcessor
from dna_cancer.utils.exceptions import FeatureExtractionError, PreprocessingError


class TestPreprocessingAndFeaturesFlow(unittest.TestCase):
    """Test the complete preprocessing and feature generation pipeline."""

    def setUp(self):
        self.cleaner = SequenceCleaner(to_upper=True, remove_whitespace=True)
        self.validator = SequenceValidator(allowed_bases="ACGTN", min_length=4)
        self.processor = SequenceProcessor(cleaner=self.cleaner, validator=self.validator)

    def test_complete_flow_standard_sequence(self):
        raw_sequence = "  atgc\r\n gatcg\t "
        # 1. Clean
        cleaned = self.cleaner.clean(raw_sequence)
        self.assertEqual(cleaned, "ATGCGATCG")

        # 2. Validate
        self.assertTrue(self.validator.is_valid(cleaned))

        # 3. K-mer extraction
        kmers = extract_kmers(cleaned, k=6, stride=1, to_lower=True, as_sentence=False)
        self.assertEqual(kmers, ["atgcga", "tgcgat", "gcgatc", "cgatcg"])

        # 4. K-mer sentence representation
        sentence = extract_kmers(cleaned, k=6, stride=1, to_lower=True, as_sentence=True)
        self.assertEqual(sentence, "atgcga tgcgat gcgatc cgatcg")

    def test_lowercase_sequences(self):
        raw = "acgtacgt"
        cleaned = self.cleaner.clean(raw)
        self.assertEqual(cleaned, "ACGTACGT")
        self.assertTrue(self.validator.is_valid(cleaned))

    def test_whitespace_and_linebreaks_removal(self):
        raw = "\n\t  A T C G \r\n C T A G \n\t "
        cleaned = self.cleaner.clean(raw)
        self.assertEqual(cleaned, "ATCGCTAG")
        self.assertTrue(self.validator.is_valid(cleaned))

    def test_ambiguous_nucleotide_characters(self):
        raw = "ATCGNATCGN"
        cleaned = self.cleaner.clean(raw)
        self.assertEqual(cleaned, "ATCGNATCGN")
        self.assertTrue(self.validator.is_valid(cleaned))

        # Disallow N in validator
        strict_validator = SequenceValidator(allowed_bases="ACGT")
        self.assertFalse(strict_validator.is_valid(cleaned))

        # Replace non-ACGT with N
        cleaner_replace = SequenceCleaner(replace_unknown_with="N")
        cleaned_rep = cleaner_replace.clean("ATCGXYZATCG")
        self.assertEqual(cleaned_rep, "ATCGNNNATCG")

    def test_empty_sequences(self):
        empty_raw = "   \n\t "
        cleaned = self.cleaner.clean(empty_raw)
        self.assertEqual(cleaned, "")
        self.assertFalse(self.validator.is_valid(cleaned))

        # Process single with drop_invalid
        self.assertIsNone(self.processor.process_single(empty_raw))

        # Process single without drop_invalid
        strict_processor = SequenceProcessor(
            cleaner=self.cleaner,
            validator=self.validator,
            drop_invalid=False,
        )
        with self.assertRaises(PreprocessingError):
            strict_processor.process_single(empty_raw)

    def test_sequences_shorter_than_k(self):
        short_seq = "ATC"
        cleaned = self.cleaner.clean(short_seq)
        # Sequence is length 3, k=6
        kmers = extract_kmers(cleaned, k=6, as_sentence=False)
        self.assertEqual(kmers, [])

        sentence = extract_kmers(cleaned, k=6, as_sentence=True)
        self.assertEqual(sentence, "")

    def test_different_kmer_lengths(self):
        seq = "ATGCGATC"  # Length 8
        # k=1: 8 kmers
        k1 = extract_kmers(seq, k=1, as_sentence=False)
        self.assertEqual(len(k1), 8)
        self.assertEqual(k1, ["a", "t", "g", "c", "g", "a", "t", "c"])

        # k=3: 8 - 3 + 1 = 6 kmers
        k3 = extract_kmers(seq, k=3, as_sentence=False)
        self.assertEqual(len(k3), 6)
        self.assertEqual(k3, ["atg", "tgc", "gcg", "cga", "gat", "atc"])

        # k=8: 8 - 8 + 1 = 1 kmer
        k8 = extract_kmers(seq, k=8, as_sentence=False)
        self.assertEqual(k8, ["atgcgatc"])

    def test_deterministic_output_for_identical_inputs(self):
        seq = "  ATCGGCTA\nCGTA\t "
        res1 = self.processor.process_single(seq)
        res2 = self.processor.process_single(seq)
        self.assertEqual(res1, res2)

        kmers1 = extract_kmers(res1, k=6, stride=1, as_sentence=True)
        kmers2 = extract_kmers(res2, k=6, stride=1, as_sentence=True)
        self.assertEqual(kmers1, kmers2)


if __name__ == "__main__":
    unittest.main()

