"""Tests for k-mer extraction."""

import sys
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.features.kmer_extractor import KMerExtractor, extract_kmers
from dna_cancer.utils.exceptions import FeatureExtractionError


class TestKMerExtractor(unittest.TestCase):
    """Test k-mer extraction on synthetic DNA sequences."""

    def test_extract_kmers_default_k6(self):
        seq = "ATGCGATCG"
        # Length 9, k=6 -> 9 - 6 + 1 = 4 kmers:
        # atgcga, tgcgat, gcgatc, cgatcg
        kmers = extract_kmers(seq, k=6, stride=1, to_lower=True, as_sentence=False)
        expected = ["atgcga", "tgcgat", "gcgatc", "cgatcg"]
        self.assertEqual(kmers, expected)

    def test_extract_kmers_stride_2(self):
        seq = "ATGCGATCG"
        # stride 2: index 0 ("atgcga"), index 2 ("gcgatc")
        kmers = extract_kmers(seq, k=6, stride=2, to_lower=True, as_sentence=False)
        expected = ["atgcga", "gcgatc"]
        self.assertEqual(kmers, expected)

    def test_extract_kmers_as_sentence(self):
        seq = "ATGCGATCG"
        sentence = extract_kmers(seq, k=6, stride=1, to_lower=True, as_sentence=True)
        self.assertEqual(sentence, "atgcga tgcgat gcgatc cgatcg")

    def test_sequence_shorter_than_k(self):
        seq = "ATC"
        kmers = extract_kmers(seq, k=6, as_sentence=False)
        self.assertEqual(kmers, [])

        sentence = extract_kmers(seq, k=6, as_sentence=True)
        self.assertEqual(sentence, "")

    def test_invalid_parameters_raise_error(self):
        with self.assertRaises(FeatureExtractionError):
            extract_kmers("ATCG", k=0)

        with self.assertRaises(FeatureExtractionError):
            extract_kmers("ATCG", k=3, stride=0)

        with self.assertRaises(FeatureExtractionError):
            extract_kmers(12345, k=3)  # type: ignore

    def test_kmer_extractor_class(self):
        extractor = KMerExtractor(k=3, stride=1, to_lower=True, as_sentence=True)
        result = extractor.extract("ATCG")
        self.assertEqual(result, "atc tcg")

        batch_result = extractor.extract_batch(["ATCG", "GCTA"])
        self.assertEqual(batch_result, ["atc tcg", "gct cta"])


if __name__ == "__main__":
    unittest.main()

