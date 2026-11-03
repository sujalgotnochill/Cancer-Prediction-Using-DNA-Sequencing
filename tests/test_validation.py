"""Tests for DNA sequence and dataset validation."""

import sys
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.data.validation import SequenceValidator, validate_dataset_schema, validate_dna_sequence
from dna_cancer.utils.exceptions import DataValidationError, SequenceFormatError


class MockDataFrame:
    """Lightweight duck-typed DataFrame mock for environments without pandas."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self.columns = list(data.keys())

    def __len__(self) -> int:
        first_col = next(iter(self._data.values())) if self._data else []
        return len(first_col)


class TestSequenceValidation(unittest.TestCase):
    """Test DNA sequence validation rules with synthetic sequences."""

    def test_valid_standard_sequence(self):
        valid_dna = "ATGCGATCGATCGATCGATC"
        self.assertTrue(validate_dna_sequence(valid_dna))

    def test_valid_lowercase_sequence(self):
        valid_dna = "atgcgatcgatcgatcgatc"
        self.assertTrue(validate_dna_sequence(valid_dna))

    def test_valid_sequence_with_n(self):
        dna_with_n = "ATCGNATCGN"
        self.assertTrue(validate_dna_sequence(dna_with_n, allowed_bases="ACGTN"))

    def test_invalid_disallowed_n(self):
        dna_with_n = "ATCGNATCGN"
        self.assertFalse(validate_dna_sequence(dna_with_n, allowed_bases="ACGT"))

    def test_invalid_characters(self):
        invalid_dna = "ATCG123XYZ!"
        self.assertFalse(validate_dna_sequence(invalid_dna))

    def test_raise_exception_on_invalid(self):
        invalid_dna = "ATCGXYZ"
        with self.assertRaises(SequenceFormatError):
            validate_dna_sequence(invalid_dna, raise_exception=True)

    def test_empty_sequence(self):
        self.assertFalse(validate_dna_sequence(""))
        self.assertTrue(validate_dna_sequence("", allow_empty=True))

    def test_sequence_length_constraints(self):
        short_seq = "ATC"
        self.assertFalse(validate_dna_sequence(short_seq, min_length=5))
        self.assertTrue(validate_dna_sequence(short_seq, min_length=2, max_length=5))
        self.assertFalse(validate_dna_sequence(short_seq, max_length=2))

    def test_sequence_validator_class(self):
        validator = SequenceValidator(allowed_bases="ACGT", min_length=4)
        self.assertTrue(validator.is_valid("ACGTACGT"))
        self.assertFalse(validator.is_valid("ACG"))
        self.assertFalse(validator.is_valid("ACGTN"))

        with self.assertRaises(SequenceFormatError):
            validator.validate("ACGTN")


class TestDatasetSchemaValidation(unittest.TestCase):
    """Test dataset schema validation."""

    def test_valid_schema(self):
        df = MockDataFrame({
            "sequence": ["ATCGATCG", "GCTAGCTA"],
            "cancer": ["Cervical Cancer", "Liver Cancer"],
            "id": ["s1", "s2"],
        })
        self.assertTrue(validate_dataset_schema(df, required_columns=["sequence", "cancer"]))

    def test_missing_column(self):
        df = MockDataFrame({
            "sequence": ["ATCGATCG"],
        })
        with self.assertRaises(DataValidationError):
            validate_dataset_schema(df, required_columns=["sequence", "cancer"])

    def test_empty_dataframe(self):
        df = MockDataFrame({})
        with self.assertRaises(DataValidationError):
            validate_dataset_schema(df, required_columns=["sequence", "cancer"])


if __name__ == "__main__":
    unittest.main()
