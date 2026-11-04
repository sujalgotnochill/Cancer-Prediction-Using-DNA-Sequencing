"""Tests for data ingestion using synthetic test files."""

import sys
import tempfile
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.data.ingestion import load_csv, load_fasta_as_dataframe, parse_fasta_records
from dna_cancer.data.validation import validate_dataset_records
from dna_cancer.utils.exceptions import DataIngestionError, DataValidationError, SequenceFormatError


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion with synthetic CSV and FASTA files."""

    def test_load_csv_valid(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("id,sequence,cancer\ns1,ATCGATCG,Colorectal Cancer\ns2,GCTAGCTA,Liver Cancer\n")
            tmp_path = tmp.name

        try:
            df = load_csv(tmp_path, expected_columns=["sequence", "cancer"])
            self.assertEqual(len(df), 2)
            self.assertIn("sequence", df.columns)
            self.assertIn("cancer", df.columns)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_csv_empty_file_raises_error(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        try:
            with self.assertRaises(DataIngestionError) as ctx:
                load_csv(tmp_path)
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_csv_missing_columns_raises_error(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("id,seq_only\ns1,ATCG\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(DataIngestionError) as ctx:
                load_csv(tmp_path, expected_columns=["sequence", "cancer"])
            self.assertIn("missing expected columns", str(ctx.exception).lower())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_parse_fasta_valid_single_and_multiple(self):
        fasta_content = (
            ">seq1 Gene Alpha\n"
            "ATCGATCG\n"
            "GCTAGCTA\n"
            ">seq2 Gene Beta\n"
            "CCCCGGGG\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp.write(fasta_content)
            tmp_path = tmp.name

        try:
            records = parse_fasta_records(tmp_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["id"], "seq1")
            self.assertEqual(records[0]["description"], "Gene Alpha")
            self.assertEqual(records[0]["sequence"], "ATCGATCGGCTAGCTA")
            self.assertEqual(records[0]["len"], 16)

            self.assertEqual(records[1]["id"], "seq2")
            self.assertEqual(records[1]["sequence"], "CCCCGGGG")
            self.assertEqual(records[1]["len"], 8)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_fasta_as_dataframe_with_label(self):
        fasta_content = ">seq1\nATCGATCG\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp.write(fasta_content)
            tmp_path = tmp.name

        try:
            df = load_fasta_as_dataframe(tmp_path, cancer_label="Cervical Cancer")
            self.assertEqual(len(df), 1)
            self.assertIn("cancer", df.columns)
            self.assertEqual(df["cancer"][0], "Cervical Cancer")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fasta_empty_file_raises_error(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp_path = tmp.name

        try:
            with self.assertRaises(DataIngestionError) as ctx:
                parse_fasta_records(tmp_path)
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fasta_sequence_before_header_raises_error(self):
        malformed = "ATCGATCG\n>seq1\nATCG\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp.write(malformed)
            tmp_path = tmp.name

        try:
            with self.assertRaises(DataIngestionError) as ctx:
                parse_fasta_records(tmp_path)
            self.assertIn("sequence data encountered before any header", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fasta_empty_header_raises_error(self):
        malformed = ">\nATCGATCG\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp.write(malformed)
            tmp_path = tmp.name

        try:
            with self.assertRaises(DataIngestionError) as ctx:
                parse_fasta_records(tmp_path)
            self.assertIn("empty record header", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fasta_record_with_no_sequence_raises_error(self):
        malformed = ">seq1\n>seq2\nATCG\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp.write(malformed)
            tmp_path = tmp.name

        try:
            with self.assertRaises(DataIngestionError) as ctx:
                parse_fasta_records(tmp_path)
            self.assertIn("has no sequence data", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_dataset_records_missing_sequences_and_labels(self):
        from dna_cancer.data.ingestion import SimpleDataFrame

        # Valid dataset
        df_valid = SimpleDataFrame([
            {"sequence": "ATCGATCG", "cancer": "Colorectal Cancer"},
            {"sequence": "GCTAGCTA", "cancer": "Liver Cancer"},
        ])
        summary = validate_dataset_records(
            df_valid,
            expected_classes=["Colorectal Cancer", "Liver Cancer"],
            raise_exception=True,
        )
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["valid_records"], 2)

        # Missing sequence
        df_missing_seq = SimpleDataFrame([
            {"sequence": "", "cancer": "Colorectal Cancer"},
        ])
        with self.assertRaises(DataValidationError) as ctx:
            validate_dataset_records(df_missing_seq, raise_exception=True)
        self.assertIn("empty nucleotide sequence", str(ctx.exception))

        # Invalid label
        df_invalid_label = SimpleDataFrame([
            {"sequence": "ATCGATCG", "cancer": "Unknown Brain Tumor"},
        ])
        with self.assertRaises(DataValidationError) as ctx:
            validate_dataset_records(
                df_invalid_label,
                expected_classes=["Colorectal Cancer", "Liver Cancer"],
                raise_exception=True,
            )
        self.assertIn("invalid label", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

