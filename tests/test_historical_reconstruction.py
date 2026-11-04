"""Tests for historical dataset reconstruction."""

import csv
import json
import unittest
from pathlib import Path


class TestHistoricalDatasetReconstruction(unittest.TestCase):
    """Verifies that the historical 1,200-sample dataset reconstruction is exact."""

    def setUp(self):
        self.csv_path = Path("data/interim/historical_reconstruction/DNAdata_reconstructed.csv")
        self.meta_path = Path("artifacts/datasets/historical_reconstruction.json")

    def test_artifacts_exist(self):
        """Reconstructed CSV and metadata JSON must exist."""
        self.assertTrue(self.csv_path.exists(), f"Missing {self.csv_path}")
        self.assertTrue(self.meta_path.exists(), f"Missing {self.meta_path}")

    def test_reconstructed_csv_structure_and_counts(self):
        """CSV must contain exactly 1,200 samples across 4 balanced classes."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        
        header = reader[0]
        self.assertEqual(header, ["Unnamed: 0.1", "Unnamed: 0", "sequence", "len", "cancer"])
        
        rows = reader[1:]
        self.assertEqual(len(rows), 1200)

        # Class counts
        class_counts = {}
        for r in rows:
            cls = r[4]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        expected_classes = {
            "Cervical Cancer": 300,
            "Colorectal Cancer": 300,
            "Endometrial Cancer": 300,
            "Liver Cancer": 300,
        }
        self.assertEqual(class_counts, expected_classes)

    def test_historical_boundary_records_match(self):
        """First record (Cervical) and last record (Liver) must match legacy notebook outputs."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        
        row_0 = reader[1]
        self.assertEqual(row_0[0], "0")
        self.assertEqual(row_0[1], "0")
        self.assertEqual(row_0[3], "399")
        self.assertEqual(row_0[4], "Cervical Cancer")
        self.assertTrue(row_0[2].startswith("ATGCTGAGCGACATGGACTTTAGAGTCAAAGAAGA"))

        row_1199 = reader[1200]
        self.assertEqual(row_1199[0], "299")
        self.assertEqual(row_1199[1], "299")
        self.assertEqual(row_1199[3], "123")
        self.assertEqual(row_1199[4], "Liver Cancer")
        self.assertTrue(row_1199[2].startswith("ATGGTTCCGGATAGTTTAGAAGTAAATATTTCTAA"))

    def test_metadata_json_schema(self):
        """JSON metadata must contain valid provenance, taxonomy, and leakage analysis."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("reconstruction_provenance", meta)
        self.assertIn("biological_provenance", meta)
        self.assertIn("model_leakage_and_validity_analysis", meta)
        self.assertIn("overall_statistics", meta)
        self.assertIn("class_statistics", meta)
        self.assertIn("duplicate_analysis", meta)

        self.assertEqual(meta["overall_statistics"]["total_records"], 1200)
        self.assertEqual(meta["overall_statistics"]["unique_sequences"], 982)


if __name__ == "__main__":
    unittest.main()

