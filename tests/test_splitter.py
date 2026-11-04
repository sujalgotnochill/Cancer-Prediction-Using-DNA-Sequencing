"""Unit tests for dataset splitting, patient leakage prevention, and partition strategies."""

import unittest
from typing import List

from dna_cancer.data.schema import DNASample
from dna_cancer.data.splitter import (
    DatasetSplitter,
    recommend_splitting_strategy,
    split_grouped,
    split_random,
    split_stratified,
)
from dna_cancer.utils.exceptions import DataValidationError


class TestDatasetSplitter(unittest.TestCase):
    """Test data splitting algorithms and patient leakage prevention."""

    def test_recommend_splitting_strategy_group(self) -> None:
        # Multiple samples from same patient
        samples = [
            DNASample("S1", "ACGTACGTACGT", "LUAD", patient_id="PAT_001"),
            DNASample("S2", "TGCATGCATGCA", "LUAD", patient_id="PAT_001"),
            DNASample("S3", "CGATCGATCGAT", "BRCA", patient_id="PAT_002"),
        ]
        strat, reason = recommend_splitting_strategy(samples, group_key="patient_id")
        self.assertEqual(strat, "group")
        self.assertIn("patient/sample data leakage", reason)

    def test_recommend_splitting_strategy_stratified(self) -> None:
        # Unique patients (or no group dependencies) with balanced labels
        samples = [
            DNASample("S1", "ACGTACGTACGT", "LUAD", patient_id="PAT_001"),
            DNASample("S2", "TGCATGCATGCA", "BRCA", patient_id="PAT_002"),
            DNASample("S3", "CGATCGATCGAT", "COAD", patient_id="PAT_003"),
        ]
        strat, reason = recommend_splitting_strategy(samples, group_key="patient_id")
        self.assertEqual(strat, "stratified")
        self.assertIn("stratified splitting recommended", reason)

    def test_recommend_splitting_strategy_random(self) -> None:
        # No labels
        samples = [
            DNASample("S1", "ACGTACGTACGT", None),
            DNASample("S2", "TGCATGCATGCA", None),
            DNASample("S3", "CGATCGATCGAT", None),
        ]
        strat, _ = recommend_splitting_strategy(samples)
        self.assertEqual(strat, "random")

    def test_split_random_deterministic_and_proportions(self) -> None:
        samples = [DNASample(f"S_{i}", "ACGTACGTACGT", "LUAD") for i in range(100)]
        train1, val1, test1 = split_random(samples, train_size=0.7, val_size=0.15, test_size=0.15, seed=42)
        train2, val2, test2 = split_random(samples, train_size=0.7, val_size=0.15, test_size=0.15, seed=42)

        self.assertEqual(len(train1), 70)
        self.assertEqual(len(val1), 15)
        self.assertEqual(len(test1), 15)
        self.assertEqual([s.sample_id for s in train1], [s.sample_id for s in train2])

    def test_split_stratified_preserves_class_distribution(self) -> None:
        # 60 LUAD and 40 BRCA samples
        samples = [
            DNASample(f"LUAD_{i}", "ACGTACGTACGT", "LUAD") for i in range(60)
        ] + [
            DNASample(f"BRCA_{i}", "TGCATGCATGCA", "BRCA") for i in range(40)
        ]

        train, val, test = split_stratified(samples, train_size=0.7, val_size=0.15, test_size=0.15, seed=42)
        self.assertEqual(len(train) + len(val) + len(test), 100)

        # Check train distribution
        train_labels = [s.label for s in train]
        luad_train = train_labels.count("LUAD")
        brca_train = train_labels.count("BRCA")
        # 60 * 0.7 = 42, 40 * 0.7 = 28
        self.assertEqual(luad_train, 42)
        self.assertEqual(brca_train, 28)

    def test_split_grouped_strictly_prevents_leakage(self) -> None:
        # 10 patients, each having between 2 and 4 samples
        samples: List[DNASample] = []
        sample_counter = 0
        for pat_idx in range(10):
            pat_id = f"PAT_{pat_idx:03d}"
            label = "LUAD" if pat_idx % 2 == 0 else "BRCA"
            num_samples = 3 if pat_idx < 5 else 4
            for _ in range(num_samples):
                samples.append(
                    DNASample(
                        sample_id=f"S_{sample_counter}",
                        sequence="ACGTACGTACGT",
                        label=label,
                        patient_id=pat_id,
                    )
                )
                sample_counter += 1

        train, val, test = split_grouped(
            samples,
            group_key="patient_id",
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            seed=42,
        )

        train_patients = {s.patient_id for s in train}
        val_patients = {s.patient_id for s in val}
        test_patients = {s.patient_id for s in test}

        # Assert absolute zero overlap between patient sets across splits
        self.assertEqual(train_patients.intersection(val_patients), set())
        self.assertEqual(train_patients.intersection(test_patients), set())
        self.assertEqual(val_patients.intersection(test_patients), set())

        # Ensure all samples were assigned
        self.assertEqual(len(train) + len(val) + len(test), len(samples))

    def test_dataset_splitter_orchestration_and_summary(self) -> None:
        samples = [
            DNASample(f"S_{i}", "ACGTACGTACGT", "LUAD" if i % 2 == 0 else "BRCA", patient_id=f"P_{i//2}")
            for i in range(20)
        ]
        splitter = DatasetSplitter(strategy="auto", seed=123)
        train, val, test = splitter.split(samples)

        self.assertEqual(len(train) + len(val) + len(test), 20)
        summary = splitter.summarize_split(train, val, test, group_key="patient_id")

        self.assertEqual(summary["strategy_used"], "group")
        self.assertEqual(summary["total_samples"], 20)
        self.assertEqual(summary["leakage_free"], True)
        self.assertIn("train", summary)
        self.assertIn("validation", summary)
        self.assertIn("test", summary)

    def test_split_validation_errors(self) -> None:
        samples = [DNASample("S1", "ACGT", "LUAD"), DNASample("S2", "TGCA", "LUAD")]
        # Less than 3 samples for 3-way split
        with self.assertRaises(DataValidationError):
            split_random(samples)

        # Invalid ratio sums
        samples_valid = [DNASample(f"S_{i}", "ACGT", "LUAD") for i in range(10)]
        with self.assertRaises(DataValidationError):
            split_random(samples_valid, train_size=0.5, val_size=0.5, test_size=0.5)


if __name__ == "__main__":
    unittest.main()

