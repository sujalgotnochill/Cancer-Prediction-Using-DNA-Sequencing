"""Tests for Phase 4 Baseline Machine Learning Benchmarks."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath("src"))

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from dna_cancer.data.schema import CancerSequenceSample, DNASample
from dna_cancer.data.splitter import (
    DatasetSplitter,
    split_duplicate_grouped,
    split_stratified,
    verify_split_leakage,
)
from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.models.baseline_models import (
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
    RandomForestClassifierWrapper,
)
from dna_cancer.tracking.tracker import ExperimentTracker


class TestBaselineBenchmarks(unittest.TestCase):
    """Test suite verifying leakage prevention, vectorizer fitting, models, and tracking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.toy_samples = [
            DNASample(sample_id="s1", sequence="ATGCATGCATGC", label="ClassA"),
            DNASample(sample_id="s2", sequence="ATGCATGCATGC", label="ClassA"),  # Duplicate of s1
            DNASample(sample_id="s3", sequence="GGCCGGCCGGCC", label="ClassA"),
            DNASample(sample_id="s4", sequence="TTTTTTTTTTTT", label="ClassB"),
            DNASample(sample_id="s5", sequence="TTTTTTTTTTTT", label="ClassB"),  # Duplicate of s4
            DNASample(sample_id="s6", sequence="AAAAAAAAAAAA", label="ClassB"),
            DNASample(sample_id="s7", sequence="CGCGCGCGCGCG", label="ClassC"),
            DNASample(sample_id="s8", sequence="CGCGCGCGCGCG", label="ClassC"),  # Duplicate of s7
            DNASample(sample_id="s9", sequence="TATATATATATA", label="ClassC"),
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_duplicate_group_splitting_prevents_leakage(self):
        """Identical sequences must never be partitioned across train/val/test."""
        train, val, test = split_duplicate_grouped(
            self.toy_samples,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            seed=42,
        )

        train_seqs = {s.sequence for s in train}
        val_seqs = {s.sequence for s in val}
        test_seqs = {s.sequence for s in test}

        # Assert pairwise disjoint sets
        self.assertEqual(len(train_seqs & val_seqs), 0, "Sequence leaked between train and val")
        self.assertEqual(len(train_seqs & test_seqs), 0, "Sequence leaked between train and test")
        self.assertEqual(len(val_seqs & test_seqs), 0, "Sequence leaked between val and test")

        audit = verify_split_leakage(train, val, test)
        self.assertTrue(audit["is_leakage_free"])
        self.assertFalse(audit["sequence_leakage"]["has_leakage"])

    def test_verify_split_leakage_detects_deliberate_overlap(self):
        """Audit function must identify overlapping sequences as leakage."""
        train = [self.toy_samples[0]]  # ATGCATGCATGC
        val = [self.toy_samples[1]]    # ATGCATGCATGC (identical!)
        test = [self.toy_samples[3]]

        audit = verify_split_leakage(train, val, test)
        self.assertFalse(audit["is_leakage_free"])
        self.assertTrue(audit["sequence_leakage"]["has_leakage"])
        self.assertEqual(audit["sequence_leakage"]["train_val_overlap_count"], 1)

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for vectorization test")
    def test_feature_builder_training_only_fitting(self):
        """FeatureBuilder must be fitted exclusively on training sequences."""
        train_seqs = ["ATGCATGC", "CGTACGTA"]
        test_seqs = ["NNNNNNNN", "TTTTAAAA"]

        fb = FeatureBuilder(k=4, vectorizer_type="count", return_sparse=True)
        fb.fit(train_seqs, labels=["A", "B"])

        vocab_before = dict(fb.vectorizer.vocabulary_)
        X_test = fb.transform_sequences(test_seqs)

        # Vocabulary must not change when transforming test data
        self.assertEqual(vocab_before, fb.vectorizer.vocabulary_)
        # Transformed shape must match vocabulary dimension
        self.assertEqual(X_test.shape[1], fb.n_features)

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for model interfaces test")
    def test_baseline_models_interfaces_and_reproducibility(self):
        """Verify baseline classifiers implement BaseDNAClassifier and are reproducible."""
        X = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]]
        y = [0, 1, 0, 1, 0, 1]

        models = [
            MultinomialNBClassifier(alpha=1.0),
            LogisticRegressionClassifier(C=1.0, random_state=42),
            RandomForestClassifierWrapper(n_estimators=10, random_state=42),
        ]

        for model in models:
            self.assertIsInstance(model, BaseDNAClassifier)
            model.fit(X, y)
            preds1 = model.predict(X)
            probs = model.predict_proba(X)
            self.assertEqual(len(preds1), len(y))
            self.assertEqual(len(probs), len(y))

            # Test persistence save and load
            save_path = Path(self.temp_dir) / f"{model.__class__.__name__}.joblib"
            model.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = model.__class__.load(save_path)
            preds2 = loaded.predict(X)
            self.assertEqual(list(preds1), list(preds2), f"{model.__class__.__name__} failed save/load parity")

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for metrics calculation test")
    def test_classification_metrics_calculation(self):
        """Verify metric calculation output dictionary and precision/recall values."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 0, 2, 2]  # 5/6 = 0.8333

        metrics = compute_classification_metrics(y_true, y_pred, target_names=["A", "B", "C"])
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6, places=3)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_experiment_tracker_logs_limitation_notice(self):
        """ExperimentTracker records must contain the mandatory scientific disclaimer."""
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            artifact_dir=self.temp_dir,
            enable_mlflow=False,
        )

        record = tracker.log_run(
            run_name="test_run",
            dataset_id="test_ds",
            dataset_checksum="dummy_checksum",
            split_config={"strategy": "duplicate_grouped", "is_leakage_free": True},
            feature_config={"k": 4, "n_features": 256},
            model_name="MultinomialNB",
            model_params={"alpha": 1.0},
            random_seed=42,
            metrics={"accuracy": 0.95, "f1_macro": 0.94},
            training_duration=0.01,
            prediction_duration=0.005,
        )

        self.assertIn("scientific_limitation_notice", record)
        self.assertIn("Educational sequence-classification benchmark", record["scientific_limitation_notice"])
        self.assertIn("not verified clinical cancer diagnoses", record["scientific_limitation_notice"])

        # Check local file was saved
        run_file = Path(self.temp_dir) / f"{record['run_id']}.json"
        self.assertTrue(run_file.exists())


if __name__ == "__main__":
    unittest.main()
