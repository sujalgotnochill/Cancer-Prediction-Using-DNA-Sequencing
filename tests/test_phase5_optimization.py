"""Unit tests for Phase 5 feature transformations, training fold isolation, and model consistency."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.features.kmer_extractor import extract_kmers
from dna_cancer.models import (
    GradientBoostingClassifierWrapper,
    LinearSVCClassifier,
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
    RandomForestClassifierWrapper,
)


class TestPhase5Optimization(unittest.TestCase):
    """Test feature transformations, sparse matrix handling, fold fitting, and serialization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_feature_transformations(self) -> None:
        """Verify count, tfidf, frequency, and canonical k-mer feature transformations."""
        seqs = ["ATGCATGCATGC", "GCATGCATGCAT", "AAAAAAAAAAAA"]
        labels = ["ClassA", "ClassB", "ClassA"]

        # 1. Count features
        fb_count = FeatureBuilder(k=3, vectorizer_type="count", return_sparse=False)
        X_count = fb_count.fit(seqs, labels).transform_sequences(seqs)
        self.assertEqual(X_count.shape[0], 3)
        self.assertTrue(np.all(X_count >= 0))

        # 2. TF-IDF features
        fb_tfidf = FeatureBuilder(k=3, vectorizer_type="tfidf", return_sparse=False)
        X_tfidf = fb_tfidf.fit(seqs, labels).transform_sequences(seqs)
        self.assertEqual(X_tfidf.shape[0], 3)
        self.assertTrue(np.all(X_tfidf >= 0))

        # 3. Frequency normalized features
        fb_freq = FeatureBuilder(k=3, vectorizer_type="frequency", return_sparse=False)
        X_freq = fb_freq.fit(seqs, labels).transform_sequences(seqs)
        self.assertEqual(X_freq.shape[0], 3)
        row_sums = X_freq.sum(axis=1)
        # All non-empty rows must sum to 1.0 (L1 relative frequency)
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], rtol=1e-5)

        # 4. Canonical representation
        kmers_std = extract_kmers("ATG", k=3, canonical=False)
        kmers_can = extract_kmers("ATG", k=3, canonical=True)
        # Revcomp of ATG is CAT; min('atg', 'cat') is 'atg'
        self.assertEqual(kmers_std, ["atg"])
        self.assertEqual(kmers_can, ["atg"])

        kmers_tac_std = extract_kmers("TAC", k=3, canonical=False)
        kmers_tac_can = extract_kmers("TAC", k=3, canonical=True)
        # Revcomp of TAC is GTA; min('tac', 'gta') is 'gta'
        self.assertEqual(kmers_tac_std, ["tac"])
        self.assertEqual(kmers_tac_can, ["gta"])

    def test_training_fold_only_fitting_prevents_leakage(self) -> None:
        """Verify vocabulary fitting is isolated strictly to the training fold."""
        train_seqs = ["AAAAACCCCC", "CCCCCAAAAA"]
        train_labels = ["ClassA", "ClassB"]
        # Validation sequence contains 'GGGGG' which is not in train
        val_seqs = ["GGGGGTTTTT"]

        fb = FeatureBuilder(k=5, vectorizer_type="count", return_sparse=True)
        fb.fit(train_seqs, train_labels)

        # Train vocabulary must contain only train k-mers
        vocab = fb.get_feature_names()
        self.assertTrue(any("aaaaa" in w for w in vocab))
        self.assertFalse(any("ggggg" in w for w in vocab))

        # Transforming val_seqs must work and ignore unseen k-mers
        X_val = fb.transform_sequences(val_seqs)
        self.assertEqual(X_val.shape[0], 1)
        self.assertEqual(X_val.shape[1], len(vocab))
        # Since 'ggggg' and 'ttttt' are not in vocab, row sum must be 0
        self.assertEqual(X_val.sum(), 0)

    def test_sparse_matrix_handling_across_models(self) -> None:
        """Verify all baseline and optimized models accept scipy CSR matrices directly."""
        np.random.seed(42)
        X_sparse = sp.csr_matrix(np.random.poisson(0.5, (40, 16)))
        y = np.random.choice([0, 1, 2, 3], size=40)

        # 1. LinearSVC
        svc = LinearSVCClassifier(C=1.0, dual="auto", random_state=42)
        svc.fit(X_sparse, y)
        preds_svc = svc.predict(X_sparse)
        self.assertEqual(len(preds_svc), 40)
        probs_svc = svc.predict_proba(X_sparse)
        self.assertEqual(probs_svc.shape, (40, 4))

        # 2. Logistic Regression
        lr = LogisticRegressionClassifier(C=1.0, max_iter=500, random_state=42)
        lr.fit(X_sparse, y)
        preds_lr = lr.predict(X_sparse)
        self.assertEqual(len(preds_lr), 40)

        # 3. MultinomialNB
        nb = MultinomialNBClassifier(alpha=1.0)
        nb.fit(X_sparse, y)
        preds_nb = nb.predict(X_sparse)
        self.assertEqual(len(preds_nb), 40)

        # 4. Random Forest
        rf = RandomForestClassifierWrapper(n_estimators=10, random_state=42)
        rf.fit(X_sparse, y)
        preds_rf = rf.predict(X_sparse)
        self.assertEqual(len(preds_rf), 40)

    def test_reproducibility_with_seed(self) -> None:
        """Verify that identical random seeds produce identical models and predictions."""
        np.random.seed(42)
        X = sp.csr_matrix(np.random.randn(30, 10))
        y = np.random.choice([0, 1], size=30)

        clf1 = LinearSVCClassifier(C=1.0, dual="auto", random_state=123)
        clf1.fit(X, y)
        p1 = clf1.predict(X)

        clf2 = LinearSVCClassifier(C=1.0, dual="auto", random_state=123)
        clf2.fit(X, y)
        p2 = clf2.predict(X)

        np.testing.assert_array_equal(p1, p2)

    def test_model_serialization_and_prediction_consistency(self) -> None:
        """Verify model and FeatureBuilder persistence and exact prediction equality."""
        seqs = ["ATGCATGCATGC", "GCATGCATGCAT", "TTTTTTTTTTTT", "GGGGGGGGGGGG"]
        labels = ["Cancer_A", "Cancer_B", "Cancer_A", "Cancer_B"]

        fb = FeatureBuilder(k=3, vectorizer_type="tfidf", return_sparse=True)
        fb.fit(seqs, labels)
        X = fb.transform_sequences(seqs)
        y = fb.transform_labels(labels)

        clf = LinearSVCClassifier(C=1.0, dual="auto", random_state=42)
        clf.fit(X, y)
        preds_orig = clf.predict(X)

        # Save to temp directory
        model_path = self.temp_path / "model.joblib"
        vec_path = self.temp_path / "vec.joblib"
        enc_path = self.temp_path / "enc.joblib"

        clf.save(model_path)
        fb.save(vectorizer_path=vec_path, encoder_path=enc_path)

        # Reload
        reloaded_clf = LinearSVCClassifier.load(model_path)
        reloaded_fb = FeatureBuilder(k=3, vectorizer_type="tfidf", return_sparse=True)
        reloaded_fb.load(vectorizer_path=vec_path, encoder_path=enc_path)

        # Predict on same sequences
        X_reloaded = reloaded_fb.transform_sequences(seqs)
        preds_reloaded = reloaded_clf.predict(X_reloaded)

        np.testing.assert_array_equal(preds_orig, preds_reloaded)
        labels_orig = fb.inverse_transform_labels(preds_orig)
        labels_reloaded = reloaded_fb.inverse_transform_labels(preds_reloaded)
        self.assertEqual(labels_orig, labels_reloaded)


if __name__ == "__main__":
    unittest.main()

