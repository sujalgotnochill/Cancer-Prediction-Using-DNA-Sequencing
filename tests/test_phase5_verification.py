"""Unit tests verifying Phase 5 frozen candidate artifacts, prediction parity, and vault lockdown."""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.features.kmer_extractor import extract_kmers
from dna_cancer.models.baseline_models import LinearSVCClassifier


class TestPhase5FrozenVerification(unittest.TestCase):
    """Test suite for Phase 5.1 candidate verification."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cand_dir = Path("artifacts/models/phase5_best_candidate")
        cls.vault_dir = Path("artifacts/vault")
        cls.eval_dir = Path("artifacts/evaluation")

    def test_artifacts_exist_and_match_checksums(self) -> None:
        """Verify all frozen candidate files exist and match checksums.sha256."""
        checksums_path = self.cand_dir / "checksums.sha256"
        self.assertTrue(checksums_path.exists(), "checksums.sha256 missing")

        with open(checksums_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if line.strip()]

        for expected_sha, fname in lines:
            fpath = self.cand_dir / fname
            self.assertTrue(fpath.exists(), f"Artifact {fname} missing")
            with open(fpath, "rb") as bf:
                actual_sha = hashlib.sha256(bf.read()).hexdigest()
            self.assertEqual(actual_sha, expected_sha, f"Checksum mismatch for {fname}")

    def test_model_and_vectorizer_compatibility(self) -> None:
        """Verify model architecture, vocabulary dimension, and class encoding."""
        model = joblib.load(self.cand_dir / "model.joblib")
        vec = joblib.load(self.cand_dir / "vectorizer.joblib")
        enc = joblib.load(self.cand_dir / "encoder.joblib")
        config = json.load(open(self.cand_dir / "model_config.json"))

        self.assertIsInstance(model, LinearSVCClassifier)
        self.assertEqual(len(vec.vocabulary_), 256)
        self.assertEqual(model.model.coef_.shape, (4, 256))
        self.assertEqual(list(enc.classes_), [
            "Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"
        ])
        self.assertTrue(config.get("is_frozen"))

    def test_prediction_reproducibility_on_synthetic_sequence(self) -> None:
        """Verify that reloaded pipeline produces deterministic predictions."""
        model = joblib.load(self.cand_dir / "model.joblib")
        vec = joblib.load(self.cand_dir / "vectorizer.joblib")
        enc = joblib.load(self.cand_dir / "encoder.joblib")

        test_seq = "ATGCATGCATGCATGCATGC" * 20
        kmers = " ".join(extract_kmers(test_seq, k=4, canonical=False))
        X = vec.transform([kmers])

        pred_idx1 = model.predict(X)
        pred_idx2 = model.predict(X)
        scores1 = model.model.decision_function(X)
        scores2 = model.model.decision_function(X)

        self.assertEqual(int(pred_idx1[0]), int(pred_idx2[0]))
        np.testing.assert_allclose(scores1, scores2, atol=1e-12)

    def test_vault_remains_locked(self) -> None:
        """Verify that the production sealed vault is locked with zero unseals."""
        vault_state_path = self.vault_dir / "vault_state.json"
        self.assertTrue(vault_state_path.exists())
        with open(vault_state_path, "r") as f:
            state = json.load(f)
        self.assertEqual(state.get("status"), "LOCKED")
        self.assertFalse(state.get("unsealed"))
        self.assertEqual(state.get("unseal_count"), 0)


if __name__ == "__main__":
    unittest.main()

