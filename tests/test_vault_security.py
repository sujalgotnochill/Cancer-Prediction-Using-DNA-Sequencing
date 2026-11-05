"""Unit tests for Sealed Test Vault security, boundaries, and evaluation safeguards."""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.development import get_development_cv_folds, load_development_cohort
from dna_cancer.data.ingestion import load_csv, load_json, load_samples, validate_not_sealed_vault_path
from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.schema import DNASample
from dna_cancer.data.vault import AUTHORIZED_UNSEAL_TOKEN, SealedVault
from dna_cancer.evaluation.final_evaluator import FinalHoldoutEvaluator
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.utils.exceptions import DataValidationError


class MockDNAClassifier(BaseDNAClassifier):
    """Mock classifier for testing evaluation boundaries without training."""

    def __init__(self, constant_prediction: str = "Cervical Cancer") -> None:
        self.constant_prediction = constant_prediction
        self.is_fitted = False

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "MockDNAClassifier":
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> List[str]:
        return [self.constant_prediction for _ in range(len(X))]

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError

    def save(self, path: Any) -> None:
        pass

    def load(self, path: Any) -> "MockDNAClassifier":
        return self


class TestVaultSecurityAndBoundaries(unittest.TestCase):
    """Verify architectural boundaries and failure modes of the sealed vault and evaluator."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.vault_dir = self.temp_path / "vault"
        self.dev_cohort_path = self.temp_path / "dev_cohort.csv"
        self.vault = SealedVault(
            vault_dir=self.vault_dir,
            dev_cohort_path=self.dev_cohort_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ordinary_ingestion_cannot_load_holdout_file(self) -> None:
        """Verify that standard dataset loaders reject direct ingestion of sealed holdout files."""
        holdout_path = "artifacts/vault/test_holdout_sealed.json"

        with self.assertRaises(PermissionError):
            load_samples(holdout_path)

        with self.assertRaises(PermissionError):
            load_json(holdout_path)

        with self.assertRaises(PermissionError):
            load_csv(holdout_path)

        with self.assertRaises(PermissionError):
            validate_not_sealed_vault_path(holdout_path)

    def test_development_cohort_cannot_load_holdout_path(self) -> None:
        """Verify load_development_cohort rejects vault holdout file paths."""
        with self.assertRaises(PermissionError):
            load_development_cohort(path="artifacts/vault/test_holdout_sealed.json")

    def test_development_cohort_never_silently_falls_back(self) -> None:
        """Verify missing development cohort raises FileNotFoundError without fallback."""
        non_existent = self.temp_path / "does_not_exist_cohort.csv"
        with self.assertRaises(FileNotFoundError):
            load_development_cohort(path=non_existent)

    def test_frozen_model_config_validation(self) -> None:
        """Verify FinalHoldoutEvaluator rejects unfrozen model configurations."""
        evaluator = FinalHoldoutEvaluator(vault=self.vault)

        # 1. Non-dict config
        with self.assertRaises(ValueError):
            evaluator.verify_readiness(model_config="not_a_dict")  # type: ignore

        # 2. Config without is_frozen
        with self.assertRaises(ValueError):
            evaluator.verify_readiness(model_config={"model": "LogisticRegression"})

        # 3. Config with is_frozen = False
        with self.assertRaises(ValueError):
            evaluator.verify_readiness(model_config={"model": "LogisticRegression", "is_frozen": False})

    def test_manifest_verification_and_checksum_integrity(self) -> None:
        """Verify that FinalHoldoutEvaluator checks manifest and detects modified files."""
        # Initialize an isolated test vault in temp directory
        self.vault.initialize_vault()
        event_log = self.temp_path / "events.jsonl"
        evaluator = FinalHoldoutEvaluator(vault=self.vault, event_log_path=event_log)

        frozen_config = {
            "model_type": "LogisticRegression",
            "is_frozen": True,
            "hyperparameters": {"C": 1.0},
        }

        # Readiness should succeed on pristine isolated vault
        readiness = evaluator.verify_readiness(model_config=frozen_config)
        self.assertTrue(readiness["is_ready"])
        self.assertTrue(readiness["vault_locked"])

        # Tamper with holdout file -> must fail checksum check
        with open(self.vault.vault_data_path, "a", encoding="utf-8") as f:
            f.write(" ")

        with self.assertRaises(DataValidationError):
            evaluator.verify_readiness(model_config=frozen_config)

    def test_modified_development_cohort_fails_verification(self) -> None:
        """Verify that tampering with development cohort fails readiness and dev loading."""
        self.vault.initialize_vault()
        event_log = self.temp_path / "events.jsonl"
        evaluator = FinalHoldoutEvaluator(vault=self.vault, event_log_path=event_log)

        # Tamper with dev cohort
        with open(self.dev_cohort_path, "a", encoding="utf-8") as f:
            f.write("tampered_extra_row\n")

        frozen_config = {"model_type": "LogisticRegression", "is_frozen": True}

        # Final evaluator must detect dev cohort checksum mismatch
        with self.assertRaises(DataValidationError):
            evaluator.verify_readiness(model_config=frozen_config)

        # Development cohort loader must also detect dev cohort checksum mismatch
        with self.assertRaises(DataValidationError):
            load_development_cohort(
                path=self.dev_cohort_path,
                verify_checksum=True,
                manifest_path=self.vault.manifest_path,
            )

    def test_missing_state_file_tampering_fails_safely(self) -> None:
        """Verify that deleting the vault state file is detected as tampering."""
        self.vault.initialize_vault()
        self.assertTrue(self.vault.is_locked())

        # Simulate adversarial or accidental state file deletion
        if self.vault.state_path.exists():
            self.vault.state_path.unlink()

        # is_locked must detect state file deletion and raise DataValidationError
        with self.assertRaises(DataValidationError):
            self.vault.is_locked()

        # unseal must also fail safely
        with self.assertRaises(DataValidationError):
            self.vault.unseal(token=AUTHORIZED_UNSEAL_TOKEN, reason="Test after state deletion")

    def test_one_shot_evaluation_enforcement_in_evaluator(self) -> None:
        """Verify that FinalHoldoutEvaluator strictly rejects repeated evaluation attempts."""
        self.vault.initialize_vault()
        event_log = self.temp_path / "events.jsonl"
        evaluator = FinalHoldoutEvaluator(vault=self.vault, event_log_path=event_log)

        model = MockDNAClassifier(constant_prediction="Cervical Cancer")
        frozen_config = {
            "model_type": "MockDNAClassifier",
            "is_frozen": True,
            "hyperparameters": {},
        }

        # First evaluation on the isolated test vault
        result = evaluator.evaluate_frozen_model(
            model=model,
            model_config=frozen_config,
            unseal_token=AUTHORIZED_UNSEAL_TOKEN,
            reason="Isolated test suite one-shot validation",
        )
        self.assertEqual(result["status"], "FINAL_EVALUATION_COMPLETED")
        self.assertEqual(result["sample_count"], 240)
        self.assertTrue(event_log.exists())

        # Second evaluation attempt MUST raise PermissionError
        with self.assertRaises(PermissionError):
            evaluator.evaluate_frozen_model(
                model=model,
                model_config=frozen_config,
                unseal_token=AUTHORIZED_UNSEAL_TOKEN,
                reason="Second evaluation attempt (must be blocked)",
            )

    def test_production_vault_lifecycle_integrity(self) -> None:
        """Verify that the production sealed vault in artifacts/vault is in a valid lifecycle state (locked or evaluated once)."""
        prod_vault = SealedVault()
        if not prod_vault.vault_data_path.exists():
            self.skipTest("Production holdout dataset is untracked and absent in fresh clone environment")
        status = prod_vault.get_vault_status()
        self.assertTrue(status["vault_exists"])
        self.assertTrue(status["manifest_exists"])
        # Production vault must either be pre-evaluation (LOCKED, count 0) or evaluated strictly once (UNSEALED, count 1)
        self.assertIn(status["unseal_count"], [0, 1])
        if status["unseal_count"] == 0:
            self.assertEqual(status["status"], "LOCKED")
            self.assertFalse(status["unsealed"])
            self.assertTrue(status["is_locked"])
        else:
            self.assertEqual(status["status"], "UNSEALED")
            self.assertTrue(status["unsealed"])
            self.assertFalse(status["is_locked"])


if __name__ == "__main__":
    unittest.main()
