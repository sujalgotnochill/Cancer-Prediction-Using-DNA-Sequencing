"""Unit tests for the SealedVault access-control mechanism and CV fold generators."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.schema import DNASample
from dna_cancer.data.splitter import (
    generate_group_shufflesplit_folds,
    split_gene_family_cv_folds,
)
from dna_cancer.data.vault import AUTHORIZED_UNSEAL_TOKEN, SealedVault
from dna_cancer.utils.exceptions import DataValidationError


class TestVaultAndCrossValidation(unittest.TestCase):
    """Test sealed vault lifecycle, procedural locking, and group-aware CV fold generation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_dir = Path(self.temp_dir.name) / "vault"
        self.dev_cohort_path = Path(self.temp_dir.name) / "dev_cohort.csv"
        self.vault = SealedVault(
            vault_dir=self.vault_dir,
            dev_cohort_path=self.dev_cohort_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_vault_lifecycle_and_one_shot_unseal(self) -> None:
        # Initialize vault using verified historical reconstruction
        status = self.vault.initialize_vault()
        self.assertTrue(status["vault_exists"])
        self.assertTrue(status["manifest_exists"])
        self.assertTrue(self.vault.is_locked())
        self.assertEqual(status["status"], "LOCKED")

        # Verify integrity
        audit = self.vault.verify_integrity()
        self.assertTrue(audit["is_valid"])
        self.assertTrue(audit["sha_match"])
        self.assertEqual(audit["sample_count"], 240)
        self.assertEqual(audit["sequence_leakage_with_dev"], 0)

        # Attempt unseal with invalid token -> should fail
        with self.assertRaises(PermissionError):
            self.vault.unseal(token="WRONG_TOKEN", reason="Testing authorization failure")

        # Attempt unseal with empty reason -> should fail
        with self.assertRaises(ValueError):
            self.vault.unseal(token=AUTHORIZED_UNSEAL_TOKEN, reason="")

        # Perform legitimate one-shot unseal
        holdout_samples = self.vault.unseal(
            token=AUTHORIZED_UNSEAL_TOKEN,
            reason="Phase 4.4 test suite verification of one-shot unsealing contract",
        )
        self.assertEqual(len(holdout_samples), 240)
        self.assertFalse(self.vault.is_locked())
        self.assertEqual(self.vault.get_vault_status()["status"], "UNSEALED")

        # Attempt SECOND unseal -> MUST raise PermissionError
        with self.assertRaises(PermissionError):
            self.vault.unseal(
                token=AUTHORIZED_UNSEAL_TOKEN,
                reason="Second evaluation attempt (should be strictly blocked)",
            )

    def test_split_gene_family_cv_folds(self) -> None:
        # Create synthetic samples representing multiple classes and gene families
        samples = []
        for c_idx, label in enumerate(["Cancer_A", "Cancer_B"]):
            for g_idx in range(6):
                gene_name = f"GENE_{c_idx}_{g_idx}"
                # Gene sizes vary (some have 4, some have 2 samples)
                n_samples = 4 if g_idx == 0 else 2
                for s_idx in range(n_samples):
                    s = DNASample(
                        sample_id=f"S_{c_idx}_{g_idx}_{s_idx}",
                        sequence=f"ACGT_{c_idx}_{g_idx}_{s_idx}_ACGT",
                        label=label,
                    )
                    s.metadata.gene = gene_name
                    samples.append(s)

        total_samples = len(samples)
        folds = split_gene_family_cv_folds(samples, n_splits=3, seed=42)
        self.assertEqual(len(folds), 3)

        val_sample_ids = []
        for fold_idx, (train_set, val_set) in enumerate(folds):
            # Assert no empty sets
            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(val_set), 0)

            # Assert disjointness of genes
            train_genes = {s.metadata.gene for s in train_set if s.metadata.gene}
            val_genes = {s.metadata.gene for s in val_set if s.metadata.gene}
            self.assertEqual(train_genes & val_genes, set(), f"Fold {fold_idx} has gene leakage!")

            # Assert disjointness of sequences
            train_seqs = {s.sequence for s in train_set}
            val_seqs = {s.sequence for s in val_set}
            self.assertEqual(train_seqs & val_seqs, set(), f"Fold {fold_idx} has sequence leakage!")

            # Collect validation sample IDs
            val_sample_ids.extend([s.sample_id for s in val_set])

        # Assert every sample appeared in validation exactly once
        self.assertEqual(len(val_sample_ids), total_samples)
        self.assertEqual(len(set(val_sample_ids)), total_samples)

    def test_generate_group_shufflesplit_folds(self) -> None:
        samples = []
        for c_idx, label in enumerate(["Cancer_A", "Cancer_B"]):
            for g_idx in range(6):
                s = DNASample(f"S_{c_idx}_{g_idx}", f"ACGT_{c_idx}_{g_idx}", label)
                s.metadata.gene = f"G_{c_idx}_{g_idx}"
                samples.append(s)

        folds = generate_group_shufflesplit_folds(samples, n_splits=4, val_size=0.25, seed=42)
        self.assertEqual(len(folds), 4)

        for train_set, val_set in folds:
            train_genes = {s.metadata.gene for s in train_set if s.metadata.gene}
            val_genes = {s.metadata.gene for s in val_set if s.metadata.gene}
            self.assertEqual(train_genes & val_genes, set())


if __name__ == "__main__":
    unittest.main()
