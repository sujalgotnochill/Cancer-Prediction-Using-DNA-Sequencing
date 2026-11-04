"""Sealed evaluation vault for physical isolation and procedural access-control of final holdout data."""

import csv
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from dna_cancer.data.ingestion import ColumnMapping, load_samples, parse_fasta_records
from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.schema import DNASample, SampleMetadata
from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

AUTHORIZED_UNSEAL_TOKEN = "PHASE_5_FINAL_UNSEAL"

CERVICAL_HOLDOUT_GENES = {
    "ATP10A": 9,
    "CYFIP1": 10,
    "GABRA5": 7,
    "GABRB3": 6,
    "POTEB": 4,
    "POTEB2": 4,
    "SNURF": 3,
    "TUBGCP5": 17,
}


class SealedVault:
    """Manages physical separation, procedural locking, and one-shot unsealing of test holdout data.

    SECURITY NOTICE:
    This module provides a local software-level procedural lock and tamper-evident audit state.
    It guarantees that automated training, hyperparameter searches, and feature extraction pipelines
    cannot accidentally load the sealed test holdout.
    It does NOT provide kernel-level, cryptographic, or hardware security against deliberate out-of-band inspection.
    """

    def __init__(
        self,
        vault_dir: Union[str, Path] = "artifacts/vault",
        dev_cohort_path: Union[str, Path] = "data/processed/development_cohort.csv",
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.dev_cohort_path = Path(dev_cohort_path)
        self.manifest_path = self.vault_dir / "test_sealed_manifest.json"
        self.vault_data_path = self.vault_dir / "test_holdout_sealed.json"
        self.state_path = self.vault_dir / "vault_state.json"

    def initialize_vault(
        self,
        source_csv_path: Union[str, Path] = "data/interim/historical_reconstruction/DNAdata_reconstructed.csv",
        fasta_dir: Union[str, Path] = "data/raw/FASTA",
        force: bool = False,
    ) -> Dict[str, Any]:
        """Partition reconstructed dataset into Development Cohort (N=960) and Sealed Vault (N=240)."""
        source_p = Path(source_csv_path)
        fasta_p = Path(fasta_dir)

        if not source_p.exists():
            raise FileNotFoundError(f"Source reconstructed dataset not found: {source_p}")

        if self.manifest_path.exists() and self.vault_data_path.exists() and not force:
            logger.info("Sealed vault already initialized at %s (use force=True to reinitialize)", self.vault_dir)
            return self.get_vault_status()

        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.dev_cohort_path.parent.mkdir(parents=True, exist_ok=True)

        mapping = ColumnMapping(sequence_col="sequence", label_col="cancer", id_col="Unnamed: 0")
        samples = load_samples(source_p, file_format="csv", column_mapping=mapping)

        # Attach FASTA metadata
        fasta_files = [
            ("Cervical Cancer", "cervical_cancer.txt"),
            ("Colorectal Cancer", "Colorectal cancer .txt"),
            ("Endometrial Cancer", "Endometrial cancer .txt"),
            ("Liver Cancer", "Liver Cancer .txt"),
        ]
        meta = {}
        for label, fname in fasta_files:
            f_path = fasta_p / fname
            if f_path.exists():
                records = parse_fasta_records(f_path)[:300]
                for idx, r in enumerate(records):
                    desc = r.get("description", "")
                    m_gene = re.search(r"\[gene=([^\]]+)\]", desc)
                    m_locus = re.search(r"\[locus_tag=([^\]]+)\]", desc)
                    meta[(label, idx)] = {
                        "gene": m_gene.group(1) if m_gene else None,
                        "locus": m_locus.group(1) if m_locus else None,
                    }

        for idx_in_class, s in enumerate(samples):
            class_idx = idx_in_class % 300
            m = meta.get((s.label, class_idx))
            if m:
                s.metadata.gene = m["gene"]
                s.metadata.locus = m["locus"]
                s.metadata.additional_metadata["gene"] = m["gene"]
                s.metadata.additional_metadata["locus"] = m["locus"]

        dev_samples: List[DNASample] = []
        holdout_samples: List[DNASample] = []

        for s in samples:
            class_idx = int(s.sample_id) % 300
            if s.label == "Cervical Cancer":
                if s.metadata.gene in CERVICAL_HOLDOUT_GENES:
                    holdout_samples.append(s)
                else:
                    dev_samples.append(s)
            elif s.label == "Colorectal Cancer":
                # Samples 90 and 256 share identical sequence; keep 256 in Dev with 90
                # Holdout gets indices 240..299 except 256, plus 239 -> exactly 60 samples
                if (class_idx >= 240 and class_idx != 256) or class_idx == 239:
                    holdout_samples.append(s)
                else:
                    dev_samples.append(s)
            elif s.label in ["Endometrial Cancer", "Liver Cancer"]:
                if class_idx >= 240:
                    holdout_samples.append(s)
                else:
                    dev_samples.append(s)

        if len(dev_samples) != 960 or len(holdout_samples) != 240:
            raise DataValidationError(
                f"Partition error: Expected 960 dev and 240 holdout, got {len(dev_samples)} dev and {len(holdout_samples)} holdout."
            )

        # 1. Write Development Cohort CSV
        with open(self.dev_cohort_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Unnamed: 0", "sequence", "cancer", "gene", "locus"])
            for s in dev_samples:
                writer.writerow([
                    s.sample_id,
                    s.sequence,
                    s.label,
                    s.metadata.gene or "",
                    s.metadata.locus or "",
                ])

        # 2. Write Sealed Holdout JSON
        holdout_records = [
            {
                "sample_id": s.sample_id,
                "sequence": s.sequence,
                "label": s.label,
                "metadata": s.metadata.to_dict(),
            }
            for s in holdout_samples
        ]
        with open(self.vault_data_path, "w", encoding="utf-8") as f:
            json.dump(holdout_records, f, indent=2)

        # 3. Write initial locked state
        initial_state = {
            "status": "LOCKED",
            "unsealed": False,
            "unseal_count": 0,
            "unsealed_at": None,
            "unseal_token": None,
            "unseal_reason": None,
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, indent=2)

        # 4. Write manifest
        source_sha = calculate_file_checksum(source_p)
        dev_sha = calculate_file_checksum(self.dev_cohort_path)
        holdout_sha = calculate_file_checksum(self.vault_data_path)

        manifest = {
            "manifest_version": "1.0.0",
            "created_at": datetime.datetime.now().isoformat(),
            "source_dataset": str(source_p).replace("\\", "/"),
            "source_sha256": source_sha,
            "development_cohort": {
                "path": str(self.dev_cohort_path).replace("\\", "/"),
                "sample_count": len(dev_samples),
                "class_distribution": {
                    "Cervical Cancer": 240,
                    "Colorectal Cancer": 240,
                    "Endometrial Cancer": 240,
                    "Liver Cancer": 240,
                },
                "sha256": dev_sha,
            },
            "sealed_holdout": {
                "path": str(self.vault_data_path).replace("\\", "/"),
                "sample_count": len(holdout_samples),
                "class_distribution": {
                    "Cervical Cancer": 60,
                    "Colorectal Cancer": 60,
                    "Endometrial Cancer": 60,
                    "Liver Cancer": 60,
                },
                "cervical_holdout_genes": [
                    {"gene": g, "sample_count": cnt} for g, cnt in sorted(CERVICAL_HOLDOUT_GENES.items())
                ],
                "sha256": holdout_sha,
            },
            "safeguard_contract": (
                "This sealed vault is protected by a software procedural lock and access-logging contract. "
                "The holdout partition must remain sealed until model architecture and hyperparameters are frozen. "
                "This mechanism prevents accidental data leakage during automated optimization loops but does not provide "
                "operating-system or cryptographic isolation."
            ),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Initialized sealed vault (Dev N=%d, Holdout N=%d). Manifest saved to %s",
            len(dev_samples), len(holdout_samples), self.manifest_path
        )
        return self.get_vault_status()

    def is_locked(self) -> bool:
        """Check whether the sealed test vault is currently locked."""
        if not self.vault_data_path.exists():
            return True
        if not self.state_path.exists():
            raise DataValidationError(
                f"Sealed vault data exists at {self.vault_data_path} but vault state file {self.state_path} is missing. "
                "State file deletion or tampering detected."
            )
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as err:
            raise DataValidationError(f"Vault state file at {self.state_path} is corrupted: {err}") from err
        return not bool(state.get("unsealed", False))

    def get_vault_status(self) -> Dict[str, Any]:
        """Return the current procedural status of the sealed vault."""
        state = {}
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                state = {"status": "CORRUPTED"}

        manifest = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        is_locked_val = False
        try:
            is_locked_val = self.is_locked()
        except DataValidationError:
            is_locked_val = False

        return {
            "vault_exists": self.vault_data_path.exists(),
            "manifest_exists": self.manifest_path.exists(),
            "status": state.get("status", "NOT_INITIALIZED" if not self.vault_data_path.exists() else "LOCKED"),
            "is_locked": is_locked_val,
            "unsealed": state.get("unsealed", False),
            "unsealed_at": state.get("unsealed_at"),
            "unseal_count": state.get("unseal_count", 0),
            "development_cohort_path": str(self.dev_cohort_path).replace("\\", "/"),
            "holdout_data_path": str(self.vault_data_path).replace("\\", "/"),
            "manifest_path": str(self.manifest_path).replace("\\", "/"),
        }

    def verify_integrity(self) -> Dict[str, Any]:
        """Perform a forensic integrity audit on the sealed vault files and disjointness."""
        if not self.manifest_path.exists():
            return {"is_valid": False, "error": f"Manifest missing at {self.manifest_path}"}
        if not self.vault_data_path.exists():
            return {"is_valid": False, "error": f"Vault data missing at {self.vault_data_path}"}
        if not self.state_path.exists():
            return {"is_valid": False, "error": f"Vault state file missing at {self.state_path}"}

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as err:
            return {"is_valid": False, "error": f"Manifest corrupted: {err}"}

        holdout_sha = calculate_file_checksum(self.vault_data_path)
        expected_holdout_sha = manifest.get("sealed_holdout", {}).get("sha256")
        sha_valid = (holdout_sha == expected_holdout_sha)

        try:
            with open(self.vault_data_path, "r", encoding="utf-8") as f:
                holdout_records = json.load(f)
        except Exception as err:
            return {"is_valid": False, "error": f"Holdout JSON corrupted: {err}"}

        count_valid = (len(holdout_records) == 240)

        # Check dev cohort SHA if available
        dev_sha_valid = True
        if self.dev_cohort_path.exists():
            expected_dev_sha = manifest.get("development_cohort", {}).get("sha256")
            if expected_dev_sha:
                actual_dev_sha = calculate_file_checksum(self.dev_cohort_path)
                dev_sha_valid = (actual_dev_sha == expected_dev_sha)

        # Check sequence disjointness with dev cohort if available
        disjoint_valid = True
        seq_leak_count = 0
        if self.dev_cohort_path.exists():
            dev_seqs = set()
            with open(self.dev_cohort_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dev_seqs.add(row["sequence"].strip().upper())
            holdout_seqs = {r["sequence"].strip().upper() for r in holdout_records if "sequence" in r}
            overlap = dev_seqs & holdout_seqs
            seq_leak_count = len(overlap)
            disjoint_valid = (seq_leak_count == 0)

        locked = False
        try:
            locked = self.is_locked()
        except DataValidationError:
            locked = False

        is_valid = bool(sha_valid and count_valid and disjoint_valid and dev_sha_valid)
        err_msg = None
        if not sha_valid:
            err_msg = f"Holdout SHA-256 mismatch: expected {expected_holdout_sha}, got {holdout_sha}"
        elif not count_valid:
            err_msg = f"Holdout count mismatch: expected 240, got {len(holdout_records)}"
        elif not disjoint_valid:
            err_msg = f"Holdout sequence leakage detected with dev cohort: {seq_leak_count} overlap"
        elif not dev_sha_valid:
            err_msg = "Development cohort SHA-256 mismatch against manifest"

        return {
            "is_valid": is_valid,
            "error": err_msg,
            "sha_match": sha_valid,
            "dev_sha_match": dev_sha_valid,
            "sample_count": len(holdout_records),
            "expected_sample_count": 240,
            "sequence_leakage_with_dev": seq_leak_count,
            "is_locked": locked,
        }

    def unseal(self, token: str, reason: str) -> List[DNASample]:
        """Unseal the holdout vault strictly once for final frozen evaluation."""
        if not self.is_locked():
            raise PermissionError(
                "Sealed test vault has ALREADY been unsealed. Final evaluation is strictly one-shot."
            )

        if token != AUTHORIZED_UNSEAL_TOKEN:
            raise PermissionError(f"Invalid access token for sealed vault. Expected '{AUTHORIZED_UNSEAL_TOKEN}'.")

        if not reason or len(reason.strip()) < 10:
            raise ValueError("A valid justification reason (at least 10 characters) is required to unseal the vault.")

        audit = self.verify_integrity()
        if not audit["is_valid"]:
            raise DataValidationError(f"Cannot unseal vault: integrity check failed: {audit.get('error')}")

        with open(self.vault_data_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        samples: List[DNASample] = []
        for r in raw_records:
            meta = SampleMetadata.from_dict(r.get("metadata", {}))
            s = DNASample(
                sample_id=r["sample_id"],
                sequence=r["sequence"],
                label=r["label"],
                metadata=meta,
            )
            samples.append(s)

        # Update state file atomically
        new_state = {
            "status": "UNSEALED",
            "unsealed": True,
            "unseal_count": 1,
            "unsealed_at": datetime.datetime.now().isoformat(),
            "unseal_token": token,
            "unseal_reason": reason.strip(),
        }
        temp_state_path = self.state_path.with_suffix(".tmp")
        with open(temp_state_path, "w", encoding="utf-8") as f:
            json.dump(new_state, f, indent=2)
        os.replace(temp_state_path, self.state_path)

        logger.warning(
            "SEALED VAULT UNSEALED: reason='%s', token='%s', sample_count=%d",
            reason.strip(), token, len(samples)
        )
        return samples

