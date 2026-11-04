"""Final frozen model evaluation entry point against the sealed holdout vault."""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.schema import DNASample
from dna_cancer.data.vault import AUTHORIZED_UNSEAL_TOKEN, SealedVault
from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class FinalHoldoutEvaluator:
    """Exclusive entry point for final frozen model evaluation against the sealed test holdout.

    Enforces all architectural boundary checks:
    1. Model configuration is explicitly frozen ('is_frozen': True).
    2. Development dataset checksum matches manifest.
    3. Sealed test holdout checksum matches manifest.
    4. Procedural vault lock is active and has not been previously unsealed.
    5. Evaluation event is permanently recorded in an append-only audit log.
    6. Repeated evaluations are strictly rejected.
    """

    def __init__(
        self,
        vault: Optional[SealedVault] = None,
        event_log_path: Union[str, Path] = "artifacts/vault/evaluation_events.jsonl",
    ) -> None:
        self.vault = vault or SealedVault()
        self.event_log_path = Path(event_log_path)

    def verify_readiness(
        self,
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify all pre-conditions before unsealing without unsealing the vault.

        Args:
            model_config: Configuration dictionary for the candidate model.

        Returns:
            Dictionary detailing readiness status.

        Raises:
            ValueError: If model configuration is not explicitly frozen.
            FileNotFoundError: If required files (manifest, data) are missing.
            DataValidationError: If checksums fail or files are corrupted.
            PermissionError: If vault has already been unsealed.
        """
        if not isinstance(model_config, dict):
            raise ValueError("model_config must be a dictionary.")

        is_frozen = bool(model_config.get("is_frozen", False) or model_config.get("frozen", False))
        if not is_frozen:
            raise ValueError(
                "Model configuration must be explicitly marked as frozen ('is_frozen': True) before final holdout evaluation."
            )

        if not self.vault.manifest_path.exists():
            raise FileNotFoundError(f"Vault manifest not found at {self.vault.manifest_path}")

        try:
            with open(self.vault.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as err:
            raise DataValidationError(f"Vault manifest is corrupted or unreadable: {err}") from err

        # Check dev cohort checksum if path exists
        dev_rel_path = manifest.get("development_cohort", {}).get("path")
        dev_path = Path(dev_rel_path) if dev_rel_path else self.vault.dev_cohort_path
        if not dev_path.exists():
            raise FileNotFoundError(f"Development cohort not found at {dev_path}")
        dev_sha = calculate_file_checksum(dev_path)
        expected_dev_sha = manifest["development_cohort"]["sha256"]
        if dev_sha != expected_dev_sha:
            raise DataValidationError(
                f"Development cohort checksum mismatch: expected {expected_dev_sha}, got {dev_sha}."
            )

        # Check holdout file checksum
        holdout_rel_path = manifest.get("sealed_holdout", {}).get("path")
        holdout_path = Path(holdout_rel_path) if holdout_rel_path else self.vault.vault_data_path
        if not holdout_path.exists():
            raise FileNotFoundError(f"Sealed holdout data not found at {holdout_path}")
        holdout_sha = calculate_file_checksum(holdout_path)
        expected_holdout_sha = manifest["sealed_holdout"]["sha256"]
        if holdout_sha != expected_holdout_sha:
            raise DataValidationError(
                f"Sealed holdout checksum mismatch: expected {expected_holdout_sha}, got {holdout_sha}."
            )

        # Check lock status
        if not self.vault.is_locked():
            raise PermissionError("Sealed test vault is already UNSEALED. Repeated evaluation is prohibited.")

        return {
            "is_ready": True,
            "model_frozen": True,
            "dev_sha_match": True,
            "holdout_sha_match": True,
            "vault_locked": True,
            "holdout_sample_count": manifest["sealed_holdout"]["sample_count"],
        }

    def evaluate_frozen_model(
        self,
        model: BaseDNAClassifier,
        model_config: Dict[str, Any],
        unseal_token: str,
        reason: str,
        feature_extractor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute strictly one-shot final evaluation on the unsealed holdout.

        Args:
            model: Trained classifier inheriting from BaseDNAClassifier.
            model_config: Frozen model configuration dictionary.
            unseal_token: Authorization token matching AUTHORIZED_UNSEAL_TOKEN.
            reason: Methodological justification for unsealing.
            feature_extractor: Optional feature transformer fitted on development data.

        Returns:
            Dictionary containing final evaluation metrics and metadata.
        """
        # 1. Verify all pre-conditions
        self.verify_readiness(model_config=model_config)

        # 2. Unseal the vault
        holdout_samples = self.vault.unseal(token=unseal_token, reason=reason)

        # 3. Extract features and evaluate
        sequences = [s.sequence for s in holdout_samples]
        labels = [s.label for s in holdout_samples]

        if feature_extractor is not None:
            if hasattr(feature_extractor, "transform"):
                X_holdout = feature_extractor.transform(sequences)
            else:
                X_holdout = feature_extractor(sequences)
        else:
            X_holdout = sequences

        y_pred = model.predict(X_holdout)

        # Compute metrics
        metrics = compute_classification_metrics(y_true=labels, y_pred=y_pred)

        # Record evaluation event in append-only log
        self._record_evaluation_event(
            model_config=model_config,
            metrics=metrics,
            token=unseal_token,
            reason=reason,
        )

        return {
            "status": "FINAL_EVALUATION_COMPLETED",
            "evaluated_at": datetime.datetime.now().isoformat(),
            "sample_count": len(holdout_samples),
            "model_config": model_config,
            "metrics": metrics,
        }

    def _record_evaluation_event(
        self,
        model_config: Dict[str, Any],
        metrics: Dict[str, Any],
        token: str,
        reason: str,
    ) -> None:
        """Append-only record of the final evaluation event."""
        event = {
            "event": "FINAL_HOLDOUT_EVALUATION",
            "timestamp": datetime.datetime.now().isoformat(),
            "unseal_token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest()[:16],
            "reason": reason,
            "model_config": model_config,
            "metrics_summary": {
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
            },
        }
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        logger.warning("Recorded final holdout evaluation event to %s", self.event_log_path)

