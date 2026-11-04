"""Experiment tracking system for DNA sequence classification benchmarks.

Supports both MLflow (when available) and local structured JSON/Markdown experiment logs.
Records dataset checksums, split configurations, k-mer feature settings, model hyperparameters,
durations, metrics, and artifact paths. Never uploads raw sensitive biological data.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore

from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Manages logging and persistence of benchmark runs."""

    def __init__(
        self,
        experiment_name: str = "dna_baseline_benchmarks",
        artifact_dir: Union[str, Path] = "artifacts/experiments",
        enable_mlflow: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.runs_summary_path = self.artifact_dir / "runs_summary.json"

        self.mlflow_enabled = enable_mlflow and (mlflow is not None)
        if self.mlflow_enabled:
            try:
                mlflow.set_experiment(self.experiment_name)
                logger.info("MLflow tracking active for experiment '%s'", self.experiment_name)
            except Exception as e:
                logger.warning("Could not initialize MLflow experiment: %s. Using local tracking only.", e)
                self.mlflow_enabled = False

    def log_run(
        self,
        run_name: str,
        dataset_id: str,
        dataset_checksum: str,
        split_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        model_name: str,
        model_params: Dict[str, Any],
        random_seed: int,
        metrics: Dict[str, Any],
        training_duration: float,
        prediction_duration: float,
        artifact_paths: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Record an experiment run's parameters, metrics, and artifact references.

        Args:
            run_name: Descriptive name for the experiment run.
            dataset_id: Identifier of the source dataset.
            dataset_checksum: SHA-256 checksum of dataset file.
            split_config: Partition strategy, sizes, and leakage audit result.
            feature_config: k-mer size, stride, vectorizer type, and n_features.
            model_name: Identifier of the classifier architecture.
            model_params: Dictionary of model hyperparameters.
            random_seed: Deterministic seed used.
            metrics: Accuracy, precision, recall, f1, etc.
            training_duration: Training duration in seconds.
            prediction_duration: Inference duration in seconds.
            artifact_paths: Map of artifact descriptions to relative/absolute file paths.
            tags: Extra key-value tags.

        Returns:
            The complete logged run record dictionary.
        """
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        run_record: Dict[str, Any] = {
            "run_id": run_id,
            "run_name": run_name,
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "dataset": {
                "dataset_id": dataset_id,
                "checksum_sha256": dataset_checksum,
            },
            "split_configuration": split_config,
            "feature_configuration": feature_config,
            "model": {
                "name": model_name,
                "parameters": model_params,
                "random_seed": random_seed,
            },
            "durations": {
                "training_seconds": float(round(training_duration, 4)),
                "prediction_seconds": float(round(prediction_duration, 4)),
            },
            "metrics": {
                "accuracy": float(round(metrics.get("accuracy", 0.0), 4)),
                "precision_macro": float(round(metrics.get("precision_macro", 0.0), 4)),
                "recall_macro": float(round(metrics.get("recall_macro", 0.0), 4)),
                "f1_macro": float(round(metrics.get("f1_macro", 0.0), 4)),
                "confusion_matrix": metrics.get("confusion_matrix", []),
                "per_class": metrics.get("classification_report", {}),
            },
            "artifacts": artifact_paths or {},
            "tags": tags or {},
            "scientific_limitation_notice": (
                "Educational sequence-classification benchmark. "
                "The evaluated dataset contains human chromosome 15 transcripts and bacterial genomic sequences. "
                "Labels are not verified clinical cancer diagnoses; results do not represent clinical prediction performance."
            ),
        }

        # 1. Local JSON file logging
        run_file = self.artifact_dir / f"{run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)

        # 2. Update runs summary index
        self._update_runs_summary(run_record)

        # 3. MLflow logging if active
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "dataset_id": dataset_id,
                        "dataset_checksum": dataset_checksum[:12],
                        "model_name": model_name,
                        "random_seed": random_seed,
                        "kmer_k": feature_config.get("k"),
                        "split_strategy": split_config.get("strategy"),
                        **{f"model_{k}": v for k, v in model_params.items() if isinstance(v, (int, float, str, bool))},
                    })
                    mlflow.log_metrics({
                        "accuracy": metrics.get("accuracy", 0.0),
                        "precision_macro": metrics.get("precision_macro", 0.0),
                        "recall_macro": metrics.get("recall_macro", 0.0),
                        "f1_macro": metrics.get("f1_macro", 0.0),
                        "training_seconds": training_duration,
                        "prediction_seconds": prediction_duration,
                    })
                    if artifact_paths:
                        for art_name, art_p in artifact_paths.items():
                            p = Path(art_p)
                            if p.exists() and p.is_file():
                                mlflow.log_artifact(str(p), artifact_path=art_name)
            except Exception as e:
                logger.warning("MLflow logging failed for run %s: %s", run_id, e)

        logger.info("Recorded benchmark run '%s' (run_id: %s) to %s", run_name, run_id, run_file)
        return run_record

    def _update_runs_summary(self, new_record: Dict[str, Any]) -> None:
        """Update runs_summary.json index file."""
        records: List[Dict[str, Any]] = []
        if self.runs_summary_path.exists():
            try:
                with open(self.runs_summary_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        # Summarized entry
        summary_entry = {
            "run_id": new_record["run_id"],
            "run_name": new_record["run_name"],
            "timestamp": new_record["timestamp"],
            "model": new_record["model"]["name"],
            "k": new_record["feature_configuration"].get("k"),
            "split_strategy": new_record["split_configuration"].get("strategy"),
            "n_features": new_record["feature_configuration"].get("n_features"),
            "accuracy": new_record["metrics"]["accuracy"],
            "f1_macro": new_record["metrics"]["f1_macro"],
            "training_seconds": new_record["durations"]["training_seconds"],
            "prediction_seconds": new_record["durations"]["prediction_seconds"],
            "is_leakage_free": new_record["split_configuration"].get("is_leakage_free", True),
        }
        records.append(summary_entry)

        with open(self.runs_summary_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def get_runs_summary(self) -> List[Dict[str, Any]]:
        """Return list of all summarized runs."""
        if not self.runs_summary_path.exists():
            return []
        try:
            with open(self.runs_summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

