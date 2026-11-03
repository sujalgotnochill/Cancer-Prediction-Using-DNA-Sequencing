"""Model evaluation orchestration module."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Orchestrates model evaluation against test data and persists performance reports."""

    def __init__(self, target_names: Optional[List[str]] = None) -> None:
        self.target_names = target_names

    def evaluate(
        self,
        model: BaseDNAClassifier,
        X_test: Any,
        y_test: Any,
    ) -> Dict[str, Any]:
        """Generate evaluation metrics for a trained classifier on test data."""
        y_pred = model.predict(X_test)
        return self.evaluate_predictions(y_true=y_test, y_pred=y_pred)

    def evaluate_predictions(
        self,
        y_true: Union[List[int], Any],
        y_pred: Union[List[int], Any],
    ) -> Dict[str, Any]:
        """Compute metrics given ground truth and predictions."""
        return compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            target_names=self.target_names,
        )

    def save_report(
        self,
        metrics: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> None:
        """Save evaluation metrics dictionary to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved evaluation report to %s", out)
