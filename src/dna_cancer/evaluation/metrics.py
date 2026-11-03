"""Classification metrics calculation module."""

from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: Union[List[int], Any],
    y_pred: Union[List[int], Any],
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute standard multi-class classification metrics.

    Args:
        y_true: True class labels or indices.
        y_pred: Predicted class labels or indices.
        target_names: Optional list of human-readable class names.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, confusion matrix, and report.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError:
        raise ImportError("scikit-learn is required to calculate classification metrics.")

    y_true_arr = np.asarray(y_true) if np is not None else y_true
    y_pred_arr = np.asarray(y_pred) if np is not None else y_pred

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    logger.info("Evaluation metrics calculated: accuracy=%.4f, f1_macro=%.4f", acc, f1_macro)
    return metrics
