"""Evaluation metrics and assessment modules."""

from dna_cancer.evaluation.evaluator import Evaluator
from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.evaluation.visualization import (
    plot_confusion_matrix,
    plot_kmer_ablation,
    plot_model_comparison,
    plot_per_class_f1,
)

__all__ = [
    "Evaluator",
    "compute_classification_metrics",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_kmer_ablation",
    "plot_per_class_f1",
]

