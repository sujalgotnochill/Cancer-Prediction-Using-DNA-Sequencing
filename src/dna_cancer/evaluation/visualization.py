"""Visualization utilities for baseline machine learning benchmarks."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None  # type: ignore
    np = None  # type: ignore

from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_matplotlib() -> None:
    if plt is None or np is None:
        raise ImportError("matplotlib and numpy are required to generate evaluation plots.")


def plot_model_comparison(
    models_metrics: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Baseline Model Performance Comparison",
) -> Path:
    """Generate and save grouped bar chart comparing multiple models across standard metrics.

    Args:
        models_metrics: List of dicts, each containing:
            - 'model_name': str
            - 'accuracy': float
            - 'precision_macro': float
            - 'recall_macro': float
            - 'f1_macro': float
        output_path: Destination image path (.png).
        title: Plot title string.

    Returns:
        Resolved Path to the saved plot.
    """
    _ensure_matplotlib()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    names = [m["model_name"] for m in models_metrics]
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    
    accs = [m.get("accuracy", 0.0) for m in models_metrics]
    precs = [m.get("precision_macro", 0.0) for m in models_metrics]
    recs = [m.get("recall_macro", 0.0) for m in models_metrics]
    f1s = [m.get("f1_macro", 0.0) for m in models_metrics]

    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5 * width, accs, width, label="Accuracy", color="#2b5c8f")
    rects2 = ax.bar(x - 0.5 * width, precs, width, label="Macro Precision", color="#4392f1")
    rects3 = ax.bar(x + 0.5 * width, recs, width, label="Macro Recall", color="#dc851f")
    rects4 = ax.bar(x + 1.5 * width, f1s, width, label="Macro F1", color="#38b000")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim([0.0, 1.08])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    # Value labels
    for rects in [rects1, rects2, rects3, rects4]:
        for r in rects:
            h = r.get_height()
            if h > 0.05:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", out)
    return out


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> Path:
    """Plot and save a styled confusion matrix heatmap with annotations."""
    _ensure_matplotlib()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate counts inside cells
    thresh = cm_arr.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_arr[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold" if val > 0 else "normal",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out)
    return out


def plot_kmer_ablation(
    k_values: List[int],
    models_accuracies: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "Effect of k-mer Size on Classification Accuracy",
) -> Path:
    """Plot accuracy curves across different k-mer values (e.g. k=3, 4, 5, 6)."""
    _ensure_matplotlib()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "d"]
    colors = ["#2b5c8f", "#dc851f", "#38b000", "#7b2cbf"]

    for idx, (model_name, accs) in enumerate(models_accuracies.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            k_values,
            accs,
            marker=marker,
            linewidth=2,
            markersize=8,
            label=model_name,
            color=color,
        )

    ax.set_xlabel("k-mer size (k)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved k-mer ablation plot to %s", out)
    return out


def plot_per_class_f1(
    class_names: List[str],
    model_class_f1s: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "Per-Class F1 Score Comparison",
) -> Path:
    """Plot per-class F1 performance across models."""
    _ensure_matplotlib()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(model_class_f1s)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / max(1, n_models)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2b5c8f", "#dc851f", "#38b000", "#7b2cbf"]

    for idx, (model_name, f1_scores) in enumerate(model_class_f1s.items()):
        offset = (idx - (n_models - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            f1_scores,
            width,
            label=model_name,
            color=colors[idx % len(colors)],
        )
        for r in rects:
            h = r.get_height()
            if h > 0.05:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim([0.0, 1.08])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved per-class F1 plot to %s", out)
    return out

