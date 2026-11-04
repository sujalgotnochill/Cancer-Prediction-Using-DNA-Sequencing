"""Dataset partitioning and splitting module with leakage prevention."""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dna_cancer.data.schema import DNASample
from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def recommend_splitting_strategy(
    samples: List[DNASample],
    group_key: str = "patient_id",
) -> Tuple[str, str]:
    """Inspect dataset metadata to recommend an optimal, leakage-free splitting strategy.

    Args:
        samples: List of DNASample objects.
        group_key: Attribute name to inspect for patient or sample grouping.

    Returns:
        Tuple of (strategy_name, rationale_string).
    """
    if not samples:
        return "random", "Empty dataset; default to random."

    group_values = [getattr(s, group_key, None) or s.metadata.additional_metadata.get(group_key) for s in samples]
    populated_groups = [g for g in group_values if g]

    if populated_groups:
        group_counts = defaultdict(int)
        for g in populated_groups:
            group_counts[g] += 1
        multi_sample_groups = [g for g, cnt in group_counts.items() if cnt > 1]
        if multi_sample_groups:
            return (
                "group",
                f"Group identifiers ('{group_key}') detected with multiple samples ({len(multi_sample_groups)} multi-sample groups). "
                "Group-based splitting is strongly recommended to prevent patient/sample data leakage."
            )

    labels = [s.label for s in samples if s.label]
    if len(labels) == len(samples) and len(set(labels)) > 1:
        return (
            "stratified",
            "Supervised classification labels present across all samples without grouping dependencies; "
            "stratified splitting recommended to preserve class proportions."
        )

    return "random", "No group dependencies or incomplete labels; random splitting recommended."


def _validate_split_ratios(train_size: float, val_size: float, test_size: float) -> None:
    """Validate that split proportions sum to 1.0 and are non-negative."""
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-4:
        raise DataValidationError(f"Split sizes must sum to 1.0, got train={train_size}, val={val_size}, test={test_size} (sum={total})")
    if train_size <= 0:
        raise DataValidationError(f"train_size must be positive, got {train_size}")
    if val_size < 0 or test_size < 0:
        raise DataValidationError("val_size and test_size cannot be negative.")


def split_random(
    samples: List[DNASample],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[DNASample], List[DNASample], List[DNASample]]:
    """Partition samples randomly using a deterministic seed."""
    _validate_split_ratios(train_size, val_size, test_size)
    n = len(samples)
    if n < 3:
        raise DataValidationError(f"Cannot perform 3-way split on dataset with only {n} samples.")

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)

    train_samples = [samples[i] for i in indices[:train_end]]
    val_samples = [samples[i] for i in indices[train_end:val_end]]
    test_samples = [samples[i] for i in indices[val_end:]]

    return train_samples, val_samples, test_samples


def split_stratified(
    samples: List[DNASample],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[DNASample], List[DNASample], List[DNASample]]:
    """Partition samples preserving target class label proportions across splits."""
    _validate_split_ratios(train_size, val_size, test_size)
    n = len(samples)
    if n < 3:
        raise DataValidationError(f"Cannot perform 3-way split on dataset with only {n} samples.")

    class_to_samples: Dict[str, List[DNASample]] = defaultdict(list)
    for s in samples:
        lbl = s.label or "Unlabeled"
        class_to_samples[lbl].append(s)

    train_out: List[DNASample] = []
    val_out: List[DNASample] = []
    test_out: List[DNASample] = []

    rng = random.Random(seed)

    for lbl, group in class_to_samples.items():
        g_len = len(group)
        group_copy = list(group)
        rng.shuffle(group_copy)

        if g_len < 3:
            # Fall back to putting minority items proportionally
            train_out.extend(group_copy[: max(1, int(train_size * g_len))])
            remaining = group_copy[max(1, int(train_size * g_len)):]
            if remaining:
                test_out.append(remaining[0])
            continue

        n_train = max(1, int(train_size * g_len))
        n_val = int(val_size * g_len)

        train_out.extend(group_copy[:n_train])
        val_out.extend(group_copy[n_train : n_train + n_val])
        test_out.extend(group_copy[n_train + n_val :])

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    rng.shuffle(test_out)

    return train_out, val_out, test_out


def split_grouped(
    samples: List[DNASample],
    group_key: str = "patient_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[DNASample], List[DNASample], List[DNASample]]:
    """Partition samples ensuring no group (e.g. patient) leaks across train/val/test splits.

    All samples belonging to a patient or group are assigned atomically to exactly one partition.
    """
    _validate_split_ratios(train_size, val_size, test_size)
    if not samples:
        raise DataValidationError("Cannot split empty sample list.")

    # Group sample objects by group identifier
    groups: Dict[str, List[DNASample]] = defaultdict(list)
    for idx, s in enumerate(samples):
        grp = getattr(s, group_key, None) or s.metadata.additional_metadata.get(group_key)
        # Fall back to unique sample_id if no group key present
        actual_group = str(grp) if grp is not None else f"ungrouped_{idx}"
        groups[actual_group].append(s)

    unique_groups = list(groups.keys())
    if len(unique_groups) < 3:
        raise DataValidationError(
            f"Group splitting requires at least 3 distinct groups, found {len(unique_groups)} groups: {unique_groups}"
        )

    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    g_train_end = int(train_size * n_groups)
    g_val_end = g_train_end + int(val_size * n_groups)

    train_groups = set(unique_groups[:g_train_end])
    val_groups = set(unique_groups[g_train_end:g_val_end])
    test_groups = set(unique_groups[g_val_end:])

    # Strictly enforce disjoint sets
    assert not (train_groups & val_groups), "Leakage detected: train & val groups overlap!"
    assert not (train_groups & test_groups), "Leakage detected: train & test groups overlap!"
    assert not (val_groups & test_groups), "Leakage detected: val & test groups overlap!"

    train_out: List[DNASample] = []
    val_out: List[DNASample] = []
    test_out: List[DNASample] = []

    for g_id, g_samples in groups.items():
        if g_id in train_groups:
            train_out.extend(g_samples)
        elif g_id in val_groups:
            val_out.extend(g_samples)
        else:
            test_out.extend(g_samples)

    logger.info(
        "Group split by '%s' (%d total groups): train=%d samples (%d groups), val=%d samples (%d groups), test=%d samples (%d groups)",
        group_key, n_groups, len(train_out), len(train_groups), len(val_out), len(val_groups), len(test_out), len(test_groups)
    )

    return train_out, val_out, test_out


class DatasetSplitter:
    """Configurable dataset splitting coordinator."""

    def __init__(
        self,
        strategy: str = "auto",
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
        group_key: str = "patient_id",
    ) -> None:
        self.strategy = strategy.lower()
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.group_key = group_key
        self.last_strategy_used: Optional[str] = None

    def split(
        self,
        samples: List[DNASample],
    ) -> Tuple[List[DNASample], List[DNASample], List[DNASample]]:
        """Split a list of DNASample objects using configured or automatically determined strategy."""
        strat = self.strategy
        if strat == "auto":
            strat, reason = recommend_splitting_strategy(samples, group_key=self.group_key)
            logger.info("Auto-selected splitting strategy '%s': %s", strat, reason)

        self.last_strategy_used = strat

        if strat == "group":
            return split_grouped(
                samples=samples,
                group_key=self.group_key,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                seed=self.seed,
            )
        elif strat == "stratified":
            return split_stratified(
                samples=samples,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                seed=self.seed,
            )
        elif strat == "random":
            return split_random(
                samples=samples,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                seed=self.seed,
            )
        else:
            raise DataValidationError(f"Unsupported splitting strategy: '{strat}'. Supported: auto, group, stratified, random")

    def summarize_split(
        self,
        train: List[DNASample],
        val: List[DNASample],
        test: List[DNASample],
        group_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate summary statistics and leakage audit for train, validation, and test splits."""
        key = group_key or self.group_key

        def _partition_stats(subset: List[DNASample]) -> Dict[str, Any]:
            class_counts = defaultdict(int)
            group_vals = set()
            for s in subset:
                lbl = s.label or "Unlabeled"
                class_counts[lbl] += 1
                g = getattr(s, key, None) or s.metadata.additional_metadata.get(key)
                if g:
                    group_vals.add(str(g))
            return {
                "count": len(subset),
                "class_distribution": dict(class_counts),
                "distinct_groups": len(group_vals),
                "group_ids": list(group_vals),
            }

        train_stats = _partition_stats(train)
        val_stats = _partition_stats(val)
        test_stats = _partition_stats(test)

        train_grp = set(train_stats.pop("group_ids"))
        val_grp = set(val_stats.pop("group_ids"))
        test_grp = set(test_stats.pop("group_ids"))

        leakage = (
            bool(train_grp.intersection(val_grp))
            or bool(train_grp.intersection(test_grp))
            or bool(val_grp.intersection(test_grp))
        )

        return {
            "total_samples": len(train) + len(val) + len(test),
            "strategy_used": self.last_strategy_used or self.strategy,
            "leakage_free": not leakage,
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        }
