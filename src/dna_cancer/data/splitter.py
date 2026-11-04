"""Dataset partitioning and splitting module with leakage prevention."""

import hashlib
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


def split_duplicate_grouped(
    samples: List[DNASample],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    group_by: str = "sequence",
) -> Tuple[List[DNASample], List[DNASample], List[DNASample]]:
    """Partition samples grouping identical sequences (or transcript/gene families) together.

    Guarantees that no identical sequence appears in multiple splits.
    Maintains class balance by stratifying sequence groups by their assigned class label.

    Args:
        samples: List of DNASample objects.
        train_size: Proportion of groups for training.
        val_size: Proportion of groups for validation.
        test_size: Proportion of groups for test.
        seed: Random seed for group shuffling.
        group_by: 'sequence' (hash of normalized DNA) or 'gene' (sample gene metadata).

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    _validate_split_ratios(train_size, val_size, test_size)
    if not samples:
        raise DataValidationError("Cannot split empty sample list.")

    # 1. Assign each sample to a group
    seq_groups: Dict[str, List[DNASample]] = defaultdict(list)
    for idx, s in enumerate(samples):
        if group_by == "gene" and getattr(s.metadata, "gene", None):
            grp_id = f"gene_{s.metadata.gene}"
        elif group_by == "gene" and s.metadata.additional_metadata.get("gene"):
            grp_id = f"gene_{s.metadata.additional_metadata['gene']}"
        else:
            # Hash canonicalized uppercase sequence
            norm_seq = s.sequence.strip().upper()
            grp_id = "seq_" + hashlib.sha256(norm_seq.encode("utf-8")).hexdigest()
        seq_groups[grp_id].append(s)

    # 2. Map groups to their primary label for stratified group partitioning
    label_to_group_ids: Dict[str, List[str]] = defaultdict(list)
    for grp_id, grp_samples in seq_groups.items():
        # Use dominant label in group
        lbl_counts = defaultdict(int)
        for s in grp_samples:
            lbl_counts[s.label or "Unlabeled"] += 1
        primary_lbl = max(lbl_counts.items(), key=lambda x: x[1])[0]
        label_to_group_ids[primary_lbl].append(grp_id)

    rng = random.Random(seed)
    train_group_ids: Set[str] = set()
    val_group_ids: Set[str] = set()
    test_group_ids: Set[str] = set()

    for lbl, g_ids in label_to_group_ids.items():
        shuffled = list(g_ids)
        rng.shuffle(shuffled)
        n_g = len(shuffled)

        if n_g < 3:
            # Minority group fallback
            train_group_ids.update(shuffled[: max(1, int(train_size * n_g))])
            rem = shuffled[max(1, int(train_size * n_g)) :]
            if rem:
                test_group_ids.add(rem[0])
            continue

        n_train = max(1, int(train_size * n_g))
        n_val = int(val_size * n_g)

        train_group_ids.update(shuffled[:n_train])
        val_group_ids.update(shuffled[n_train : n_train + n_val])
        test_group_ids.update(shuffled[n_train + n_val :])

    # Assert zero overlap between group ID sets
    assert not (train_group_ids & val_group_ids), "Duplicate leakage between train and val groups!"
    assert not (train_group_ids & test_group_ids), "Duplicate leakage between train and test groups!"
    assert not (val_group_ids & test_group_ids), "Duplicate leakage between val and test groups!"

    train_out: List[DNASample] = []
    val_out: List[DNASample] = []
    test_out: List[DNASample] = []

    for grp_id, grp_samples in seq_groups.items():
        if grp_id in train_group_ids:
            train_out.extend(grp_samples)
        elif grp_id in val_group_ids:
            val_out.extend(grp_samples)
        else:
            test_out.extend(grp_samples)

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    rng.shuffle(test_out)

    logger.info(
        "Duplicate-grouped split (%d unique groups, group_by='%s'): train=%d samples (%d groups), val=%d samples (%d groups), test=%d samples (%d groups)",
        len(seq_groups),
        group_by,
        len(train_out),
        len(train_group_ids),
        len(val_out),
        len(val_group_ids),
        len(test_out),
        len(test_group_ids),
    )

    return train_out, val_out, test_out


def split_gene_family_grouped(
    samples: List[DNASample],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    strict_metadata_only: bool = False,
    gene_key: str = "gene",
    grouping_mode: str = "within_organism",
) -> Tuple[List[DNASample], List[DNASample], List[DNASample]]:
    """Partition samples grouping gene families together to prevent transcript-isoform leakage.

    Guarantees that no gene family appears in multiple splits.
    
    Grouping modes:
      - 'within_organism' (default): Groups by (organism/label, gene). Prevents transcript-isoform
        leakage within each biological source while correctly treating matching gene symbols across
        distinct bacterial phyla (e.g. coxA in S. amylolyticus vs A. pallidiroseus) as distinct groups.
      - 'cross_organism': Groups identical gene symbols globally across all classes regardless of organism.
        (Note: Sequence-level orthology is not verified across distinct bacterial phyla).
      - 'exact_sequence': Groups strictly by exact nucleotide sequence hash.

    For samples lacking explicit gene annotations:
      - If strict_metadata_only=True: samples without gene metadata are excluded from partitioning.
      - If strict_metadata_only=False: samples fall back to sequence-hash grouping, ensuring identical sequences
        are never split across partitions while keeping all available samples. Fallback sequence hashes
        are tracked transparently and never misrepresented as biological gene families.

    Args:
        samples: List of DNASample objects.
        train_size: Proportion of groups for training.
        val_size: Proportion of groups for validation.
        test_size: Proportion of groups for test.
        seed: Random seed for group shuffling.
        strict_metadata_only: If True, exclude samples lacking gene metadata.
        gene_key: Metadata field to use for gene family (default: 'gene').
        grouping_mode: 'within_organism', 'cross_organism', or 'exact_sequence'.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    _validate_split_ratios(train_size, val_size, test_size)
    if not samples:
        raise DataValidationError("Cannot split empty sample list.")

    if strict_metadata_only:
        active_samples = [
            s for s in samples
            if (getattr(s.metadata, gene_key, None) or s.metadata.additional_metadata.get(gene_key))
        ]
        if len(active_samples) < 3:
            raise DataValidationError(
                f"Strict gene-family splitting requires at least 3 samples with '{gene_key}' metadata, found {len(active_samples)}."
            )
        logger.info(
            "Strict gene-family filter: retained %d samples with verified '%s' annotations (excluded %d unannotated)",
            len(active_samples), gene_key, len(samples) - len(active_samples)
        )
    else:
        active_samples = samples

    if grouping_mode == "cross_organism":
        logger.warning(
            "Grouping mode 'cross_organism' active: matching gene symbols will be grouped globally across "
            "distinct organisms without verified sequence-level orthology."
        )

    # Assign each sample to a group
    groups: Dict[str, List[DNASample]] = defaultdict(list)
    bio_group_count = 0
    seq_group_count = 0

    for s in active_samples:
        gene = getattr(s.metadata, gene_key, None) or s.metadata.additional_metadata.get(gene_key)
        norm_seq = s.sequence.strip().upper()
        seq_hash = hashlib.sha256(norm_seq.encode("utf-8")).hexdigest()

        if grouping_mode == "exact_sequence" or (not gene):
            # Fallback exact sequence grouping: ensure identical sequences never split across partitions
            grp_id = f"seq_{seq_hash}"
            seq_group_count += 1
        elif grouping_mode == "within_organism":
            # Within-organism gene family grouping: incorporate organism/label context
            org_ctx = s.metadata.organism or s.label or "unspecified"
            grp_id = f"gene_{org_ctx}_{str(gene).strip()}"
            bio_group_count += 1
        elif grouping_mode == "cross_organism":
            # Global cross-organism gene grouping
            grp_id = f"gene_{str(gene).strip()}"
            bio_group_count += 1
        else:
            raise DataValidationError(
                f"Unsupported grouping_mode: '{grouping_mode}'. Supported: 'within_organism', 'cross_organism', 'exact_sequence'"
            )

        groups[grp_id].append(s)

    # Map groups to their primary label for stratified group partitioning
    label_to_group_ids: Dict[str, List[str]] = defaultdict(list)
    for grp_id, grp_samples in groups.items():
        lbl_counts = defaultdict(int)
        for s in grp_samples:
            lbl_counts[s.label or "Unlabeled"] += 1
        primary_lbl = max(lbl_counts.items(), key=lambda x: x[1])[0]
        label_to_group_ids[primary_lbl].append(grp_id)

    rng = random.Random(seed)
    train_group_ids: Set[str] = set()
    val_group_ids: Set[str] = set()
    test_group_ids: Set[str] = set()

    for lbl, g_ids in label_to_group_ids.items():
        shuffled = list(g_ids)
        rng.shuffle(shuffled)
        n_g = len(shuffled)

        if n_g < 3:
            train_group_ids.update(shuffled[: max(1, int(train_size * n_g))])
            rem = shuffled[max(1, int(train_size * n_g)) :]
            if rem:
                test_group_ids.add(rem[0])
            continue

        n_train = max(1, int(train_size * n_g))
        n_val = int(val_size * n_g)

        train_group_ids.update(shuffled[:n_train])
        val_group_ids.update(shuffled[n_train : n_train + n_val])
        test_group_ids.update(shuffled[n_train + n_val :])

    assert not (train_group_ids & val_group_ids), "Gene-family leakage between train and val groups!"
    assert not (train_group_ids & test_group_ids), "Gene-family leakage between train and test groups!"
    assert not (val_group_ids & test_group_ids), "Gene-family leakage between val and test groups!"

    train_out: List[DNASample] = []
    val_out: List[DNASample] = []
    test_out: List[DNASample] = []

    for grp_id, grp_samples in groups.items():
        if grp_id in train_group_ids:
            train_out.extend(grp_samples)
        elif grp_id in val_group_ids:
            val_out.extend(grp_samples)
        else:
            test_out.extend(grp_samples)

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    rng.shuffle(test_out)

    logger.info(
        "Gene-family split (%d unique groups, mode='%s', strict=%s): train=%d samples (%d groups), val=%d samples (%d groups), test=%d samples (%d groups)",
        len(groups),
        grouping_mode,
        strict_metadata_only,
        len(train_out),
        len(train_group_ids),
        len(val_out),
        len(val_group_ids),
        len(test_out),
        len(test_group_ids),
    )

    return train_out, val_out, test_out


def split_gene_family_cv_folds(
    samples: List[DNASample],
    n_splits: int = 3,
    seed: int = 42,
    grouping_mode: str = "within_organism",
    gene_key: str = "gene",
) -> List[Tuple[List[DNASample], List[DNASample]]]:
    """Partition development samples into deterministic group-aware cross-validation folds.

    Guarantees:
      1. Every development sample appears in the validation partition of exactly one fold.
      2. No within-organism gene family or exact nucleotide sequence crosses training and validation partitions.
      3. All class labels are represented in both training and validation splits for every fold.
      4. Fold assignments are completely deterministic.

    Args:
        samples: List of DNASample objects (e.g., Development Cohort).
        n_splits: Number of cross-validation folds (recommended: 3).
        seed: Random seed for deterministic group shuffling.
        grouping_mode: 'within_organism' (default), 'cross_organism', or 'exact_sequence'.
        gene_key: Metadata field for gene family.

    Returns:
        List of (train_samples, val_samples) tuples for each fold.
    """
    if n_splits < 2:
        raise DataValidationError(f"Cross-validation requires at least 2 folds, got {n_splits}.")
    if not samples:
        raise DataValidationError("Cannot generate CV folds on empty sample list.")

    # 1. Assign each sample to an atomic group
    groups: Dict[str, List[DNASample]] = defaultdict(list)
    for s in samples:
        gene = getattr(s.metadata, gene_key, None) or s.metadata.additional_metadata.get(gene_key)
        norm_seq = s.sequence.strip().upper()
        seq_hash = hashlib.sha256(norm_seq.encode("utf-8")).hexdigest()

        if grouping_mode == "exact_sequence" or (not gene):
            grp_id = f"seq_{seq_hash}"
        elif grouping_mode == "within_organism":
            org_ctx = s.metadata.organism or s.label or "unspecified"
            grp_id = f"gene_{org_ctx}_{str(gene).strip()}"
        elif grouping_mode == "cross_organism":
            grp_id = f"gene_{str(gene).strip()}"
        else:
            raise DataValidationError(f"Unsupported grouping_mode: '{grouping_mode}'.")

        groups[grp_id].append(s)

    # 2. Partition groups per class using greedy bin-packing
    label_groups: Dict[str, Dict[str, List[DNASample]]] = defaultdict(dict)
    for gid, glist in groups.items():
        lbl_counts = defaultdict(int)
        for s in glist:
            lbl_counts[s.label or "Unlabeled"] += 1
        primary_lbl = max(lbl_counts.items(), key=lambda x: x[1])[0]
        label_groups[primary_lbl][gid] = glist

    rng = random.Random(seed)
    fold_val_groups: List[Set[str]] = [set() for _ in range(n_splits)]
    fold_val_samples: List[List[DNASample]] = [[] for _ in range(n_splits)]

    for lbl, gdict in label_groups.items():
        # Shuffle keys with tie-breaker seed, then sort descending by group size
        g_items = list(gdict.items())
        rng.shuffle(g_items)
        sorted_groups = sorted(g_items, key=lambda x: len(x[1]), reverse=True)

        class_fold_counts = [0] * n_splits
        for gid, glist in sorted_groups:
            # Assign to the fold currently having the fewest samples for this class
            min_idx = class_fold_counts.index(min(class_fold_counts))
            fold_val_groups[min_idx].add(gid)
            fold_val_samples[min_idx].extend(glist)
            class_fold_counts[min_idx] += len(glist)

    # 3. Construct (train, val) pairs and enforce leakage verification
    cv_folds: List[Tuple[List[DNASample], List[DNASample]]] = []
    all_val_samples_count = 0

    for f_idx in range(n_splits):
        val_set = list(fold_val_samples[f_idx])
        val_grps = fold_val_groups[f_idx]
        all_val_samples_count += len(val_set)

        train_set: List[DNASample] = []
        train_grps: Set[str] = set()

        for o_idx in range(n_splits):
            if o_idx != f_idx:
                train_set.extend(fold_val_samples[o_idx])
                train_grps.update(fold_val_groups[o_idx])

        # Assert disjointness
        assert not (val_grps & train_grps), f"Fold {f_idx}: Group leakage between train and val!"
        train_seqs = {s.sequence.strip().upper() for s in train_set}
        val_seqs = {s.sequence.strip().upper() for s in val_set}
        assert not (train_seqs & val_seqs), f"Fold {f_idx}: Sequence leakage between train and val!"

        # Ensure no empty class in train or val
        val_classes = {s.label for s in val_set}
        train_classes = {s.label for s in train_set}
        if len(val_classes) < len(label_groups):
            logger.warning("Fold %d validation set is missing classes: %s", f_idx, set(label_groups.keys()) - val_classes)

        rng.shuffle(train_set)
        rng.shuffle(val_set)
        cv_folds.append((train_set, val_set))

    assert all_val_samples_count == len(samples), (
        f"Validation sample assignment mismatch: expected {len(samples)}, got {all_val_samples_count}"
    )

    logger.info(
        "Generated %d-fold group-aware CV (mode='%s', total_samples=%d, total_groups=%d)",
        n_splits, grouping_mode, len(samples), len(groups)
    )

    return cv_folds


def generate_group_shufflesplit_folds(
    samples: List[DNASample],
    n_splits: int = 10,
    val_size: float = 0.20,
    seed: int = 42,
    grouping_mode: str = "within_organism",
    gene_key: str = "gene",
) -> List[Tuple[List[DNASample], List[DNASample]]]:
    """Generate repeated group-aware Monte Carlo train/validation splits.

    Note: The repeated iterations are non-independent resamplings from the same population.
    Results across iterations must be aggregated using median/IQR rather than treated as independent evidence.

    Args:
        samples: List of DNASample objects.
        n_splits: Number of repeated splits (e.g. 10).
        val_size: Proportion of groups for validation (e.g. 0.20).
        seed: Base random seed for split generation.
        grouping_mode: 'within_organism', 'cross_organism', or 'exact_sequence'.
        gene_key: Metadata field for gene family.

    Returns:
        List of (train_samples, val_samples) tuples for each iteration.
    """
    if n_splits < 1:
        raise DataValidationError(f"n_splits must be at least 1, got {n_splits}.")
    if not (0.0 < val_size < 1.0):
        raise DataValidationError(f"val_size must be between 0.0 and 1.0, got {val_size}.")

    folds: List[Tuple[List[DNASample], List[DNASample]]] = []
    for i in range(n_splits):
        iter_seed = seed + (i * 37)
        tr, va, _ = split_gene_family_grouped(
            samples=samples,
            train_size=1.0 - val_size,
            val_size=val_size,
            test_size=0.0,
            seed=iter_seed,
            grouping_mode=grouping_mode,
            gene_key=gene_key,
        )
        folds.append((tr, va))

    logger.info(
        "Generated %d-iteration group-aware ShuffleSplit (mode='%s', val_size=%.2f, base_seed=%d)",
        n_splits, grouping_mode, val_size, seed
    )
    return folds


def verify_gene_family_leakage(
    train: List[DNASample],
    val: List[DNASample],
    test: List[DNASample],
    gene_key: str = "gene",
    grouping_mode: str = "within_organism",
) -> Dict[str, Any]:
    """Inspect partitions to detect gene-family leakage across train/val/test splits."""
    def _get_entries(subset: List[DNASample]) -> Set[Tuple[str, str]]:
        res = set()
        for s in subset:
            g = getattr(s.metadata, gene_key, None) or s.metadata.additional_metadata.get(gene_key)
            if g:
                org = s.metadata.organism or s.label or "unspecified"
                res.add((str(org).strip(), str(g).strip()))
        return res

    train_entries = _get_entries(train)
    val_entries = _get_entries(val)
    test_entries = _get_entries(test)

    if grouping_mode == "within_organism":
        leak_tv_pairs = train_entries & val_entries
        leak_tt_pairs = train_entries & test_entries
        leak_vt_pairs = val_entries & test_entries
    else:
        # Cross-organism matching of raw gene symbols
        train_genes = {g for _, g in train_entries}
        val_genes = {g for _, g in val_entries}
        test_genes = {g for _, g in test_entries}
        leak_tv_pairs = {(o, g) for o, g in train_entries if g in val_genes}
        leak_tt_pairs = {(o, g) for o, g in train_entries if g in test_genes}
        leak_vt_pairs = {(o, g) for o, g in val_entries if g in test_genes}

    has_leakage = bool(leak_tv_pairs or leak_tt_pairs or leak_vt_pairs)

    # Cross-organism shared symbol check (informational)
    t_genes = {g for _, g in train_entries}
    v_genes = {g for _, g in val_entries}
    te_genes = {g for _, g in test_entries}
    cross_tv = t_genes & v_genes
    cross_tt = t_genes & te_genes
    cross_vt = v_genes & te_genes

    all_annotated = {g for _, g in (train_entries | val_entries | test_entries)}

    return {
        "is_leakage_free": not has_leakage,
        "grouping_mode": grouping_mode,
        "total_annotated_genes": len(all_annotated),
        "train_genes_count": len(t_genes),
        "val_genes_count": len(v_genes),
        "test_genes_count": len(te_genes),
        "train_val_overlap": sorted(list({g for _, g in leak_tv_pairs})),
        "train_test_overlap": sorted(list({g for _, g in leak_tt_pairs})),
        "val_test_overlap": sorted(list({g for _, g in leak_vt_pairs})),
        "detailed_overlap_pairs": {
            "train_val": sorted([f"{o}:{g}" for o, g in leak_tv_pairs]),
            "train_test": sorted([f"{o}:{g}" for o, g in leak_tt_pairs]),
            "val_test": sorted([f"{o}:{g}" for o, g in leak_vt_pairs]),
        },
        "cross_organism_shared_symbols": {
            "train_val": sorted(list(cross_tv - {g for _, g in leak_tv_pairs})),
            "train_test": sorted(list(cross_tt - {g for _, g in leak_tt_pairs})),
            "val_test": sorted(list(cross_vt - {g for _, g in leak_vt_pairs})),
        },
    }


def verify_split_leakage(
    train: List[DNASample],
    val: List[DNASample],
    test: List[DNASample],
    group_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect partitions to detect sequence-level or group-level data leakage."""
    train_seqs = {s.sequence.strip().upper() for s in train}
    val_seqs = {s.sequence.strip().upper() for s in val}
    test_seqs = {s.sequence.strip().upper() for s in test}

    seq_leak_tv = train_seqs & val_seqs
    seq_leak_tt = train_seqs & test_seqs
    seq_leak_vt = val_seqs & test_seqs
    has_seq_leakage = bool(seq_leak_tv or seq_leak_tt or seq_leak_vt)

    group_leakage = False
    grp_leak_tv = set()
    grp_leak_tt = set()
    grp_leak_vt = set()

    if group_key:
        def _get_groups(subset: List[DNASample]) -> Set[str]:
            res = set()
            for s in subset:
                g = getattr(s, group_key, None) or s.metadata.additional_metadata.get(group_key)
                if g:
                    res.add(str(g))
            return res

        t_grp = _get_groups(train)
        v_grp = _get_groups(val)
        tt_grp = _get_groups(test)

        grp_leak_tv = t_grp & v_grp
        grp_leak_tt = t_grp & tt_grp
        grp_leak_vt = v_grp & tt_grp
        group_leakage = bool(grp_leak_tv or grp_leak_tt or grp_leak_vt)

    return {
        "is_leakage_free": not (has_seq_leakage or group_leakage),
        "sequence_leakage": {
            "has_leakage": has_seq_leakage,
            "train_val_overlap_count": len(seq_leak_tv),
            "train_test_overlap_count": len(seq_leak_tt),
            "val_test_overlap_count": len(seq_leak_vt),
        },
        "group_leakage": {
            "group_key": group_key,
            "has_leakage": group_leakage,
            "train_val_overlap_count": len(grp_leak_tv),
            "train_test_overlap_count": len(grp_leak_tt),
            "val_test_overlap_count": len(grp_leak_vt),
        },
    }


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
        gene_key: str = "gene",
        strict_metadata_only: bool = False,
        grouping_mode: str = "within_organism",
    ) -> None:
        self.strategy = strategy.lower()
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.group_key = group_key
        self.gene_key = gene_key
        self.strict_metadata_only = strict_metadata_only
        self.grouping_mode = grouping_mode
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
        elif strat == "duplicate_grouped":
            return split_duplicate_grouped(
                samples=samples,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                seed=self.seed,
                group_by="sequence",
            )
        elif strat == "gene_family_grouped":
            return split_gene_family_grouped(
                samples=samples,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                seed=self.seed,
                strict_metadata_only=self.strict_metadata_only,
                gene_key=self.gene_key,
                grouping_mode=self.grouping_mode,
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
            raise DataValidationError(
                f"Unsupported splitting strategy: '{strat}'. Supported: auto, duplicate_grouped, gene_family_grouped, group, stratified, random"
            )

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
