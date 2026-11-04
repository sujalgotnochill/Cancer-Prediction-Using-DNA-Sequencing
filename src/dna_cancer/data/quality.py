"""Data validation, quality assurance, duplicate analysis, and conflict detection."""

import json
import math
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from dna_cancer.data.schema import DNASample, compute_reverse_complement
from dna_cancer.data.validation import STANDARD_DNA_BASES, SequenceValidator
from dna_cancer.utils.exceptions import DataValidationError, SequenceFormatError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicatePolicy(str, Enum):
    """Policy for handling duplicate sequences in a dataset."""
    REPORT_ONLY = "report_only"
    KEEP_FIRST = "keep_first"
    REMOVE_ALL = "remove_all"
    ERROR = "error"


class QualityCheckResult:
    """Detailed summary report of dataset quality and validation checks."""

    def __init__(
        self,
        total_samples: int,
        valid_samples: int,
        duplicate_sequences: List[Dict[str, Any]],
        conflicting_labels: List[Dict[str, Any]],
        reverse_complement_matches: List[Dict[str, Any]],
        duplicate_ids: List[str],
        invalid_sequences: List[Dict[str, Any]],
        missing_labels: List[str],
        unexpected_labels: List[Dict[str, Any]],
        length_outliers: List[Dict[str, Any]],
        class_distribution: Dict[str, int],
        metadata_completeness: Dict[str, float],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        self.total_samples = total_samples
        self.valid_samples = valid_samples
        self.duplicate_sequences = duplicate_sequences
        self.conflicting_labels = conflicting_labels
        self.reverse_complement_matches = reverse_complement_matches
        self.duplicate_ids = duplicate_ids
        self.invalid_sequences = invalid_sequences
        self.missing_labels = missing_labels
        self.unexpected_labels = unexpected_labels
        self.length_outliers = length_outliers
        self.class_distribution = class_distribution
        self.metadata_completeness = metadata_completeness
        self.warnings = warnings
        self.errors = errors

    @property
    def is_valid(self) -> bool:
        """Returns True if no critical errors (e.g. invalid sequences, conflicts) occurred."""
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality check results to a dictionary."""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "duplicate_sequence_count": len(self.duplicate_sequences),
            "conflicting_label_count": len(self.conflicting_labels),
            "reverse_complement_match_count": len(self.reverse_complement_matches),
            "duplicate_id_count": len(self.duplicate_ids),
            "invalid_sequence_count": len(self.invalid_sequences),
            "missing_label_count": len(self.missing_labels),
            "unexpected_label_count": len(self.unexpected_labels),
            "length_outlier_count": len(self.length_outliers),
            "class_distribution": self.class_distribution,
            "metadata_completeness": self.metadata_completeness,
            "duplicates": self.duplicate_sequences[:50],  # sample up to 50
            "conflicts": self.conflicting_labels,
            "reverse_complements": self.reverse_complement_matches[:50],
        }

    def to_markdown(self) -> str:
        """Render quality check results as a GitHub Flavored Markdown report."""
        status_badge = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"# Dataset Quality & Validation Report: {status_badge}",
            "",
            f"- **Total Samples Evaluated:** {self.total_samples}",
            f"- **Valid Sequences:** {self.valid_samples} ({self.valid_samples / max(1, self.total_samples) * 100:.1f}%)",
            f"- **Critical Errors:** {len(self.errors)}",
            f"- **Warnings:** {len(self.warnings)}",
            "",
            "## Quality Metrics Summary",
            "",
            "| Metric | Count | Status |",
            "| :--- | :--- | :--- |",
            f"| Duplicate Sequences | {len(self.duplicate_sequences)} | {'Notice' if self.duplicate_sequences else 'Clean'} |",
            f"| Conflicting Labels (Same Sequence) | {len(self.conflicting_labels)} | {'CRITICAL' if self.conflicting_labels else 'Clean'} |",
            f"| Reverse-Complement Pairs | {len(self.reverse_complement_matches)} | {'Notice' if self.reverse_complement_matches else 'Clean'} |",
            f"| Duplicate Sample IDs | {len(self.duplicate_ids)} | {'CRITICAL' if self.duplicate_ids else 'Clean'} |",
            f"| Invalid Nucleotide Sequences | {len(self.invalid_sequences)} | {'CRITICAL' if self.invalid_sequences else 'Clean'} |",
            f"| Missing Labels | {len(self.missing_labels)} | {'Warning' if self.missing_labels else 'Clean'} |",
            f"| Unexpected Labels | {len(self.unexpected_labels)} | {'Warning' if self.unexpected_labels else 'Clean'} |",
            f"| Length Outliers | {len(self.length_outliers)} | {'Notice' if self.length_outliers else 'Clean'} |",
            "",
            "## Class Distribution",
            "",
            "| Class Name | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]
        total_lbl = sum(self.class_distribution.values()) or 1
        for lbl, cnt in sorted(self.class_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / total_lbl) * 100
            lines.append(f"| {lbl} | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "## Metadata Completeness",
            "",
            "| Field | Populated Ratio |",
            "| :--- | :--- |",
        ])
        for field_name, pct in self.metadata_completeness.items():
            lines.append(f"| {field_name} | {pct * 100:.1f}% |")

        if self.conflicting_labels:
            lines.extend([
                "",
                "## Critical: Conflicting Labels for Identical Sequences",
                "> [!CAUTION]",
                "> Identical sequences were assigned conflicting classification labels across samples. This represents potential misannotation or cross-sample contamination.",
                "",
                "| Sample IDs | Sequence Length | Conflicting Labels |",
                "| :--- | :--- | :--- |",
            ])
            for conflict in self.conflicting_labels[:10]:
                ids_str = ", ".join(conflict["sample_ids"][:5])
                lbls_str = ", ".join(conflict["labels"])
                lines.append(f"| {ids_str} | {conflict['sequence_length']} | {lbls_str} |")

        if self.errors:
            lines.extend([
                "",
                "## Error Log",
                "",
            ])
            for err in self.errors[:20]:
                lines.append(f"- ❌ {err}")

        if self.warnings:
            lines.extend([
                "",
                "## Warning Log",
                "",
            ])
            for warn in self.warnings[:20]:
                lines.append(f"- ⚠️ {warn}")

        return "\n".join(lines)

    def save_reports(self, output_dir: Union[str, Path], dataset_id: str) -> Tuple[Path, Path]:
        """Save quality report as both JSON and readable Markdown."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{dataset_id}_quality.json"
        md_path = out_dir / f"{dataset_id}_quality.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        md_path.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Saved dataset quality reports to %s and %s", json_path, md_path)
        return json_path, md_path


class DatasetValidator:
    """Comprehensive validator and quality checker for DNA sequence datasets."""

    def __init__(
        self,
        validator: Optional[SequenceValidator] = None,
        expected_classes: Optional[Iterable[str]] = None,
        min_length: int = 10,
        max_length: int = 1000000,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.REPORT_ONLY,
    ) -> None:
        self.validator = validator or SequenceValidator(min_length=min_length, max_length=max_length)
        self.expected_classes = set(expected_classes) if expected_classes else None
        self.min_length = min_length
        self.max_length = max_length
        self.duplicate_policy = duplicate_policy

    def validate_dataset(
        self,
        samples: List[DNASample],
        raise_on_error: bool = False,
    ) -> QualityCheckResult:
        """Run all quality and consistency checks on a list of DNASample objects.

        Args:
            samples: List of DNASample instances.
            raise_on_error: If True, raises DataValidationError on critical failure.

        Returns:
            QualityCheckResult containing detailed metrics and logs.
        """
        total = len(samples)
        errors: List[str] = []
        warnings: List[str] = []

        if total == 0:
            msg = "Dataset contains 0 samples."
            errors.append(msg)
            if raise_on_error:
                raise DataValidationError(msg)
            return QualityCheckResult(
                total_samples=0,
                valid_samples=0,
                duplicate_sequences=[],
                conflicting_labels=[],
                reverse_complement_matches=[],
                duplicate_ids=[],
                invalid_sequences=[],
                missing_labels=[],
                unexpected_labels=[],
                length_outliers=[],
                class_distribution={},
                metadata_completeness={},
                warnings=warnings,
                errors=errors,
            )

        # Tracking structures
        id_counts: Counter = Counter()
        seq_to_samples: Dict[str, List[DNASample]] = defaultdict(list)
        rc_to_seq: Dict[str, str] = {}
        invalid_sequences: List[Dict[str, Any]] = []
        missing_labels: List[str] = []
        unexpected_labels: List[Dict[str, Any]] = []
        length_outliers: List[Dict[str, Any]] = []
        class_dist: Counter = Counter()

        metadata_fields = ["patient_id", "organism", "gene", "locus", "chromosome", "source", "sequencing_platform"]
        meta_present_counts: Counter = Counter()

        valid_count = 0

        for idx, sample in enumerate(samples):
            s_id = sample.sample_id
            seq = sample.sequence
            lbl = sample.label

            # 1. ID Check
            if not s_id:
                errors.append(f"Sample at index {idx} has an empty sample_id.")
            else:
                id_counts[s_id] += 1

            # 2. Sequence Check
            is_valid_seq = True
            if not seq:
                invalid_sequences.append({"index": idx, "id": s_id, "reason": "Empty sequence"})
                is_valid_seq = False
            elif not self.validator.is_valid(seq):
                invalid_sequences.append({"index": idx, "id": s_id, "length": len(seq), "reason": "Invalid characters or length"})
                is_valid_seq = False

            if is_valid_seq:
                valid_count += 1
                seq_upper = seq.upper()
                seq_to_samples[seq_upper].append(sample)
                rc_to_seq[compute_reverse_complement(seq_upper)] = seq_upper

                if len(seq) < self.min_length or len(seq) > self.max_length:
                    length_outliers.append({"id": s_id, "length": len(seq)})

            # 3. Label Check
            if not lbl:
                missing_labels.append(s_id)
            else:
                class_dist[lbl] += 1
                if self.expected_classes and lbl not in self.expected_classes:
                    unexpected_labels.append({"id": s_id, "label": lbl})

            # 4. Metadata Completeness
            for mf in metadata_fields:
                if getattr(sample, mf, None) or sample.metadata.additional_metadata.get(mf):
                    meta_present_counts[mf] += 1

        # Check duplicate IDs
        duplicate_ids = [s_id for s_id, count in id_counts.items() if count > 1]
        if duplicate_ids:
            errors.append(f"Found {len(duplicate_ids)} duplicate sample IDs: {duplicate_ids[:10]}")

        # Check duplicate sequences and conflicts
        duplicate_sequences: List[Dict[str, Any]] = []
        conflicting_labels: List[Dict[str, Any]] = []

        for seq_str, sample_list in seq_to_samples.items():
            if len(sample_list) > 1:
                labels_found = set(s.label for s in sample_list if s.label)
                dup_entry = {
                    "sequence_length": len(seq_str),
                    "count": len(sample_list),
                    "sample_ids": [s.sample_id for s in sample_list],
                    "labels": list(labels_found),
                }
                duplicate_sequences.append(dup_entry)

                # Conflicting labels for identical sequence
                if len(labels_found) > 1:
                    conflicting_labels.append(dup_entry)
                    errors.append(
                        f"Conflicting labels for identical sequence ({len(seq_str)} bp): "
                        f"labels={list(labels_found)}, sample_ids={[s.sample_id for s in sample_list][:5]}"
                    )

        if duplicate_sequences:
            msg = f"Found {len(duplicate_sequences)} duplicate sequence groups ({sum(d['count'] for d in duplicate_sequences)} total samples)."
            if self.duplicate_policy == DuplicatePolicy.ERROR:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Check reverse-complement matches
        reverse_complement_matches: List[Dict[str, Any]] = []
        checked_pairs: Set[Tuple[str, str]] = set()

        for seq_upper in seq_to_samples.keys():
            if seq_upper in rc_to_seq and rc_to_seq[seq_upper] != seq_upper:
                other_seq = rc_to_seq[seq_upper]
                pair_key = tuple(sorted([seq_upper, other_seq]))
                if pair_key not in checked_pairs:
                    checked_pairs.add(pair_key)
                    rc_matches = {
                        "sequence_length": len(seq_upper),
                        "forward_sample_ids": [s.sample_id for s in seq_to_samples[seq_upper]],
                        "reverse_complement_sample_ids": [s.sample_id for s in seq_to_samples[other_seq]],
                    }
                    reverse_complement_matches.append(rc_matches)

        if reverse_complement_matches:
            warnings.append(
                f"Found {len(reverse_complement_matches)} reverse-complement match pairs. "
                "Note: Reverse complements are not treated as duplicates by default, as biological "
                "strandedness dictates whether opposite strands carry distinct biological significance."
            )

        if invalid_sequences:
            errors.append(f"Found {len(invalid_sequences)} invalid nucleotide sequences.")

        if missing_labels:
            warnings.append(f"Found {len(missing_labels)} samples with missing target labels.")

        if unexpected_labels:
            warnings.append(
                f"Found {len(unexpected_labels)} samples with unexpected labels not in expected_classes."
            )

        # Check class imbalance
        if class_dist and len(class_dist) > 1:
            min_c = min(class_dist.values())
            max_c = max(class_dist.values())
            imbalance_ratio = min_c / max(1, max_c)
            if imbalance_ratio < 0.2:
                warnings.append(
                    f"Severe class imbalance detected: min_class={min_c}, max_class={max_c} (ratio={imbalance_ratio:.2f})."
                )

        completeness = {mf: meta_present_counts[mf] / total for mf in metadata_fields}

        result = QualityCheckResult(
            total_samples=total,
            valid_samples=valid_count,
            duplicate_sequences=duplicate_sequences,
            conflicting_labels=conflicting_labels,
            reverse_complement_matches=reverse_complement_matches,
            duplicate_ids=duplicate_ids,
            invalid_sequences=invalid_sequences,
            missing_labels=missing_labels,
            unexpected_labels=unexpected_labels,
            length_outliers=length_outliers,
            class_distribution=dict(class_dist),
            metadata_completeness=completeness,
            warnings=warnings,
            errors=errors,
        )

        if raise_on_error and not result.is_valid:
            raise DataValidationError(f"Dataset validation failed with {len(errors)} errors: {errors[:3]}")

        return result

    def apply_duplicate_policy(
        self,
        samples: List[DNASample],
    ) -> List[DNASample]:
        """Filter dataset according to configured duplicate handling policy."""
        if self.duplicate_policy in (DuplicatePolicy.REPORT_ONLY, DuplicatePolicy.ERROR):
            return list(samples)

        seen_sequences: Set[str] = set()
        all_sequences = Counter(s.sequence.upper() for s in samples)

        filtered: List[DNASample] = []
        for s in samples:
            seq_up = s.sequence.upper()
            if self.duplicate_policy == DuplicatePolicy.KEEP_FIRST:
                if seq_up not in seen_sequences:
                    seen_sequences.add(seq_up)
                    filtered.append(s)
            elif self.duplicate_policy == DuplicatePolicy.REMOVE_ALL:
                if all_sequences[seq_up] == 1:
                    filtered.append(s)

        logger.info(
            "Applied duplicate policy '%s': reduced from %d to %d samples",
            self.duplicate_policy.value, len(samples), len(filtered)
        )
        return filtered

