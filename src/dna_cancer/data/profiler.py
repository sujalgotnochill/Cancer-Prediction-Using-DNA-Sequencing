"""Dataset profiling and statistical reporting module."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dna_cancer.data.quality import DatasetValidator, QualityCheckResult
from dna_cancer.data.schema import DNASample
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def _calculate_length_statistics(lengths: List[int]) -> Dict[str, float]:
    """Compute summary statistics for sequence lengths without requiring external numpy."""
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0}

    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    min_val = float(sorted_lens[0])
    max_val = float(sorted_lens[-1])
    mean_val = float(sum(sorted_lens) / n)

    # Median
    mid = n // 2
    if n % 2 == 1:
        median_val = float(sorted_lens[mid])
    else:
        median_val = float((sorted_lens[mid - 1] + sorted_lens[mid]) / 2.0)

    # Variance and Standard Deviation
    variance = sum((x - mean_val) ** 2 for x in sorted_lens) / max(1, n - 1)
    std_val = math.sqrt(variance)

    # Quartiles
    q25_idx = int(0.25 * (n - 1))
    q75_idx = int(0.75 * (n - 1))
    q25_val = float(sorted_lens[q25_idx])
    q75_val = float(sorted_lens[q75_idx])

    return {
        "min": min_val,
        "max": max_val,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "q25": round(q25_val, 2),
        "q75": round(q75_val, 2),
    }


def _calculate_nucleotide_frequencies(sequences: List[str]) -> Dict[str, Any]:
    """Compute base frequencies and GC-content across sequences."""
    counts: Counter = Counter()
    total_bases = 0

    for seq in sequences:
        upper = seq.upper()
        counts.update(upper)
        total_bases += len(upper)

    if total_bases == 0:
        return {"total_bases": 0, "gc_content": 0.0, "base_counts": {}, "base_percentages": {}}

    a = counts.get("A", 0)
    c = counts.get("C", 0)
    g = counts.get("G", 0)
    t = counts.get("T", 0)
    n = counts.get("N", 0)

    gc_content = ((g + c) / total_bases) * 100.0

    return {
        "total_bases": total_bases,
        "gc_content": round(gc_content, 2),
        "base_counts": {"A": a, "C": c, "G": g, "T": t, "N": n},
        "base_percentages": {
            "A": round((a / total_bases) * 100, 2),
            "C": round((c / total_bases) * 100, 2),
            "G": round((g / total_bases) * 100, 2),
            "T": round((t / total_bases) * 100, 2),
            "N": round((n / total_bases) * 100, 2),
        },
    }


class DatasetProfile:
    """Comprehensive statistical profile of a DNA sequence dataset."""

    def __init__(
        self,
        dataset_id: str,
        total_samples: int,
        unique_sequences: int,
        length_stats: Dict[str, float],
        nucleotide_stats: Dict[str, Any],
        quality_result: QualityCheckResult,
    ) -> None:
        self.dataset_id = dataset_id
        self.total_samples = total_samples
        self.unique_sequences = unique_sequences
        self.length_stats = length_stats
        self.nucleotide_stats = nucleotide_stats
        self.quality_result = quality_result

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "total_samples": self.total_samples,
            "unique_sequences": self.unique_sequences,
            "redundancy_ratio": round(1.0 - (self.unique_sequences / max(1, self.total_samples)), 4),
            "length_statistics": self.length_stats,
            "nucleotide_statistics": self.nucleotide_stats,
            "class_distribution": self.quality_result.class_distribution,
            "quality_summary": {
                "valid_samples": self.quality_result.valid_samples,
                "is_valid": self.quality_result.is_valid,
                "duplicate_sequence_count": len(self.quality_result.duplicate_sequences),
                "conflicting_label_count": len(self.quality_result.conflicting_labels),
                "reverse_complement_count": len(self.quality_result.reverse_complement_matches),
                "duplicate_id_count": len(self.quality_result.duplicate_ids),
                "invalid_sequence_count": len(self.quality_result.invalid_sequences),
                "missing_label_count": len(self.quality_result.missing_labels),
                "unexpected_label_count": len(self.quality_result.unexpected_labels),
            },
            "metadata_completeness": self.quality_result.metadata_completeness,
            "conflicts": self.quality_result.conflicting_labels,
        }

    def to_markdown(self) -> str:
        """Format profile as a rich GitHub Flavored Markdown report."""
        q = self.quality_result
        lines = [
            f"# Dataset Statistical Profile: `{self.dataset_id}`",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Total Samples:** {self.total_samples}",
            f"- **Unique DNA Sequences:** {self.unique_sequences}",
            f"- **Sequence Redundancy:** {(1.0 - (self.unique_sequences / max(1, self.total_samples))) * 100:.2f}%",
            f"- **Total Base Pairs:** {self.nucleotide_stats.get('total_bases', 0):,}",
            f"- **Overall GC-Content:** {self.nucleotide_stats.get('gc_content', 0.0):.2f}%",
            f"- **Data Quality Status:** {'✅ PASSED' if q.is_valid else '❌ ATTENTION NEEDED'}",
            "",
            "## 2. Sequence Length Statistics",
            "",
            "| Statistic | Value (bp) |",
            "| :--- | :--- |",
            f"| Minimum Length | {int(self.length_stats.get('min', 0))} |",
            f"| Maximum Length | {int(self.length_stats.get('max', 0))} |",
            f"| Mean Length | {self.length_stats.get('mean', 0.0):.2f} |",
            f"| Median Length | {self.length_stats.get('median', 0.0):.2f} |",
            f"| Standard Deviation | {self.length_stats.get('std', 0.0):.2f} |",
            f"| 25th Percentile (Q1) | {self.length_stats.get('q25', 0.0):.2f} |",
            f"| 75th Percentile (Q3) | {self.length_stats.get('q75', 0.0):.2f} |",
            "",
            "## 3. Nucleotide Composition",
            "",
            "| Base | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]

        b_counts = self.nucleotide_stats.get("base_counts", {})
        b_pcts = self.nucleotide_stats.get("base_percentages", {})
        for base in ("A", "C", "G", "T", "N"):
            lines.append(f"| {base} | {b_counts.get(base, 0):,} | {b_pcts.get(base, 0.0):.2f}% |")

        lines.extend([
            "",
            "## 4. Class Distribution & Imbalance",
            "",
            "| Cancer Class | Sample Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        total_lbl = sum(q.class_distribution.values()) or 1
        for lbl, cnt in sorted(q.class_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / total_lbl) * 100
            lines.append(f"| {lbl} | {cnt} | {pct:.2f}% |")

        lines.extend([
            "",
            "## 5. Metadata Completeness",
            "",
            "| Metadata Attribute | Populated Percentage |",
            "| :--- | :--- |",
        ])
        for field_name, ratio in q.metadata_completeness.items():
            lines.append(f"| `{field_name}` | {ratio * 100:.1f}% |")

        lines.extend([
            "",
            "## 6. Quality & Anomaly Assessment",
            "",
            f"- **Duplicate Sequence Groups:** {len(q.duplicate_sequences)}",
            f"- **Conflicting Label Groups:** {len(q.conflicting_labels)}",
            f"- **Reverse-Complement Matches:** {len(q.reverse_complement_matches)}",
            f"- **Duplicate Sample IDs:** {len(q.duplicate_ids)}",
            f"- **Invalid Sequences:** {len(q.invalid_sequences)}",
            f"- **Missing Target Labels:** {len(q.missing_labels)}",
            f"- **Unexpected Target Labels:** {len(q.unexpected_labels)}",
        ])

        if q.conflicting_labels:
            lines.extend([
                "",
                "### Conflicting Sequence Annotations",
                "> [!CAUTION]",
                "> Identical nucleotide sequences were found annotated with different cancer classes.",
                "",
                "| Sample IDs | Sequence Length | Conflicting Labels |",
                "| :--- | :--- | :--- |",
            ])
            for conflict in q.conflicting_labels[:10]:
                ids = ", ".join(conflict["sample_ids"][:5])
                lbls = ", ".join(conflict["labels"])
                lines.append(f"| {ids} | {conflict['sequence_length']} bp | {lbls} |")

        return "\n".join(lines)

    def save_reports(self, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """Save profile to JSON and Markdown under output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_file = out / f"{self.dataset_id}_profile.json"
        md_file = out / f"{self.dataset_id}_profile.md"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        md_file.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Saved dataset profile reports to %s and %s", json_file, md_file)
        return json_file, md_file


class DatasetProfiler:
    """Orchestrates comprehensive profiling of DNA sequence datasets."""

    def __init__(self, validator: Optional[DatasetValidator] = None) -> None:
        self.validator = validator or DatasetValidator()

    def profile(
        self,
        samples: List[DNASample],
        dataset_id: str = "dataset",
    ) -> DatasetProfile:
        """Generate statistical and quality profile for a list of DNASample objects."""
        quality_res = self.validator.validate_dataset(samples, raise_on_error=False)

        valid_sequences = [s.sequence for s in samples if s.sequence and self.validator.validator.is_valid(s.sequence)]
        unique_seqs = len(set(s.upper() for s in valid_sequences))

        lengths = [len(s) for s in valid_sequences]
        length_stats = _calculate_length_statistics(lengths)
        nuc_stats = _calculate_nucleotide_frequencies(valid_sequences)

        profile = DatasetProfile(
            dataset_id=dataset_id,
            total_samples=len(samples),
            unique_sequences=unique_seqs,
            length_stats=length_stats,
            nucleotide_stats=nuc_stats,
            quality_result=quality_res,
        )
        return profile

