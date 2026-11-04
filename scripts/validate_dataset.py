#!/usr/bin/env python
"""Command-line script to run quality validation and duplicate detection on a DNA dataset.

Usage Examples:
    python scripts/validate_dataset.py --file data/raw/samples.csv --expected-classes "Colorectal Cancer" "Liver Cancer"
    python scripts/validate_dataset.py --id ds_samples --duplicate-policy report_only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dna_cancer.config.settings import resolve_path
from dna_cancer.data.ingestion import load_samples
from dna_cancer.data.quality import DatasetValidator, DuplicatePolicy
from dna_cancer.data.registry import DatasetRegistry
from dna_cancer.utils.logger import get_logger

logger = get_logger("validate_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comprehensive quality, sequence validity, and conflict checks on a DNA sequence dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Direct path to sequence dataset file.")
    group.add_argument("--id", help="Registered dataset ID in the registry.")

    parser.add_argument("--min-length", type=int, default=10, help="Minimum permissible nucleotide length.")
    parser.add_argument("--max-length", type=int, default=1000000, help="Maximum permissible nucleotide length.")
    parser.add_argument("--expected-classes", nargs="*", help="List of allowed cancer target class labels.")
    parser.add_argument(
        "--duplicate-policy",
        choices=["report_only", "keep_first", "remove_all", "error"],
        default="report_only",
        help="Action to take when duplicate sequences are detected.",
    )
    parser.add_argument("--output-dir", default="artifacts/datasets", help="Directory where validation reports are saved.")
    parser.add_argument("--registry-file", help="Path to custom registry JSON file.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any validation errors are detected.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)

    if args.id:
        registry = DatasetRegistry(registry_file=args.registry_file)
        if not registry.has_dataset(args.id):
            logger.error("Dataset '%s' not found in registry.", args.id)
            return 1
        samples = registry.load_dataset_samples(args.id)
        ds_id = args.id
    else:
        file_path = resolve_path(args.file)
        if not file_path.exists():
            logger.error("File does not exist: %s", file_path)
            return 1
        samples = load_samples(file_path)
        ds_id = file_path.stem

    logger.info("Validating dataset '%s' with %d samples...", ds_id, len(samples))
    validator = DatasetValidator(
        min_length=args.min_length,
        max_length=args.max_length,
        expected_classes=args.expected_classes,
        duplicate_policy=DuplicatePolicy(args.duplicate_policy),
    )

    quality_res = validator.validate_dataset(samples, raise_on_error=False)
    json_path, md_path = quality_res.save_reports(out_dir, dataset_id=ds_id)

    print("\n" + "=" * 60)
    print(f"Validation Result for: {ds_id}")
    print(f"   Status:                 {'[PASSED]' if quality_res.is_valid else '[FAILED]'}")
    print(f"   Total Evaluated:        {quality_res.total_samples}")
    print(f"   Valid Samples:          {quality_res.valid_samples}")
    print(f"   Duplicate Sequences:    {len(quality_res.duplicate_sequences)}")
    print(f"   Conflicting Labels:     {len(quality_res.conflicting_labels)}")
    print(f"   Reverse-Complement:     {len(quality_res.reverse_complement_matches)}")
    print(f"   Duplicate Sample IDs:   {len(quality_res.duplicate_ids)}")
    print(f"   Missing Labels:         {len(quality_res.missing_labels)}")
    print(f"   Errors:                 {len(quality_res.errors)}")
    print(f"   Warnings:               {len(quality_res.warnings)}")
    print(f"   Detailed Report:        {md_path}")
    print("=" * 60 + "\n")

    if args.strict and not quality_res.is_valid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
