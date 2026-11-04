#!/usr/bin/env python
"""Command-line script to compute length, GC-content, and distribution profiles for a DNA dataset.

Usage Examples:
    python scripts/profile_dataset.py --file data/raw/samples.csv
    python scripts/profile_dataset.py --id ds_colorectal --output-dir artifacts/datasets/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dna_cancer.config.settings import resolve_path
from dna_cancer.data.ingestion import load_samples
from dna_cancer.data.profiler import DatasetProfiler
from dna_cancer.data.registry import DatasetRegistry
from dna_cancer.utils.logger import get_logger

logger = get_logger("profile_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistical, length, GC-content, and class distribution profiles for a DNA sequence dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Direct path to sequence dataset file.")
    group.add_argument("--id", help="Registered dataset ID in the registry.")

    parser.add_argument("--output-dir", default="artifacts/datasets", help="Directory where profile reports are saved.")
    parser.add_argument("--registry-file", help="Path to custom registry JSON file.")
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

    logger.info("Profiling dataset '%s' (%d samples)...", ds_id, len(samples))
    profiler = DatasetProfiler()
    profile = profiler.profile(samples, dataset_id=ds_id)
    json_path, md_path = profile.save_reports(out_dir)

    print("\n" + "=" * 60)
    print(f"Dataset Profile for: {ds_id}")
    print(f"   Total Sequences:     {profile.total_samples}")
    print(f"   Unique Sequences:    {profile.unique_sequences}")
    print(f"   Sequence Length:     min={profile.length_stats['min']:.0f}, max={profile.length_stats['max']:.0f}, mean={profile.length_stats['mean']:.1f}, median={profile.length_stats['median']:.1f}")
    print(f"   GC-Content:          {profile.nucleotide_stats['gc_content']:.2f}%")
    print(f"   Classes Found:       {len(profile.quality_result.class_distribution)}")
    for lbl, cnt in profile.quality_result.class_distribution.items():
        print(f"     - {lbl}: {cnt}")
    print(f"   JSON Profile:        {json_path}")
    print(f"   Markdown Report:     {md_path}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

