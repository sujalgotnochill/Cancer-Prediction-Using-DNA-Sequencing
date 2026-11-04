#!/usr/bin/env python
"""Command-line script to list all registered DNA datasets and their version history.

Usage Examples:
    python scripts/list_datasets.py
    python scripts/list_datasets.py --tag "colorectal" --all-versions
    python scripts/list_datasets.py --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dna_cancer.data.registry import DatasetRegistry
from dna_cancer.utils.logger import get_logger

logger = get_logger("list_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List datasets registered in the project registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry-file", help="Path to custom registry JSON file.")
    parser.add_argument("--tag", help="Filter datasets by specific tag.")
    parser.add_argument("--all-versions", action="store_true", help="Include all historical versions of each dataset.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = DatasetRegistry(registry_file=args.registry_file)
    datasets = registry.list_datasets(tag=args.tag, all_versions=args.all_versions)

    if args.json:
        data = [d.to_dict() for d in datasets]
        print(json.dumps(data, indent=2))
        return 0

    print("\n" + "=" * 90)
    print(f"Registered Datasets ({len(datasets)} found in {registry.registry_file})")
    print("=" * 90)
    print(f"{'Dataset ID':<20} {'Ver':<5} {'Format':<8} {'Samples':<10} {'Created':<22} {'Checksum (SHA256)':<18}")
    print("-" * 90)

    for d in datasets:
        created_short = d.created_at[:19].replace("T", " ") if d.created_at else "Unknown"
        cs_short = d.checksum_sha256[:16] + ".." if d.checksum_sha256 else "-"
        print(f"{d.dataset_id:<20} {d.version:<5} {d.format:<8} {d.sample_count:<10} {created_short:<22} {cs_short:<18}")

    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

