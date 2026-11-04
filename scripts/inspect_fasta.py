#!/usr/bin/env python
"""Command-line utility to stream-inspect a directory of FASTA sequence files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dna_cancer.config.settings import resolve_path
from dna_cancer.data.fasta_inspector import inspect_collection
from dna_cancer.utils.logger import get_logger

logger = get_logger("inspect_fasta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-inspect FASTA genomic files, compute base distributions, and analyze cross-file overlaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir", "-d",
        default="data/raw/FASTA",
        help="Directory containing FASTA files.",
    )
    parser.add_argument(
        "--output-json", "-o",
        default="artifacts/datasets/fasta_inspection.json",
        help="Path where structured inspection JSON will be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fasta_dir = resolve_path(args.dir)
    out_json = resolve_path(args.output_json)

    if not fasta_dir.exists():
        logger.error("FASTA directory does not exist: %s", fasta_dir)
        return 1

    print("\n" + "=" * 80)
    print(f"Starting Streaming Forensic Inspection of: {fasta_dir}")
    print("=" * 80)

    results = inspect_collection(fasta_dir, output_json_path=out_json)

    print("\n" + "=" * 105)
    print(f"{'Filename':<26} {'Size':<10} {'Records':<10} {'Total Bases':<15} {'Avg Len':<10} {'GC-Content':<12} {'Duplicates':<12}")
    print("-" * 105)

    for name, f_info in sorted(results["files"].items()):
        gc = f_info["alphabet_percentages"].get("G", 0.0) + f_info["alphabet_percentages"].get("C", 0.0)
        dups = f_info["internal_duplicates"]
        print(
            f"{name:<26} {f_info['size_human']:<10} {f_info['total_records']:<10,}"
            f" {f_info['total_nucleotides']:<15,} {f_info['avg_length']:<10.1f} {gc:<12.2f}% {dups:<12,}"
        )

    print("=" * 105)
    print(f"Total Collection Records:      {results['total_collection_records']:,}")
    print(f"Total Collection Nucleotides:  {results['total_collection_nucleotides']:,}")
    print(f"Inspection Report JSON Saved:  {out_json}")
    print("=" * 105 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

