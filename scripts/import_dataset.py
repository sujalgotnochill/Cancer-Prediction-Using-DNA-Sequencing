#!/usr/bin/env python
"""Command-line script to ingest, validate, profile, and register a DNA sequence dataset.

Usage Examples:
    python scripts/import_dataset.py --file data/raw/colorectal.fasta --id ds_colorectal --default-label "Colorectal Cancer" --profile
    python scripts/import_dataset.py --file data/raw/samples.csv --id ds_samples --source "NCBI GenBank" --validate
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dna_cancer.config.settings import resolve_path
from dna_cancer.data.ingestion import ColumnMapping, FastaLabelMapper, load_samples
from dna_cancer.data.profiler import DatasetProfiler
from dna_cancer.data.quality import DatasetValidator
from dna_cancer.data.registry import DatasetRegistry
from dna_cancer.utils.exceptions import DNACancerError
from dna_cancer.utils.logger import get_logger

logger = get_logger("import_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest, validate, profile, and register a DNA cancer dataset into the project registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", "-f", required=True, help="Path to input dataset file (CSV, FASTA, JSON, JSONL).")
    parser.add_argument("--id", help="Unique dataset identifier. Defaults to filename stem.")
    parser.add_argument("--name", help="Human-readable dataset display name.")
    parser.add_argument("--format", choices=["csv", "fasta", "json", "jsonl"], help="Explicitly specify file format.")
    parser.add_argument("--source", help="Genomic data source or repository (e.g. NCBI, TCGA).")
    parser.add_argument("--description", help="Description of dataset contents, provenance, and experiment.")
    parser.add_argument("--default-label", help="Default target cancer label assigned if not present in records.")
    parser.add_argument("--label-map", help="JSON string or path to JSON/CSV mapping sequence IDs to labels.")
    parser.add_argument("--column-map", help="JSON string mapping custom column names (e.g. '{\"sequence_col\": \"seq\"}').")
    parser.add_argument("--tags", nargs="*", default=[], help="Categorization tags for dataset.")
    parser.add_argument("--validate", action="store_true", help="Execute quality checks during import.")
    parser.add_argument("--profile", action="store_true", help="Generate statistical profiling reports in artifacts/datasets/.")
    parser.add_argument("--registry-file", help="Custom registry JSON file location.")
    parser.add_argument("--output-dir", default="artifacts/datasets", help="Directory where reports are stored.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_path = resolve_path(args.file)
    if not file_path.exists():
        logger.error("File does not exist: %s", file_path)
        return 1

    ds_id = args.id or file_path.stem
    logger.info("Starting dataset import for ID: '%s' from %s", ds_id, file_path)

    # 1. Parse column mapping if supplied
    col_mapping = None
    if args.column_map:
        try:
            if Path(args.column_map).exists():
                with open(args.column_map, "r", encoding="utf-8") as f:
                    c_dict = json.load(f)
            else:
                c_dict = json.loads(args.column_map)
            col_mapping = ColumnMapping(**c_dict)
        except Exception as exc:
            logger.error("Failed to parse --column-map: %s", exc)
            return 1

    # 2. Parse label mapping if supplied
    label_mapper = None
    if args.label_map:
        try:
            if Path(args.label_map).exists():
                label_mapper = FastaLabelMapper.from_file(args.label_map, default_label=args.default_label)
            else:
                l_dict = json.loads(args.label_map)
                label_mapper = FastaLabelMapper(mapping=l_dict, default_label=args.default_label)
        except Exception as exc:
            logger.error("Failed to parse --label-map: %s", exc)
            return 1
    elif args.default_label:
        label_mapper = FastaLabelMapper(default_label=args.default_label)

    # 3. Ingest samples
    try:
        samples = load_samples(
            file_path=file_path,
            file_format=args.format,
            column_mapping=col_mapping,
            label_mapper=label_mapper,
            default_label=args.default_label,
            dataset_id=ds_id,
        )
    except DNACancerError as exc:
        logger.error("Data ingestion failed: %s", exc)
        return 1

    out_dir = resolve_path(args.output_dir)
    val_report_path = None
    prof_report_path = None

    # 4. Optional Validation
    if args.validate or args.profile:
        validator = DatasetValidator()
        quality_res = validator.validate_dataset(samples, raise_on_error=False)
        json_val, md_val = quality_res.save_reports(out_dir, dataset_id=ds_id)
        val_report_path = str(json_val)
        if not quality_res.is_valid:
            logger.warning("Dataset quality check reported %d errors and %d warnings.", len(quality_res.errors), len(quality_res.warnings))
        else:
            logger.info("Dataset quality check passed with 0 errors.")

    # 5. Optional Profiling
    if args.profile:
        profiler = DatasetProfiler()
        profile = profiler.profile(samples, dataset_id=ds_id)
        json_prof, md_prof = profile.save_reports(out_dir)
        prof_report_path = str(json_prof)
        logger.info("Generated profile report: %s and %s", json_prof, md_prof)

    # 6. Register dataset
    registry = DatasetRegistry(registry_file=args.registry_file)
    meta = registry.register_dataset(
        dataset_id=ds_id,
        file_path=file_path,
        name=args.name,
        file_format=args.format,
        source=args.source,
        description=args.description,
        sample_count=len(samples),
        column_mapping=col_mapping.__dict__ if col_mapping else None,
        label_mapping=label_mapper.mapping if label_mapper else None,
        validation_report_path=val_report_path,
        profile_report_path=prof_report_path,
        tags=args.tags,
    )

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Successfully registered dataset: {meta.dataset_id} (Version {meta.version})")
    print(f"   Name:         {meta.name}")
    print(f"   Samples:      {meta.sample_count}")
    print(f"   Format:       {meta.format}")
    print(f"   SHA-256:      {meta.checksum_sha256[:16]}...")
    print(f"   File Path:    {meta.file_path}")
    if val_report_path:
        print(f"   Validation:   {val_report_path}")
    if prof_report_path:
        print(f"   Profile:      {prof_report_path}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
