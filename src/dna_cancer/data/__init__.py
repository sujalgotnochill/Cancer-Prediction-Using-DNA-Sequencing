"""Data ingestion, dataset abstractions, quality validation, profiling, and registry management."""

from dna_cancer.data.dataset import DNADataset
from dna_cancer.data.fasta_inspector import (
    FastaFileSummary,
    format_bytes,
    inspect_collection,
    inspect_fasta_file,
)
from dna_cancer.data.ingestion import (
    ColumnMapping,
    FastaLabelMapper,
    SimpleDataFrame,
    load_csv,
    load_fasta_as_dataframe,
    load_json,
    load_jsonl,
    load_samples,
    parse_fasta_records,
)
from dna_cancer.data.profiler import DatasetProfile, DatasetProfiler
from dna_cancer.data.quality import DatasetValidator, DuplicatePolicy, QualityCheckResult
from dna_cancer.data.registry import DatasetMetadata, DatasetRegistry, calculate_file_checksum
from dna_cancer.data.schema import CancerSequenceSample, DNASample, SampleMetadata, compute_reverse_complement
from dna_cancer.data.splitter import (
    DatasetSplitter,
    generate_group_shufflesplit_folds,
    recommend_splitting_strategy,
    split_duplicate_grouped,
    split_gene_family_cv_folds,
    split_gene_family_grouped,
    split_grouped,
    split_random,
    split_stratified,
    verify_gene_family_leakage,
    verify_split_leakage,
)
from dna_cancer.data.validation import (
    SequenceValidator,
    validate_dataset_records,
    validate_dataset_schema,
    validate_dna_sequence,
)
from dna_cancer.data.vault import SealedVault

__all__ = [
    # Schema
    "DNASample",
    "CancerSequenceSample",
    "SampleMetadata",
    "compute_reverse_complement",
    # Ingestion
    "ColumnMapping",
    "FastaLabelMapper",
    "SimpleDataFrame",
    "load_csv",
    "load_fasta_as_dataframe",
    "load_json",
    "load_jsonl",
    "load_samples",
    "parse_fasta_records",
    # Dataset container
    "DNADataset",
    # Validation & Quality
    "SequenceValidator",
    "DatasetValidator",
    "DuplicatePolicy",
    "QualityCheckResult",
    "validate_dna_sequence",
    "validate_dataset_schema",
    "validate_dataset_records",
    # Profiler
    "DatasetProfile",
    "DatasetProfiler",
    # Registry
    "DatasetMetadata",
    "DatasetRegistry",
    "calculate_file_checksum",
    # Splitting
    "DatasetSplitter",
    "recommend_splitting_strategy",
    "split_random",
    "split_stratified",
    "split_grouped",
    "split_duplicate_grouped",
    "split_gene_family_grouped",
    "split_gene_family_cv_folds",
    "generate_group_shufflesplit_folds",
    "verify_split_leakage",
    "verify_gene_family_leakage",
    # Inspector
    "FastaFileSummary",
    "inspect_fasta_file",
    "inspect_collection",
    "format_bytes",
    # Vault
    "SealedVault",
]
