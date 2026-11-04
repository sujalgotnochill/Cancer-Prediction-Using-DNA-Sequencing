# Dataset Engineering & Management Guide

This guide describes how to import, validate, profile, register, and partition DNA sequence datasets within the **DNA Cancer Classification** system.

---

## 1. Overview & Architecture

The dataset management system provides a reproducible, leak-proof bioinformatics data engineering layer. It guarantees that:
1. Heterogeneous sequence formats (CSV, FASTA, JSON, JSONL) are converted into a unified schema (`DNASample`).
2. Phenotypic/disease labels are never fabricated or inferred without explicit user mapping.
3. Every dataset registered into `artifacts/datasets/registry.json` is version-tracked and verified via SHA-256 checksums.
4. Data partitioning eliminates patient leakage (`patient_id`), ensuring multi-sample donors do not leak between training and evaluation splits.

---

## 2. Multi-Format Ingestion & Column Mapping

The universal sequence loader [`load_samples()`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/ingestion.py#L341) auto-detects formats or accepts an explicit format specification.

### 2.1 Tabular CSV Ingestion

By default, CSV files are expected to have `sequence` (or `seq`) and `cancer` (or `label`) columns. For datasets with custom schemas, pass a `ColumnMapping`:

```python
from dna_cancer.data.ingestion import ColumnMapping, load_samples

mapping = ColumnMapping(
    sequence_col="RAW_NUCLEOTIDES",
    label_col="TUMOR_TYPE",
    id_col="SAMPLE_BARCODE",
    patient_id_col="PATIENT_ID",
    gene_col="TARGET_GENE",
)

samples = load_samples("data/raw/custom_tcga_export.csv", column_mapping=mapping)
```

### 2.2 FASTA Ingestion with Explicit Label Mapping

FASTA headers (`>seq_id description`) do not inherently contain cancer diagnosis labels. The system prohibits guessing labels from sequence headers. Instead, `FastaLabelMapper` is used:

```python
from dna_cancer.data.ingestion import FastaLabelMapper, load_samples

# Option A: In-memory dictionary
mapper = FastaLabelMapper(
    mapping={"seq1": "Colorectal Cancer", "seq2": "Liver Cancer"},
    default_label="Unannotated",
)

# Option B: External CSV mapping file (id,label)
mapper = FastaLabelMapper.from_file("configs/fasta_labels.csv")

# Option C: External JSON mapping file {"seq1": "Lung Adenocarcinoma"}
mapper = FastaLabelMapper.from_file("configs/fasta_labels.json")

samples = load_samples("data/raw/sequences.fasta", label_mapper=mapper)
```

### 2.3 JSON and JSONL Ingestion

The loader supports standard JSON arrays, dictionaries with a `"records"` key, and line-delimited JSON (`.jsonl`):

```python
samples = load_samples("data/raw/sequencing_records.jsonl")
```

---

## 3. Dataset Registry & Version Management

The [`DatasetRegistry`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/registry.py#L71) stores metadata in a transparent, persistent JSON file at `artifacts/datasets/registry.json`.

### 3.1 Registry Operations

```python
from dna_cancer.data.registry import DatasetRegistry

registry = DatasetRegistry()

# 1. Register a new dataset
metadata = registry.register_dataset(
    dataset_id="tcga_colorectal_v1",
    file_path="data/raw/colorectal_samples.csv",
    name="TCGA-COAD Sequence Cohort",
    source="TCGA Genomic Data Commons",
    description="Exonic sequence reads from primary colorectal adenocarcinomas",
    tags=["colorectal", "tcga", "exome"],
)

# 2. Retrieve metadata
entry = registry.get_dataset("tcga_colorectal_v1")
print(f"Dataset: {entry.name}, Samples: {entry.sample_count}, SHA-256: {entry.checksum_sha256}")

# 3. Load samples directly through the registry
samples = registry.load_dataset_samples("tcga_colorectal_v1")

# 4. List registered datasets
all_datasets = registry.list_datasets(tag="colorectal")
```

### 3.2 Semantic Versioning & Checksums
- When a file registered under an existing `dataset_id` is re-registered with modified content, the SHA-256 checksum changes and the version number increments automatically (`v1 -> v2`).
- If the file is identical (checksum matches), the existing entry is updated without incrementing the version.

---

## 4. Quality Assurance, Duplicates & Conflicts

The [`DatasetValidator`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/quality.py#L186) performs 10 essential bioinformatic quality checks:

1. **Character Validation:** Verifies allowed IUPAC nucleotide characters (`ACGTN`).
2. **Length Boundaries:** Flags sequences below `min_length` (default: 10 bp) or above `max_length`.
3. **Ambiguous Nucleotide Ratio:** Profiles the proportion of ambiguous base calls (`N`).
4. **Target Label Audit:** Identifies missing labels (`None`, empty, or `"nan"`) and unexpected classes.
5. **Exact Duplicate Sequences:** Identifies identical sequences present multiple times.
6. **Conflicting Label Annotations:** Flags critical data errors where identical nucleotide sequences are annotated with differing cancer classes.
7. **Reverse-Complement Matches:** Detects when two sequences are reverse complements of each other (distinguishing forward vs reverse strand reads).
8. **Duplicate Sample IDs:** Identifies conflicting or duplicated sample identifiers.
9. **Severe Class Imbalance:** Emits a warning when the minority class comprises less than 20% of the majority class.
10. **Metadata Completeness:** Calculates the percentage of samples populating biological fields (`organism`, `gene`, `patient_id`, `sequencing_platform`).

### 4.1 Duplicate Handling Policies (`DuplicatePolicy`)

```python
from dna_cancer.data.quality import DatasetValidator, DuplicatePolicy

# Available policies:
# - DuplicatePolicy.REPORT_ONLY: Flags duplicates in the report without modifying samples (default)
# - DuplicatePolicy.KEEP_FIRST: Retains the first occurrence of each unique sequence
# - DuplicatePolicy.REMOVE_ALL: Discards all samples that appear more than once
# - DuplicatePolicy.ERROR: Treats duplicate sequences as a fatal validation error

validator = DatasetValidator(duplicate_policy=DuplicatePolicy.KEEP_FIRST)
filtered_samples = validator.apply_duplicate_policy(samples)
```

---

## 5. Statistical Profiling

The [`DatasetProfiler`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/profiler.py#L239) produces rich statistical distributions without requiring heavy external libraries:

- **Length Summary:** Min, max, mean, median, standard deviation, 25th percentile (Q1), and 75th percentile (Q3).
- **Nucleotide Composition:** Individual counts and percentages for `A`, `C`, `G`, `T`, and `N`.
- **GC-Content:** Percentage of guanine and cytosine bases:
  $$\text{GC-Content (\%)} = \frac{\text{Count}(G) + \text{Count}(C)}{\text{Total Bases}} \times 100$$
- **Class & Metadata Breakdowns:** Sample distribution across cancer types and metadata fill rates.

Dual reports are exported to `artifacts/datasets/`:
- `<dataset_id>_profile.json`: Machine-readable parameters for automated pipelines.
- `<dataset_id>_profile.md`: Formatted Markdown report with tables and alerts for human review.

---

## 6. Leakage-Free Data Splitting

### 6.1 The Patient Leakage Problem

In cancer genomics, a patient or study participant may contribute multiple sequence fragments, multiple tumor biopsies, or repeated sequencing reads.

> [!CAUTION]
> If a naive random or stratified split places fragment A from Patient 001 in the training set and fragment B from Patient 001 in the test set, the classifier can memorize patient-specific genomic background variants (e.g., private SNPs or germline variations) rather than true cancer-discriminative motifs.
> This produces **artificially inflated evaluation metrics** that fail completely on external cohorts.

### 6.2 Group-Based Splitting

[`split_grouped()`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/splitter.py#L122) groups all sequences by `patient_id` (or specified `group_key`) and assigns the entire group atomically to either Train, Validation, or Test:

```python
from dna_cancer.data.splitter import DatasetSplitter

splitter = DatasetSplitter(
    strategy="auto",         # Automatically selects 'group' when patient_id is present
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    group_key="patient_id",
    seed=42,
)

train_set, val_set, test_set = splitter.split(samples)
summary = splitter.summarize_split(train_set, val_set, test_set)

assert summary["leakage_free"] is True
```

---

## 7. Command-Line Interface (CLI) Reference

The project provides 4 command-line tools in `scripts/`:

### 7.1 Import a Dataset (`import_dataset.py`)

Ingests, validates, profiles, and registers a dataset into the registry:

```bash
# Ingest CSV with validation and profiling enabled
python scripts/import_dataset.py \
    --file data/raw/cancer_data.csv \
    --id dataset_cancer_v1 \
    --name "Initial 4-Class Cancer Dataset" \
    --source "College Project Archive" \
    --validate \
    --profile

# Ingest FASTA with an explicit label mapping file
python scripts/import_dataset.py \
    --file data/raw/exome_reads.fasta \
    --id fasta_exomes \
    --label-map configs/fasta_labels.csv \
    --validate
```

### 7.2 Validate a Dataset (`validate_dataset.py`)

Runs quality assurance checks on a file or an existing registered dataset:

```bash
# Validate registered dataset with strict exit code on error
python scripts/validate_dataset.py \
    --id dataset_cancer_v1 \
    --expected-classes "Cervical Cancer" "Colorectal Cancer" "Endometrial Cancer" "Liver Cancer" \
    --strict

# Validate a standalone raw file
python scripts/validate_dataset.py \
    --file data/raw/external_test.csv \
    --duplicate-policy keep_first
```

### 7.3 Profile a Dataset (`profile_dataset.py`)

Computes statistical distributions and saves reports to `artifacts/datasets/`:

```bash
python scripts/profile_dataset.py --id dataset_cancer_v1
```

### 7.4 List Registered Datasets (`list_datasets.py`)

Lists cataloged datasets, versions, sample counts, and SHA-256 hashes:

```bash
# Standard table view
python scripts/list_datasets.py

# Filter by tag
python scripts/list_datasets.py --tag "tcga"

# Output in JSON format
python scripts/list_datasets.py --json
```

