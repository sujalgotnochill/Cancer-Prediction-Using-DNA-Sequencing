# Data Directory Structure

This directory organizes data through the sequential stages of the machine learning lifecycle:

```text
data/
├── raw/         # Immutable raw source files (FASTA, raw CSVs)
├── interim/     # Cleaned, standardized, and filtered sequence data
└── processed/   # Tokenized k-mers, vectorized matrices, and engineered features
```

---

## 1. `data/raw/`
- **Purpose:** Stores unmodified source genomic datasets (e.g., raw FASTA sequence files, original disease datasets).
- **Policy:** Files in this directory must remain **read-only and immutable**. Never manually edit files in `raw/`.
- **Ignored by Git:** Raw genomic files are excluded from version control via `.gitignore`. Place actual sequence files here locally or download via the data pipeline.
- **Reference link:** Legacy dataset download link is preserved in `legacy/Data.txt`.

## 2. `data/interim/`
- **Purpose:** Stores intermediate data resulting from sequence normalization, cleaning, and validation.
- **Transformations applied:**
  - Nucleotide capitalization (converted to uppercase `ACGTN`).
  - Removal of whitespace, line wraps, and non-nucleotide characters.
  - Filtering of invalid or undersized sequences.
- **Policy:** Can be safely regenerated at any time by executing the preprocessing pipeline.

## 3. `data/processed/`
- **Purpose:** Final feature matrices and tokenized datasets ready for model consumption.
- **Contents:**
  - $k$-mer sequence representations.
  - Numerical feature matrices and encoded target label vectors.
  - Partitioned train/val/test splits.
- **Policy:** Generated automatically by `dna_cancer.features.feature_builder` and `dna_cancer.training.pipeline`.

