# Data Dictionary & Unified Schema: DNA Cancer Classification

This document specifies the unified dataset schema, field definitions, biological metadata taxonomy, and ground-truth cancer classes for the **DNA Cancer Classification** system.

> [!IMPORTANT]
> The schema strictly separates biological sequence data, experimental metadata, and target disease labels.
> Biological metadata (such as organism, gene, chromosome, or patient ID) must **never be fabricated** when absent from raw datasets. Missing values remain explicitly `None` or omitted.

---

## 1. Unified Dataset Schema (`DNASample` & `SampleMetadata`)

All datasets ingested into the system—regardless of file format (CSV, FASTA, JSON, JSONL)—are transformed into standardized `DNASample` objects defined in [`src/dna_cancer/data/schema.py`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/schema.py).

### 1.1 Core Fields (`DNASample`)

| Field Name | Type | Required? | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `sample_id` | `str` | **Yes** | Unique identifier for the individual sequence sample or read. | `"TCGA-02-0001-01A"` |
| `sequence` | `str` | **Yes** | Nucleotide character sequence (`A`, `C`, `G`, `T`, `N`). Standardized to uppercase during cleaning. | `"ATGCATGCATGC"` |
| `label` | `Optional[str]` | Conditional | Supervised classification ground-truth label. Required for training/evaluation; optional for raw unannotated inference. | `"Colorectal Cancer"` |
| `length` | `int` | Derived | Nucleotide character count (`len(sequence)`). Computed automatically as a property. | `1200` |
| `metadata` | `SampleMetadata` | Optional | Structured biological and experimental metadata container. | See Section 1.2 |

---

### 1.2 Biological & Experimental Metadata (`SampleMetadata`)

Biological provenance is captured through the `SampleMetadata` dataclass.

| Attribute | Type | Default | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `dataset_id` | `Optional[str]` | `None` | Identifier of the registered dataset from which this sample originated. | `"ds_tcga_luad_v1"` |
| `source` | `Optional[str]` | `None` | Public archive, laboratory, or repository source. | `"TCGA"`, `"NCBI GenBank"` |
| `organism` | `Optional[str]` | `None` | Scientific taxonomic name of the organism. | `"Homo sapiens"` |
| `gene` | `Optional[str]` | `None` | HGNC gene symbol associated with the sequence locus. | `"TP53"`, `"EGFR"`, `"KRAS"` |
| `locus` | `Optional[str]` | `None` | Specific genomic locus or coordinate description. | `"17p13.1"` |
| `chromosome` | `Optional[str]` | `None` | Chromosome identifier. | `"chr17"`, `"chr7"` |
| `sequence_type` | `Optional[str]` | `None` | Molecular nature of sequence (e.g. genomic DNA, cDNA, mRNA read). | `"cDNA"`, `"genomic_DNA"` |
| `collection_date` | `Optional[str]` | `None` | Date or year of clinical collection/sequencing. | `"2023-05-12"` |
| `sequencing_platform`| `Optional[str]` | `None` | Sequencing technology used. | `"Illumina NovaSeq 6000"`, `"PacBio Sequel II"` |
| `patient_id` | `Optional[str]` | `None` | Anonymized patient or donor identifier. **Critical for preventing patient data leakage across train/test splits.** | `"PATIENT_84920"` |
| `additional_metadata`| `Dict[str, Any]`| `{}` | Unstructured key-value pairs for dataset-specific attributes (e.g. tumor stage, histology). | `{"tumor_stage": "IIIA"}` |

---

## 2. FASTA Label Mapping Policy

FASTA sequence files (`.fasta`, `.fa`, `.fna`) contain header identifiers (e.g., `>seq1 Homo sapiens TP53`) but **do not inherently contain verified cancer diagnosis labels**.

> [!CAUTION]
> In legacy scripts, labels were occasionally assumed from sequence header text or hardcoded folder locations without audit trails.
> The modernized system enforces an **explicit metadata mapping mechanism** via `FastaLabelMapper`:
> - Labels must be associated with sequence IDs through an explicit lookup dictionary, CSV file (`id,label`), or JSON mapping file (`{"seq_id": "cancer_label"}`).
> - In the absence of an explicit mapping, samples are ingested with `label=None` or a user-provided fallback default label (e.g., `"Unannotated"`).
> - Cancer labels are **never** automatically guessed or parsed from header comments without verified user configuration.

---

## 3. Legacy College Dataset Schema (`legacy/old_notebooks/`)

The original second-year college project operated on four cancer types compiled into tabular CSV format:

| Class Index | Cancer Class Name | Verified in Legacy Code | Description / Legacy Context |
| :---: | :--- | :---: | :--- |
| **0** | `Cervical Cancer` | Yes | Extracted from legacy experiments |
| **1** | `Colorectal Cancer` | Yes | Extracted from legacy experiments / FASTA conversion |
| **2** | `Endometrial Cancer`| Yes | Extracted from legacy experiments |
| **3** | `Liver Cancer` | Yes | Extracted from legacy experiments |

### Legacy Field Mapping:
When ingesting legacy CSV files from `data/raw/` or `legacy/`, use the following `ColumnMapping`:
- `id_col="id"` (maps to `sample_id`)
- `sequence_col="sequence"` (maps to `sequence`)
- `label_col="cancer"` (maps to `label`)
- `len` is omitted from raw ingestion as it is dynamically computed by `DNASample.length`.

---

## 4. Derived & Processed Representations

During preprocessing and feature extraction, the following interim and processed representations are generated:

| Stage | Field / Representation | Type | Purpose |
| :--- | :--- | :--- | :--- |
| **Cleaning** | `cleaned_sequence` | `str` | Normalized uppercase nucleotide sequence containing only allowed IUPAC characters (`ACGTN`). Whitespace and carriage returns removed. |
| **Feature Extraction** | `kmers` | `List[str]` | Overlapping nucleotide sub-sequences of length $k$ (default $k=6$, stride $=1$). |
| **Tokenization** | `kmer_sentence` | `str` | Space-delimited string of $k$-mers suitable for NLP tokenizers (e.g., `"ATGCAT TGCATG GCATGC"`). |
| **Vectorization** | `feature_vector` | `scipy.sparse.csr_matrix` or `ndarray` | Bag-of-words count matrix or TF-IDF matrix representing $k$-mer frequencies across the sequence. |
| **Encoding** | `encoded_label` | `int` | Zero-indexed numerical target class generated by `LabelEncoder`. |
