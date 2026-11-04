# Real Dataset Acceptance Review (Phase 3.5)

**Author:** Senior Data Engineer & Bioinformatics Software Architect  
**Date:** 2026-11-04  
**Project:** Cancer Prediction Using DNA Sequencing  
**Review Status:** **COMPLETED — ORIGINAL DATASET NOT LOCALLY HOSTED (EXTERNAL DOWNLOAD REQUIRED)**  
**All Existing Tests:** 104 passed (0 failures, 0 errors in 0.35s)

---

## 1. Executive Summary & Forensic Findings

In Phase 3.5, we conducted a rigorous inspection of the repository to locate, inspect, and evaluate the original project dataset prior to model experimentation in Phase 4.

### Key Inspection Results:
1. **Local Availability:** The dataset files (`DNAdata.csv`, component CSV files, and FASTA files) are **NOT present locally** anywhere within the workspace (`data/raw/`, `data/interim/`, `data/processed/`, or `legacy/`).
2. **Git Commit History:** The dataset was **never committed to the Git repository**. From the initial commit (`6da4a98: "complete model"`), only Jupyter notebooks and a reference file (`legacy/Data.txt`) were tracked.
3. **Download Source Identified:** The authoritative dataset repository is documented in [`legacy/Data.txt`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/legacy/Data.txt):
   $$\text{Google Drive Repository: } \texttt{https://drive.google.com/drive/folders/1g-cAyQwqpSoG7nrmaLuzgmn7AtzVgAa2?usp=sharing}$$
4. **Execution Constraint Applied:** In accordance with the prompt's explicit instruction:
   > *"If the dataset is not available locally, report the original download source and stop. Do not download from an unverified source or fabricate data."*
   We halted the automated ingestion, preserved all raw files unchanged, did not fabricate mock records, and documented the exact dataset structure reconstructed from the legacy notebook execution outputs.

---

## 2. Dataset Reconstruction from Legacy Notebook Outputs

While the raw CSV file was not checked into Git, forensic analysis of the preserved execution outputs in [`legacy/old_notebooks/data_preprocessing.ipynb`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/legacy/old_notebooks/data_preprocessing.ipynb) and [`legacy/old_notebooks/DNA_model.ipynb`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/legacy/old_notebooks/DNA_model.ipynb) revealed the exact structure, size, and properties of the dataset.

### 2.1 File Structure & Dimensions

- **Primary Consolidated File:** `/content/drive/MyDrive/disease datasets/DNAdata.csv`
- **Total Record Count:** Exactly **1,200 sequence samples**
  - Confirmed via `DNA_model.ipynb` Cell 8: `print(x.shape) -> (1200, 230368)`
  - Confirmed via `data_preprocessing.ipynb` Cell 3: row index range `0` to `1199`.
- **Columns in `DNAdata.csv`:**
  1. `Unnamed: 0.1`: Global integer index ($0$ to $1199$).
  2. `Unnamed: 0`: Class-batch integer index ($0$ to $299$).
  3. `sequence`: Raw nucleotide DNA sequence string (uppercase IUPAC characters).
  4. `len`: Integer sequence character length (`len(sequence)`).
  5. `cancer`: Target cancer classification string label.

### 2.2 Class Distribution

The dataset contains exactly **4 target classes**, perfectly balanced with approximately 300 samples per class:

| Class Index | Cancer Class Name | Sample Count (Estimated) | Test Split Count ($20\%$) | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `Cervical Cancer` | 300 | 49 | 1.00 | 1.00 | 1.00 |
| **1** | `Colorectal Cancer` | 300 | 61 | 0.92 | 1.00 | 0.96 |
| **2** | `Endometrial Cancer`| 300 | 63 | 1.00 | 0.92 | 0.96 |
| **3** | `Liver Cancer` | 300 | 67 | 1.00 | 1.00 | 1.00 |
| **Total** | | **1,200** | **240** | **0.98** | **0.98** | **0.98** |

*Note: Test counts, precision, recall, and F1-scores are retrieved directly from the preserved classification report and confusion matrix outputs in `DNA_model.ipynb` (Cells 20 & 23).*

### 2.3 Sequence Characteristics

- **Observed Length Range:** Samples observed in the preserved outputs display substantial length variation:
  - Minimum observed: 267 bp
  - Intermediate lengths: 399 bp, 1,020 bp, 1,266 bp, 1,635 bp
  - Maximum observed: 2,175 bp
- **Observed Nucleotides:** Standard canonical DNA bases (`A`, `C`, `G`, `T`). No ambiguous IUPAC characters (`N`, `R`, `Y`) appeared in the sampled output heads.

---

## 3. Critical Methodological & Data Quality Issues in Legacy Code

Our audit revealed four severe methodological flaws in the legacy experiments that must be resolved in our modernized framework:

### 3.1 Unverified FASTA Blanket Labeling
In `fasta2csv.ipynb` (Cell 1 & 2):
```python
with open('/content/drive/MyDrive/disease datasets/FASTA/Colorectal cancer .txt') as fasta_file:
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        identifiers.append(str(seq_record.seq))
data['cancer'] = "Colorectal Cancer"
```
**Problem:** The legacy project assumed that *every* sequence inside a FASTA file labeled "Colorectal cancer .txt" was an instance of colorectal cancer, without verifying individual gene locus identifiers, accession numbers, somatic status, or control status.

### 3.2 Feature Vectorizer Data Leakage
In `DNA_model.ipynb` (Cell 6 & 10):
```python
cv = CountVectorizer(ngram_range=(4, 4))
x = cv.fit_transform(x)  # Fitted on all 1,200 samples!
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
**Problem:** The vocabulary and $N$-gram frequency representation were learned on the **entire 1,200-sample dataset** prior to data partitioning. In rigorous ML engineering, vectorizers must be fitted strictly on the training partition (`x_train`) and applied downstream to validation/test partitions.

### 3.3 Overlapping Validation Split Bug
In `DNA_model.ipynb` (Cell 10):
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
```
**Problem:** The second call split the original unsplit matrices (`x`, `y`) instead of `x_train`. This caused `x_val` to overlap with `x_test`, corrupting validation and test separation.

### 3.4 Complete Lack of Biological Provenance
Neither `DNAdata.csv` nor the individual CSVs stored:
- Patient or donor identifiers (`patient_id`).
- Gene locus or symbol (`gene`).
- Chromosome coordinate (`chromosome`).
- Sequencing platform or read technology (`sequencing_platform`).
- Somatic vs. germline designation.

---

## 4. Import & Acceptance Procedure for the Real Dataset

To accept and ingest the actual dataset once downloaded from Google Drive, execute the following standardized steps using the Phase 3 architecture:

### Step 1: Place Downloaded File in Raw Storage
Download `DNAdata.csv` from:
`https://drive.google.com/drive/folders/1g-cAyQwqpSoG7nrmaLuzgmn7AtzVgAa2?usp=sharing`  
Save it to:
```text
data/raw/DNAdata.csv
```

### Step 2: Ingest, Validate, Profile, and Register via CLI
Run the Phase 3 ingestion script with explicit column mapping and quality checks enabled:

```powershell
$env:PYTHONPATH="src"
python scripts/import_dataset.py `
    --file data/raw/DNAdata.csv `
    --id cancer_dna_1200_v1 `
    --name "Original 4-Class Cancer DNA Cohort" `
    --source "Google Drive (legacy/Data.txt)" `
    --description "Legacy college benchmark dataset comprising 1,200 sequences across Cervical, Colorectal, Endometrial, and Liver cancers" `
    --column-map '{"sequence_col": "sequence", "label_col": "cancer", "id_col": "Unnamed: 0.1"}' `
    --tags legacy benchmark 4-class `
    --validate `
    --profile
```

### Step 3: Run Standalone Validation & Quality Inspection
```powershell
python scripts/validate_dataset.py `
    --id cancer_dna_1200_v1 `
    --expected-classes "Cervical Cancer" "Colorectal Cancer" "Endometrial Cancer" "Liver Cancer" `
    --strict
```

### Step 4: Verify Generated Artifacts
The CLI will generate:
- `artifacts/datasets/registry.json`: Updated with SHA-256 checksum and dataset metadata.
- `artifacts/datasets/cancer_dna_1200_v1_quality.json` & `.md`: Duplicate sequence audit, reverse complement matches, and conflict detection.
- `artifacts/datasets/cancer_dna_1200_v1_profile.json` & `.md`: Sequence length distribution, GC-content, and base frequencies.

---

## 5. Biological Data Issues & Clinical Disclaimer

1. **Non-Canonical & Ambiguous Nucleotides:**  
   The Phase 3 cleaning and validation modules handle ambiguous base calls (`N`) cleanly. When the full dataset is imported, any unexpected non-IUPAC characters will be flagged automatically in the validation report.
2. **Reverse Complements & Strandedness:**  
   The legacy notebooks did not check for strandedness. In our framework, `compute_reverse_complement()` detects sequences matching opposite strands without automatically deleting them, allowing researchers to evaluate stranded library protocols.
3. **No Clinical Validity:**  
   > [!CAUTION]
   > The 1,200-sample dataset was constructed as an exploratory undergraduate prototype. It lacks verified accession provenance, clinical staging, patient demographics, and matched normal germline controls.
   > **Under no circumstances should this dataset or models trained upon it be considered clinically valid, diagnostic, or suitable for healthcare decision-making.**

---

## 6. Data Integrity Verification

| Metric | Status / Value | Explanation |
| :--- | :--- | :--- |
| **Original Record Count** | 1,200 | Verified from legacy notebook execution outputs |
| **Local Raw Files Found** | 0 | Dataset was never committed to Git |
| **Existing Repository Integrity** | Preserved | No files modified or deleted in `legacy/` or `data/` |
| **Test Suite Status** | 104 Passing | 0 regressions across all foundation and data tests |
| **Ingestion Readiness** | Ready | CLI tools and schema contracts tested and fully operational |

---

## 7. Next Steps for Phase 4 (Baseline Models)

1. **User Action:** Download `DNAdata.csv` from the Google Drive link (`legacy/Data.txt`) into `data/raw/DNAdata.csv`.
2. **Automated Registration:** Run `python scripts/import_dataset.py` as detailed in Section 4.
3. **Proceed to Phase 4:** Proceed with baseline machine learning model development (Multinomial Naive Bayes, L2 Logistic Regression, Random Forest) using strictly leakage-free train/validation/test partitions.

