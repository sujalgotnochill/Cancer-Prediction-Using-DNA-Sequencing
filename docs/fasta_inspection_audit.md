# Audit Report: FASTA Inspection Verification (Phase 3.6.1)

**Author:** Senior Data Engineer & Bioinformatics Software Architect  
**Date:** 2026-11-04  
**Audit Target:** Review and Verification of Phase 3.6 Inspection Findings  
**Structured Audit Artifact:** [`artifacts/datasets/fasta_inspection_audit.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/datasets/fasta_inspection_audit.json)  
**Test Suite Status:** 108 tests passing (0 failures, 0 errors in 0.13s)

---

## 1. Executive Summary & Purpose of Audit

This audit was conducted to resolve discrepancies identified in the Phase 3.6 inspection of the raw FASTA collection in `data/raw/FASTA/`:
1. **File Size Audit:** Resolving the apparent discrepancy between the reported size of `Gastric cancer .txt` (491,527,857 bytes) and the prompt's reference to ~4,800,008 KB.
2. **Duplicate Calculation Audit:** Reconciling the pairwise overlap matrix sum (21,537) against the text statement (21,520), and establishing strict definitions for unique sequence identities, duplicate record instances, and conflicting label assignments.
3. **Hash Matching Verification:** Upgrading candidate matching from truncated 8-byte hashes to full 256-bit SHA-256 digests, verifying disk-backed sequence equality, and isolating reverse-complement matches.
4. **Biological Provenance Evidence:** Strictly separating directly observed FASTA header text from inferred organism taxonomy.
5. **Historical Experiment Critique:** Separating proven code behaviors and observed outputs from unverified hypotheses regarding the legacy project's 98% accuracy.

---

## 2. File Size Verification & Gastric Dataset Resolution

### 2.1 Filesystem Measurements

All five files were measured directly using Python's `os.stat` and PowerShell filesystem APIs. No file was modified, truncated, or replaced:

| File Name | Exact Bytes | KiB ($/1024$) | MiB ($/1024^2$) | GiB ($/1024^3$) | Windows Explorer Display |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cervical_cancer.txt` | 30,324,062 | 29,613.34 KiB | 28.92 MiB | 0.0282 GiB | **29,613 KB** |
| `Colorectal cancer .txt`| 37,840,477 | 36,953.59 KiB | 36.09 MiB | 0.0352 GiB | **36,954 KB** |
| `Endometrial cancer .txt`| 3,571,164 | 3,487.46 KiB | 3.41 MiB | 0.0033 GiB | **3,487 KB** |
| `Gastric cancer .txt` | **491,527,857** | **480,007.67 KiB**| **468.76 MiB** | **0.4578 GiB** | **480,008 KB** |
| `Liver Cancer .txt` | 77,637,445 | 75,817.82 KiB | 74.04 MiB | 0.0723 GiB | **75,818 KB** |

### 2.2 Resolution of the Gastric File Size Discrepancy

- **Observation in Windows Explorer:** Windows Explorer formats file sizes in KB by computing $\text{Bytes} \div 1024$ and rounding to the nearest integer:
  $$491,527,857 \text{ bytes} \div 1024 = 480,007.67285\dots \text{ KiB} \implies \mathbf{480,008\text{ KB}}$$
- **Discrepancy Cause:** The prompt referred to *"approximately 4,800,008 KB"*. Comparing $480,008\text{ KB}$ with $4,800,008\text{ KB}$ reveals an extra zero in the prompt's figure (a clerical decimal/typographical error: 480 thousand KB vs. 4.8 million KB).
- **Audit Verdict:** The file `Gastric cancer .txt` is **491,527,857 bytes** (~468.76 MiB / ~480,008 KB in Windows Explorer). It was **not** partially inspected, truncated, or replaced; it was read completely from byte 0 to byte 491,527,857.

---

## 3. Duplicate Calculations & Metric Definitions

### 3.1 Explanation of the 21,520 vs. 21,537 Discrepancy

In the Phase 3.6 report:
- The pairwise overlap matrix listed:
  - Cervical–Colorectal: **17**
  - Colorectal–Gastric: **1,857**
  - Colorectal–Liver: **7,360**
  - Gastric–Liver: **12,303**
  - **Sum of Pairwise Overlaps:** $17 + 1,857 + 7,360 + 12,303 = \mathbf{21,537}$
- The text summary stated: *"If ingested with filename labels, 21,520 sequences would be assigned contradictory cancer diagnoses."*
- **Mathematical Root Cause:** The text author computed $1,857 + 7,360 + 12,303 = 21,520$, inadvertently omitting the 17 shared sequences between Cervical and Colorectal ($21,520 + 17 = 21,537$).

### 3.2 Formal Metric Separation (Avoiding Metric Conflation)

To prevent confusion, we formally separate four distinct duplicate metrics:

1. **Unique Sequence Identities Shared Across Files:**  
   There are exactly **17,823 unique sequence strings** in the collection that appear in more than one file:
   - Sequences appearing in **exactly 2 files:** **15,966**
     - Cervical + Colorectal: 17
     - Colorectal + Liver only: 5,503
     - Gastric + Liver only: 10,446
   - Sequences appearing in **exactly 3 files:** **1,857**
     - Colorectal + Gastric + Liver simultaneously: 1,857
   - Total unique shared sequences: $15,966 + 1,857 = \mathbf{17,823}$.
2. **Sum of Pairwise Overlaps (21,537):**  
   Because 1,857 sequences appear in three files simultaneously, they are counted in (Colorectal, Gastric), in (Colorectal, Liver), and in (Gastric, Liver). Thus:
   $$\text{Pairwise Sum} = 17,823 + (2 \times 1,857) = 17,823 + 3,714 = \mathbf{21,537}$$
3. **Total Record Instances Participating in Cross-File Overlaps:**  
   Because sequences can appear multiple times within each file, the total number of physical records associated with conflicting sequence identities is **64,630 records**:
   - `cervical_cancer.txt`: 34 records
   - `Colorectal cancer .txt`: 14,694 records
   - `Gastric cancer .txt`: 19,356 records
   - `Liver Cancer .txt`: 30,546 records
   - `Endometrial cancer .txt`: 0 records
4. **Internal Redundant Records Within Files:**  
   Sequences appearing more than once within the same file:
   - `cervical_cancer.txt`: 7,029 redundant records (13,389 total records $\to$ 6,360 unique)
   - `Colorectal cancer .txt`: 4,744 redundant records (33,362 total records $\to$ 28,618 unique)
   - `Endometrial cancer .txt`: 8 redundant records (2,962 total records $\to$ 2,954 unique)
   - `Gastric cancer .txt`: 43,365 redundant records (436,244 total records $\to$ 392,879 unique)
   - `Liver Cancer .txt`: 11,099 redundant records (66,826 total records $\to$ 55,727 unique)

---

## 4. Hash Verification & Reverse-Complement Audit

### 4.1 Full 256-Bit SHA-256 Verification

The Phase 3.6 streaming inspector used truncated 8-byte SHA-256 prefixes for fast profiling. In this audit:
- Every sequence was indexed using its **full 32-byte (256-bit) SHA-256 digest**.
- Disk-backed character equality verification was performed across candidate matches by seeking to the exact byte offsets in each file.
- **Result:** Exactly 100% character-for-character sequence identity confirmed; **zero hash collisions** detected across all 552,783 records.

### 4.2 Reverse-Complement Audit

We audited whether any sequence matches represented reverse complements (sense vs. antisense strands):
- **Palindromic sequences ($\text{seq} == \text{RC}(\text{seq})$):** **0**
- **Non-palindromic reverse-complement matches within files:** **0**
- **Non-palindromic reverse-complement matches across files:** **0**
- **Verdict:** All cross-file and internal sequence duplicates are **100% exact forward-strand sequence identities**.

---

## 5. Biological Provenance Audit: Direct Evidence vs. Inferred Taxonomy

We strictly separate **directly observed FASTA header text** from **inferred taxonomic identities**:

| File | Directly Observed in FASTA Headers | Inferred / RefSeq Cross-Referenced Identity |
| :--- | :--- | :--- |
| `cervical_cancer.txt` | Accessions `NC_000015.10` (8,535 records), `LC270024`–`LC270123`, `CP097885.1`, `NZ_`. Protein tags: `E6 protein`, `E7 protein`, `E1 protein`, `GOLGA6L6`, `POTEB2`. | **RefSeq verified:** `NC_000015.10` is *Homo sapiens* Chromosome 15. The `LC` series is Human Papillomavirus type 16 (HPV-16). `CP097885` and `NZ_` are bacterial contigs. |
| `Colorectal cancer .txt`| Accessions `ADVG01000001.1` (3,094 records), `CP071092`–`CP071099`, `CP003116`, `CP003121`. Locus tags: `Krac_9242`–`Krac_9246`. Proteins: MFS transporters, ABC transporters. | **NCBI WGS verified:** `ADVG01000001.1` is *Ktedonobacter racemifer* SCRC/b-89 (bacterium). `CP003116`/`CP003121` are bacterial genomes. No human genes observed. |
| `Endometrial cancer .txt`| **Single accession** across all 2,962 records: `CP029347.1`. Locus tags: `HMF8227_00001`–`02962`. Proteins: `DnaA`, `DnaN`, `RecF`, `GyrB`. UniProt: `C1DFU2`, `P0A990`. | **GenBank verified:** `CP029347.1` is a complete circular bacterial chromosome. `DnaA` and `dnaN` are bacterial DNA replication machinery. |
| `Gastric cancer .txt` | Accessions `CP017839.1` and hundreds of WGS contigs (`AKHM`, `AKHN`, `AKHO`, `QBP`, `QBQ`, `BNKL`). Proteins: 30S ribosomal proteins S7/S12, outer membrane proteins. | **WGS assembly verified:** Microbial Whole-Genome Shotgun contigs from diverse environmental and bacterial isolates. |
| `Liver Cancer .txt` | Accessions `QASA01000001.1`, `CP003322`, `CP014486`, plus identical accessions `ADVG01000001.1`, `CP003116`, `CP003121`. Proteins: `DnaA`, `IscS`, `PrdX`. | **GenBank verified:** Bacterial genomes. Exactly 7,360 unique sequences are shared with `Colorectal cancer .txt` because identical bacterial genomes were included in both files. |

> [!IMPORTANT]
> **Audit Conclusion on Provenance:**  
> The observed header evidence proves that the files in `data/raw/FASTA/` do **not** represent human tumor tissue samples. They consist of human chromosome 15 CDS, HPV-16 viral isolates, and multiple bacterial genomes. The cancer filenames represent unverified external file labels.

---

## 6. Historical 98% Accuracy Explanation Audit

In Phase 3.6, it was suggested that the 98% accuracy achieved in the legacy notebook was driven by the model distinguishing human DNA from bacterial DNA. Here we strictly separate what the code proves from what is an unverified hypothesis:

### 6.1 Proven Facts from Notebook Code & Outputs
1. **Proven:** In `fasta2csv.ipynb`, the script read `Colorectal cancer .txt` and applied a blanket label `data['cancer'] = "Colorectal Cancer"` to all records without individual sequence validation.
2. **Proven:** In `data_preprocessing.ipynb`, the dataset `DNAdata.csv` was loaded with 1,200 rows (300 rows per class for 4 classes). The first 3 rows match the first 3 records of `cervical_cancer.txt`, and the first 2 rows of `Colorectal_Cancer.csv` match the first 2 records of `Colorectal cancer .txt`.
3. **Proven:** In `DNA_model.ipynb` Cell 6, `CountVectorizer(ngram_range=(4, 4))` was fitted on all 1,200 samples **before** train/test splitting. This represents data leakage.
4. **Proven:** In `DNA_model.ipynb` Cell 10, the validation split re-partitioned the full unsplit dataset `x`, causing the validation set to overlap with the test set.
5. **Proven:** The classification report output showed 0.98 accuracy (235/240 correct test predictions, with 5 Endometrial samples misclassified as Colorectal).

### 6.2 Clarification of Unverified Hypotheses
- **Hypothesis:** *"Taxonomic divergence (human vs. bacterial DNA) was the definitive cause of the 98% accuracy."*
- **Audit Correction:** This explanation remains a plausible scientific **hypothesis**, but it **cannot be stated as a proven fact** without reproducing the historical training run.
- **Alternative / Confounding Factors:** Multiple severe methodological flaws could independently or jointly explain the high reported accuracy:
  1. **Vectorizer Leakage:** Fitting the vectorizer on held-out test data.
  2. **Overlapping Validation Splits:** Leaking test representations into model tuning.
  3. **Sequence Length Artifacts:** The mean sequence length in Cervical (1,873 bp) is double that of the other classes (~948 bp), allowing classifiers to exploit length-correlated token counts.
  4. **Memorization of Internal Duplicates:** 52.5% of records in Cervical and 14–16% in Colorectal/Liver are duplicate sequences that could have spanned both train and test splits.

---

## 7. Deliverables & Preserved Records

1. **Audit Documentation:** This file (`docs/fasta_inspection_audit.md`).
2. **Machine-Readable Audit Data:** [`artifacts/datasets/fasta_inspection_audit.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/datasets/fasta_inspection_audit.json).
3. **Preserved History:** The original report [`docs/fasta_dataset_inspection.md`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/docs/fasta_dataset_inspection.md) and JSON [`artifacts/datasets/fasta_inspection.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/datasets/fasta_inspection.json) are preserved intact with audit notes added.
4. **Test Suite:** All **108 unit and integration tests passing** in `0.13s`.

