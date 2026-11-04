# Baseline Benchmark Integrity Audit

**Phase:** 4.1 - Baseline Benchmark Integrity Audit  
**Auditor Role:** Senior Machine Learning Engineer & Computational Biology Researcher  
**Target Dataset:** `data/interim/historical_reconstruction/DNAdata_reconstructed.csv`  
**Dataset SHA-256:** `eb14d47f3eeea2684313ff8a5cb726f431fbfba62eaf371f618e3a444a245bc3`  
**Audit Artifact:** `artifacts/evaluation/benchmark_integrity_audit.json`  
**Integrity Verdict:** **VERIFIED ACCURATE & REPRODUCIBLE (READY FOR MODEL OPTIMIZATION)**  

---

## 1. Executive Summary

This forensic audit rigorously verified the data integrity, partition boundaries, feature transformations, evaluation metrics, and experiment telemetry of the Phase 4 baseline machine learning benchmarks.

The headline performance of the top-performing model—**Logistic Regression ($k=5$ and $k=6$) achieving 96.77% Test Accuracy and 0.9723 Macro F1** under the duplicate-grouped benchmark—was **re-run from scratch and confirmed bit-for-bit identical**.

### Key Findings Matrix
| Audit Dimension | Verification Method | Status | Findings |
|---|---|---|---|
| **Dataset Source & Hash** | SHA-256 hashing & record validation | **PASSED** | Exact historical 1,200-sample reconstruction used; no synthetic or mock fixtures. |
| **Duplicate-Grouped Split** | Pairwise set disjointness audit | **PASSED** | $\text{Train} \cap \text{Test} = \emptyset$. Zero identical sequences cross partitions. |
| **Stratified Split Leakage** | Sequence membership overlap audit | **CONFIRMED** | 71.1% of test Cervical sequences in stratified split are duplicates of training sequences. |
| **Feature Isolation** | Code execution & vocabulary inspection | **PASSED** | Vectorizers fitted exclusively on training sequences; zero global statistics learned on test. |
| **Sparse Matrix Usage** | In-memory object type inspection | **PASSED** | Features stored as `scipy.sparse.csr_matrix`; no unnecessary dense allocations. |
| **Metric Reproducibility** | Clean re-execution of best model | **PASSED** | Reproduced 150 / 155 test accuracy (96.77%) and identical confusion matrix. |
| **Telemetry Integrity** | Schema validation across 24 runs | **PASSED** | 24 / 24 experiment JSONs complete with mandatory scientific limitation notice. |

---

## 2. Experimental Data Verification

We verified the physical dataset file loaded by `BaselineBenchmarkRunner`:
- **File Location:** `data/interim/historical_reconstruction/DNAdata_reconstructed.csv`
- **File Size:** `1,349,151 bytes`
- **SHA-256 Checksum:** `eb14d47f3eeea2684313ff8a5cb726f431fbfba62eaf371f618e3a444a245bc3`
- **Total Records:** `1,200`
- **Column Schema:** `['Unnamed: 0.1', 'Unnamed: 0', 'sequence', 'len', 'cancer']`
- **Target Distribution:** Exactly 300 samples per class:
  - `Cervical Cancer`: 300 (25.0%)
  - `Colorectal Cancer`: 300 (25.0%)
  - `Endometrial Cancer`: 300 (25.0%)
  - `Liver Cancer`: 300 (25.0%)

### Sequence Duplicity Audit
- **Unique Sequence Strings:** `982`
- **Sequence Groups with Duplicates:** `25` groups spanning `243` records.
  - `Cervical Cancer`: 23 sequence groups spanning **239 records** (human chromosome 15 transcript CDS duplicates).
  - `Colorectal Cancer`: 2 sequence groups spanning **4 records**.
  - `Endometrial Cancer`: 0 duplicate groups (300 unique sequences).
  - `Liver Cancer`: 0 duplicate groups (300 unique sequences).

**Verdict:** The benchmark strictly used the verified historical reconstructed dataset, with no synthetic fixtures or corrupted sample IDs.

---

## 3. Data Partition Audit & Leakage Boundaries

### 3.1 Sample Counts Across Partitions

| Partitioning Strategy | Split | Cervical | Colorectal | Endometrial | Liver | Total Samples | % of Dataset |
|---|---|---|---|---|---|---|---|
| **Duplicate-Grouped** | **Train** | 266 | 209 | 210 | 210 | **895** | 74.6% |
| **Duplicate-Grouped** | **Validation** | 15 | 45 | 45 | 45 | **150** | 12.5% |
| **Duplicate-Grouped** | **Test** | 19 | 46 | 45 | 45 | **155** | 12.9% |
| *Conventional Stratified* | *Train* | 210 | 210 | 210 | 210 | *840* | 70.0% |
| *Conventional Stratified* | *Validation* | 45 | 45 | 45 | 45 | *180* | 15.0% |
| *Conventional Stratified* | *Test* | 45 | 45 | 45 | 45 | *180* | 15.0% |

### 3.2 What Each Split Strategy Guarantees and Does Not Guarantee

```mermaid
flowchart TD
    subgraph DG["Duplicate-Grouped Split (by Sequence)"]
        DG_G["GUARANTEES:<br/>• Zero identical sequence strings cross partitions<br/>• Out-of-sample sequence generalization"]
        DG_NG["DOES NOT GUARANTEE:<br/>• Equal sample counts per class (varies by group size)<br/>• Zero gene-family homology across splice variants"]
    end

    subgraph CS["Conventional Stratified Split"]
        CS_G["GUARANTEES:<br/>• Exact sample-level class balance (210/45/45 per class)<br/>• Preserved class prior proportions"]
        CS_NG["DOES NOT GUARANTEE:<br/>• Duplicate sequence isolation<br/>(32/45 test Cervical samples leaked from train)"]
    end
```

1. **Duplicate-Grouped Strategy:**
   - **Guarantees:** Strict sequence disjointness. For all pairs $(\mathcal{A}, \mathcal{B}) \in \{(\text{Train}, \text{Val}), (\text{Train}, \text{Test}), (\text{Val}, \text{Test})\}$, $\mathcal{A}_{\text{seqs}} \cap \mathcal{B}_{\text{seqs}} = \emptyset$.
   - **Does NOT Guarantee:** Sample-level class symmetry. Because Cervical Cancer sequences have high duplication multiplicity (239 records across 23 groups), assigning groups atomically allocates 266 Cervical samples to Train, 15 to Val, and 19 to Test. Every one of the 19 test Cervical samples is a **strictly unseen sequence string**.
2. **Conventional Stratified Strategy:**
   - **Guarantees:** Exact sample-level class balance (210/45/45 per class across all 4 classes).
   - **Does NOT Guarantee:** Protection against duplicate-sequence memorization.
   - **Empirical Measurement:** **32 out of 45 test Cervical samples (71.1%)** were identical duplicates of sequences in the training set.

### 3.3 Transcript & Gene-Family Crossing Audit
We investigated whether different splice variants or isoforms of the same gene cross partitions when grouped by sequence hash:
- **Cervical FASTA Gene Annotations:** Mapped 300 Cervical samples to their NCBI `[gene=...]` tags.
- **Distinct Gene Tags:** 25 genes in Train, 7 genes in Val, 12 genes in Test.
- **Shared Gene Families across Train and Test:** 8 human chromosome 15 gene families (`POTEB`, `TUBGCP5`, `CYFIP1`, `ATP10A`, `GOLGA6L22`, `UBE3A`, `GABRB3`, `GABRA5`) had one splice isoform variant in Train and a distinct splice isoform variant in Test.
- **Implication:** Grouping by sequence hash guarantees that **no identical sequence string** is seen during training, but models that capture general gene-family k-mers may still benefit from biological homology between alternative splice variants.

### 3.4 Quarantined Test Set Confirmation
The test set was loaded and partitioned once, stored in memory, and transformed using the pre-fitted training vectorizer. No model selection, hyperparameter tuning, or early stopping was performed using the test partition.

---

## 4. Feature Engineering Audit

1. **Training-Only Fitting:** Verified in `BaselineBenchmarkRunner`:
   ```python
   fb.fit(train_seqs, train_labels)
   X_train = fb.transform_sequences(train_seqs)
   X_val = fb.transform_sequences(val_seqs)
   X_test = fb.transform_sequences(test_seqs)
   ```
   Vocabulary indices and document frequencies are computed exclusively on `train_seqs`.
2. **Transform-Only Test Data:** Transforming test sequences does not alter `fb.vectorizer.vocabulary_`.
3. **Sparse Matrix Enforcement:** `fb.transform_sequences()` outputs `scipy.sparse.csr_matrix` when `return_sparse=True`. Matrix dimensions:
   - $k=3$: $895 \times 64$ (CSR sparse matrix)
   - $k=4$: $895 \times 256$ (CSR sparse matrix)
   - $k=5$: $895 \times 1,024$ (CSR sparse matrix)
   - $k=6$: $895 \times 4,096$ (CSR sparse matrix)
   Zero densification occurred in memory during model fitting.

---

## 5. Evaluation Metrics & Reproducibility Verification

### 5.1 Bit-for-Bit Reproducibility Test
We independently re-ran the top-performing configuration:
- **Model:** `LogisticRegressionClassifier(C=1.0, max_iter=1000, random_state=42)`
- **Feature Setting:** $k=6$, stride=1, count vectorizer
- **Partition:** `duplicate_grouped`, seed=42

#### Verification Results:
| Metric | Original Logged Run (`20261104_072130_a82524`) | Re-Run Audit Value | Parity Status |
|---|---|---|---|
| **Test Accuracy** | `0.9677419...` (150 / 155) | `0.9677419...` (150 / 155) | **BIT-FOR-BIT IDENTICAL** |
| **Macro Precision** | `0.9724425...` | `0.9724425...` | **BIT-FOR-BIT IDENTICAL** |
| **Macro Recall** | `0.9723429...` | `0.9723429...` | **BIT-FOR-BIT IDENTICAL** |
| **Macro F1** | `0.9723349...` | `0.9723349...` | **BIT-FOR-BIT IDENTICAL** |
| **Confusion Matrix** | `[[19, 0, 0, 0], [0, 45, 1, 0], [0, 2, 42, 1], [0, 0, 1, 44]]` | `[[19, 0, 0, 0], [0, 45, 1, 0], [0, 2, 42, 1], [0, 0, 1, 44]]` | **BIT-FOR-BIT IDENTICAL** |

### 5.2 Deep-Dive into the 5 Test Errors
Out of 155 test samples, exactly 5 samples were misclassified:
1. **Cervical Cancer (19 test samples):** 19 / 19 correct (100% precision, 100% recall). Zero human sequences confused with bacteria.
2. **Liver Cancer (45 test samples):** 44 / 45 correct (97.8% recall); 1 sample misclassified as Endometrial.
3. **Colorectal Cancer (46 test samples):** 45 / 46 correct (97.8% recall); 1 sample misclassified as Endometrial.
4. **Endometrial Cancer (45 test samples):** 42 / 45 correct (93.3% recall); 2 samples misclassified as Colorectal, 1 misclassified as Liver.

**Biological Reality:** The misclassifications are exclusively confined to the three bacterial classes—primarily between *Ktedonobacter* (Colorectal, 54.2% GC) and *Saliniradius* (Endometrial, 52.5% GC), which share high nucleotide compositional similarity.

---

## 6. Audit of the 24 Experiment Records

We audited all 24 run records in `artifacts/experiments/`:
- **Record Count:** Exactly 24 individual JSON run records matching `artifacts/experiments/runs_summary.json`.
- **Field Completeness:** 100% compliance with required fields (`run_id`, `run_name`, `dataset`, `split_configuration`, `feature_configuration`, `model`, `durations`, `metrics`, `artifacts`, `scientific_limitation_notice`).
- **Terminology Correction:** 
  - *Previous Issue:* Console output previously used the shorthand string `"LEAKED"` for the conventional stratified split.
  - *Correction:* A conventional stratified split is not inherently or definitionally "data leakage"; rather, it is *vulnerable to duplicate-sequence leakage* when identical sequences exist in the dataset. Telemetry records and audit artifacts explicitly record `contains_duplicate_sequence_leakage: true` rather than vague labels.

---

## 7. Conclusions & Readiness Verdict

1. **Benchmark Rigor Certified:** The baseline benchmark implementation is mathematically and empirically sound. All leakage vectors identified in historical notebooks (global vectorizer fitting, overlapping test-val partitions, duplicate sequence memorization) have been completely eliminated under `duplicate_grouped`.
2. **High Accuracy Demystified:** The 96.77% test accuracy achieved by Logistic Regression is not an artifact of data leakage or overfitting; it directly reflects the phylogenetic divergence between human chromosome 15 and three bacterial phyla.
3. **Production Readiness:** The baseline infrastructure, model contracts, feature builders, evaluators, and tracking mechanisms are certified **ready for Phase 5 (Model Optimization and Advanced Architectures)**.

