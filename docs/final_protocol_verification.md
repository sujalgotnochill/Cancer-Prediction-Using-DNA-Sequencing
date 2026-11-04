# Final Evaluation Protocol Verification & Gene Inventory Audit

## 1. Executive Summary & Purpose

Phase 4.2 and Phase 4.3 exposed fundamental methodological limitations in baseline benchmark evaluations: test-set selection contamination, splice-isoform sequence leakage, and extreme gene-family size skewness in the human *Homo sapiens* Chromosome 15 subset (labeled "Cervical Cancer").

This document provides the independent mathematical, biological, and software verification of the finalized evaluation protocol before beginning Phase 5 model optimization.

### Key Verification Findings
1. **Gene Inventory Reconciliation**: All 31 human gene families present in the 300-sample Cervical Cancer subset have been forensically reconciled, cataloging NCBI GeneIDs, accessions, and sample counts.
2. **Holdout Independence**: The 240-sample Tier 2 Sealed Test Vault achieves mathematically proven zero sequence overlap and zero gene-family overlap with the 960-sample Tier 1 Development Cohort.
3. **Cross-Validation Realism**: The discrepancy between Phase 4.3 preliminary estimates (112, 99, 89 on $N=300$) and the actual Development Cohort 3-fold split (112, 64, 64 on $N=240$) is reconciled. The statistical consequence of single-gene validation folds (Fold 0 validating exclusively on `SNRPN`) is formally documented.
4. **Procedural Safeguards**: The Tier 2 Test Vault is protected by physical file separation, SHA-256 manifest integrity verification, and one-shot programmatic unseal safeguards.

---

## 2. Reconciled Cervical Gene-Family Inventory

The 300 human transcript samples (derived from *Homo sapiens* Chromosome 15 RefSeq coding sequences labeled as "Cervical Cancer") were partitioned into 240 Development samples and 60 Sealed Holdout samples.

The complete, verified inventory of all 31 gene families is detailed below:

| # | Gene Symbol | NCBI GeneID | Sample Count | Representative Accessions | Cohort Assignment | Evidence Source |
|---|---|---|---|---|---|---|
| 1 | `SNRPN` | 6638 | 112 | `NP_001336391.1` .. `NP_001387696.1` | **Development** | RefSeq `[gene=SNRPN]` (recs 118..229) |
| 2 | `NIPA2` | 81614 | 51 | `XP_016878134.1` .. `XP_006720430.1` | **Development** | RefSeq `[gene=NIPA2]` (recs 25..75) |
| 3 | `UBE3A` | 7337 | 48 | `NP_001341479.1` .. `NP_001341475.1` | **Development** | RefSeq `[gene=UBE3A]` (recs 230..277) |
| 4 | `TUBGCP5` | 114791 | 17 | `NP_001341303.1` .. `NP_001341307.1` | **Sealed Holdout** | RefSeq `[gene=TUBGCP5]` (recs 86..102) |
| 5 | `CYFIP1` | 23191 | 10 | `NP_001311048.1` .. `NP_001311051.1` | **Sealed Holdout** | RefSeq `[gene=CYFIP1]` (recs 76..85) |
| 6 | `ATP10A` | 57194 | 9 | `XP_011520130.1` .. `XP_011520131.1` | **Sealed Holdout** | RefSeq `[gene=ATP10A]` (recs 278..286) |
| 7 | `GABRA5` | 2558 | 7 | `NP_001158509.1` .. `XP_047288294.1` | **Sealed Holdout** | RefSeq `[gene=GABRA5]` (recs 293..299) |
| 8 | `GABRB3` | 2562 | 6 | `NP_068712.1` .. `NP_001265560.1` | **Sealed Holdout** | RefSeq `[gene=GABRB3]` (recs 287..292) |
| 9 | `POTEB` | 100996331 | 4 | `NP_001264233.1` .. `XP_011542096.1` | **Sealed Holdout** | RefSeq `[gene=POTEB]` (recs 14..17) |
| 10 | `POTEB2` | 100287399 | 4 | `NP_001264232.1` .. `XP_011542095.1` | **Sealed Holdout** | RefSeq `[gene=POTEB2]` (recs 2..5) |
| 11 | `POTEB3` | 102724631 | 4 | `NP_997238.2` .. `XP_011542098.1` | **Development** | RefSeq `[gene=POTEB3]` (recs 6..9) |
| 12 | `GOLGA6L2` | 283685 | 3 | `XP_047288353.1` .. `NP_001291317.1` | **Development** | RefSeq `[gene=GOLGA6L2]` (recs 108..110) |
| 13 | `SNURF` | 8926 | 3 | `NP_005669.2` .. `NP_001381263.1` | **Sealed Holdout** | RefSeq `[gene=SNURF]` (recs 115..117) |
| 14 | `GOLGA6L22` | 440243 | 2 | `NP_001383885.1` .. `NP_001383886.1` | **Development** | RefSeq `[gene=GOLGA6L22]` (recs 21..22) |
| 15 | `GOLGA6L26` | 102723623 | 2 | `XP_016878267.1` .. `NP_001369375.2` | **Development** | RefSeq `[gene=GOLGA6L26]` (recs 104..105) |
| 16 | `GOLGA8S` | 653061 | 2 | `NP_001382302.1` .. `NP_001342394.1` | **Development** | RefSeq `[gene=GOLGA8S]` (recs 106..107) |
| 17 | `NIPA1` | 123606 | 2 | `NP_653200.2` .. `NP_001135747.1` | **Development** | RefSeq `[gene=NIPA1]` (recs 23..24) |
| 18 | `GOLGA6L1` | 283767 | 1 | `NP_001001413.3` | **Development** | RefSeq `[gene=GOLGA6L1]` (rec 103) |
| 19 | `GOLGA6L6` | 727832 | 1 | `NP_001138476.2` | **Development** | RefSeq `[gene=GOLGA6L6]` (rec 1) |
| 20 | `LOC102724737` | 102724737 | 1 | `XP_047289356.1` | **Development** | RefSeq `[gene=LOC102724737]` (rec 12) |
| 21 | `LOC102724971` | 102724971 | 1 | `XP_011542106.1` | **Development** | RefSeq `[gene=LOC102724971]` (rec 13) |
| 22 | `LOC107983983` | 107983983 | 1 | `XP_016878258.1` | **Development** | RefSeq `[gene=LOC107983983]` (rec 20) |
| 23 | `LOC124903442` | 124903442 | 1 | `XP_047289354.1` | **Development** | RefSeq `[gene=LOC124903442]` (rec 0) |
| 24 | `MAGEL2` | 54551 | 1 | `NP_061939.3` | **Development** | RefSeq `[gene=MAGEL2]` (rec 112) |
| 25 | `MKRN3` | 7681 | 1 | `NP_005655.1` | **Development** | RefSeq `[gene=MKRN3]` (rec 111) |
| 26 | `NDN` | 4692 | 1 | `NP_002478.1` | **Development** | RefSeq `[gene=NDN]` (rec 113) |
| 27 | `NPAP1` | 23742 | 1 | `NP_061831.2` | **Development** | RefSeq `[gene=NPAP1]` (rec 114) |
| 28 | `OR4M2` | 390538 | 1 | `NP_001004719.2` | **Development** | RefSeq `[gene=OR4M2]` (rec 18) |
| 29 | `OR4M2B` | 118568804 | 1 | `NP_001382225.1` | **Development** | RefSeq `[gene=OR4M2B]` (rec 10) |
| 30 | `OR4N4` | 283694 | 1 | `NP_001005241.2` | **Development** | RefSeq `[gene=OR4N4]` (rec 19) |
| 31 | `OR4N4C` | 105369274 | 1 | `NP_001352318.2` | **Development** | RefSeq `[gene=OR4N4C]` (rec 11) |

### Summary Totals
- **Development Cohort ($N=240$)**: 23 unique gene families ($112 + 51 + 48 + 4 + 3 + 2\times 4 + 1\times 14 = 240$ samples).
- **Sealed Test Holdout ($N=60$)**: 8 unique gene families ($17 + 10 + 9 + 7 + 6 + 4 + 4 + 3 = 60$ samples).
- **Total Cervical Records ($N=300$)**: 31 unique gene families.
- **Overlap Between Development and Holdout**: **0 gene families (0.00%)**.

---

## 3. Forensic Gene Reconciliation & Investigation

During Phase 4.2 audit, cross-partition leakage was identified in the historical duplicate-grouped split. When comparing the Phase 4.2 leakage inventory with the Phase 4.3 proposed holdout allocation, four specific genes require explicit forensic clarification:

### 1. `GOLGA6L22` (2 samples)
- **Phase 4.2 Finding**: In the baseline duplicate-grouped split, `GOLGA6L22` leaked across partitions (1 sample in training, 1 sample in test).
- **Reconciliation Decision**: Assigned to the **Development Cohort**.
- **Justification**: Both samples (NCBI GeneID 440243, records 21 and 22) are kept strictly grouped together. Allocating `GOLGA6L22` to the holdout was unnecessary because the holdout already achieved its exact 60-sample target through the 8 selected genes ($17+10+9+7+6+4+4+3=60$). Retaining `GOLGA6L22` in development avoids unnecessary fragmentation of small gene groups.

### 2. `UBE3A` (48 samples)
- **Phase 4.2 Finding**: Leaked heavily in historical duplicate splits (44 samples in train, 4 samples in test).
- **Reconciliation Decision**: Assigned to the **Development Cohort**.
- **Justification**: The sealed holdout budget for Cervical Cancer is strictly 60 samples (to preserve 1:1:1:1 balance with the 60 samples from Colorectal, Endometrial, and Liver). If `UBE3A` (48 samples) were assigned to the holdout:
  1. Only 12 remaining slots would exist for all other human genes.
  2. A single gene family would comprise **80.0%** of the entire Cervical holdout evaluation.
  3. The holdout would measure generalization to `UBE3A` almost exclusively, rather than diverse out-of-gene generalization across multiple gene families.
  Therefore, keeping `UBE3A` in the Development cohort provides diverse high-count training representation while allowing the holdout to evaluate 8 distinct unseen gene families.

### 3. `POTEB2` (4 samples)
- **Phase 4.2 Finding**: Did not leak in Phase 4.2 duplicate splits.
- **Reconciliation Decision**: Assigned to the **Sealed Test Holdout**.
- **Justification**: `POTEB2` (NCBI GeneID 100287399, records 2..5) represents a distinct locus from `POTEB` (NCBI GeneID 100996331, records 14..17) and `POTEB3` (NCBI GeneID 102724631, records 6..9). Under strict within-organism grouping, `POTEB2` is kept intact as an 4-sample group and placed into the holdout alongside `POTEB` (4 samples), while `POTEB3` remains in development.

### 4. `SNURF` (3 samples)
- **Phase 4.2 Finding**: Did not leak into the test set in Phase 4.2 duplicate splits.
- **Reconciliation Decision**: Assigned to the **Sealed Test Holdout**.
- **Justification**: `SNURF` (NCBI GeneID 8926, records 115..117, bicistronic transcript locus with `SNRPN` but distinct NCBI gene symbol annotation) contains exactly 3 samples. Selecting `SNURF` brings the 8-gene holdout partition to exactly 60 samples:
  $$\text{Holdout} = 17 (\text{TUBGCP5}) + 10 (\text{CYFIP1}) + 9 (\text{ATP10A}) + 7 (\text{GABRA5}) + 6 (\text{GABRB3}) + 4 (\text{POTEB}) + 4 (\text{POTEB2}) + 3 (\text{SNURF}) = 60$$

---

## 4. Comprehensive 3-Fold Grouped Cross-Validation Audit

### 4.1 Cohort Re-Partitioning and the Phase 4.3 Count Reconciliation
In Phase 4.3 preliminary feasibility modeling (`cv_feasibility.json`), a theoretical 3-fold split was computed across the *entire* historical dataset ($N=1,200$, Cervical $N=300$). That preliminary bin-packing yielded Cervical validation fold counts of **112, 99, 89**.

However, under the actual finalized evaluation protocol:
1. 240 samples (60 per class) were permanently removed to form the Tier 2 Sealed Test Vault.
2. The remaining **Tier 1 Development Cohort contains exactly 960 samples** (240 Cervical, 240 Colorectal, 240 Endometrial, 240 Liver).
3. The 3-fold cross-validation must be executed exclusively on the 240 Cervical Development samples across the 23 remaining gene families.

When greedy bin-packing is applied to the 240 Cervical Development samples:
- `SNRPN` (112 samples) is the single largest group. It cannot be split, so it occupies an entire fold by itself.
- Remaining Cervical samples: $240 - 112 = 128$.
- Splitting the remaining 128 samples across 2 folds yields exactly **64 samples per fold** ($64 + 64 = 128$).
- Bacterial classes (Colorectal, Endometrial, Liver) partition uniformly into $240 / 3 = 80$ samples per fold.

### 4.2 Verified Per-Fold Partition Distribution
The exact sample distributions across all 3 validation folds on the 960-sample Development Cohort are:

| Metric | Fold 0 (Val) | Fold 1 (Val) | Fold 2 (Val) |
|---|---|---|---|
| **Cervical Cancer Samples** | **112** | **64** | **64** |
| Colorectal Cancer Samples | 80 | 80 | 80 |
| Endometrial Cancer Samples | 80 | 80 | 80 |
| Liver Cancer Samples | 80 | 80 | 80 |
| **Total Validation Samples** | **352** | **304** | **304** |
| **Total Training Samples** | **608** | **656** | **656** |
| **Cervical Gene Families in Val** | **1** (`SNRPN`) | **10** (`NIPA2` 51 + 9 small) | **12** (`UBE3A` 48 + 11 small) |
| Group Leakage (Train vs Val) | **0** | **0** | **0** |
| Sequence Leakage (Train vs Val)| **0** | **0** | **0** |
| Empty Classes in Train or Val | **0** | **0** | **0** |

### 4.3 Biological & Statistical Limitations of Single-Gene Validation Folds
The partition structure reveals an unavoidable mathematical constraint arising from the distribution of human transcript isoforms:
1. **Fold 0 Cervical Validation is Single-Gene**: Because `SNRPN` contains 112 samples (46.7% of the Cervical development cohort), Fold 0 validates Cervical classification performance **exclusively against transcripts of the `SNRPN` gene family**. If a model fails to generalize to `SNRPN` sequence motifs, Fold 0 Cervical recall will drop catastrophically, while Folds 1 and 2 may show near-perfect recall.
2. **Asymmetric Training Set Size**: Model training on Fold 0 uses 608 samples (only 128 Cervical samples from 22 gene families), whereas training on Folds 1 and 2 uses 656 samples (176 Cervical samples including `SNRPN`).
3. **Statistical Consequence**: K-Fold cross-validation variance across folds is not merely random sampling noise; it reflects structural biological heterogeneity between specific human gene families. Evaluation metrics in Phase 5 must report fold-level metrics alongside the aggregate mean to ensure transparency.

---

## 5. Audit of Secondary 10-Iteration GroupShuffleSplit

To complement 3-fold cross-validation, a secondary Monte Carlo Group-Stratified ShuffleSplit was audited (`generate_group_shufflesplit_folds` with 10 iterations, 25% validation holdout per iteration).

### 5.1 Verification Findings
1. **Group Disjointness**: In all 10 iterations, zero gene families cross the train/validation boundary.
2. **Zero Sequence Leakage**: No identical sequence appears in both partitions in any iteration.
3. **Class Representation**: Every class is represented in both partitions across all 10 iterations.

### 5.2 Non-Independence of Resamplings
A critical statistical limitation must be emphasized:
> [!WARNING] Non-Independence of Repeated Resamplings
> Repeated ShuffleSplit iterations do NOT constitute independent experimental observations. All 10 iterations sample from the exact same 23 Cervical gene families and fixed bacterial sequences. A gene assigned to validation in iteration 1 will appear in training in iteration 2. Consequently, fold metrics are highly correlated.

### 5.3 Aggregation Standards for Reporting
To prevent invalid statistical inferences during model selection:
1. **Median and Interquartile Range (IQR)**: Experiment results across the 10 iterations must be reported as $\text{Median} \pm \text{IQR}$, rather than standard error of the mean (SEM).
2. **Prohibition of Parametric T-Tests**: Analysts must **never** perform paired Student's $t$-tests treating the 10 resamplings as 10 independent degrees of freedom, as this severely underestimates variance and produces spurious $p$-values.

---

## 6. Sealed Vault Safeguards: Procedural vs Technical Protection

The Tier 2 Sealed Test Holdout ($N=240$) is protected by `SealedVault` (`src/dna_cancer/data/vault.py`).

### 6.1 Architecture of Safeguards
1. **Physical File Separation**:
   - Development cohort is exported to `data/processed/development_cohort.csv` ($N=960$). Training, feature extraction, cross-validation, and hyperparameter search scripts read *exclusively* from this path.
   - Sealed test holdout is stored in `artifacts/vault/test_holdout_sealed.json` ($N=240$).
2. **Procedural Software Lock**:
   - `artifacts/vault/test_sealed_manifest.json` records SHA-256 checksums (`eb14d47...` for reconstructed dataset, `7c35fc4...` for Dev, `430a2e6...` for Holdout).
   - Calling `vault.get_vault_status()` confirms `is_locked: True` and `status: "LOCKED"`.
3. **One-Shot Programmatic Authorization**:
   - Calling `vault.unseal(token=...)` requires the exact authorization token `"PHASE_5_FINAL_UNSEAL"`.
   - The unseal event is permanently logged with timestamp and reason.
   - Any secondary call to `unseal()` raises a `PermissionError("Vault has already been unsealed and cannot be unsealed again.")`.

### 6.2 Realistic Boundary of Protection
It is vital to distinguish between software safeguards and absolute system-level barriers:
- **What it provides**: A robust procedural fence that prevents accidental automated leakage. Training scripts, grid search loops, and feature extractors cannot accidentally ingest holdout samples because the holdout is not in the data directory and is stored in a separate sealed structure.
- **What it does NOT provide**: It does not provide cryptographic hardware encryption, kernel-level access control, or adversarial protection against an engineer intentionally inspecting `test_holdout_sealed.json`. The integrity of the sealed vault relies upon professional adherence to the evaluation protocol.

---

## 7. Final Protocol Recommendation for Phase 5

Based on forensic verification, the binding evaluation protocol for Phase 5 model optimization is:

1. **Development & Tuning (Tier 1)**:
   - Primary metric: Mean Macro F1 across **Group-Balanced 3-Fold Cross-Validation** on `data/processed/development_cohort.csv` ($N=960$).
   - Report per-fold metrics to track `SNRPN` sensitivity.
   - Secondary sanity check: 10-Iteration GroupShuffleSplit (reported as Median / IQR).
2. **Architecture & Feature Freezing**:
   - Feature extractors ($k$-mer vocabulary, tokenizers, scaling parameters) must be fitted strictly inside each cross-validation training fold.
   - All hyperparameter decisions ($C$, tree depth, learning rate, network architecture) must be finalized prior to vault unsealing.
3. **One-Shot Final Generalization Evaluation (Tier 2)**:
   - Upon freezing the single best-performing model architecture, call `vault.unseal(token="PHASE_5_FINAL_UNSEAL")` exactly once.
   - Evaluate model on the 240 sealed test samples.
   - Report Final Accuracy and Class Recalls with **95% Wilson Score Confidence Intervals**, and Macro F1 with **1,000-iteration Bootstrap Confidence Intervals**.
   - No subsequent model tuning or hyperparameter adjustments are permitted after unsealing.

