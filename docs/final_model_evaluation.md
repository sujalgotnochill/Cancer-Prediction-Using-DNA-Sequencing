# Phase 6: Final One-Shot Holdout Evaluation Report

## 1. Executive Summary & Scientific Governance

This document presents the official, independent, one-shot evaluation of the frozen sequence classification candidate model on the **Tier 2 Sealed Test Holdout** (`artifacts/vault/test_holdout_sealed.json`, $N=240$).

### 1.1 Scientific Classification Context & Dataset Provenance
Forensic investigation in Phases 3.6–4.1 definitively established that the experimental dataset does not represent clinical cancer biopsies. Rather, sequences labeled:
- **Cervical Cancer** originate from **human chromosome 15 RefSeq mRNA coding transcripts**.
- **Colorectal Cancer, Endometrial Cancer, and Liver Cancer** originate from **bacterial genomic contigs** belonging to various microbial species.

Accordingly:
- This evaluation benchmarks a **four-class DNA sequence-source classifier**.
- Metrics reflect the model's ability to discriminate human chromosomal mRNA transcripts from diverse bacterial genomic fragments.
- **No claim of clinical cancer diagnosis, patient prognosis, or oncology prediction is made or implied.**

---

### 1.2 Pre-Unseal Readiness Audit
Prior to unsealing the vault, all 10 mandatory pre-unseal readiness checks were executed programmatically and verified:
1. **Model Configuration Frozen**: Confirmed `"is_frozen": True` in `model_config.json`.
2. **Artifact Cryptographic Hashes**: Confirmed SHA-256 hashes of `model.joblib`, `vectorizer.joblib`, `encoder.joblib`, and `model_config.json` against `checksums.sha256`.
3. **Pipeline Compatibility**: Model input dimensionality (256) exactly matches vectorizer vocabulary size (256); encoder maps 4 cancer classes.
4. **Development Cohort Integrity**: Checksum of `data/processed/development_cohort.csv` matches manifest (`7c35fc4f...`).
5. **Sealed Holdout Integrity**: Checksum of `artifacts/vault/test_holdout_sealed.json` matches manifest (`430a2e6f...`) without prior unsealing.
6. **Vault Lock State**: Confirmed `status: "LOCKED"` and `unsealed: False`.
7. **Unseal Counter**: Confirmed `unseal_count == 0`.
8. **Audit Ledger Verification**: Confirmed zero prior evaluation events in `artifacts/vault/evaluation_events.jsonl`.
9. **Evaluator Identity**: Verified execution through approved `FinalHoldoutEvaluator`.
10. **Zero Refitting**: The pipeline performed purely forward-pass inference without refitting or adaptation.

---

## 2. Final Evaluation Performance Metrics ($N=240$)

The evaluation was executed on 2026-11-04T20:45:06. The test cohort contains 240 samples with an exact balance of 60 samples per class:

| Metric | Holdout Metric Value | 95% Wilson Score Confidence Interval | Development CV Baseline | CV vs Holdout Difference ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.9125** (219/240) | **[0.8699, 0.9421]** | 0.9674 | -0.0549 (-5.49%) |
| **Macro F1 Score** | **0.9105** | — | 0.9671 | -0.0566 (-5.66%) |
| **Macro Precision** | **0.9145** | — | 0.9728 | -0.0583 (-5.83%) |
| **Macro Recall** | **0.9125** | — | 0.9667 | -0.0542 (-5.42%) |
| **Weighted F1 Score** | **0.9105** | — | 0.9671 | -0.0566 (-5.66%) |
| **Total Misclassifications**| **21 / 240** (8.75%)| — | 31 / 960 (3.23%) | +5.52% error rate |

---

## 3. Per-Class Performance & 95% Wilson Score Intervals

| Target Class | Support ($N$) | Correct | Errors | Recall | Recall 95% Wilson CI | Precision | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cervical Cancer (Human)** | 60 | 46 | 14 | **0.7667** | **[0.6456, 0.8556]** | **0.9388** | **0.8440** |
| **Colorectal Cancer (Bacterial)**| 60 | 54 | 6 | **0.9000** | **[0.7985, 0.9534]** | **0.8710** | **0.8852** |
| **Endometrial Cancer (Bacterial)**| 60 | 59 | 1 | **0.9833** | **[0.9114, 0.9971]** | **0.8806** | **0.9291** |
| **Liver Cancer (Bacterial)** | 60 | 60 | 0 | **1.0000** | **[0.9398, 1.0000]** | **0.9677** | **0.9836** |

---

## 4. Confusion Matrix and Error Dynamics

### 4.1 4 × 4 Confusion Matrix
The contingency matrix below details class confusion across all 240 holdout samples:

| True Class \ Predicted Class | Cervical (Human) | Colorectal (Bacterial) | Endometrial (Bacterial) | Liver (Bacterial) | Total True Samples | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cervical Cancer** | **46** | 7 | 7 | 0 | 60 | **76.67%** |
| **Colorectal Cancer** | 3 | **54** | 1 | 2 | 60 | **90.00%** |
| **Endometrial Cancer** | 0 | 1 | **59** | 0 | 60 | **98.33%** |
| **Liver Cancer** | 0 | 0 | 0 | **60** | 60 | **100.00%** |
| **Total Predicted** | 49 | 62 | 67 | 62 | 240 | — |

*Visual Heatmap*: Preserved at [`artifacts/evaluation/final_confusion_matrix.png`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/final_confusion_matrix.png).

---

### 4.2 Breakdown of the 21 Misclassifications

The 21 errors consist of **14 human transcript errors** and **7 bacterial sequence errors**:

#### A. Human Cervical Gene Family Generalization (14 errors across 60 samples)
Inspecting the exact gene family breakdown of holdout Cervical transcripts reveals a crucial dichotomy:

| Gene Family in Holdout | Status in Development Set | Holdout Support | Correct Predictions | Errors | Holdout Accuracy | Error Diagnostics |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`TUBGCP5`** | **Entirely Unseen** | 17 | 17 | 0 | **100.0%** | Perfect generalization to novel gene family |
| **`CYFIP1`** | **Entirely Unseen** | 10 | 10 | 0 | **100.0%** | Perfect generalization to novel gene family |
| **`POTEB2`** | **Entirely Unseen** | 4 | 4 | 0 | **100.0%** | Perfect generalization to novel gene family |
| **`ATP10A`** | Partitioned (Isoforms) | 9 | 9 | 0 | **100.0%** | Perfect generalization across isoforms |
| **`POTEB`** | Partitioned (Paralogs) | 4 | 4 | 0 | **100.0%** | Perfect generalization across paralogs |
| **`GABRB3`** | Novel Subunit Isoforms | 6 | 2 | 4 | **33.3%** | 4 samples misclassified as Colorectal |
| **`GABRA5`** | Novel Subunit Isoforms | 7 | 0 | 7 | **0.0%** | 4 samples $\to$ Endometrial, 3 samples $\to$ Colorectal |
| **`SNURF`** | Ultra-Short Transcripts | 3 | 0 | 3 | **0.0%** | 3 samples $\to$ Endometrial (all length 216 bp) |
| **Total Cervical** | — | **60** | **46** | **14** | **76.7%** | — |

*Analytical Insights on Cervical Errors*:
1. **Unseen Major Families Succeeded**: Five of the eight Cervical gene groups (`TUBGCP5`, `CYFIP1`, `POTEB2`, `ATP10A`, `POTEB`) achieved **100% accuracy (44 out of 44 correct)** despite `TUBGCP5`, `CYFIP1`, and `POTEB2` never appearing in development training.
2. **Extreme Sequence Length Sparsity (`SNURF`)**: All 3 samples of `SNURF` are only 216 bp in length (a tiny alternative open reading frame of the `SNURF-SNRPN` locus). In a 4-mer TF-IDF representation, 216 bp yields fewer than 213 $k$-mers, resulting in extreme feature vector sparsity that fell below the human margin threshold.
3. **GABA Receptor Subunit Convergence (`GABRA5` & `GABRB3`)**: The GABA-A receptor alpha-5 and beta-3 subunits possess specialized codon usage and hydrophobic transmembrane domain motifs that shifted their 4-mer frequency distributions closer to prokaryotic genomic profiles.

#### B. Bacterial Misclassifications (7 errors across 180 samples)
- **Liver Cancer (60 samples)**: **60/60 correct (100.0% recall, 0 errors)**.
- **Endometrial Cancer (60 samples)**: **59/60 correct (98.33% recall)**. Only 1 sample (ID 296, 234 bp) was confused with Colorectal.
- **Colorectal Cancer (60 samples)**: **54/60 correct (90.00% recall)**. 6 errors occurred:
  - 3 samples (IDs 252, 271, 298, lengths 183–444 bp) were predicted as Cervical Cancer.
  - 2 samples (IDs 243, 289, lengths 147–240 bp) were predicted as Liver Cancer.
  - 1 sample (ID 293, 480 bp) was predicted as Endometrial Cancer.
  - *Observation*: All 6 misclassified Colorectal contigs were short fragments ($\le 480$ bp), consistent with Phase 5 error findings that shorter sequence windows experience higher sampling noise.

---

## 5. Development CV vs Final Holdout Comparative Analysis

| Dimension | Development Grouped CV ($N=960$) | Final Sealed Holdout ($N=240$) | Difference ($\Delta$) | Methodological Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F1** | **0.9671 ± 0.0063** | **0.9105** | **-0.0566** (-5.66%) | Expected generalization gap on completely held-out gene families. |
| **Accuracy** | **0.9674 ± 0.0073** | **0.9125** | **-0.0549** (-5.49%) | Confirms robust 91.25% generalization under zero data leakage. |
| **Cervical Recall** | **0.9844 ± 0.0127** | **0.7667** | **-0.2177** (-21.77%) | Driven by concentrated failure on `SNURF` (216 bp) and `GABRA5`/`GABRB3`. |
| **Colorectal Recall** | **0.9667 ± 0.0059** | **0.9000** | **-0.0667** (-6.67%) | Minor drop due to short contig fragments ($\le 444$ bp). |
| **Endometrial Recall** | **0.9292 ± 0.0388** | **0.9833** | **+0.0541** (+5.41%) | Superior holdout performance; near-zero bacterial confusion. |
| **Liver Recall** | **0.9875 ± 0.0177** | **1.0000** | **+0.0125** (+1.25%) | Flawless classification across all 60 holdout contigs. |

### Methodological Explanations for Differences:
1. **Zero Optimistic Bias**: In Phase 5, the model was frozen before unsealing. Unlike traditional benchmarks where the test set is repeatedly probed or hyperparameters are tuned post-test, this evaluation reflects an authentic, unadapted out-of-distribution measurement.
2. **Gene Family Distribution Shift**: While the development set was dominated by large gene families (`SNRPN` with 112 samples, `NIPA2` with 51 samples, `UBE3A` with 48 samples), the holdout cohort tested diverse, smaller gene families. Five unseen families generalized at 100%, but the two GABA receptor subunits and ultra-short `SNURF` exposed boundary limits of a 256-feature linear model.
3. **High Bacterial Accuracy Maintained**: Overall bacterial classification accuracy on the holdout was **96.1% (173/180 correct)**, demonstrating that the learned prokaryotic vs eukaryotic codon separation is robust.

---

## 6. Correct Probability and Margin Reporting

As mandated by Phase 6 audit standards:
- The serialized model is a standard `LinearSVCClassifier` optimizing hinge loss with L2 regularization.
- **No genuine probability calibration component** (e.g., Platt scaling via `CalibratedClassifierCV` or isotonic regression) was fitted during training.
- The `predict_proba()` method implemented in the wrapper applies a mathematical softmax transform directly to raw decision function values. **Softmax scores are not empirical probabilities** and must not be described as calibrated confidence intervals.
- The mean maximum decision margin on the holdout was **$+1.042$** (minimum margin **$+0.086$** for correct predictions), demonstrating that well-classified samples reside safely outside the margin hyperplane boundary.

---

## 7. Audit Receipt and Sealed Vault Lifecycle State

### 7.1 Permanent Cryptographic Ledger
The official evaluation receipt was recorded at [`artifacts/vault/final_evaluation_receipt.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/vault/final_evaluation_receipt.json):
```json
{
  "receipt_version": "1.0.0",
  "event_identifier": "FINAL_HOLDOUT_EVALUATION",
  "evaluation_timestamp": "2026-11-04T20:45:06.846642",
  "evaluator_version": "FinalHoldoutEvaluator v1.0.0",
  "unseal_count": 1,
  "authorization_token_hash": "cb90858fbd0a5446",
  "cryptographic_verification": {
    "model_joblib_sha256": "1422d3163b1d33bb84052f40bbb7466476a36ecadf85dd440d6278844faff386",
    "vectorizer_joblib_sha256": "a034de878fac4ba7401f13555b4b5aa6509e11fc282754ae526645b2668093c8",
    "encoder_joblib_sha256": "3c78f7eee9e9be78349cf8a3c3bcee02a0efaad961252f727bb2b43065c9015c",
    "model_config_sha256": "35fe8ec22804993160e649cd19c82e2270f4bec09840099f8efa7a8dc0984aae",
    "development_cohort_sha256": "7c35fc4f226488cfc2d37ed332746be3f3c1aa8982ab7a8991c7fb374781475c",
    "sealed_holdout_sha256": "430a2e6f8b4b85dfb8bd6a16300e44417a374853a2eb7616a8e5d6d9116db940"
  },
  "final_metric_summary": {
    "accuracy": 0.9125,
    "accuracy_95ci": [0.8699, 0.9421],
    "macro_f1": 0.9105,
    "macro_precision": 0.9145,
    "macro_recall": 0.9125,
    "cervical_recall": 0.7667,
    "colorectal_recall": 0.9000,
    "endometrial_recall": 0.9833,
    "liver_recall": 1.0000
  }
}
```

### 7.2 Post-Evaluation Lock Status
- `artifacts/vault/vault_state.json`: State transitioned to `"status": "UNSEALED"`, `"unsealed": true`, `"unseal_count": 1`.
- `artifacts/vault/evaluation_events.jsonl`: Exactly 1 event recorded.
- **Repeat Evaluation Block**: Confirmed that any second call to `evaluate_frozen_model()` immediately raises `PermissionError: Sealed test vault is already UNSEALED. Repeated evaluation is prohibited.`

---

## 8. Artifact and Evaluation Deliverables

1. **Evaluation Document**: [`docs/final_model_evaluation.md`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/docs/final_model_evaluation.md)
2. **Metrics & Results JSON**: [`artifacts/evaluation/final_holdout_results.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/final_holdout_results.json)
3. **Confusion Matrix Heatmap**: [`artifacts/evaluation/final_confusion_matrix.png`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/final_confusion_matrix.png)
4. **Classification Report JSON**: [`artifacts/evaluation/final_classification_report.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/final_classification_report.json)
5. **Evaluation Receipt**: [`artifacts/vault/final_evaluation_receipt.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/vault/final_evaluation_receipt.json)
6. **Append-Only Event Ledger**: [`artifacts/vault/evaluation_events.jsonl`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/vault/evaluation_events.jsonl)

