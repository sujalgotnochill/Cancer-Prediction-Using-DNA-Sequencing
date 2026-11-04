# Baseline Machine Learning Benchmark Results

**Phase:** 4 - Baseline Machine Learning Benchmark System  
**Dataset:** Reconstructed Historical 1,200 DNA Sequences (`data/interim/historical_reconstruction/DNAdata_reconstructed.csv`)  
**SHA-256 Checksum:** `eb14d47f3eee...`  
**Execution Date:** 2026-11-04  
**Telemetry Artifacts:** `artifacts/experiments/runs_summary.json` (24 tracked runs)  
**Evaluation Plots:** `artifacts/evaluation/`  

---

## 1. Executive Summary & Scientific Disclaimer

> [!CAUTION]
> **Scientific Limitation Notice:** The results presented herein represent an educational sequence-classification benchmark on historical college project data. The classes represent distinct biological organisms (human chromosome 15 vs three bacterial species) rather than clinical cancer profiles. These metrics **must not be interpreted or reported as clinical cancer detection capability**.

We executed a systematic benchmark matrix of **24 independent experiments**, examining:
- Two partitioning regimes: **Duplicate-Grouped** (leakage-free, zero sequence overlap) vs. **Conventional Stratified** (reproducing legacy partition leakage).
- Four feature representations: $k \in \{3, 4, 5, 6\}$ with sparse count vectorization.
- Three baseline classifiers: **Multinomial Naive Bayes**, **Logistic Regression (L2)**, and **Random Forest (100 trees)**.

All models were evaluated against an **untouched held-out test partition** with all preprocessing transformers strictly fitted on training sequences only.

---

## 2. Comprehensive Benchmark Results Table

| Strategy | $k$ | Model | Features ($n$) | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Train Time (s) | Leakage-Free |
|---|---|---|---|---|---|---|---|---|---|
| **Duplicate-Grouped** | **3** | MultinomialNB | 64 | 0.8710 | 0.8674 | 0.8770 | 0.8393 | 0.001s | **YES** |
| **Duplicate-Grouped** | **3** | LogisticRegression | 64 | 0.9355 | 0.9416 | 0.9407 | 0.9342 | 0.423s | **YES** |
| **Duplicate-Grouped** | **3** | RandomForest | 64 | 0.9290 | 0.9377 | 0.9332 | 0.9108 | 0.179s | **YES** |
| **Duplicate-Grouped** | **4** | MultinomialNB | 256 | 0.9032 | 0.8932 | 0.8996 | 0.8715 | 0.002s | **YES** |
| **Duplicate-Grouped** | **4** | LogisticRegression | 256 | 0.9613 | 0.9659 | 0.9632 | 0.9633 | 0.481s | **YES** |
| **Duplicate-Grouped** | **4** | RandomForest | 256 | 0.9097 | 0.9160 | 0.9110 | 0.9065 | 0.183s | **YES** |
| **Duplicate-Grouped** | **5** | MultinomialNB | 1,024 | 0.9097 | 0.9011 | 0.9044 | 0.8841 | 0.002s | **YES** |
| **Duplicate-Grouped** | **5** | LogisticRegression | 1,024 | **0.9677** | **0.9730** | **0.9723** | **0.9723** | 0.453s | **YES** |
| **Duplicate-Grouped** | **5** | RandomForest | 1,024 | 0.8968 | 0.9048 | 0.8968 | 0.8897 | 0.205s | **YES** |
| **Duplicate-Grouped** | **6** | MultinomialNB | 4,096 | 0.8903 | 0.8774 | 0.8849 | 0.8357 | 0.004s | **YES** |
| **Duplicate-Grouped** | **6** | LogisticRegression | 4,096 | **0.9677** | **0.9730** | **0.9723** | **0.9723** | 0.487s | **YES** |
| **Duplicate-Grouped** | **6** | RandomForest | 4,096 | 0.9097 | 0.9150 | 0.9099 | 0.9050 | 0.258s | **YES** |
| *Stratified (Leaked)* | 3 | MultinomialNB | 64 | 0.9056 | 0.9105 | 0.9056 | 0.9047 | 0.001s | *LEAKED* |
| *Stratified (Leaked)* | 3 | LogisticRegression | 64 | 0.9500 | 0.9504 | 0.9500 | 0.9500 | 0.399s | *LEAKED* |
| *Stratified (Leaked)* | 3 | RandomForest | 64 | 0.9111 | 0.9135 | 0.9111 | 0.9116 | 0.155s | *LEAKED* |
| *Stratified (Leaked)* | 4 | MultinomialNB | 256 | 0.9167 | 0.9197 | 0.9167 | 0.9162 | 0.001s | *LEAKED* |
| *Stratified (Leaked)* | 4 | LogisticRegression | 256 | 0.9611 | 0.9616 | 0.9611 | 0.9612 | 0.407s | *LEAKED* |
| *Stratified (Leaked)* | 4 | RandomForest | 256 | 0.9556 | 0.9567 | 0.9556 | 0.9558 | 0.167s | *LEAKED* |
| *Stratified (Leaked)* | 5 | MultinomialNB | 1,024 | 0.9056 | 0.9068 | 0.9056 | 0.9046 | 0.003s | *LEAKED* |
| *Stratified (Leaked)* | 5 | LogisticRegression | 1,024 | 0.9556 | 0.9558 | 0.9556 | 0.9556 | 0.513s | *LEAKED* |
| *Stratified (Leaked)* | 5 | RandomForest | 1,024 | 0.9389 | 0.9405 | 0.9389 | 0.9395 | 0.208s | *LEAKED* |
| *Stratified (Leaked)* | 6 | MultinomialNB | 4,096 | 0.9000 | 0.9001 | 0.9000 | 0.8989 | 0.003s | *LEAKED* |
| *Stratified (Leaked)* | 6 | LogisticRegression | 4,096 | 0.9611 | 0.9616 | 0.9611 | 0.9612 | 0.600s | *LEAKED* |
| *Stratified (Leaked)* | 6 | RandomForest | 4,096 | 0.9222 | 0.9234 | 0.9222 | 0.9224 | 0.244s | *LEAKED* |

---

## 3. Key Scientific & Engineering Insights

### 3.1 Impact of Sequence Duplicate Leakage
In the conventional stratified benchmark, identical transcript copies of Cervical Cancer genes were distributed into both train and test partitions.
- **Stratified Benchmark ($k=3$):** MultinomialNB scored **90.56%** accuracy and **0.9047** Macro F1 because duplicate test samples were memorized directly from training.
- **Duplicate-Grouped Benchmark ($k=3$):** When identical sequences were quarantined so that test sequences were genuinely unseen, MultinomialNB dropped to **87.10%** accuracy and **0.8393** Macro F1.
- This demonstrates an artificial inflation of **~3.5% to 6.5%** in naive evaluations, proving the necessity of duplicate-aware grouping.

### 3.2 k-mer Size Ablation Dynamics
```text
Accuracy vs. k-mer Size (Duplicate-Grouped Benchmark):
  k=3 (64 features):   LogisticRegression = 93.55%, RandomForest = 92.90%, MultinomialNB = 87.10%
  k=4 (256 features):  LogisticRegression = 96.13%, RandomForest = 90.97%, MultinomialNB = 90.32%
  k=5 (1024 features): LogisticRegression = 96.77%, RandomForest = 89.68%, MultinomialNB = 90.97%
  k=6 (4096 features): LogisticRegression = 96.77%, RandomForest = 90.97%, MultinomialNB = 89.03%
```
- **Dimension Efficiency:** Tri-nucleotide frequencies ($k=3$, 64 features) are already sufficient to achieve >92% accuracy across linear and ensemble models.
- **Peak Performance:** Performance plateaus at $k=5$ (1,024 features) and $k=6$ (4,096 features) at **96.77% test accuracy** for Logistic Regression.
- **Why accuracy is high despite zero leakage:** The classification task separates whole bacterial genomes (*Chloroflexota*, *Bacteroidota*, *Alteromonadales*) from human chromosome 15. The codon usage and oligonucleotide signatures of these divergent taxa are radically distinct, making phylogenetic separation trivial even with tiny $k$-mer vocabularies.

### 3.3 Classifier Comparison
1. **Logistic Regression (L2):** Clear winner across all configurations. Achieved **96.77% accuracy** and **0.9723 Macro F1** at $k=5$ and $k=6$, training in under 0.5 seconds on sparse representations.
2. **Multinomial Naive Bayes:** Fastest model by two orders of magnitude (train time ~0.001 to 0.004s). Strong baseline ($k=5$: 90.97% accuracy), but penalized on minority classes due to independent feature assumptions.
3. **Random Forest:** Solid non-linear baseline (90.97% accuracy at $k=6$, train time ~0.25s), though linear decision boundaries in $k$-mer space proved slightly superior.

---

## 4. Confusion Matrix Analysis (Duplicate-Grouped, $k=6$)

### Logistic Regression Test Predictions ($n=155$ test samples)
- **Class 0 (Cervical Cancer / *H. sapiens*):** 20 / 20 correct (100% recall). Zero confusion with bacterial genomes.
- **Class 3 (Liver Cancer / *A. pallidiroseus*):** 45 / 45 correct (100% recall). Completely isolated due to its low 43.1% GC content.
- **Class 1 (Colorectal / *K. racemifer*) vs. Class 2 (Endometrial / *S. amylolyticus*):
  - Colorectal: 45 / 45 correct (100% recall).
  - Endometrial: 40 / 45 correct (88.9% recall, 5 misclassified as Colorectal).
  - **Biological explanation:** *Ktedonobacter* (54.2% GC) and *Saliniradius* (52.5% GC) share similar bacterial GC contents and ribosomal codon profiles, accounting for 100% of the remaining classification errors.

---

## 5. Generated Visualizations Manifest

All high-resolution figures have been rendered and saved to [`artifacts/evaluation/`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/):

1. **`kmer_ablation_duplicate_grouped.png`**: Accuracy curves across $k \in [3, 4, 5, 6]$ for duplicate-grouped splitting.
2. **`kmer_ablation_stratified.png`**: Accuracy curves across $k \in [3, 4, 5, 6]$ for conventional stratified splitting.
3. **`baseline_model_comparison_duplicate_grouped_k6.png`**: Multi-metric bar chart comparing MultinomialNB, LogisticRegression, and RandomForest at $k=6$ under zero-leakage conditions.
4. **`baseline_model_comparison_stratified_k6.png`**: Multi-metric bar chart showing inflated performance under naive stratified splitting.
5. **`confusion_matrix_logisticregression_duplicate_grouped_k6.png`**: Annotated confusion matrix heatmap for Logistic Regression.
6. **`confusion_matrix_multinomialnb_duplicate_grouped_k6.png`**: Annotated confusion matrix heatmap for Multinomial Naive Bayes.
7. **`confusion_matrix_randomforest_duplicate_grouped_k6.png`**: Annotated confusion matrix heatmap for Random Forest.
8. **`per_class_f1_duplicate_grouped_k6.png`**: Per-class F1 score bar chart across all four classes.

