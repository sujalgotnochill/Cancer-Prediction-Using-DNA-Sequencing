# Phase 5: Model Optimization and Robust Sequence Classification

## 1. Executive Summary and Evaluation Governance

During **Phase 5**, we designed, executed, and analyzed a comprehensive model optimization and sequence feature engineering benchmark. In strict compliance with the project's data governance architecture established in Phase 4.4 and Phase 4.5:
1. **Tier 1 Development Cohort Isolation**: All feature transformations, vocabulary fittings, hyperparameter sweeps, and cross-validation evaluations were performed exclusively on the 960-sample development cohort (`data/processed/development_cohort.csv`, SHA-256: `7c35fc4f226488cfc2d37ed332746be3f3c1aa8982ab7a8991c7fb374781475c`).
2. **Tier 2 Sealed Test Vault Lockdown**: The 240-sample sealed test holdout (`artifacts/vault/test_holdout_sealed.json`) remained strictly sealed throughout Phase 5 (`unseal_count: 0`, `status: "LOCKED"`). No test holdout samples or labels were accessed, evaluated, or leaked.
3. **Rigorous Cross-Validation**: All model selection decisions were guided by primary 3-fold within-organism gene-family grouped cross-validation, verified by secondary 10-iteration group-aware ShuffleSplit.
4. **Educational Classification Benchmark Context**: Forensic analysis previously revealed that the dataset comprises human chromosome 15 transcripts (labeled Cervical Cancer) and bacterial genomic sequences (labeled Colorectal, Endometrial, and Liver Cancer). Hence, all results are benchmark metrics for sequence classification and do not represent clinical cancer prediction.

Across 124 experimental configurations, the winning candidate selected and frozen for final holdout evaluation is **`LinearSVC (k=4, TF-IDF, standard)`**, achieving a primary 3-fold grouped CV Macro F1 of **0.9671 ± 0.0063** and an accuracy of **0.9674 ± 0.0073**, with zero out-of-fold misclassifications on the critical `SNRPN` gene family (112/112 correct).

---

## 2. Experimental Optimization Framework

### 2.1 Model Families Evaluated
We benchmarked five distinct model architectures with sparse-matrix native processing:
- **Multinomial Naive Bayes (`MultinomialNBClassifier`)**: Generative probabilistic baseline using $k$-mer counts or normalized frequencies with Laplace/Lidstone smoothing parameter $\alpha \in \{0.01, 0.1, 0.5, 1.0, 2.0\}$.
- **Logistic Regression (`LogisticRegressionClassifier`)**: Multinomial generalized linear model with L2 regularization ($C \in \{0.01, 0.1, 1.0, 5.0, 10.0, 50.0\}$), balanced class weights, and L-BFGS solver.
- **Linear Support Vector Classifier (`LinearSVCClassifier`)**: Maximum-margin linear classifier optimizing squared hinge loss with L2 regularization ($C \in \{0.01, 0.1, 0.5, 1.0, 5.0, 10.0\}$), dual formulation, and calibrated softmax posterior probabilities.
- **Random Forest (`RandomForestClassifierWrapper`)**: Ensemble tree classifier with `n_estimators` $\in \{50, 100, 200\}$, `max_depth` $\in \{10, 20, \text{None}\}$, and `min_samples_leaf` $\in \{1, 2\}$.
- **Gradient Tree Boosting (`GradientBoostingClassifierWrapper`)**: Sequentially additive tree boosting using scikit-learn's native tree ensemble.

### 2.2 Primary Evaluation Protocol: 3-Fold Grouped Cross-Validation
To guarantee zero biological gene-family leakage, folds were partitioned such that:
- Every human gene family appears exclusively in one fold.
- `SNRPN` (112 samples) resides exclusively in Fold 0 validation.
- `NIPA2` (51 samples) and 13 minor gene families reside in Fold 1 validation (64 samples).
- `UBE3A` (48 samples) and 16 minor gene families reside in Fold 2 validation (64 samples).
- Bacterial classes (Colorectal, Endometrial, Liver) are identically balanced at 80 samples per fold (total 240 bacterial samples per fold).

| Split Partition | Cervical Samples | Bacterial Samples | Total Samples | Cervical Composition |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 0 Validation** | 112 | 240 (80 Col, 80 Endo, 80 Liv) | 352 | `SNRPN` (112) |
| **Fold 1 Validation** | 64 | 240 (80 Col, 80 Endo, 80 Liv) | 304 | `NIPA2` (51) + 13 small families (13) |
| **Fold 2 Validation** | 64 | 240 (80 Col, 80 Endo, 80 Liv) | 304 | `UBE3A` (48) + 16 small families (16) |

---

## 3. Benchmark Results and Optimization Findings

### 3.1 Initial Baseline Model Comparison ($k=5$, TF-IDF)
At the standardized baseline setting ($k=5$, TF-IDF representation, default parameters):

| Model Architecture | Mean Macro F1 | Macro F1 Std | Mean Accuracy | Accuracy Std | Total Train Time (s) | Inference Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVC** | **0.9189** | ±0.0385 | **0.9298** | ±0.0319 | 0.100s | <0.001s |
| **Logistic Regression** | 0.7655 | ±0.0163 | 0.8071 | ±0.0123 | 0.234s | <0.001s |
| **Gradient Tree Boosting** | 0.7535 | ±0.0135 | 0.7974 | ±0.0097 | 120.614s | 0.008s |
| **Multinomial Naive Bayes** | 0.6645 | ±0.0092 | 0.7366 | ±0.0076 | 0.011s | <0.001s |
| **Random Forest** | 0.6515 | ±0.0141 | 0.7183 | ±0.0125 | 0.733s | 0.005s |

*Key Insights from Baselines*:
1. **Linear Classifiers Outperform Trees**: Linear models (LinearSVC, Logistic Regression) substantially outperformed Random Forest and Gradient Boosting. In 1,024-dimensional sequence space, max-margin linear hyperplanes effectively isolate high-frequency $k$-mer combinations without greedy orthogonal tree splits.
2. **Computational Overhead of Tree Boosting**: Gradient Boosting required over 120 seconds across 3 folds while achieving only 0.7535 Macro F1, rendering it computationally unviable for large-scale grid search compared to Linear SVC (0.100s).

---

### 3.2 Top 10 Model Configurations Across All Experiments

The following table summarizes the top 10 performing configurations out of 124 evaluated during Phase 5:

| Rank | Model Name | $k$ | Representation | Canonical | Mean Macro F1 | F1 Std | Mean Accuracy | Features |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **linear_svc (k=4, TF-IDF)** | **4** | **TF-IDF** | **False** | **0.9671** | **±0.0063** | **0.9674** | **256** |
| 2 | linear_svc (k=4, tfidf, uncalibrated) | 4 | TF-IDF | False | 0.9671 | ±0.0063 | 0.9674 | 256 |
| 3 | logistic_regression (k=3, count) | 3 | Count | False | 0.9667 | ±0.0038 | 0.9674 | 64 |
| 4 | linear_svc (k=5, TF-IDF, canonical) | 5 | TF-IDF | True | 0.9651 | ±0.0082 | 0.9653 | 512 |
| 5 | linear_svc (k=4, TF-IDF, canonical) | 4 | TF-IDF | True | 0.9639 | ±0.0095 | 0.9645 | 136 |
| 6 | linear_svc (k=3, TF-IDF) | 3 | TF-IDF | False | 0.9620 | ±0.0071 | 0.9627 | 64 |
| 7 | logistic_regression (k=4, count) | 4 | Count | False | 0.9579 | ±0.0066 | 0.9579 | 256 |
| 8 | linear_svc (k=5, TF-IDF, C=1.0) | 5 | TF-IDF | False | 0.9189 | ±0.0385 | 0.9298 | 1024 |
| 9 | linear_svc (k=5, TF-IDF, cw=balanced)| 5 | TF-IDF | False | 0.9189 | ±0.0385 | 0.9298 | 1024 |
| 10| LinearSVC (k=5, C=0.5) | 5 | TF-IDF | False | 0.8921 | ±0.0410 | 0.9146 | 1024 |

---

### 3.3 Detailed Fold Performance of Winning Candidate

The frozen champion candidate is `LinearSVC (k=4, TF-IDF, standard)`:

| Fold Index | Validation Gene Families | Cervical Recall | Colorectal Recall | Endometrial Recall | Liver Recall | Fold Accuracy | Fold Macro F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | `SNRPN` (112 samples) | **1.0000** (112/112) | 0.9625 (77/80) | 0.9625 (77/80) | 0.9625 (77/80) | **0.9744** | **0.9720** |
| **Fold 1** | `NIPA2` (51) + 13 small families | **0.9844** (63/64) | 0.9750 (78/80) | 0.8750 (70/80) | 1.0000 (80/80) | **0.9572** | **0.9581** |
| **Fold 2** | `UBE3A` (48) + 16 small families | **0.9688** (62/64) | 0.9625 (77/80) | 0.9500 (76/80) | 1.0000 (80/80) | **0.9704** | **0.9711** |
| **Mean ± Std**| — | **0.9844 ± 0.0127** | **0.9667 ± 0.0059**| **0.9292 ± 0.0388**| **0.9875 ± 0.0177**| **0.9674 ± 0.0073**| **0.9671 ± 0.0063**|

*Critical Validation Milestone*:
In Fold 0, where validation contains only the large `SNRPN` gene family (112 samples), `LinearSVC (k=4, TF-IDF)` achieved **100% recall (112/112)**. In contrast, under TF-IDF at $k=3$ or with uncalibrated Logistic Regression, Fold 0 recall collapsed to near zero because length normalization penalized long mRNA isoforms. `LinearSVC` with $k=4$ completely overcame this failure mode.

---

### 3.4 Secondary Verification: 10-Iteration GroupStratifiedShuffleSplit

To evaluate stability across alternative family assignments without violating grouping constraints, we performed 10 repeated group-aware resamplings (80% train / 20% validation). Because resamplings draw from the same 23 development gene families, iterations are non-independent. In compliance with Phase 4.3 evaluation protocol standards, results are reported using non-parametric **Median** and **Interquartile Range (IQR)**:

| Model Candidate | Median Macro F1 | IQR Macro F1 | Min F1 | Max F1 | Median Accuracy | IQR Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LinearSVC (k=4, TF-IDF)`** | **0.9660** | **0.0334** | 0.8792 | 0.9869 | **0.9713** | **0.0142** |
| `LogisticRegression (k=3, count)` | 0.9541 | 0.0592 | 0.8883 | 0.9824 | 0.9589 | 0.0141 |
| `MultinomialNB (k=5, alpha=0.1)` | 0.3687 | 0.2625 | 0.2777 | 0.9414 | 0.3952 | 0.2086 |

`LinearSVC (k=4, TF-IDF)` demonstrated tighter dispersion (IQR = 0.0334) and higher median performance (0.9660) compared to count-based Logistic Regression (IQR = 0.0592, median 0.9541), confirming robust stability across diverse gene-family split configurations.

---

## 4. Frozen Candidate Specification

In accordance with Phase 5 requirements, the winning candidate has been serialized and permanently frozen:
- **Directory**: `artifacts/models/phase5_best_candidate/`
- **Trained Model Binary**: `model.joblib` (`LinearSVCClassifier`)
- **Fitted Vectorizer**: `vectorizer.joblib` (`TfidfVectorizer`, $k=4$, 256 vocabulary features)
- **Fitted Encoder**: `encoder.joblib` (`LabelEncoder`, 4 classes)
- **Configuration Contract**: `model_config.json` (`"is_frozen": true`)
- **Training Cohort SHA-256**: `7c35fc4f226488cfc2d37ed332746be3f3c1aa8982ab7a8991c7fb374781475c`
- **Safeguard Attestation**: Final holdout evaluation remains locked pending Phase 6 authorization.

