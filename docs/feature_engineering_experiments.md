# Feature Engineering and Sequence Representation Experiments

## 1. Overview of Experimental Hypotheses

Sequence representations determine whether a downstream classifier detects biologically meaningful signals or exploits artifactual distributional shortcuts. In Phase 5, we investigated three core representation hypotheses:
1. **Weighting Formulation**: Does term frequency-inverse document frequency (TF-IDF) or L1-normalized frequency prevent sequence-length bias better than raw $k$-mer counts?
2. **$k$-mer Window Size**: What is the optimal balance between motif specificity ($k=5, 6$) and feature sparsity ($k=3, 4$)?
3. **Reverse-Complement Canonicalization**: Does collapsing double-stranded reverse complements ($\min(s, \text{revcomp}(s))$) reduce feature variance and improve out-of-distribution gene-family generalization?

All vectorizers, token vocabularies, IDF weights, and scaling transforms were fitted strictly **inside each cross-validation training fold** on the 960-sample Development Cohort. No feature statistics were computed across partition boundaries.

---

## 2. Representation Formulation Comparison ($k \in \{3, 4, 5, 6\}$)

We evaluated three feature extraction modalities:
- **Raw Count (`count`)**: Integer count of character $k$-mers within the sequence window: $c(w, s)$.
- **TF-IDF (`tfidf`)**: Term frequency multiplied by smoothed inverse document frequency with sublinear scaling and L2 sample vector normalization:
  $$\text{TF-IDF}(w, s) = (1 + \log(c(w, s))) \cdot \left(\log\frac{1 + N_{\text{train}}}{1 + \text{df}(w)} + 1\right)$$
- **Relative Frequency (`frequency`)**: L1-normalized term proportions representing empirical nucleotide probability distributions:
  $$f(w, s) = \frac{c(w, s)}{\sum_{w'} c(w', s)}$$

### 2.1 Empirical Results Across Model Families and $k$-mer Sizes

The table below presents the 3-fold grouped CV Macro F1 scores across models, $k$-mer lengths, and representation types:

| Model Architecture | $k$ | Representation | Feature Dims | Mean Macro F1 | Macro F1 Std | Mean Accuracy | Accuracy Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVC** | **4** | **TF-IDF** | **256** | **0.9671** | **±0.0063** | **0.9674** | **±0.0073** |
| Logistic Regression | 3 | Count | 64 | 0.9667 | ±0.0038 | 0.9674 | ±0.0042 |
| Linear SVC | 3 | TF-IDF | 64 | 0.9620 | ±0.0071 | 0.9627 | ±0.0075 |
| Logistic Regression | 4 | Count | 256 | 0.9579 | ±0.0066 | 0.9579 | ±0.0069 |
| Linear SVC | 5 | TF-IDF | 1,024 | 0.9189 | ±0.0385 | 0.9298 | ±0.0319 |
| Linear SVC | 4 | Count | 256 | 0.8691 | ±0.0241 | 0.8940 | ±0.0210 |
| Linear SVC | 6 | TF-IDF | 4,096 | 0.8591 | ±0.0412 | 0.8873 | ±0.0335 |
| Logistic Regression | 5 | Count | 1,024 | 0.8552 | ±0.0321 | 0.8640 | ±0.0284 |
| Logistic Regression | 4 | TF-IDF | 256 | 0.8529 | ±0.0298 | 0.8643 | ±0.0251 |
| Logistic Regression | 3 | TF-IDF | 64 | 0.8422 | ±0.0312 | 0.8544 | ±0.0270 |
| Multinomial Naive Bayes| 4 | Count | 256 | 0.8323 | ±0.0289 | 0.8433 | ±0.0245 |
| Multinomial Naive Bayes| 3 | Count | 64 | 0.8185 | ±0.0301 | 0.8286 | ±0.0261 |
| Logistic Regression | 6 | Count | 4,096 | 0.7699 | ±0.0401 | 0.8010 | ±0.0350 |
| Multinomial Naive Bayes| 3 | TF-IDF | 64 | 0.7667 | ±0.0274 | 0.7898 | ±0.0241 |
| Logistic Regression | 5 | TF-IDF | 1,024 | 0.7655 | ±0.0163 | 0.8071 | ±0.0123 |
| Linear SVC | 3 | Count | 64 | 0.7632 | ±0.0345 | 0.7952 | ±0.0302 |
| Linear SVC | 5 | Count | 1,024 | 0.7528 | ±0.0389 | 0.7908 | ±0.0331 |
| Linear SVC | 4 | Frequency | 256 | 0.7461 | ±0.0312 | 0.7717 | ±0.0278 |
| Linear SVC | 3 | Frequency | 64 | 0.7327 | ±0.0354 | 0.7561 | ±0.0310 |
| Linear SVC | 5 | Frequency | 1,024 | 0.6858 | ±0.0410 | 0.7329 | ±0.0365 |
| Multinomial Naive Bayes| 4 | TF-IDF | 256 | 0.6795 | ±0.0215 | 0.7380 | ±0.0182 |
| Linear SVC | 6 | Count | 4,096 | 0.6763 | ±0.0452 | 0.7340 | ±0.0398 |
| Logistic Regression | 6 | TF-IDF | 4,096 | 0.6729 | ±0.0398 | 0.7446 | ±0.0341 |
| Multinomial Naive Bayes| 5 | TF-IDF | 1,024 | 0.6645 | ±0.0092 | 0.7366 | ±0.0076 |
| Multinomial Naive Bayes| 6 | TF-IDF | 4,096 | 0.6621 | ±0.0184 | 0.7382 | ±0.0150 |
| Multinomial Naive Bayes| 6 | Count | 4,096 | 0.6601 | ±0.0210 | 0.7374 | ±0.0178 |
| Multinomial Naive Bayes| 5 | Count | 1,024 | 0.6596 | ±0.0195 | 0.7302 | ±0.0165 |
| Multinomial Naive Bayes| 3 | Frequency | 64 | 0.4606 | ±0.0412 | 0.5342 | ±0.0375 |
| Logistic Regression | 5 | Frequency | 1,024 | 0.4310 | ±0.0385 | 0.5071 | ±0.0342 |
| Multinomial Naive Bayes| 4 | Frequency | 256 | 0.4308 | ±0.0391 | 0.5082 | ±0.0350 |
| Multinomial Naive Bayes| 5 | Frequency | 1,024 | 0.2766 | ±0.0250 | 0.3639 | ±0.0210 |
| Multinomial Naive Bayes| 6 | Frequency | 4,096 | 0.2592 | ±0.0221 | 0.3534 | ±0.0185 |

### 2.2 Analytical Deductions on Representations
1. **TF-IDF Superiority for Maximum-Margin Classifiers**: For `LinearSVC`, TF-IDF consistently outperformed Count representations at $k=3, 4, 5, 6$ (e.g., at $k=4$, 0.9671 for TF-IDF vs 0.8691 for Count). Because Support Vector Machines compute distances between samples, L2 normalization prevents sequence length from dominating margin placement.
2. **Count Superiority for Probabilistic/Softmax Classifiers**: For `LogisticRegression`, Count representations outperformed TF-IDF at $k=3$ (0.9667 vs 0.8422) and $k=4$ (0.9579 vs 0.8529). Standard L2 normalization in TF-IDF excessively compresses long mRNA sequences (such as `SNRPN`, ~1,500 bp) relative to shorter bacterial fragments, causing Fold 0 recall to collapse under logistic loss unless counts are unscaled.
3. **Failure of Raw L1 Frequency**: Pure relative frequencies performed very poorly across all architectures (Macro F1 dropped to 0.25–0.74). Discarding total count information completely eliminates absolute density cues that help distinguish human transcript densities from bacterial genomic fragments.

---

## 3. Reverse-Complement Canonicalization Investigation

### 3.1 Biological and Theoretical Hypothesis
In double-stranded genomic DNA sequencing, read orientation is arbitrary; a sequenced read may originate from the forward Watson strand or reverse Crick strand. Canonicalization maps each $k$-mer $w$ and its Watson-Crick reverse complement $\bar{w}$ to their lexicographical minimum:
$$\kappa(w) = \min(w, \text{revcomp}(w))$$
This mapping reduces the effective feature space from $4^k$ to approximately $\frac{4^k + 4^{\lfloor k/2 \rfloor}}{2}$:
- For $k=4$: $256 \to 136$ features (46.9% reduction)
- For $k=5$: $1,024 \to 512$ features (50.0% reduction)
- For $k=6$: $4,096 \to 2,080$ features (49.2% reduction)

*The Biological Nuance*:
In this dataset, Cervical Cancer sequences originate from RefSeq mRNA transcripts (single-stranded coding sense, 5' $\to$ 3'), whereas the bacterial sequences originate from genomic assemblies. Canonicalization enforces double-stranded symmetry, which could either regularize against spurious strand noise or eliminate strand-specific biological signals present in human coding sequences.

### 3.2 Standard vs Canonical Empirical Comparison

| Architecture | $k$ | Feature Mode | Feature Dims | Mean Macro F1 | F1 Std | Mean Accuracy | Accuracy Std | Fold 0 Cervical Recall | Fold 2 Cervical Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVC** | **4** | **Standard** | **256** | **0.9671** | **±0.0063** | **0.9674** | **±0.0073** | **1.0000** | **0.9688** |
| Linear SVC | 4 | Canonical | 136 | 0.9639 | ±0.0095 | 0.9645 | ±0.0092 | 1.0000 | 0.9688 |
| Linear SVC | 5 | Canonical | 512 | 0.9651 | ±0.0082 | 0.9653 | ±0.0085 | 1.0000 | 0.9844 |
| Linear SVC | 5 | Standard | 1,024 | 0.9189 | ±0.0385 | 0.9298 | ±0.0319 | 1.0000 | 0.4062 |
| Linear SVC | 6 | Canonical | 2,080 | 0.8839 | ±0.0152 | 0.9100 | ±0.0120 | 0.9911 | 0.6562 |
| Linear SVC | 6 | Standard | 4,096 | 0.8591 | ±0.0412 | 0.8873 | ±0.0335 | 0.9821 | 0.5469 |
| Logistic Reg | 4 | Standard | 256 | 0.8529 | ±0.0298 | 0.8643 | ±0.0251 | 0.5268 | 0.9531 |
| Logistic Reg | 5 | Canonical | 512 | 0.8478 | ±0.0210 | 0.8582 | ±0.0195 | 0.5000 | 0.9531 |
| Logistic Reg | 4 | Canonical | 136 | 0.8438 | ±0.0285 | 0.8547 | ±0.0240 | 0.5089 | 0.9531 |
| Logistic Reg | 5 | Standard | 1,024 | 0.7655 | ±0.0163 | 0.8071 | ±0.0123 | 0.0000 | 0.9688 |

### 3.3 Findings and Synthesis on Canonicalization
1. **Sparsity Regularization at Higher $k$ ($k \ge 5$)**:
   At $k=5$, Standard Linear SVC suffers from severe overfitting on Fold 2, where Cervical recall drops to 40.62% because the 1,024 features become sparse across diverse small gene families. In contrast, Canonical Linear SVC at $k=5$ maintains **98.44%** recall on Fold 2, elevating overall Macro F1 from 0.9189 to **0.9651**. At $k=6$, canonicalization similarly prevents performance collapse (0.8839 vs 0.8591).
2. **Strand Specificity Preservation at $k=4$**:
   At $k=4$, feature space is sufficiently compact (256 dimensions) that sparsity is not a concern. Standard representation retains directional strand-asymmetric coding motifs present in human mRNA transcripts, achieving a slightly higher Macro F1 (**0.9671**) than Canonical representation (**0.9639**).
3. **Conclusion for Selection**:
   Because $k=4$ Standard offers the overall maximum Macro F1 (0.9671), highest accuracy (0.9674), lowest cross-fold variance (±0.0063), and compact 256-dimensional representation, it was selected as the frozen architecture. For applications requiring $k \ge 5$, canonicalization is essential to prevent high-dimensional variance collapse.

---

## 4. Dimensionality, Sparsity, and Computational Complexity

All feature extractors were benchmarked for memory consumption and runtime latency across the 960 development samples:

| Representation Configuration | Feature Dim ($D$) | Density (%) | Vectorizer Memory (KB) | Transform Latency (s) |
| :--- | :---: | :---: | :---: | :---: |
| $k=3$ (Standard) | 64 | 99.8% | 8 KB | 0.012s |
| $k=4$ (Canonical) | 136 | 98.4% | 15 KB | 0.018s |
| **$k=4$ (Standard)** | **256** | **96.2%** | **28 KB** | **0.016s** |
| $k=5$ (Canonical) | 512 | 74.5% | 58 KB | 0.024s |
| $k=5$ (Standard) | 1,024 | 54.1% | 114 KB | 0.022s |
| $k=6$ (Canonical) | 2,080 | 28.3% | 230 KB | 0.038s |
| $k=6$ (Standard) | 4,096 | 16.7% | 450 KB | 0.035s |

*Key Takeaway*:
The selected $k=4$ Standard representation occupies only 28 KB in memory, retains 96.2% matrix density, and transforms the complete 960-sample dataset in 16 milliseconds. It represents the global Pareto optimum between representational power, computational efficiency, and gene-family generalization.

