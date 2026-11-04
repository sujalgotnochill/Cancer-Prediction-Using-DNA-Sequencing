# Phase 4.2: Gene-Family Generalization & Isoform Leakage Audit

**Project:** Cancer Prediction Using DNA Sequencing  
**Phase:** 4.2 (Gene-Family Generalization & Evaluation Protocol Audit)  
**Author:** Senior Machine Learning Engineer & Computational Biology Researcher  
**Status:** Completed  
**Artifact References:**  
- [`artifacts/evaluation/gene_family_generalization.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/gene_family_generalization.json)  
- [`artifacts/evaluation/evaluation_protocol_audit.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/evaluation_protocol_audit.json)  
- [`docs/benchmark_integrity_audit.md`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/docs/benchmark_integrity_audit.md)  

---

## 1. Executive Summary

During Phase 4 baseline benchmarking, the top-performing baseline model (Logistic Regression with $k=5$ and $k=6$) achieved **96.77% test accuracy** and a **1.0000 Cervical Cancer test recall** on a duplicate-grouped split. While Phase 4.1 confirmed that identical nucleotide sequence duplicates were successfully quarantined from crossing partitions, a deeper forensic examination revealed a critical biological phenomenon:

> **Human chromosome 15 transcripts from the exact same gene family crossed between train, validation, and test partitions in the duplicate-grouped split.**

Because splice variants (transcript isoforms) of the same gene share substantial nucleotide homology (often 90%–98% identical sequence, differing only by exon inclusions, alternate splice sites, or terminal truncations), they produced different SHA-256 hashes and were distributed independently across splits.

To quantify the degree of memorization versus true sequence generalization, we implemented **Protocol 3: Gene-Family-Grouped Partitioning**, ensuring that all transcripts of a given gene family were isolated exclusively to a single partition.

### Key Audit Findings

1. **Gene-Family Leakage in Historical Duplicate Split:** Exactly 8 human gene families (`ATP10A`, `CYFIP1`, `GABRA5`, `GABRB3`, `GOLGA6L22`, `POTEB`, `TUBGCP5`, `UBE3A`) leaked across train, validation, and test partitions under duplicate-grouped splitting.
2. **Generalization Deficit on Unseen Gene Families:** When gene families were strictly held out, Logistic Regression performance dropped:
   - Non-stratified holdout test accuracy dropped from **96.77%** to **90.26%** ($k=6$).
   - Test **Cervical Cancer recall plummeted from 100.0% to 77.97%** (13 of 59 unseen human transcripts misclassified as bacterial sequences).
   - Stratified gene-family test recall dropped from **100.0%** to **85.71%**.
   - Validation accuracy dropped from **98.00%** to **90.97%–92.36%**.
3. **Biological Diagnosis:** The model's historical near-perfect Cervical classification was partially driven by **gene-specific $k$-mer memorization** across splice isoforms rather than general eukaryotic sequence features.

---

## 2. Genomic Metadata Mapping & Availability

To audit gene-family overlap without fabricating biological labels, we analyzed the NCBI FASTA description headers of the reconstructed dataset across all four classes ($N=300$ per class, 1,200 total samples).

| Target Class Label | Actual Biological Source | NCBI Accession Scope | Samples with `[gene=...]` | Samples with `[locus_tag=...]` | Unique Genes | Notes on Metadata Integrity |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Cervical Cancer** | *Homo sapiens* (Chr 15) | mRNA transcripts | **300 / 300 (100%)** | 0 / 300 (0%) | 31 | Full NCBI gene annotations (`ATP10A`, `UBE3A`, `GABRB3`, etc.) with multiple splice variants per gene. |
| **Colorectal Cancer** | *Ktedonobacter racemifer* | Bacterial genome | **0 / 300 (0%)** | 300 / 300 (100%) | 0 | Entirely unannotated by gene symbol; indexed exclusively by locus tags (`Krac_0001` to `Krac_0300`). |
| **Endometrial Cancer** | *Saliniradius amylolyticus* | Bacterial genome | **73 / 300 (24.3%)** | 227 / 300 (75.7%) | 73 | Only 73 samples possess explicit gene symbols; remaining 227 possess only locus tags (`HMF8227_...`). |
| **Liver Cancer** | *Adhaeribacter pallidiroseus* | Bacterial genome | **75 / 300 (25.0%)** | 225 / 75.0%) | 75 | Only 75 samples possess explicit gene symbols; remaining 225 possess only locus tags (`AHMF7616_...`). |

### Bacterial Ortholog Sharing
Five conserved bacterial metabolic genes are shared between the Endometrial and Liver cancer files:
- `coxA`, `coxB`, `coxC` (Cytochrome c oxidase subunits I, II, III)
- `cyoE` (Protoheme IX farnesyltransferase)
- `priA` (Primosomal protein N')

Zero gene symbols are shared between the human *Homo sapiens* transcripts (Cervical) and the bacterial genomes.

> [!IMPORTANT]
> **Scientific Constraint on Metadata:** We strictly rejected the practice of imputing or fabricating gene symbols for the 752 bacterial samples that lack NCBI `[gene=...]` tags. In Protocol 3a, unannotated records fall back to exact-sequence hashes; in Protocol 3b, unannotated records are formally excluded, leaving a verified 3-class cohort of $N=448$.

---

## 3. Evaluation Protocol Comparison

We conducted a side-by-side empirical benchmark of Logistic Regression ($C=1.0$, `max_iter=1000`) across four splitting protocols for $k=5$ and $k=6$:

1. **Protocol 1: Conventional Stratified Split ($N=1,200$)**
   - Preserves class ratios randomly without grouping.
   - Sequence leakage: **Present** (11 sequence overlaps between train and test).
   - Gene leakage: **Present** (13 human gene families overlap between train and test).
2. **Protocol 2: Exact-Sequence Duplicate-Grouped Split ($N=1,200$)**
   - Partitions identical sequence hashes atomically.
   - Sequence leakage: **Zero** (0 sequence overlaps).
   - Gene leakage: **Present** (8 human gene families overlap between train and test).
3. **Protocol 3a: Gene-Family-Grouped Split — Full Cohort ($N=1,200$)**
   - Groups sequences by `gene` when present, falling back to sequence hashes for unannotated bacterial records.
   - Sequence leakage: **Zero**.
   - Gene leakage: **Zero** across all partitions.
4. **Protocol 3b: Gene-Family-Grouped Split — Strict Annotated Cohort ($N=448$)**
   - Filters dataset to only samples possessing verified NCBI gene symbols (Cervical: 300, Endometrial: 73, Liver: 75; Colorectal: 0).
   - Sequence leakage: **Zero**.
   - Gene leakage: **Zero**.

---

## 4. Empirical Benchmark Results

All metrics below are drawn directly from [`artifacts/evaluation/gene_family_generalization.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/evaluation/gene_family_generalization.json).

### Performance Summary Table

| Protocol | $k$ | Train / Val / Test | Val Acc | Test Acc | Test Macro F1 | Cervical Precision | Cervical Recall | Cervical F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protocol 1: Stratified** | 5 | 840 / 180 / 180 | 0.9889 | 0.9556 | 0.9556 | 1.0000 | 0.9778 | 0.9888 |
| **Protocol 1: Stratified** | 6 | 840 / 180 / 180 | 0.9722 | 0.9611 | 0.9612 | 1.0000 | 0.9556 | 0.9773 |
| **Protocol 2: Duplicate-Grouped** | 5 | 895 / 150 / 155 | 0.9800 | 0.9677 | 0.9723 | 1.0000 | **1.0000** | **1.0000** |
| **Protocol 2: Duplicate-Grouped** | 6 | 895 / 150 / 155 | 0.9600 | 0.9677 | 0.9723 | 1.0000 | **1.0000** | **1.0000** |
| **Protocol 3a: Gene-Family Full** | 5 | 905 / 144 / 151 | 0.9236 | 0.9603 | 0.9534 | 1.0000 | **0.8571** | 0.9231 |
| **Protocol 3a: Gene-Family Full** | 6 | 905 / 144 / 151 | 0.9097 | 0.9603 | 0.9536 | 1.0000 | **0.8571** | 0.9231 |
| **Protocol 3b: Gene-Family Strict** | 5 | 379 / 31 / 38 | 0.9032 | 0.9474 | 0.9487 | 1.0000 | **0.8571** | 0.9231 |
| **Protocol 3b: Gene-Family Strict** | 6 | 379 / 31 / 38 | 0.9032 | 0.9474 | 0.9487 | 1.0000 | **0.8571** | 0.9231 |

### Performance Under Non-Stratified Group Partitioning

When all transcripts of unseen human genes are quarantined together without forcing artificial class balance (Train: 861, Val: 144, Test: 195; test Cervical genes: `GABRA5`, `LOC102724737`, `SNURF`, `UBE3A`), the impact of isoform leakage is even starker:

| Experiment Configuration | Test Accuracy | Test Macro F1 | Cervical Precision | Cervical Recall | Cervical Misclassifications |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Duplicate-Grouped ($k=6$)** | 0.9677 | 0.9723 | 1.0000 | **1.0000** | 0 / 30 |
| **Gene-Family Grouped ($k=5$)** | 0.9282 | 0.9242 | 1.0000 | **0.9322** | 4 / 59 |
| **Gene-Family Grouped ($k=6$)** | **0.9026** | **0.9015** | 1.0000 | **0.7797** | **13 / 59** |

---

## 5. In-Depth Failure & Error Analysis

### Where Did Misclassified Human Sequences Go?
In the strict gene-family holdout ($k=6$, non-stratified), the confusion matrix was:

$$\begin{bmatrix}
46 & 7 & 4 & 2 \\
0 & 46 & 4 & 0 \\
0 & 1 & 35 & 0 \\
0 & 0 & 1 & 49
\end{bmatrix}$$

Out of 59 unseen human Cervical transcripts:
- **46** were correctly classified as human Cervical Cancer.
- **7** were misclassified as *Ktedonobacter racemifer* (Colorectal Cancer).
- **4** were misclassified as *Saliniradius amylolyticus* (Endometrial Cancer).
- **2** were misclassified as *Adhaeribacter pallidiroseus* (Liver Cancer).

### Why Did Isoform Memorization Occur?
1. **High Isoform Homology:** Transcripts of human genes like `UBE3A` and `POTEB` share massive stretches of identical coding sequences and exon boundaries. When isoform A is in train and isoform B is in test, $k$-mer count vectors between train and test share near-identical profiles.
2. **High Dimensionality of $k=6$:** At $k=6$, the vocabulary contains 4,096 canonical features. A linear model easily assigns large weights to gene-specific 6-mers (e.g., specific splice junction motifs or 3' UTR signatures) rather than general human-genome $k$-mer background distributions.
3. **Overestimation of Baseline Capability:** In duplicate-grouped splits, the model achieved 100% recall on Cervical Cancer because every test Cervical sequence had near-identical splice cousins in the training set. Once true gene families were held out, recall fell by **22.03%** (from 100% to 77.97%).

---

## 6. Conclusions & Architectural Decisions for Phase 5

1. **Protocol 2 is Invalid for Transcript Generalization:** Duplicate-grouped splitting is strictly insufficient when evaluating multi-transcript eukaryotic data. It creates an illusion of generalizability while masking isoform memorization.
2. **Gene-Family Grouping is Mandatory:** For any dataset containing multi-isoform eukaryotic sequences, `split_gene_family_grouped` (or `strategy="gene_family_grouped"`) must be utilized to measure true out-of-gene generalization.
3. **Educational Reality Check:** The fact that human transcripts can be misclassified as bacterial genomes when genes are held out emphasizes that even distinguishing eukaryotic from bacterial sequences requires robust models that do not overfit to specific transcripts.
4. **Clinical Boundaries Maintained:** This finding further solidifies our Phase 3.7 and Phase 4 determination: this benchmark measures biological sequence classification across heterogeneous organisms, not clinical cancer detection.

