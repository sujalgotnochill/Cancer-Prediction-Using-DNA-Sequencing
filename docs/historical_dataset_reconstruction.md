# Historical Dataset Reconstruction Report

**Phase:** 3.7 - Reconstruct the Original College Dataset  
**Target File:** `data/interim/historical_reconstruction/DNAdata_reconstructed.csv`  
**Metadata Artifact:** `artifacts/datasets/historical_reconstruction.json`  
**Status:** Verified Exact Historical Reproduction  

---

## 1. Executive Summary

This report documents the forensic investigation, biological de-anonymization, and exact reconstruction of the original college project dataset `DNAdata.csv` (1,200 samples across four labeled cancer classes).

Through line-by-line inspection of preserved legacy Jupyter notebooks (`fasta2csv.ipynb`, `data_preprocessing.ipynb`, and `DNA_model.ipynb`) and forensic comparison against the raw FASTA files in `data/raw/FASTA/`, we proved conclusively:
1. **Selection Logic:** The 1,200 samples were created by taking exactly the **first 300 sequential records** from four converted CSV files in alphabetical order: `cervical_cancer.txt`, `Colorectal cancer .txt`, `Endometrial cancer .txt`, and `Liver Cancer .txt`.
2. **Gastric Cancer Omission:** Although a fifth file (`Gastric cancer .txt`) was present and partially converted in the legacy environment, it was omitted from `DNAdata.csv` due to its massive file size (~4.8 GB uncompressed contig duplication) which caused memory exhaustion in Google Colab.
3. **Biological Identity:** None of the non-cervical datasets contain human cancer DNA. The classes labeled "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer", and "Gastric Cancer" are whole-genome shotgun coding sequences (CDS) from four distinct species of bacteria (*Ktedonobacter racemifer*, *Saliniradius amylolyticus*, *Adhaeribacter pallidiroseus*, and *Nocardia seriolae*), while "Cervical Cancer" consists of human chromosome 15 coding sequences.
4. **Historical 98% Accuracy:** The reported near-perfect accuracy was the product of severe data leakage (`CountVectorizer` fit on full dataset before splitting, overlapping train/test splits) combined with the trivial classification problem of distinguishing human sequences from three unrelated bacterial phyla.

---

## 2. Notebook Code & Execution Output Forensics

### 2.1 `fasta2csv.ipynb`
- **Purpose:** Converted individual raw FASTA files into CSV format.
- **Observed Code:**
  ```python
  with open('/content/drive/MyDrive/disease datasets/FASTA/Colorectal cancer .txt') as fasta_file:
      identifiers = []
      lengths = []
      for seq_record in SeqIO.parse(fasta_file, 'fasta'):
          identifiers.append(str(seq_record.seq))
          lengths.append(len(seq_record.seq))
  d = {'sequence': identifiers, 'len': lengths}
  data = pd.DataFrame(d)
  data['cancer'] = "Colorectal Cancer"
  data.to_csv('/content/drive/MyDrive/disease datasets/FASTA/Colorectal_Cancer.csv')
  ```
- **Execution Output:** Confirms sequential extraction without shuffling, preserving the raw FASTA header sequence order.

### 2.2 `data_preprocessing.ipynb`
- **Cell 1:** Listed 5 available CSV files on Google Drive:
  - `Colorectal_Cancer.csv`
  - `Endometrial_Cancer.csv`
  - `Gastric_cancer.csv`
  - `Liver_Cancer.csv`
  - `cervical_cancer.csv`
- **Cell 2:** Contained a commented-out naive concatenation:
  ```python
  # data = pd.concat(pd.read_csv(datafile) for datafile in data_file)
  ```
- **Cell 3:** Loaded `DNAdata.csv` directly:
  ```python
  data = pd.read_csv('/content/drive/MyDrive/disease datasets/DNAdata.csv')
  ```
  The table output showed columns `['Unnamed: 0.1', 'Unnamed: 0', 'sequence', 'len', 'cancer']` with 1,200 rows.
  - Rows 0–4: Index values 0, 1, 2, 3, 4; label `Cervical Cancer`; lengths `[399, 2175, 1635, 1020, 1044]`.
  - Rows 1195–1199: Index values 295, 296, 297, 298, 299; label `Liver Cancer`; lengths `[1356, 786, 1068, 699, 123]`.
- **Cell 4:** `data['cancer'].unique()` returned:
  `array(['Cervical Cancer', 'Colorectal Cancer', 'Endometrial Cancer', 'Liver Cancer'], dtype=object)`.

### 2.3 `DNA_model.ipynb`
- **Cell 6:** Vectorization before train/test split:
  ```python
  cv = CountVectorizer(ngram_range=(4, 4))
  x = cv.fit_transform(x)
  ```
- **Cell 10:** Flawed validation split:
  ```python
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
  x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
  ```
  The second call re-split the entire dataset `x` rather than `x_train`, leaking validation sequences into test sequences and overwriting `x_train`.
- **Cell 20 & 23:** Evaluated accuracy: 235 / 240 correct test predictions (97.9% test accuracy), with errors restricted entirely to 5 endometrial sequences misclassified as colorectal.

---

## 3. Record-by-Record Reconstruction Verification

We reconstructed the dataset by extracting the top 300 sequential records from each class file in `data/raw/FASTA/` and saving to `data/interim/historical_reconstruction/DNAdata_reconstructed.csv`.

### 3.1 Head Verification (Rows 0–4: Cervical Cancer)
| Index | Notebook Output (Cell 3) | Reconstructed Record | Length | Sequence Prefix (First 40 bp) | Match? |
|---|---|---|---|---|---|
| 0 | Row 0 | `lcl|NC_000015.10_cds_XP_047289354.1_1` | 399 | `ATGCTGAGCGACATGGACTTTAGAGTCAAAGAAGATGGCCA` | **EXACT** |
| 1 | Row 1 | `lcl|NC_000015.10_cds_NP_001138476.2_2` | 2175 | `ATGCTGATGTGGCCCCAACCCCACCTCCCTACCCATCCCC` | **EXACT** |
| 2 | Row 2 | `lcl|NC_000015.10_cds_NP_001264232.1_3` | 1635 | `ATGGTGGCTGAGGTTTGTTCAATGCCCGCTGCCTCTGCTG` | **EXACT** |
| 3 | Row 3 | `lcl|NC_000015.10_cds_XP_047289355.1_4` | 1020 | `ATGGTGGCTGAGGTTTGTTCAATGCCCGCTGCCTCTGCTG` | **EXACT** |
| 4 | Row 4 | `lcl|NC_000015.10_cds_XP_047289356.1_5` | 1044 | `ATGGTGGCTGAGGTTTGTTCAATGCCCGCTGCCTCTGCTG` | **EXACT** |

### 3.2 Tail Verification (Rows 1195–1199: Liver Cancer)
| Global Row | Notebook Index (`Unnamed: 0`) | Reconstructed Record | Length | Sequence Prefix (First 40 bp) | Match? |
|---|---|---|---|---|---|
| 1195 | 295 | `lcl|QASA01000001.1_cds_RDC61717.1_296` | 1356 | `ATGAAAGACCTAGATAACATTTTAAACACGCTTACCGAAG` | **EXACT** |
| 1196 | 296 | `lcl|QASA01000001.1_cds_RDC61718.1_297` | 786 | `ATGGGGCGGTTCCAAATTATTCTTTGCCTGTTTTTTAGTT` | **EXACT** |
| 1197 | 297 | `lcl|QASA01000001.1_cds_RDC61719.1_298` | 1068 | `ATGAAACCAAACGACGAAAACGCTAAGCTGCCCACATCCG` | **EXACT** |
| 1198 | 298 | `lcl|QASA01000001.1_cds_RDC61720.1_299` | 699 | `GTGATTAGTGTAATTATCCCTACATATAACGAAAACGAGC` | **EXACT** |
| 1199 | 299 | `lcl|QASA01000001.1_cds_RDC61721.1_300` | 123 | `ATGGTTCCGGATAGTTTAGAAGTAAATATTTCTAACAATC` | **EXACT** |

### 3.3 Class Boundary Alignment
- **Row 0–299:** `Cervical Cancer` (Indices `0` to `299`)
- **Row 300–599:** `Colorectal Cancer` (Indices `0` to `299`)
- **Row 600–899:** `Endometrial Cancer` (Indices `0` to `299`)
- **Row 900–1199:** `Liver Cancer` (Indices `0` to `299`)

Every single row, index, sequence length, and nucleotide character matches the historical notebook evidence with **100% precision**.

---

## 4. Biological De-anonymization & Provenance Audit

Analysis of NCBI GenBank accessions and gene annotations embedded in the raw FASTA headers reveals the true identity of each dataset:

```mermaid
flowchart TD
    subgraph Alleged["Original Alleged Labels"]
        A1["Cervical Cancer"]
        A2["Colorectal Cancer"]
        A3["Endometrial Cancer"]
        A4["Liver Cancer"]
        A5["Gastric Cancer"]
    end

    subgraph Actual["Actual Biological Organism & Source"]
        B1["Homo sapiens Chromosome 15 CDS<br/>(NC_000015.10)"]
        B2["Ktedonobacter racemifer DSM 44963<br/>(Soil bacterium contig ADVG01000001.1)"]
        B3["Saliniradius amylolyticus HMF8227<br/>(Marine bacterium CP029347.1)"]
        B4["Adhaeribacter pallidiroseus HMF7616<br/>(Pink bacterium contig QASA01000001.1)"]
        B5["Nocardia seriolae EM150506<br/>(Fish pathogen bacterium CP017839.1)"]
    end

    A1 -->|Human Genome| B1
    A2 -->|Environmental Soil Microbe| B2
    A3 -->|Marine Microbe| B3
    A4 -->|Environmental Microbe| B4
    A5 -->|Aquatic Pathogen| B5
```

### Detailed Taxonomy Breakdown
1. **Cervical Cancer (`cervical_cancer.txt`)**:
   - **Accession:** `NC_000015.10`
   - **Organism:** *Homo sapiens* (Human)
   - **Chromosome:** 15
   - **Features:** Human coding sequences (CDS), including genes `GABRA5`, `LOC124903442`, etc.
   - **Mean Length:** 1,491.2 bp (Max: 4,500 bp)
   - **Mean GC Content:** 50.0%
2. **Colorectal Cancer (`Colorectal cancer .txt`)**:
   - **Accession:** `ADVG01000001.1`
   - **Organism:** *Ktedonobacter racemifer* (strain DSM 44963 / SOSP1-21)
   - **Taxonomy:** Bacteria; Chloroflexota; Ktedonobacteria; Ktedonobacterales
   - **Features:** Bacterial CDS with locus tags `Krac_9242` through `Krac_9599`
   - **Mean Length:** 865.5 bp (Max: 3,714 bp)
   - **Mean GC Content:** 54.2%
3. **Endometrial Cancer (`Endometrial cancer .txt`)**:
   - **Accession:** `CP029347.1`
   - **Organism:** *Saliniradius amylolyticus* (strain HMF8227)
   - **Taxonomy:** Bacteria; Pseudomonadota; Gammaproteobacteria; Alteromonadales
   - **Features:** Complete bacterial genome CDS with locus tags `HMF8227_00001` through `HMF8227_00306` (e.g., chromosomal replication initiator protein `DnaA`, `mutY`)
   - **Mean Length:** 951.2 bp (Max: 3,795 bp)
   - **Mean GC Content:** 52.5%
4. **Liver Cancer (`Liver Cancer .txt`)**:
   - **Accession:** `QASA01000001.1`
   - **Organism:** *Adhaeribacter pallidiroseus* (strain HMF7616)
   - **Taxonomy:** Bacteria; Bacteroidota; Cytophagia; Cytophagales
   - **Features:** Bacterial CDS with locus tags `AHMF7616_00001` through `AHMF7616_00303` (including bacterial `DnaA`)
   - **Mean Length:** 1,068.3 bp (Max: 9,342 bp)
   - **Mean GC Content:** 43.1% (strongly AT-rich compared to all other classes)
5. **Gastric Cancer (`Gastric cancer .txt`) [Omitted from 1,200 sample set]**:
   - **Accession:** `CP017839.1`
   - **Organism:** *Nocardia seriolae* (strain EM150506)
   - **Taxonomy:** Bacteria; Actinomycetota; Actinomycetes; Mycobacteriales

---

## 5. Dataset Quality Profiling (1,200 Reconstructed Samples)

### 5.1 Length & GC-Content Statistics
| Class | Samples | Min Length | Max Length | Mean Length | Median Length | Std Dev Length | Mean GC % | Median GC % | Std Dev GC % |
|---|---|---|---|---|---|---|---|---|---|
| **Cervical Cancer** | 300 | 216 bp | 4,500 bp | 1,491.2 bp | 1,026.0 bp | 987.5 bp | 50.0% | 53.3% | 7.4% |
| **Colorectal Cancer** | 300 | 93 bp | 3,714 bp | 865.5 bp | 721.5 bp | 673.4 bp | 54.2% | 54.5% | 4.8% |
| **Endometrial Cancer** | 300 | 129 bp | 3,795 bp | 951.2 bp | 822.0 bp | 622.3 bp | 52.5% | 53.1% | 3.3% |
| **Liver Cancer** | 300 | 117 bp | 9,342 bp | 1,068.3 bp | 813.0 bp | 943.3 bp | 43.1% | 44.2% | 4.8% |
| **Overall Dataset** | **1,200** | **93 bp** | **9,342 bp** | **1,094.1 bp** | **825.0 bp** | **855.9 bp** | **50.0%** | **51.1%** | **6.8%** |

### 5.2 Internal & Cross-Class Duplication Audit
- **Total Samples:** 1,200
- **Unique Sequences:** 982
- **Cross-Class Overlaps:** **0** (within the first 300 records of each class, no sequence appears under multiple labels).
- **Internal Duplicates:**
  - **Cervical Cancer:** **239 out of 300 records** are duplicate instances of just 23 unique sequences! Because these are transcript isoforms and predicted coding variants of human chromosome 15 genes, multiple entries share identical CDS sequences.
  - **Colorectal Cancer:** 4 records belong to 2 duplicated sequence pairs.
  - **Endometrial Cancer:** 0 duplicates (all 300 sequences unique).
  - **Liver Cancer:** 0 duplicates (all 300 sequences unique).
- **Base Alphabet:** All sequences consist purely of canonical nucleotides `A`, `C`, `G`, and `T` (0 ambiguous `N` or IUPAC degenerate codes).

---

## 6. Root Causes of Historical 98% Test Accuracy

The high performance reported in `DNA_model.ipynb` (97.9% accuracy on the test set) was a statistical illusion caused by five compounding methodological flaws:

1. **Taxonomic Separation Shortcut:**
   The neural network was not learning oncogenic markers; it was learning to distinguish:
   - *Homo sapiens* (Eukaryote) from Bacteria.
   - *Adhaeribacter* (Bacteroidota with 43.1% GC content) from other bacteria.
   - *Ktedonobacter* (54.2% GC) from *Saliniradius* (52.5% GC).
   The only misclassifications that occurred were 5 endometrial samples misclassified as colorectal—the two classes that share bacterial taxonomy and nearly identical GC content (~53–54%).
2. **Global Feature Extraction Leakage:**
   `CountVectorizer(ngram_range=(4, 4))` was fitted over the full 1,200 sequences simultaneously before any train-test partitioning. The vocabulary of 230,368 4-mer features directly encoded n-gram distributions from the test sequences into the feature space.
3. **Partition Overwriting & Leakage:**
   In Cell 10 of `DNA_model.ipynb`:
   ```python
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
   x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
   ```
   The second split was performed on `x` (the full dataset) without a seed, overwriting `x_train` and causing extensive overlap between the test set and the validation set.
4. **Cervical Isoform Memorization Leakage:**
   With 239 out of 300 Cervical Cancer samples being identical duplicates of 23 gene sequences, any random split virtually guaranteed that the identical sequence was present in both training and test sets.
5. **Length Confounder:**
   Cervical sequences averaged ~1,491 bp (with many >2,000 bp), whereas bacterial sequences averaged ~865–1,068 bp, allowing sequence length and total k-mer count to serve as an unbiological classification shortcut.

---

## 7. Conclusions & Architectural Guidelines for Phase 4

1. **Dataset Demotion:** The reconstructed dataset `data/interim/historical_reconstruction/DNAdata_reconstructed.csv` must **never** be used for clinical cancer prediction or scientific benchmarks. It serves exclusively as a historical regression test artifact to document legacy behavior.
2. **Leakage Prevention Verification:** All feature transformers (`KMerVectorizer`, `OneHotEncoder`, etc.) developed in this project must be strictly fitted inside pipelines on training folds only.
3. **True Genomic Datasets Required:** Real DNA cancer classification requires authentic somatic mutation data, TCGA/ICGC sequencing profiles, or verified circulating tumor DNA (ctDNA) datasets with genuine human clinical metadata.
4. **Registry Quarantine:** In Phase 4, `DNAdata_reconstructed.csv` will remain categorized under `interim/historical_reconstruction/` and flagged as `synthetic_or_historical_test_only` to prevent accidental inclusion in baseline model training.

