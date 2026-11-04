# Forensic Inspection Report: Original FASTA Dataset Collection (Phase 3.6)

**Author:** Senior Data Engineer & Bioinformatics Software Architect  
**Date:** 2026-11-04  
**Target Directory:** `data/raw/FASTA/`  
**Total Collection Records:** 552,783 sequences  
**Total Nucleotide Bases:** 532,771,862 bp (~532.8 Mb)  
**Total Disk Size:** 640,895,005 bytes (~611.2 MB)  
**Structured Output:** [`artifacts/datasets/fasta_inspection.json`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/artifacts/datasets/fasta_inspection.json)  
**Test Suite Status:** 108 tests passing (0 failures, 0 errors)

---

## 1. Executive Summary & Core Verdict

A read-only streaming forensic inspection was conducted across the five FASTA sequence files located in `data/raw/FASTA/`.

### Definitive Recommendation:
$$\mathbf{Recommendation:\ 3.\ Unsuitable\ for\ the\ Current\ Classification\ Experiment}$$

### Core Forensic Findings:
1. **Severe Cross-File Sequence Conflicts:**  
   The files share thousands of identical DNA sequences across supposedly distinct cancer classes:
   - **12,303 identical sequences** appear in both `Gastric cancer .txt` and `Liver Cancer .txt`.
   - **7,360 identical sequences** appear in both `Colorectal cancer .txt` and `Liver Cancer .txt`.
   - **1,857 identical sequences** appear in both `Colorectal cancer .txt` and `Gastric cancer .txt`.
   Any multi-class classifier trained on these raw files would receive identical sequences annotated with conflicting disease targets.
2. **Spurious Biological Provenance:**  
   The filenames (`cervical_cancer`, `Colorectal cancer`, `Endometrial cancer`, `Liver Cancer`, `Gastric cancer`) **do not represent human somatic tumor biopsies**. Header accessions prove the collection is an uncurated combination of:
   - Human Chromosome 15 reference CDS + Human Papillomavirus (HPV-16) isolates in `cervical_cancer.txt`.
   - Complete bacterial genomes and soil bacteria (e.g. *Ktedonobacter racemifer*, *Bacillus*, *Streptomyces*) in `Colorectal cancer .txt`, `Endometrial cancer .txt`, and `Liver Cancer .txt`.
   - Massive shotgun contigs and metagenomic assemblies in `Gastric cancer .txt`.
3. **The 98% Accuracy Artifact Explained:**  
   The legacy college project took the first ~300 records of each file. Because the classes represented fundamentally different biological domains (human chromosome 15 vs. unrelated bacterial species with divergent GC-content), simple $k$-mer counts achieved high accuracy by distinguishing human DNA from bacterial DNA, rather than learning true oncological biomarkers.

---

## 2. File-by-File Inspection

### 2.1 File: `cervical_cancer.txt`

| Property | Value |
| :--- | :--- |
| **Exact Filename** | `cervical_cancer.txt` (lowercase with underscore, `.txt` extension) |
| **File Size** | 30,324,062 bytes (28.92 MB / 27.58 MiB) |
| **Encoding** | `utf-8` |
| **Valid FASTA Format** | **Yes** (standard `>` headers followed by multiline uppercase sequences) |
| **Total FASTA Records** | 13,389 records |
| **Total Nucleotides** | 25,087,631 bp |
| **Length Statistics** | Min: 75 bp, Max: 14,730 bp, Mean: 1,873.75 bp |
| **GC-Content** | **47.05%** |
| **Alphabet Distribution** | `A`: 7,276,863 (29.01%), `C`: 5,681,915 (22.65%), `G`: 6,121,305 (24.40%), `T`: 6,007,548 (23.95%), `N`: 0 (0.00%) |
| **Internal Duplicates** | 7,029 duplicate sequences (6,360 unique sequences) |

#### First 5 FASTA Headers:
1. `>lcl|NC_000015.10_cds_XP_047289354.1_1 [gene=LOC124903442] [db_xref=GeneID:124903442] [protein=uncharacterized protein LOC124903442 isoform X1] [protein_id=XP_047289354.1] [location=join(20295770..20295794,20300384..20300550,20301413..20301540,20330102..20330180)] [gbkey=CDS]`
2. `>lcl|NC_000015.10_cds_NP_001138476.2_2 [gene=GOLGA6L6] [db_xref=CCDS:CCDS45184.1,Ensembl:ENSP00000480376.1,GeneID:727832] [protein=golgin subfamily A member 6-like protein 6] [protein_id=NP_001138476.2] [location=complement(join(20534259..20535689,20536342..20536483,20536814..20536872,20538252..20538358,20538454..20538547,20538636..20538686,20539773..20539901,20541548..20541709))] [gbkey=CDS]`
3. `>lcl|NC_000015.10_cds_NP_001264232.1_3 [gene=POTEB2] [db_xref=CCDS:CCDS59248.1,Ensembl:ENSP00000456953.1,GeneID:100287399] [protein=POTE ankyrin domain family member B2] [protein_id=NP_001264232.1] [location=complement(join(20835391..20835603,20837244..20837367,20845833..20845999,20847045..20847089,20848493..20848563,20854057..20854127,20856636..20856773,20858111..20858217,20861042..20861215,20861375..20861489,20865872..20866281))] [gbkey=CDS]`
4. `>lcl|NC_000015.10_cds_XP_016877342.1_4 [gene=POTEB2] [db_xref=GeneID:100287399] [protein=POTE ankyrin domain family member B2 isoform X3] [protein_id=XP_016877342.1] [location=complement(join(20845995..20845999,20854057..20854127,20856636..20856773,20858111..20858217,20861042..20861215,20861375..20861489,20865872..20866281))] [gbkey=CDS]`
5. `>lcl|NC_000015.10_cds_XP_006720417.1_5 [gene=POTEB2] [db_xref=GeneID:100287399] [protein=POTE ankyrin domain family member B2 isoform X1] [protein_id=XP_006720417.1] [location=complement(join(20847061..20847089,20854057..20854127,20856636..20856773,20858111..20858217,20861042..20861215,20861375..20861489,20865872..20866281))] [gbkey=CDS]`

#### First 5 Sequence Previews:
1. `ATGCTGAGCGACATGGACTTTAGAGTCAAAGAAGATGGCCACGGAGACAGCAGCGTGGTCAGAGTGGTAGGAGCCGGCCA` (80 bp)
2. `ATGCTGATGTGGCCCCAACCCCACCTCCCTACCCATCCCCACCTCCCTACCCATCCCCACCTCCCTACCCATCCCCACCT` (80 bp)
3. `ATGGTGGCTGAGGTTTGTTCAATGCCCGCTGCCTCTGCTGTGAAGAAGCCATTTGATCTCAGGAGCAAGATGGGCAAGTG` (80 bp)
4. `ATGGTGGCTGAGGTTTGTTCAATGCCCGCTGCCTCTGCTGTGAAGAAGCCATTTGATCTCAGGAGCAAGATGGGCAAGTG` (80 bp)
5. `ATGGTGGCTGAGGTTTGTTCAATGCCCGCTGCCTCTGCTGTGAAGAAGCCATTTGATCTCAGGAGCAAGATGGGCAAGTG` (80 bp)

---

### 2.2 File: `Colorectal cancer .txt`

| Property | Value |
| :--- | :--- |
| **Exact Filename** | `Colorectal cancer .txt` (trailing space before `.txt`) |
| **File Size** | 37,840,477 bytes (36.09 MB / 34.42 MiB) |
| **Encoding** | `utf-8` |
| **Valid FASTA Format** | **Yes** |
| **Total FASTA Records** | 33,362 records |
| **Total Nucleotides** | 31,631,444 bp |
| **Length Statistics** | Min: 46 bp, Max: 14,412 bp, Mean: 948.13 bp |
| **GC-Content** | **33.59%** |
| **Alphabet Distribution** | `A`: 11,579,355 (36.61%), `C`: 4,400,040 (13.91%), `G`: 6,224,030 (19.68%), `T`: 9,427,998 (29.81%), `N`: 21 (0.00%) |
| **Internal Duplicates** | 4,744 duplicate sequences (28,618 unique sequences) |

#### First 5 FASTA Headers:
1. `>lcl|ADVG01000001.1_cds_EFH87891.1_1 [locus_tag=Krac_9242] [protein=hypothetical protein] [protein_id=EFH87891.1] [location=complement(<1..267)] [gbkey=CDS]`
2. `>lcl|ADVG01000001.1_cds_EFH87892.1_2 [locus_tag=Krac_9243] [protein=major facilitator superfamily MFS_1] [protein_id=EFH87892.1] [location=536..1801] [gbkey=CDS]`
3. `>lcl|ADVG01000001.1_cds_EFH87893.1_3 [locus_tag=Krac_9244] [protein=hypothetical protein] [protein_id=EFH87893.1] [location=complement(2568..3041)] [gbkey=CDS]`
4. `>lcl|ADVG01000001.1_cds_EFH87894.1_4 [locus_tag=Krac_9245] [protein=hypothetical protein] [protein_id=EFH87894.1] [location=complement(3143..3337)] [gbkey=CDS]`
5. `>lcl|ADVG01000001.1_cds_EFH87895.1_5 [locus_tag=Krac_9246] [protein=transcriptional regulator, XRE family] [protein_id=EFH87895.1] [location=complement(3330..4211)] [gbkey=CDS]`

#### First 5 Sequence Previews:
1. `ATGAATAAACAGTCCGAAGAGAACTCCTTTCACACCTCGTCGATACACAACATTCGGGATGGTATCCCACATGAAGCCAT` (80 bp)
2. `ATGAATTCGCCCTTGTCGCAAATGCAAACAGCACGGCCCACAACGGCATCACATTCGTTGCAGAATGTCTGGCTCTGGAT` (80 bp)
3. `GTGAACCACTCCCTTGAGGAAATTGTGCATGCTGAGAAGCAGTTTCAACAGGCAATGGTGAACAATGATGTACAAGCCCT` (80 bp)
4. `GTGCTTGATTGGTCCTGGAGTGCTGTACAAGCGCGTTGGTGTGATATCTACTCCTGCTATTCCCCCCATGAGCAACGATT` (80 bp)
5. `ATGAAGATGAAGAATGTGCTCAAGCGGGCCCGTCAAGAGCGTAGCTATTCGCAGGAGGAGGTTGCGGGCGCATTGGGAGT` (80 bp)

---

### 2.3 File: `Endometrial cancer .txt`

| Property | Value |
| :--- | :--- |
| **Exact Filename** | `Endometrial cancer .txt` (trailing space before `.txt`) |
| **File Size** | 3,571,164 bytes (3.41 MB / 3.25 MiB) |
| **Encoding** | `utf-8` |
| **Valid FASTA Format** | **Yes** |
| **Total FASTA Records** | 2,962 records |
| **Total Nucleotides** | 2,940,588 bp |
| **Length Statistics** | Min: 36 bp, Max: 5,508 bp, Mean: 992.77 bp |
| **GC-Content** | **52.70%** |
| **Alphabet Distribution** | `A`: 705,113 (23.98%), `C`: 736,792 (25.06%), `G`: 812,685 (27.64%), `T`: 685,998 (23.33%), `N`: 0 (0.00%) |
| **Internal Duplicates** | 8 duplicate sequences (2,954 unique sequences) |

#### First 5 FASTA Headers:
1. `>lcl|CP029347.1_cds_AWL10515.1_1 [locus_tag=HMF8227_00001] [db_xref=UniProtKB/Swiss-Prot:C1DFU2] [protein=Chromosomal replication initiator protein DnaA] [protein_id=AWL10515.1] [location=1..1398] [gbkey=CDS]`
2. `>lcl|CP029347.1_cds_AWL10516.1_2 [gene=dnaN] [locus_tag=HMF8227_00002] [db_xref=UniProtKB/Swiss-Prot:P0A990] [protein=DNA-directed DNA polymerase] [protein_id=AWL10516.1] [location=1418..2521] [gbkey=CDS]`
3. `>lcl|CP029347.1_cds_AWL10517.1_3 [locus_tag=HMF8227_00003] [db_xref=UniProtKB/Swiss-Prot:Q03D52] [protein=DNA replication and repair protein RecF] [protein_id=AWL10517.1] [location=2550..3656] [gbkey=CDS]`
4. `>lcl|CP029347.1_cds_AWL10518.1_4 [gene=gyrB] [locus_tag=HMF8227_00004] [db_xref=UniProtKB/Swiss-Prot:Q9I7C2] [protein=DNA topoisomerase (ATP-hydrolyzing)] [protein_id=AWL10518.1] [location=3657..6077] [gbkey=CDS]`
5. `>lcl|CP029347.1_cds_AWL10519.1_5 [locus_tag=HMF8227_00005] [protein=hypothetical protein] [protein_id=AWL10519.1] [location=complement(6182..6616)] [gbkey=CDS]`

#### First 5 Sequence Previews:
1. `ATGGCTTTGTGGGAGCAGTGTCTTAGACGGTTGCAACAGGAGTTGCCAACACAGCAGTTCAGCATGTGGATAAGACCGCT` (80 bp)
2. `ATGAAATTTACCATCAGCAGGGAAAACTTTTTACAACCACTTCAATTGGTCGCCGGTGCCGTTGAGCGTCGTCATACCTT` (80 bp)
3. `ATGAAACTGGACCACGTCCAGATAAAGCAGTTTCGGAATTTGACGGATGTAAGCCTGCATCCGTCCACCTCCTTAAACGT` (80 bp)
4. `ATGTCAGAAGAGCAGAATTACGACTCGTCCAGCATCAAGGTCCTAAAAGGACTGGATGCGGTGAGAAAGAGACCCGGTAT` (80 bp)
5. `ATGAAATACGCTTTTATCTCATCCACCATAGCGCTGGCTACGCTGTTAGCCGGTTGCTCCGTAACTAGGGACATACTTCC` (80 bp)

---

### 2.4 File: `Gastric cancer .txt`

| Property | Value |
| :--- | :--- |
| **Exact Filename** | `Gastric cancer .txt` (trailing space before `.txt`) |
| **File Size** | 491,527,857 bytes (468.76 MB / 447.06 MiB) |
| **Encoding** | `utf-8` |
| **Valid FASTA Format** | **Yes** |
| **Total FASTA Records** | **436,244 records** |
| **Total Nucleotides** | 409,780,217 bp |
| **Length Statistics** | Min: 45 bp, Max: 44,382 bp, Mean: 939.34 bp |
| **GC-Content** | **44.02%** |
| **Alphabet Distribution** | `A`: 121,013,778 (29.53%), `C`: 84,299,843 (20.57%), `G`: 96,097,799 (23.45%), `T`: 108,362,110 (26.44%), `N`: 6,664 (0.00%), IUPAC degenerate bases: `R` (17), `Y` (4), `W` (1), `M` (1) |
| **Internal Duplicates** | 43,365 duplicate sequences (392,879 unique sequences) |

#### First 5 FASTA Headers:
1. `>lcl|CP017839.1_cds_APA94092.1_1 [locus_tag=NS506_00001] [protein=hypothetical protein] [protein_id=APA94092.1] [location=243..1175] [gbkey=CDS]`
2. `>lcl|CP017839.1_cds_APA94093.1_2 [locus_tag=NS506_00002] [db_xref=UniProtKB/Swiss-Prot:A0LRL6] [protein=30S ribosomal protein S7] [protein_id=APA94093.1] [location=complement(1312..1782)] [gbkey=CDS]`
3. `>lcl|CP017839.1_cds_APA94094.1_3 [locus_tag=NS506_00003] [db_xref=UniProtKB/Swiss-Prot:Q1KVX0] [protein=30S ribosomal protein S12, chloroplastic] [protein_id=APA94094.1] [location=complement(1782..2156)] [gbkey=CDS]`
4. `>lcl|CP017839.1_cds_APA94095.1_4 [locus_tag=NS506_00004] [protein=hypothetical protein] [protein_id=APA94095.1] [location=complement(2634..2990)] [gbkey=CDS]`
5. `>lcl|CP017839.1_cds_APA94096.1_5 [locus_tag=NS506_00005] [protein=hypothetical protein] [protein_id=APA94096.1] [location=complement(3002..3640)] [gbkey=CDS]`

#### First 5 Sequence Previews:
1. `GTGGATGATCTCGTCTTCCTTGTTCGGCACGTGGCCGTCGGCGCCGCCGTCGTCCAGCGGGGTCGGCAGGTAGTCGATGA` (80 bp)
2. `ATGCCGCGTAAGGGTCCCGCTCCCAAGCGTCCCCTGATCAACGACCCGGTCTACGGCTCCCCGCTGGTGACGCAGCTGGT` (80 bp)
3. `ATGCCAACCATCAACCAGCTGGTCCGCAAGGGTCGCCGCGACAAGGTCGCCAAGACCAAGACTGCGGCCCTCCAGGGGAG` (80 bp)
4. `ATGCGCTCCCGGACCGCCCTGGCCCTGACCGTGCTCGCCCTGACCGTCACCCTGGCCGGCTGCGGCAAGACCGTGCCCGG` (80 bp)
5. `ATGGTTCGCGGCCGTCTCGATTCTGGACTGACGGAGCGGGGGAGCGAAGCGGAGGAGGGAAGAATCGAGACCTCAGGGCC` (80 bp)

---

### 2.5 File: `Liver Cancer .txt`

| Property | Value |
| :--- | :--- |
| **Exact Filename** | `Liver Cancer .txt` (trailing space before `.txt`, capital C) |
| **File Size** | 77,637,445 bytes (74.04 MB / 70.61 MiB) |
| **Encoding** | `utf-8` |
| **Valid FASTA Format** | **Yes** |
| **Total FASTA Records** | 66,826 records |
| **Total Nucleotides** | 63,331,982 bp |
| **Length Statistics** | Min: 75 bp, Max: 31,251 bp, Mean: 947.71 bp |
| **GC-Content** | **51.21%** |
| **Alphabet Distribution** | `A`: 16,357,899 (25.83%), `C`: 15,469,669 (24.43%), `G`: 16,958,740 (26.78%), `T`: 14,545,651 (22.97%), `N`: 23 (0.00%) |
| **Internal Duplicates** | 11,099 duplicate sequences (55,727 unique sequences) |

#### First 5 FASTA Headers:
1. `>lcl|QASA01000001.1_cds_RDC61422.1_1 [locus_tag=AHMF7616_00001] [db_xref=UniProtKB/Swiss-Prot:B5YGT9] [protein=Chromosomal replication initiator protein DnaA] [protein_id=RDC61422.1] [location=1..1425] [gbkey=CDS]`
2. `>lcl|QASA01000001.1_cds_RDC61423.1_2 [gene=iscS] [locus_tag=AHMF7616_00002] [db_xref=UniProtKB/Swiss-Prot:P9WME2] [protein=Cysteine desulfurase] [protein_id=RDC61423.1] [location=1447..1875] [gbkey=CDS]`
3. `>lcl|QASA01000001.1_cds_RDC61424.1_3 [locus_tag=AHMF7616_00003] [protein=hypothetical protein] [protein_id=RDC61424.1] [location=2106..4970] [gbkey=CDS]`
4. `>lcl|QASA01000001.1_cds_RDC61425.1_4 [locus_tag=AHMF7616_00004] [protein=hypothetical protein] [protein_id=RDC61425.1] [location=5176..5715] [gbkey=CDS]`
5. `>lcl|QASA01000001.1_cds_RDC61426.1_5 [gene=prdX] [locus_tag=AHMF7616_00005] [db_xref=UniProtKB/Swiss-Prot:O77834] [protein=Peroxiredoxin] [protein_id=RDC61426.1] [location=5951..6586] [gbkey=CDS]`

#### First 5 Sequence Previews:
1. `ATGATTAAAGACTGTACGACAGTATGGAAAAACTGTCTGCAGGTGATAAAGGACAGCATAGGGGAGCAGAGCTACAAAAC` (80 bp)
2. `ATGCTTTCAAAAAAAGCGAAATACGCTCTAAAAGCCCTCTTATACCTCACCAAAAACGCCGATAAAGGCCTCATTCTTAT` (80 bp)
3. `ATGAAAAAAACTTTTTTACTCCTTTACGTTCTGCTTATAGCTGTGCTAACCACCTATGGTCAGGAGAATAATATTATTGA` (80 bp)
4. `ATGAAAGCCATTACCTCTTTATATTCCAATGCCCTTGTTTCCGAAAGCAATTTCCGGAGTTCTTCTGATAAATTACGTAC` (80 bp)
5. `ATGGCTTTACGATTAGGGGATTTAGCTCCCGATTTTACAGCAGAAACGTCGCAAGGCGTTATCCATTTTCATGAGTGGCT` (80 bp)

---

## 3. Large-File Streaming Handling

The Gastric cancer file is **491,527,857 bytes** (~491.5 MB, containing 436,244 records and ~410 million nucleotides).

To fulfill the streaming requirements:
1. **Constant-Memory Generator:** Sequences were read line-by-line using a single-pass streaming parser in [`src/dna_cancer/data/fasta_inspector.py`](file:///d:/Cancer-Prediction-Using-DNA-Sequencing/src/dna_cancer/data/fasta_inspector.py).
2. **Zero In-Memory Sequence Storage:** Sequences were never collected into an in-memory list or DataFrame. Each sequence was processed immediately and discarded.
3. **C-Level Optimized Counting:** Nucleotide counting was delegated to C-level string counting (`seq.count()`), achieving execution speeds of **~55,000 records per second**. The 491.5 MB file streamed end-to-end in **7.74 seconds**.
4. **Memory-Capped Duplicate Hashing:** Duplicates and cross-file overlaps were tracked using 8-byte SHA-256 prefixes (`hashlib.sha256(seq).digest()[:8]`). Storing 436,244 8-byte hashes required under 25 MB of RAM, easily adhering to standard system limits without disk swaps or duplicate files.

---

## 4. Header & Biological Provenance Analysis

Header parsing revealed that **not a single header contains cancer diagnosis, tumor stage, histology, patient ID, or clinical provenance**.

Instead, headers contain standard NCBI GenBank CDS export annotations:

```text
>lcl|<accession>_cds_<protein_id>_<idx> [gene=<symbol>] [locus_tag=<locus>] [protein=<desc>] [location=<coords>] [gbkey=CDS]
```

### 4.1 Organism and Genomic Provenance by File:
- **`cervical_cancer.txt`**:
  - `NC_000015.10` (8,535 records): **Human Chromosome 15** reference assembly (`GRCh38.p14`). Genes include `GOLGA6L6`, `POTEB2`, `LOC124903442`.
  - `LC270024`–`LC270123`, `LC779559`–`LC779618`: **Human Papillomavirus type 16 (HPV-16)** complete genomes and oncogenes (`E6 protein`, `E7 protein`, `E1 protein`).
  - *Provenance Reality:* Human Chromosome 15 + HPV viral sequences.
- **`Colorectal cancer .txt`**:
  - `ADVG01000001` (3,094 records): *Ktedonobacter racemifer* SCRC/b-89 (filamentous soil bacterium).
  - `CP071092`–`CP071099` (15,000+ records): Enteric/bacterial complete genomes.
  - Bacterial proteins: ABC transporters, MFS multidrug transporters, toxin-antitoxin systems.
  - *Provenance Reality:* Bacterial genomes, likely downloaded from gut microbiome studies or general NCBI searches.
- **`Endometrial cancer .txt`**:
  - `CP029347.1` (2,962 records): A **single bacterial chromosome** in its entirety.
  - Bacterial replication genes: `dnaA`, `dnaN` (DNA polymerase III beta), `recF`, `gyrB` (DNA gyrase subunit B).
  - *Provenance Reality:* One single bacterial genome.
- **`Gastric cancer .txt`**:
  - `CP017839.1` and hundreds of whole-genome shotgun (WGS) contig series (`AKHM`, `AKHN`, `AKHO`, `QBP`, `QBQ`, `QBR`).
  - Over 105,000 hypothetical proteins, 30S ribosomal proteins (S7, S12), bacterial outer membrane proteins.
  - *Provenance Reality:* Metagenomic/WGS assembly contigs from bacterial and microbial isolates.
- **`Liver Cancer .txt`**:
  - `QASA01000001`, `CP003322`, `CP014486`, `CP021669`, `CP017247`, plus identical accessions (`ADVG01000001`, `CP003121`, `CP003116`).
  - Bacterial genes: `dnaA`, `iscS`, `prdX`.
  - *Provenance Reality:* Bacterial genomes overlapping directly with Colorectal and Gastric datasets.

---

## 5. Cross-File Comparison & Overlap Matrix

| Metric | `cervical_cancer` | `Colorectal cancer` | `Endometrial cancer` | `Gastric cancer` | `Liver Cancer` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FASTA Extension** | `.txt` | ` .txt` | ` .txt` | ` .txt` | ` .txt` |
| **Record Count** | 13,389 | 33,362 | 2,962 | **436,244** | 66,826 |
| **Total Bases** | 25,087,631 | 31,631,444 | 2,940,588 | 409,780,217 | 63,331,982 |
| **Mean Length (bp)** | 1,873.75 | 948.13 | 992.77 | 939.34 | 947.71 |
| **GC-Content (%)** | 47.05% | 33.59% | 52.70% | 44.02% | 51.21% |
| **Internal Duplicates**| 7,029 (52.5%) | 4,744 (14.2%) | 8 (0.3%) | 43,365 (9.9%) | 11,099 (16.6%) |
| **Biological Domain** | Human Chr15 + HPV | Bacterial Genomes | Single Bacterium | Metagenomic Contigs | Bacterial Genomes |

### Pairwise Sequence Overlap Matrix (Exact Sequence Identity):

```text
                  cervical   Colorectal   Endometrial   Gastric   Liver
cervical             -           17            0           0        0
Colorectal          17            -            0       1,857    7,360
Endometrial          0            0            -           0        0
Gastric              0        1,857            0           -   12,303
Liver                0        7,360            0      12,303        -
```

> [!CAUTION]
> **Conflicting Annotations Across Files:**
> - `Colorectal cancer .txt` and `Liver Cancer .txt` share **7,360 identical sequences**.
> - `Gastric cancer .txt` and `Liver Cancer .txt` share **12,303 identical sequences**.
> - `Colorectal cancer .txt` and `Gastric cancer .txt` share **1,857 identical sequences**.
>
> If ingested with filename labels, **21,520 sequences** would be assigned contradictory cancer diagnoses.

---

## 6. Gastric Dataset Investigation

The user specifically requested whether the Gastric file can reasonably be treated as a fifth class.

**Finding: The Gastric file CANNOT reasonably be treated as a fifth class.**

### Scientific Rationale:
1. **Catastrophic Imbalance:** At 436,244 records, Gastric contains **78.9%** of all sequences in the entire collection. The smallest class (Endometrial) has 2,962 records (a 147:1 imbalance ratio).
2. **Massive Sequence Conflict:** 12,303 sequences in Gastric are 100% identical to sequences labeled Liver Cancer, and 1,857 sequences are identical to sequences labeled Colorectal Cancer.
3. **Metagenomic Disparity:** While Endometrial is a single bacterial isolate and Cervical is human chromosome 15, Gastric is a massive metagenomic library of diverse environmental/microbial contigs.
4. **Historical Confirmation:** In `data_preprocessing.ipynb` Cell 1, the original authors saw `Gastric_cancer.csv` in their Drive directory but explicitly commented out its inclusion and dropped it from the final 4-class `DNAdata.csv` benchmark.

---

## 7. Recommended Experimental Strategy for Phase 4

Given that the raw FASTA collection contains cross-file duplicate contamination and conflates human with bacterial genomes:

1. **Do NOT ingest the raw 552k FASTA collection as cancer classes.** Ingesting them directly would build a model that classifies human vs. bacterial taxonomy rather than cancer biology.
2. **Use the Verified 1,200-sample College Benchmark (`DNAdata.csv`):**  
   The 1,200-sample dataset created in `data_preprocessing.ipynb` (300 sequences per class) represents the exact historical benchmark from the second-year college project. Once placed in `data/raw/DNAdata.csv`, it should be imported via `scripts/import_dataset.py`.
3. **Implement Proper ML Guardrails in Phase 4:**  
   - Vectorizers must be fitted strictly on `x_train`.
   - Use stratified 5-fold cross-validation.
   - Benchmark classic algorithms (Multinomial Naive Bayes, Logistic Regression, Random Forest).
   - Acknowledge that the high benchmark accuracy stems from microbial/chromosomal taxonomy differences rather than true clinical oncology.

