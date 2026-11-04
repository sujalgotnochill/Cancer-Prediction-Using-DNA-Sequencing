# Sealed Holdout Integrity & Disjointness Audit

## 1. Executive Summary

This audit independently verifies the mathematical integrity, sequence disjointness, and biological validity of the **Tier 2 Sealed Test Holdout** ($N=240$) established in `artifacts/vault/test_holdout_sealed.json`.

The sealed holdout represents the final, unbiased benchmark partition reserved strictly for Phase 5 one-shot model evaluation. This audit confirms that the holdout satisfies all isolation criteria:
- **Zero cross-partition identical sequences** across the entire 1,200-sample dataset.
- **Zero cross-partition human gene families** between Development ($N=960$) and Holdout ($N=240$).
- **Balanced class distribution**: Exactly 60 samples per class across all four diagnostic labels.
- **Quarantine of bacterial duplicate sequences**: Proven resolution of the `d6707574` duplicate pair in Colorectal cancer.
- **Intact procedural software locks**: Status confirmed as `LOCKED` with persistent SHA-256 integrity checksums.

---

## 2. Partition Architecture & Exact Sample Counts

The original historical dataset of 1,200 samples was partitioned into two physically separated cohorts:

```
Total Reconstructed Dataset (N = 1,200)
├── Tier 1: Development Cohort (N = 960, 80.0%)
│   ├── Cervical Cancer:    240 samples (23 human gene families)
│   ├── Colorectal Cancer:  240 samples (Ktedonobacter racemifer loci)
│   ├── Endometrial Cancer: 240 samples (Saliniradius amylolyticus loci)
│   └── Liver Cancer:        240 samples (Adhaeribacter pallidiroseus loci)
└── Tier 2: Sealed Test Vault (N = 240, 20.0%)
    ├── Cervical Cancer:     60 samples (8 unseen human gene families)
    ├── Colorectal Cancer:   60 samples (unseen K. racemifer loci)
    ├── Endometrial Cancer:  60 samples (unseen S. amylolyticus loci)
    └── Liver Cancer:        60 samples (unseen A. pallidiroseus loci)
```

---

## 3. Cervical Cancer Holdout Composition: 8 Unseen Gene Families

The 60 Cervical Cancer samples allocated to the sealed holdout comprise transcripts from **8 distinct human gene families** on *Homo sapiens* Chromosome 15. None of these 8 gene families appear anywhere in the Tier 1 Development Cohort:

| Gene Family Symbol | NCBI GeneID | Holdout Sample Count | Record Indices | Representative Accession |
|---|---|---|---|---|
| `TUBGCP5` | 114791 | 17 | 86..102 | `NP_001341303.1` |
| `CYFIP1` | 23191 | 10 | 76..85 | `NP_001311048.1` |
| `ATP10A` | 57194 | 9 | 278..286 | `XP_011520130.1` |
| `GABRA5` | 2558 | 7 | 293..299 | `NP_001158509.1` |
| `GABRB3` | 2562 | 6 | 287..292 | `NP_068712.1` |
| `POTEB` | 100996331 | 4 | 14..17 | `NP_001264233.1` |
| `POTEB2` | 100287399 | 4 | 2..5 | `NP_001264232.1` |
| `SNURF` | 8926 | 3 | 115..117 | `NP_005669.2` |
| **Total** | — | **60** | — | — |

### Verification of Gene Disjointness
- Total human gene families in reconstructed dataset: **31**
- Gene families assigned to Development: **23** ($N=240$)
- Gene families assigned to Sealed Holdout: **8** ($N=60$)
- **Gene Family Intersection: $\emptyset$ (0 genes, 0.00% overlap)**

---

## 4. Bacterial Classes: Holdout Selection & Duplicate Sequence Handling

### 4.1 Selection Methodology
In the reconstructed dataset, each class consists of 300 sequential records extracted from the original FASTA files. For the bacterial classes, selection proceeded as follows:

1. **Endometrial Cancer (*Saliniradius amylolyticus*)**:
   - Total records: 300.
   - Unique sequences: 300 (0 duplicates).
   - Holdout allocation: Records 240..299 ($N=60$).
   - Development allocation: Records 0..239 ($N=240$).
2. **Liver Cancer (*Adhaeribacter pallidiroseus*)**:
   - Total records: 300.
   - Unique sequences: 300 (0 duplicates).
   - Holdout allocation: Records 240..299 ($N=60$).
   - Development allocation: Records 0..239 ($N=240$).
3. **Colorectal Cancer (*Ktedonobacter racemifer*)**:
   - Total records: 300.
   - Unique sequences: 298 (2 duplicate pairs present).

### 4.2 Forensic Resolution of Colorectal Sequence Duplicates
Forensic inspection of the 300 Colorectal cancer records revealed exactly two duplicate pairs:
- **Duplicate Pair 1 (Hash `dc4730e5...`)**: Records 39 and 40.
  - Both records fall well within the Development partition (records 0..239). They remain together in development.
- **Duplicate Pair 2 (Hash `d6707574...`)**: Records 90 and 256.
  - If records 240..299 had been naively sliced as the holdout, record 256 would have been placed in the holdout while record 90 was placed in development.
  - This would have created an **exact-sequence data leakage channel**, artificially inflating holdout accuracy on Colorectal cancer.

### Resolution Protocol
To prevent duplicate sequence leakage while maintaining exactly 60 holdout samples:
1. Record index **256 is retained in the Development cohort** alongside its identical duplicate at record 90.
2. The holdout partition takes records **240..299 excluding index 256**, and recruits record **index 239** to complete the 60-sample requirement:
   $$\text{Holdout}_{\text{Colorectal}} = \{239\} \cup \{240..299\} \setminus \{256\} \quad (|S| = 60)$$
3. Both samples of the duplicate pair remain isolated within the Development cohort.

### Verification of Cross-Partition Sequence Disjointness
A full hash collision scan was performed comparing all 960 sequences in `data/processed/development_cohort.csv` against all 240 sequences in `artifacts/vault/test_holdout_sealed.json`:
- **Identical Sequence Overlap: 0 sequences (0.00%)**
- **Hash Collisions: 0**

---

## 5. Biological Scope: What Does the Bacterial Holdout Test?

It is essential to clarify the scientific and genomic meaning of the bacterial holdout partitions to prevent misinterpretation:

### 5.1 Bacterial vs Human Genome Architecture
1. **Human Transcripts (Cervical Cancer)**:
   - Transcripts originate from human Chromosome 15 protein-coding genes.
   - Alternative splicing produces multiple distinct transcript isoforms from the same gene locus (e.g., 112 isoforms for `SNRPN`, 48 for `UBE3A`).
   - Spliced isoforms share extensive exon sequences, creating high within-gene sequence similarity.
   - Therefore, the human holdout tests **out-of-gene generalization**: can a model recognize human coding sequences from gene families it has never seen during training?
2. **Bacterial Genomes (Colorectal, Endometrial, Liver)**:
   - The sequences originate from distinct bacterial species (*K. racemifer*, *S. amylolyticus*, *A. pallidiroseus*).
   - Bacterial genomes possess single, continuous circular chromosomes without alternative splice isoforms.
   - In the reconstructed dataset, the bacterial records represent distinct coding sequences and genomic loci along their respective circular genomes.
   - Therefore, the bacterial holdout tests generalization to **unseen genomic loci / sequences** within each bacterial species genome. It does **not** test multi-transcript gene family generalization.

---

## 6. Vault Manifest & Integrity Checksums

The integrity of all dataset artifacts is persistently tracked in `artifacts/vault/test_sealed_manifest.json`:

| Artifact | File Path | Sample Count | SHA-256 Checksum |
|---|---|---|---|
| **Source Reconstructed Dataset** | `data/interim/historical_reconstruction/DNAdata_reconstructed.csv` | 1,200 | `eb14d47f3eeea2684313ff8a5cb726f431fbfba62eaf371f618e3a444a245bc3` |
| **Tier 1 Development Cohort** | `data/processed/development_cohort.csv` | 960 | `7c35fc4f226488cfc2d37ed332746be3f3c1aa8982ab7a8991c7fb374781475c` |
| **Tier 2 Sealed Test Vault** | `artifacts/vault/test_holdout_sealed.json` | 240 | `430a2e6f8b4b85dfb8bd6a16300e44417a374853a2eb7616a8e5d6d9116db940` |

### Procedural Safeguard Verification
1. `vault.get_vault_status()` confirms `is_locked: True` and `status: "LOCKED"`.
2. `vault.verify_integrity()` validates that the current on-disk files match their manifest SHA-256 checksums exactly.
3. The sealed vault remains completely untouched and unsealed during Phase 4.4.

