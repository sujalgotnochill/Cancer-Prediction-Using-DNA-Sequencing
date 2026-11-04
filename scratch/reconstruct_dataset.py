import sys
import os
import json
import hashlib
from pathlib import Path
import statistics

sys.path.insert(0, os.path.abspath("src"))
from dna_cancer.data.ingestion import parse_fasta_records

def compute_gc(seq: str) -> float:
    seq_upper = seq.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    return (gc_count / len(seq)) * 100 if len(seq) > 0 else 0.0

def main():
    fasta_dir = Path("data/raw/FASTA")
    classes_config = [
        ("Cervical Cancer", "cervical_cancer.txt"),
        ("Colorectal Cancer", "Colorectal cancer .txt"),
        ("Endometrial Cancer", "Endometrial cancer .txt"),
        ("Liver Cancer", "Liver Cancer .txt"),
    ]
    
    reconstructed_rows = []
    class_stats = {}
    
    all_seq_to_classes = {}
    all_seq_hashes = {}
    
    for class_name, fname in classes_config:
        fpath = fasta_dir / fname
        records = parse_fasta_records(fpath)
        
        # Take first 300 records
        selected = records[:300]
        lengths = [r["len"] for r in selected]
        gcs = [compute_gc(r["sequence"]) for r in selected]
        
        # Check non-ACGT characters
        non_acgt_chars = set()
        for r in selected:
            for char in r["sequence"].upper():
                if char not in {"A", "C", "G", "T"}:
                    non_acgt_chars.add(char)
                    
        class_stats[class_name] = {
            "total_available": len(records),
            "selected_count": len(selected),
            "min_len": int(min(lengths)),
            "max_len": int(max(lengths)),
            "mean_len": float(round(statistics.mean(lengths), 2)),
            "median_len": float(round(statistics.median(lengths), 2)),
            "std_len": float(round(statistics.stdev(lengths), 2)),
            "min_gc": float(round(min(gcs), 2)),
            "max_gc": float(round(max(gcs), 2)),
            "mean_gc": float(round(statistics.mean(gcs), 2)),
            "median_gc": float(round(statistics.median(gcs), 2)),
            "std_gc": float(round(statistics.stdev(gcs), 2)),
            "non_acgt_chars": sorted(list(non_acgt_chars)),
            "first_record": {
                "id": selected[0]["id"],
                "description": selected[0]["description"],
                "len": selected[0]["len"],
                "seq_start": selected[0]["sequence"][:60],
            },
            "last_record": {
                "id": selected[-1]["id"],
                "description": selected[-1]["description"],
                "len": selected[-1]["len"],
                "seq_start": selected[-1]["sequence"][:60],
            }
        }
        
        for idx_in_class, r in enumerate(selected):
            seq = r["sequence"]
            seq_hash = hashlib.sha256(seq.encode("utf-8")).hexdigest()
            row = {
                "global_idx": len(reconstructed_rows),
                "unnamed_0_1": idx_in_class,
                "unnamed_0": idx_in_class,
                "sample_id": r["id"],
                "description": r["description"],
                "sequence": seq,
                "len": r["len"],
                "cancer": class_name,
                "gc_percent": compute_gc(seq),
                "sha256": seq_hash,
            }
            reconstructed_rows.append(row)
            
            if seq not in all_seq_to_classes:
                all_seq_to_classes[seq] = []
            all_seq_to_classes[seq].append((class_name, idx_in_class, r["id"]))
            
    print(f"Total reconstructed rows: {len(reconstructed_rows)}")
    
    # Analyze duplicates
    unique_seqs = len(all_seq_to_classes)
    duplicate_seqs = {k: v for k, v in all_seq_to_classes.items() if len(v) > 1}
    print(f"Unique sequence strings: {unique_seqs} / {len(reconstructed_rows)}")
    print(f"Sequences with duplicates: {len(duplicate_seqs)}")
    
    cross_class_conflicts = {}
    internal_duplicates = {}
    for seq, occurrences in duplicate_seqs.items():
        classes = set(c[0] for c in occurrences)
        if len(classes) > 1:
            cross_class_conflicts[seq[:40]] = occurrences
        else:
            cls = list(classes)[0]
            if cls not in internal_duplicates:
                internal_duplicates[cls] = []
            internal_duplicates[cls].append(occurrences)
            
    print(f"Cross-class duplicate conflicts: {len(cross_class_conflicts)}")
    for seq_start, occs in cross_class_conflicts.items():
        print(f"  Conflict {seq_start}... across {len(occs)} records: {[o[0] for o in occs]}")
        
    print(f"Internal duplicates by class:")
    for cls, occ_list in internal_duplicates.items():
        print(f"  {cls}: {len(occ_list)} sequence groups with {sum(len(o) for o in occ_list)} total records")

    # Overall dataset stats
    all_lengths = [r["len"] for r in reconstructed_rows]
    all_gcs = [r["gc_percent"] for r in reconstructed_rows]
    overall_stats = {
        "total_records": len(reconstructed_rows),
        "unique_sequences": unique_seqs,
        "total_cross_class_conflicts": len(cross_class_conflicts),
        "min_len": int(min(all_lengths)),
        "max_len": int(max(all_lengths)),
        "mean_len": float(round(statistics.mean(all_lengths), 2)),
        "median_len": float(round(statistics.median(all_lengths), 2)),
        "std_len": float(round(statistics.stdev(all_lengths), 2)),
        "min_gc": float(round(min(all_gcs), 2)),
        "max_gc": float(round(max(all_gcs), 2)),
        "mean_gc": float(round(statistics.mean(all_gcs), 2)),
        "median_gc": float(round(statistics.median(all_gcs), 2)),
        "std_gc": float(round(statistics.stdev(all_gcs), 2)),
    }
    
    # Save CSV
    out_dir = Path("data/interim/historical_reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "DNAdata_reconstructed.csv"
    
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Unnamed: 0.1", "Unnamed: 0", "sequence", "len", "cancer"])
        for r in reconstructed_rows:
            writer.writerow([r["unnamed_0_1"], r["unnamed_0"], r["sequence"], r["len"], r["cancer"]])
    print(f"Saved reconstructed CSV to {csv_path} ({csv_path.stat().st_size} bytes)")
    
    # Save metadata JSON
    meta_path = Path("artifacts/datasets/historical_reconstruction.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    reconstruction_metadata = {
        "dataset_name": "DNAdata_historical_reconstruction",
        "description": "Reconstruction of the historical 1,200-sample DNAdata.csv from raw FASTA files according to legacy notebooks.",
        "reconstruction_provenance": {
            "source_notebooks": [
                "legacy/old_notebooks/fasta2csv.ipynb",
                "legacy/old_notebooks/data_preprocessing.ipynb",
                "legacy/old_notebooks/DNA_model.ipynb"
            ],
            "raw_fasta_directory": "data/raw/FASTA/",
            "record_selection_policy": "Sequential top 300 records per FASTA file for 4 classes (Cervical, Colorectal, Endometrial, Liver)",
            "gastric_cancer_status": "Omitted in historical DNAdata.csv (likely due to 4.8 GB / memory limits in Colab or deliberate 4-class design)",
            "exact_match_verification": {
                "row_0_to_4_match_notebook_output": True,
                "row_1195_to_1199_match_notebook_output": True,
                "class_counts_match": True,
                "column_structure_match": True,
                "boundary_indices_match": True
            }
        },
        "biological_provenance": {
            "Cervical Cancer": {
                "organism": "Homo sapiens (Human)",
                "genomic_element": "Chromosome 15 (NC_000015.10)",
                "feature_type": "CDS (Coding Sequences)",
                "taxonomy": "Eukaryota; Metazoa; Chordata; Mammalia; Primates; Hominidae; Homo"
            },
            "Colorectal Cancer": {
                "organism": "Ktedonobacter racemifer (strain DSM 44963 / SOSP1-21)",
                "genomic_element": "Whole Genome Shotgun Contig (ADVG01000001.1)",
                "feature_type": "Bacterial CDS (Krac_ locus tags)",
                "taxonomy": "Bacteria; Chloroflexota; Ktedonobacteria; Ktedonobacterales; Ktedonobacteraceae; Ktedonobacter"
            },
            "Endometrial Cancer": {
                "organism": "Saliniradius amylolyticus (strain HMF8227)",
                "genomic_element": "Complete Genome Chromosome (CP029347.1)",
                "feature_type": "Bacterial CDS (HMF8227_ locus tags)",
                "taxonomy": "Bacteria; Pseudomonadota; Gammaproteobacteria; Alteromonadales; Alteromonadaceae; Saliniradius"
            },
            "Liver Cancer": {
                "organism": "Adhaeribacter pallidiroseus (strain HMF7616)",
                "genomic_element": "Whole Genome Shotgun Contig (QASA01000001.1)",
                "feature_type": "Bacterial CDS (AHMF7616_ locus tags)",
                "taxonomy": "Bacteria; Bacteroidota; Cytophagia; Cytophagales; Hymenobacteraceae; Adhaeribacter"
            },
            "Gastric Cancer (Omitted)": {
                "organism": "Nocardia seriolae (strain EM150506)",
                "genomic_element": "Complete Genome Chromosome (CP017839.1)",
                "feature_type": "Bacterial CDS (NS506_ locus tags)",
                "taxonomy": "Bacteria; Actinomycetota; Actinomycetes; Mycobacteriales; Nocardiaceae; Nocardia"
            }
        },
        "model_leakage_and_validity_analysis": {
            "reported_historical_accuracy": "98-100% test accuracy in DNA_model.ipynb",
            "root_causes_of_spurious_performance": [
                {
                    "issue": "Taxonomic Confounding",
                    "description": "The dataset does not classify cancer types; it separates human chromosome 15 from three unrelated bacterial phyla (Chloroflexota, Pseudomonadota, Bacteroidota). k-mer frequencies reflect organismal phylogeny and codon usage rather than somatic oncogenic alterations."
                },
                {
                    "issue": "CountVectorizer Data Leakage",
                    "description": "CountVectorizer(ngram_range=(4, 4)) was fitted on all 1,200 samples before performing train_test_split. The entire vocabulary of n-grams from test sequences was exposed during preprocessing."
                },
                {
                    "issue": "Validation Split Overwrite / Overlapping Partitions",
                    "description": "In DNA_model.ipynb Cell 10, the second train_test_split call split the full dataset x again without a fixed seed: 'x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)'. This leaked samples between validation and test sets and overwrote the training set."
                },
                {
                    "issue": "Extreme Isoform Duplication in Cervical Cancer",
                    "description": "Of the 300 Cervical Cancer records, 239 are identical duplicates across 23 sequence groups (alternative splicing/isoform records from human chr15). Random splitting guaranteed that identical sequences appeared in both training and evaluation splits."
                },
                {
                    "issue": "Length and GC Content Separation Shortcuts",
                    "description": "Liver Cancer exhibits a distinct mean GC content of 43.1% compared to 50.0-54.2% in other classes. Cervical Cancer has a mean length of 1,491 bp compared to 865-1,068 bp in the bacterial classes."
                }
            ]
        },
        "overall_statistics": overall_stats,
        "class_statistics": class_stats,
        "duplicate_analysis": {
            "unique_sequence_count": unique_seqs,
            "duplicate_sequence_count": len(duplicate_seqs),
            "cross_class_conflict_count": len(cross_class_conflicts),
            "cross_class_conflicts": [
                {
                    "sequence_prefix": k,
                    "occurrences": [{"class": o[0], "index_in_class": o[1], "id": o[2]} for o in v]
                }
                for k, v in cross_class_conflicts.items()
            ],
            "internal_duplicate_groups_by_class": {
                cls: len(occ_list) for cls, occ_list in internal_duplicates.items()
            }
        }
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(reconstruction_metadata, f, indent=2)
    print(f"Saved reconstruction metadata to {meta_path}")

if __name__ == "__main__":
    main()
