import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict

# Add src
sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.ingestion import load_samples, ColumnMapping
from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.splitter import DatasetSplitter, verify_split_leakage
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models.baseline_models import (
    MultinomialNBClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifierWrapper,
)
from dna_cancer.evaluation.evaluator import Evaluator
from dna_cancer.evaluation.metrics import compute_classification_metrics

def main():
    print("=" * 80)
    print("  PHASE 4.1 AUDIT: FORENSIC INVESTIGATION OF BASELINE EXPERIMENTS")
    print("=" * 80)
    
    # 1. Dataset verification
    dataset_path = Path("data/interim/historical_reconstruction/DNAdata_reconstructed.csv")
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
        
    actual_checksum = calculate_file_checksum(dataset_path)
    file_size_bytes = dataset_path.stat().st_size
    print(f"\n1. DATASET FILE VERIFICATION:")
    print(f"  Path: {dataset_path}")
    print(f"  Size: {file_size_bytes} bytes")
    print(f"  SHA-256 Checksum: {actual_checksum}")
    
    mapping = ColumnMapping(sequence_col="sequence", label_col="cancer", id_col="Unnamed: 0")
    samples = load_samples(dataset_path, file_format="csv", column_mapping=mapping)
    print(f"  Total Samples Loaded: {len(samples)}")
    
    class_counts = defaultdict(int)
    for s in samples:
        class_counts[s.label] += 1
    print(f"  Class Distribution: {dict(class_counts)}")
    
    # Verify sequence groups
    seq_groups = defaultdict(list)
    for idx, s in enumerate(samples):
        seq_groups[s.sequence.strip().upper()].append((idx, s.label, s.sample_id))
    print(f"  Unique Sequences: {len(seq_groups)}")
    
    duplicate_seq_counts = {k: len(v) for k, v in seq_groups.items() if len(v) > 1}
    print(f"  Number of sequence groups with duplicates: {len(duplicate_seq_counts)}")
    print(f"  Total records in duplicate groups: {sum(duplicate_seq_counts.values())}")
    
    # Check duplicates by class
    class_duplicates = defaultdict(list)
    for seq, records in seq_groups.items():
        if len(records) > 1:
            classes = set(r[1] for r in records)
            for c in classes:
                class_duplicates[c].append(records)
    for c, groups in class_duplicates.items():
        print(f"    Class '{c}': {len(groups)} duplicate sequence groups containing {sum(len(g) for g in groups)} total records")
        
    # 2. Partitions Audit
    print(f"\n2. DATA PARTITION AUDIT:")
    for strat in ["duplicate_grouped", "stratified"]:
        splitter = DatasetSplitter(strategy=strat, train_size=0.70, val_size=0.15, test_size=0.15, seed=42)
        train_s, val_s, test_s = splitter.split(samples)
        
        def _get_class_dist(sub):
            d = defaultdict(int)
            for x in sub:
                d[x.label] += 1
            return dict(d)
            
        train_dist = _get_class_dist(train_s)
        val_dist = _get_class_dist(val_s)
        test_dist = _get_class_dist(test_s)
        
        leak_audit = verify_split_leakage(train_s, val_s, test_s)
        
        # Sequence overlaps
        t_seqs = {x.sequence.strip().upper() for x in train_s}
        v_seqs = {x.sequence.strip().upper() for x in val_s}
        te_seqs = {x.sequence.strip().upper() for x in test_s}
        
        print(f"\n  --- Strategy: '{strat}' ---")
        print(f"    Total Counts: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)} (Sum={len(train_s)+len(val_s)+len(test_s)})")
        print(f"    Train Class Counts: {train_dist}")
        print(f"    Val Class Counts:   {val_dist}")
        print(f"    Test Class Counts:  {test_dist}")
        print(f"    Leakage-Free (Strict Sequence Disjointness): {leak_audit['is_leakage_free']}")
        print(f"    Train-Val Sequence Overlap: {len(t_seqs & v_seqs)}")
        print(f"    Train-Test Sequence Overlap: {len(t_seqs & te_seqs)}")
        print(f"    Val-Test Sequence Overlap: {len(v_seqs & te_seqs)}")
        
        if len(t_seqs & te_seqs) > 0:
            overlap_labels = defaultdict(int)
            for s in test_s:
                if s.sequence.strip().upper() in t_seqs:
                    overlap_labels[s.label] += 1
            print(f"    Test samples whose sequences were present in Train: {dict(overlap_labels)}")

    # 3. Check gene / isoform metadata crossing
    print(f"\n3. TRANSCRIPT / GENE ISOFORM OVERLAP AUDIT:")
    # Check if raw FASTA descriptions have gene or locus info
    # Load raw FASTA for Cervical Cancer to map sequence to gene name
    from dna_cancer.data.ingestion import parse_fasta_records
    cervical_fasta = parse_fasta_records("data/raw/FASTA/cervical_cancer.txt")[:300]
    cervical_genes = []
    import re
    gene_pattern = re.compile(r"\[gene=([^\]]+)\]")
    for r in cervical_fasta:
        m = gene_pattern.search(r.get("description", ""))
        gene = m.group(1) if m else "unknown"
        cervical_genes.append((r["sequence"], gene))
        
    seq_to_gene = {s: g for s, g in cervical_genes}
    
    # Audit duplicate_grouped split for Cervical gene crossing
    splitter_dg = DatasetSplitter(strategy="duplicate_grouped", train_size=0.70, val_size=0.15, test_size=0.15, seed=42)
    train_dg, val_dg, test_dg = splitter_dg.split(samples)
    
    train_cervical_genes = {seq_to_gene.get(s.sequence, "unknown") for s in train_dg if s.label == "Cervical Cancer"}
    val_cervical_genes = {seq_to_gene.get(s.sequence, "unknown") for s in val_dg if s.label == "Cervical Cancer"}
    test_cervical_genes = {seq_to_gene.get(s.sequence, "unknown") for s in test_dg if s.label == "Cervical Cancer"}
    
    print(f"  Cervical Distinct Genes in Train: {len(train_cervical_genes)}")
    print(f"  Cervical Distinct Genes in Val:   {len(val_cervical_genes)}")
    print(f"  Cervical Distinct Genes in Test:  {len(test_cervical_genes)}")
    print(f"  Train & Test Gene Name Overlap: {train_cervical_genes & test_cervical_genes}")
    print(f"  Train & Val Gene Name Overlap:  {train_cervical_genes & val_cervical_genes}")

    # 4. Feature engineering audit
    print(f"\n4. FEATURE ENGINEERING AUDIT:")
    for k in [3, 4, 5, 6]:
        fb = FeatureBuilder(k=k, vectorizer_type="count", return_sparse=True)
        train_seqs = [s.sequence for s in train_dg]
        fb.fit(train_seqs, [s.label for s in train_dg])
        
        expected_dim = 4 ** k
        actual_dim = fb.n_features
        print(f"  k={k}: Expected Max Dim=4^{k}={expected_dim}, Actual Fitted Dim={actual_dim}")
        
    # 5. Reproducibility test of best Logistic Regression (k=5 and k=6)
    print(f"\n5. REPRODUCIBILITY VERIFICATION (Logistic Regression, duplicate_grouped, k=5 and k=6):")
    for k in [5, 6]:
        fb = FeatureBuilder(k=k, vectorizer_type="count", return_sparse=True)
        train_seqs = [s.sequence for s in train_dg]
        train_labels = [s.label for s in train_dg]
        val_seqs = [s.sequence for s in val_dg]
        val_labels = [s.label for s in val_dg]
        test_seqs = [s.sequence for s in test_dg]
        test_labels = [s.label for s in test_dg]
        
        fb.fit(train_seqs, train_labels)
        X_train = fb.transform_sequences(train_seqs)
        y_train = fb.transform_labels(train_labels)
        X_val = fb.transform_sequences(val_seqs)
        y_val = fb.transform_labels(val_labels)
        X_test = fb.transform_sequences(test_seqs)
        y_test = fb.transform_labels(test_labels)
        
        clf = LogisticRegressionClassifier(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        evaluator = Evaluator(target_names=list(fb.label_encoder.classes_))
        test_m = evaluator.evaluate(clf, X_test, y_test)
        
        print(f"  --- k={k} Re-run Results ---")
        print(f"    Test Accuracy:     {test_m['accuracy']:.4f}")
        print(f"    Test Macro Prec:   {test_m['precision_macro']:.4f}")
        print(f"    Test Macro Recall: {test_m['recall_macro']:.4f}")
        print(f"    Test Macro F1:     {test_m['f1_macro']:.4f}")
        print(f"    Confusion Matrix:  {test_m['confusion_matrix']}")
        
        # Check against logged values in runs_summary.json
        with open("artifacts/experiments/runs_summary.json", "r") as f:
            runs = json.load(f)
        logged_match = [r for r in runs if r["model"] == "LogisticRegression" and r["split_strategy"] == "duplicate_grouped" and r["k"] == k]
        if logged_match:
            logged_acc = logged_match[0]["accuracy"]
            logged_f1 = logged_match[0]["f1_macro"]
            print(f"    Logged Accuracy:   {logged_acc:.4f} (Match: {abs(logged_acc - test_m['accuracy']) < 1e-6})")
            print(f"    Logged Macro F1:   {logged_f1:.4f} (Match: {abs(logged_f1 - test_m['f1_macro']) < 1e-6})")

    # 6. Audit 24 runs records
    print(f"\n6. EXPERIMENT RECORDS AUDIT (artifacts/experiments/):")
    runs_dir = Path("artifacts/experiments")
    run_files = list(runs_dir.glob("*.json"))
    summary_file = runs_dir / "runs_summary.json"
    individual_runs = [f for f in run_files if f.name != "runs_summary.json"]
    print(f"  Total Run JSON Files: {len(individual_runs)}")
    
    with open(summary_file, "r") as f:
        summary_records = json.load(f)
    print(f"  Total Summary Entries: {len(summary_records)}")
    
    # Check for parameter completeness and uniqueness
    run_names = set()
    missing_fields = []
    for r in individual_runs:
        with open(r, "r") as f:
            data = json.load(f)
        run_name = data.get("run_name")
        if run_name in run_names:
            print(f"    Duplicate run name: {run_name}")
        run_names.add(run_name)
        
        # Check required fields
        required = ["run_id", "run_name", "dataset", "split_configuration", "feature_configuration", "model", "durations", "metrics", "scientific_limitation_notice"]
        for field in required:
            if field not in data:
                missing_fields.append((r.name, field))
                
    print(f"  Unique Run Names: {len(run_names)}")
    print(f"  Missing Required Fields: {len(missing_fields)}")
    print("=" * 80)

if __name__ == "__main__":
    main()

