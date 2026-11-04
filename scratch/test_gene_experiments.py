import sys
import os
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.ingestion import load_samples, ColumnMapping, parse_fasta_records
from dna_cancer.data.schema import DNASample
from dna_cancer.data.splitter import DatasetSplitter, verify_split_leakage
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models.baseline_models import LogisticRegressionClassifier
from dna_cancer.evaluation.evaluator import Evaluator

def attach_metadata(samples):
    files = [
        ("Cervical Cancer", "cervical_cancer.txt"),
        ("Colorectal Cancer", "Colorectal cancer .txt"),
        ("Endometrial Cancer", "Endometrial cancer .txt"),
        ("Liver Cancer", "Liver Cancer .txt")
    ]
    meta = {}
    for label, fname in files:
        records = parse_fasta_records(Path("data/raw/FASTA") / fname)[:300]
        for idx, r in enumerate(records):
            desc = r.get("description", "")
            m_gene = re.search(r"\[gene=([^\]]+)\]", desc)
            m_locus = re.search(r"\[locus_tag=([^\]]+)\]", desc)
            meta[(label, idx)] = {
                "gene": m_gene.group(1) if m_gene else None,
                "locus": m_locus.group(1) if m_locus else None,
            }
            
    for idx_in_class, s in enumerate(samples):
        class_idx = idx_in_class % 300
        m = meta.get((s.label, class_idx))
        if m:
            s.metadata.gene = m["gene"]
            s.metadata.locus = m["locus"]
            s.metadata.additional_metadata["gene"] = m["gene"]
            s.metadata.additional_metadata["locus"] = m["locus"]
    return samples

def main():
    dataset_path = Path("data/interim/historical_reconstruction/DNAdata_reconstructed.csv")
    mapping = ColumnMapping(sequence_col="sequence", label_col="cancer", id_col="Unnamed: 0")
    samples = load_samples(dataset_path, file_format="csv", column_mapping=mapping)
    samples = attach_metadata(samples)
    
    print("=" * 80)
    print("1. EVALUATING EXPERIMENT 1: GENE-FAMILY GROUPING ON FULL 1,200 DATASET")
    print("=" * 80)
    
    # Assign group keys: For Cervical -> gene_<gene_name>; For others -> gene_<label>_<gene> or seq_<hash>
    for s in samples:
        if s.label == "Cervical Cancer" and s.metadata.gene:
            s.metadata.additional_metadata["gene_family_group"] = f"gene_cervical_{s.metadata.gene}"
        elif s.metadata.gene:
            s.metadata.additional_metadata["gene_family_group"] = f"gene_{s.label}_{s.metadata.gene}"
        else:
            norm_seq = s.sequence.strip().upper()
            s.metadata.additional_metadata["gene_family_group"] = "seq_" + hashlib.sha256(norm_seq.encode("utf-8")).hexdigest()
            
    splitter_gf = DatasetSplitter(strategy="group", group_key="gene_family_group", train_size=0.70, val_size=0.15, test_size=0.15, seed=42)
    train_gf, val_gf, test_gf = splitter_gf.split(samples)
    
    def _dist(subset):
        d = defaultdict(int)
        for x in subset:
            d[x.label] += 1
        return dict(d)
        
    print(f"Full 1200 Gene-Family Split Counts:")
    print(f"  Train: {len(train_gf)} -> {_dist(train_gf)}")
    print(f"  Val:   {len(val_gf)} -> {_dist(val_gf)}")
    print(f"  Test:  {len(test_gf)} -> {_dist(test_gf)}")
    
    # Check gene disjointness for Cervical
    t_cerv_genes = {s.metadata.gene for s in train_gf if s.label == "Cervical Cancer"}
    v_cerv_genes = {s.metadata.gene for s in val_gf if s.label == "Cervical Cancer"}
    te_cerv_genes = {s.metadata.gene for s in test_gf if s.label == "Cervical Cancer"}
    print(f"  Cervical Gene Families:")
    print(f"    Train: {len(t_cerv_genes)} genes: {sorted(list(t_cerv_genes))[:5]}...")
    print(f"    Val:   {len(v_cerv_genes)} genes: {sorted(list(v_cerv_genes))}")
    print(f"    Test:  {len(te_cerv_genes)} genes: {sorted(list(te_cerv_genes))}")
    print(f"    Train & Test Gene Overlap: {t_cerv_genes & te_cerv_genes}")
    print(f"    Train & Val Gene Overlap:  {t_cerv_genes & v_cerv_genes}")
    print(f"    Val & Test Gene Overlap:   {v_cerv_genes & te_cerv_genes}")
    
    # Run Logistic Regression on k=5 and k=6
    for k in [5, 6]:
        fb = FeatureBuilder(k=k, vectorizer_type="count", return_sparse=True)
        fb.fit([s.sequence for s in train_gf], [s.label for s in train_gf])
        X_train = fb.transform_sequences([s.sequence for s in train_gf])
        y_train = fb.transform_labels([s.label for s in train_gf])
        X_val = fb.transform_sequences([s.sequence for s in val_gf])
        y_val = fb.transform_labels([s.label for s in val_gf])
        X_test = fb.transform_sequences([s.sequence for s in test_gf])
        y_test = fb.transform_labels([s.label for s in test_gf])
        
        clf = LogisticRegressionClassifier(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        evaluator = Evaluator(target_names=list(fb.label_encoder.classes_))
        val_m = evaluator.evaluate(clf, X_val, y_val)
        test_m = evaluator.evaluate(clf, X_test, y_test)
        
        print(f"\n  [k={k}] Gene-Family Grouped Model Performance:")
        print(f"    Val Accuracy:  {val_m['accuracy']:.4f}, Val Macro F1:  {val_m['f1_macro']:.4f}")
        print(f"    Test Accuracy: {test_m['accuracy']:.4f}, Test Macro F1: {test_m['f1_macro']:.4f}")
        print(f"    Test Confusion Matrix: {test_m['confusion_matrix']}")
        print(f"    Test Cervical F1: {test_m['classification_report']['Cervical Cancer']['f1-score']:.4f} (Recall: {test_m['classification_report']['Cervical Cancer']['recall']:.4f})")

    print("\n" + "=" * 80)
    print("2. EVALUATING EXPERIMENT 2: STRICT GENE-ANNOTATED SUBSET (N=448)")
    print("=" * 80)
    
    subset_samples = [s for s in samples if s.metadata.gene is not None]
    print(f"Filtered to {len(subset_samples)} samples with verified gene tags:")
    sub_counts = Counter(s.label for s in subset_samples)
    for lbl, count in sub_counts.items():
        print(f"  {lbl}: {count} samples")
        
    for s in subset_samples:
        s.metadata.additional_metadata["strict_gene_group"] = f"gene_{s.label}_{s.metadata.gene}"
        
    splitter_strict = DatasetSplitter(strategy="group", group_key="strict_gene_group", train_size=0.70, val_size=0.15, test_size=0.15, seed=42)
    train_st, val_st, test_st = splitter_strict.split(subset_samples)
    
    print(f"\nStrict Gene Subset Partition Counts:")
    print(f"  Train: {len(train_st)} -> {_dist(train_st)}")
    print(f"  Val:   {len(val_st)} -> {_dist(val_st)}")
    print(f"  Test:  {len(test_st)} -> {_dist(test_st)}")
    
    for k in [5, 6]:
        fb = FeatureBuilder(k=k, vectorizer_type="count", return_sparse=True)
        fb.fit([s.sequence for s in train_st], [s.label for s in train_st])
        X_train = fb.transform_sequences([s.sequence for s in train_st])
        y_train = fb.transform_labels([s.label for s in train_st])
        X_val = fb.transform_sequences([s.sequence for s in val_st])
        y_val = fb.transform_labels([s.label for s in val_st])
        X_test = fb.transform_sequences([s.sequence for s in test_st])
        y_test = fb.transform_labels([s.label for s in test_st])
        
        clf = LogisticRegressionClassifier(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        evaluator = Evaluator(target_names=list(fb.label_encoder.classes_))
        val_m = evaluator.evaluate(clf, X_val, y_val)
        test_m = evaluator.evaluate(clf, X_test, y_test)
        
        print(f"\n  [k={k}] Strict Gene Subset Performance (3-class):")
        print(f"    Val Accuracy:  {val_m['accuracy']:.4f}, Val Macro F1:  {val_m['f1_macro']:.4f}")
        print(f"    Test Accuracy: {test_m['accuracy']:.4f}, Test Macro F1: {test_m['f1_macro']:.4f}")
        print(f"    Test Confusion Matrix: {test_m['confusion_matrix']}")
        print(f"    Test Cervical F1: {test_m['classification_report']['Cervical Cancer']['f1-score']:.4f} (Recall: {test_m['classification_report']['Cervical Cancer']['recall']:.4f})")

if __name__ == "__main__":
    main()

