import sys
import os
import re
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.ingestion import load_samples, ColumnMapping, parse_fasta_records
from dna_cancer.data.schema import DNASample

def get_fasta_metadata():
    files = [
        ("Cervical Cancer", "cervical_cancer.txt"),
        ("Colorectal Cancer", "Colorectal cancer .txt"),
        ("Endometrial Cancer", "Endometrial cancer .txt"),
        ("Liver Cancer", "Liver Cancer .txt")
    ]
    meta_by_label_idx = {}
    for label, fname in files:
        records = parse_fasta_records(Path("data/raw/FASTA") / fname)[:300]
        for idx, r in enumerate(records):
            desc = r.get("description", "")
            m_gene = re.search(r"\[gene=([^\]]+)\]", desc)
            m_locus = re.search(r"\[locus_tag=([^\]]+)\]", desc)
            m_prot = re.search(r"\[protein=([^\]]+)\]", desc)
            meta_by_label_idx[(label, idx)] = {
                "gene": m_gene.group(1) if m_gene else None,
                "locus": m_locus.group(1) if m_locus else None,
                "protein": m_prot.group(1) if m_prot else None,
                "id": r.get("id"),
            }
    return meta_by_label_idx

def main():
    print("=== Testing Gene Family Metadata and Splitting ===")
    meta = get_fasta_metadata()
    print(f"Total mapped FASTA metadata records: {len(meta)}")
    
    # Load 1200 samples
    dataset_path = Path("data/interim/historical_reconstruction/DNAdata_reconstructed.csv")
    mapping = ColumnMapping(sequence_col="sequence", label_col="cancer", id_col="Unnamed: 0")
    samples = load_samples(dataset_path, file_format="csv", column_mapping=mapping)
    print(f"Loaded {len(samples)} samples from CSV")
    
    # Attach gene metadata to samples
    c_counts = defaultdict(int)
    for idx_in_class, s in enumerate(samples):
        # Index in class: 0..299
        class_idx = idx_in_class % 300
        m = meta.get((s.label, class_idx))
        if m:
            s.metadata.gene = m["gene"]
            s.metadata.locus = m["locus"]
            s.metadata.additional_metadata["protein"] = m["protein"]
            if m["gene"]:
                c_counts[s.label] += 1
                
    print("Samples with explicit 'gene' metadata:")
    for lbl, count in c_counts.items():
        print(f"  {lbl}: {count} / 300 ({count/300*100:.1f}%)")

if __name__ == "__main__":
    main()

