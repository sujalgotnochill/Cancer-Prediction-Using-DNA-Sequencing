import sys
import os
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.ingestion import parse_fasta_records

def main():
    fasta_dir = Path("data/raw/FASTA")
    files = {
        "Cervical Cancer": "cervical_cancer.txt",
        "Colorectal Cancer": "Colorectal cancer .txt",
        "Endometrial Cancer": "Endometrial cancer .txt",
        "Liver Cancer": "Liver Cancer .txt"
    }
    
    for label, fname in files.items():
        fpath = fasta_dir / fname
        print(f"=== {label} ({fname}) ===")
        records = parse_fasta_records(fpath)
        print(f"Total parsed: {len(records)}")
        print(f"First 3 records:")
        for i in range(min(3, len(records))):
            r = records[i]
            print(f"  [{i}] id={r['id']} len={r['len']} seq_start={r['sequence'][:35]}...")
        if len(records) >= 300:
            print(f"Records 295-299:")
            for i in range(295, min(300, len(records))):
                r = records[i]
                print(f"  [{i}] id={r['id']} len={r['len']} seq_start={r['sequence'][:35]}...")
        print()

if __name__ == "__main__":
    main()

