"""Unit tests for gene-family grouped splitting and leakage verification."""

import os
import sys
import unittest
from typing import List

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.schema import DNASample, SampleMetadata
from dna_cancer.data.splitter import (
    DatasetSplitter,
    split_gene_family_grouped,
    verify_gene_family_leakage,
)
from dna_cancer.utils.exceptions import DataValidationError


class TestGeneFamilySplitting(unittest.TestCase):
    """Test gene-family grouping algorithms, strict filtering, and leakage verification."""

    def test_split_gene_family_grouped_prevents_leakage(self) -> None:
        # Create samples representing 10 different gene families
        samples: List[DNASample] = []
        for i in range(10):
            gene_name = f"GENE_{i}"
            label = "Cancer_A" if i % 2 == 0 else "Cancer_B"
            for j in range(4):  # 4 splice variants/transcripts per gene
                sample = DNASample(
                    sample_id=f"S_{i}_{j}",
                    sequence=f"ACGTACGT{i}{j}ACGT",
                    label=label,
                )
                sample.metadata.gene = gene_name
                samples.append(sample)

        train, val, test = split_gene_family_grouped(
            samples,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            seed=42,
        )

        audit = verify_gene_family_leakage(train, val, test)
        self.assertTrue(audit["is_leakage_free"])
        self.assertEqual(audit["train_val_overlap"], [])
        self.assertEqual(audit["train_test_overlap"], [])
        self.assertEqual(audit["val_test_overlap"], [])
        self.assertEqual(audit["total_annotated_genes"], 10)
        self.assertEqual(len(train) + len(val) + len(test), 40)

    def test_split_gene_family_grouped_fallback_to_seq_hash(self) -> None:
        # Create samples where some have genes, some do not (unannotated duplicates)
        samples: List[DNASample] = []
        # 3 with genes
        for i in range(3):
            s = DNASample(f"G_{i}", f"ACGT_{i}", "LUAD")
            s.metadata.gene = f"GENE_{i}"
            samples.append(s)

        # 6 unannotated duplicates (2 unique sequences, each repeated 3 times)
        for i in range(3):
            samples.append(DNASample(f"U1_{i}", "CCCCGGGGAAAA", "LUAD"))
            samples.append(DNASample(f"U2_{i}", "TTTTAAAACCCC", "BRCA"))

        train, val, test = split_gene_family_grouped(
            samples,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            seed=42,
            strict_metadata_only=False,
        )

        self.assertEqual(len(train) + len(val) + len(test), 9)

        # Check identical unannotated sequences do not cross partitions
        train_seqs = {s.sequence for s in train}
        val_seqs = {s.sequence for s in val}
        test_seqs = {s.sequence for s in test}

        self.assertEqual(train_seqs & val_seqs, set())
        self.assertEqual(train_seqs & test_seqs, set())
        self.assertEqual(val_seqs & test_seqs, set())

    def test_split_gene_family_grouped_strict_metadata_only(self) -> None:
        samples: List[DNASample] = []
        # 6 samples with gene tags
        for i in range(6):
            s = DNASample(f"G_{i}", f"ACGT_{i}", "LUAD" if i % 2 == 0 else "BRCA")
            s.metadata.gene = f"GENE_{i}"
            samples.append(s)

        # 4 samples without gene tags
        for i in range(4):
            samples.append(DNASample(f"U_{i}", f"TTTT_{i}", "LUAD"))

        train, val, test = split_gene_family_grouped(
            samples,
            train_size=0.5,
            val_size=0.25,
            test_size=0.25,
            seed=42,
            strict_metadata_only=True,
        )

        # Should only partition the 6 samples with gene tags
        self.assertEqual(len(train) + len(val) + len(test), 6)
        for s in train + val + test:
            self.assertIsNotNone(s.metadata.gene)

    def test_split_gene_family_grouped_strict_insufficient_samples(self) -> None:
        samples = [
            DNASample("S1", "ACGT", "LUAD"),
            DNASample("S2", "TGCA", "LUAD"),
        ]
        samples[0].metadata.gene = "GENE_1"
        # Only 1 sample has gene metadata
        with self.assertRaises(DataValidationError):
            split_gene_family_grouped(samples, strict_metadata_only=True)

    def test_verify_gene_family_leakage_detects_leak(self) -> None:
        s_train = DNASample("T1", "ACGT", "LUAD")
        s_train.metadata.gene = "TP53"

        s_val = DNASample("V1", "TGCA", "LUAD")
        s_val.metadata.gene = "BRCA1"

        s_test = DNASample("TE1", "GGCC", "LUAD")
        s_test.metadata.gene = "TP53"  # Same gene as train!

        audit = verify_gene_family_leakage([s_train], [s_val], [s_test])
        self.assertFalse(audit["is_leakage_free"])
        self.assertIn("TP53", audit["train_test_overlap"])
        self.assertEqual(audit["train_val_overlap"], [])

    def test_dataset_splitter_with_gene_family_grouped(self) -> None:
        samples: List[DNASample] = []
        for i in range(8):
            s = DNASample(f"S_{i}", f"ACGT_{i}", "LUAD" if i % 2 == 0 else "BRCA")
            s.metadata.gene = f"GENE_{i}"
            samples.append(s)

        splitter = DatasetSplitter(strategy="gene_family_grouped", seed=42)
        train, val, test = splitter.split(samples)

        self.assertEqual(len(train) + len(val) + len(test), 8)
        self.assertEqual(splitter.last_strategy_used, "gene_family_grouped")

    def test_within_organism_vs_cross_organism_modes(self) -> None:
        # Create samples from two different organisms sharing matching gene symbol 'coxA'
        s1 = DNASample("S1", "ACGTACGT", "Endometrial_Cancer")
        s1.metadata.gene = "coxA"
        s1.metadata.organism = "Saliniradius_amylolyticus"

        s2 = DNASample("S2", "TGCATGCA", "Liver_Cancer")
        s2.metadata.gene = "coxA"
        s2.metadata.organism = "Adhaeribacter_pallidiroseus"

        # Additional samples to allow 3-way split
        other_samples = [
            DNASample(f"O_{i}", f"GGCC_{i}", "Endometrial_Cancer" if i % 2 == 0 else "Liver_Cancer")
            for i in range(6)
        ]
        for i, s in enumerate(other_samples):
            s.metadata.gene = f"unique_gene_{i}"

        all_samples = [s1, s2] + other_samples

        # In within_organism mode (default), s1 and s2 are distinct groups
        tr, va, te = split_gene_family_grouped(all_samples, grouping_mode="within_organism", seed=42)
        audit_within = verify_gene_family_leakage(tr, va, te, grouping_mode="within_organism")
        self.assertTrue(audit_within["is_leakage_free"])

        # Check cross_organism mode
        tr_c, va_c, te_c = split_gene_family_grouped(all_samples, grouping_mode="cross_organism", seed=42)
        audit_cross = verify_gene_family_leakage(tr_c, va_c, te_c, grouping_mode="cross_organism")
        self.assertTrue(audit_cross["is_leakage_free"])


if __name__ == "__main__":
    unittest.main()
