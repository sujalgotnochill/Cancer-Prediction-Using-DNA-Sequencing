"""Generate final evaluation protocol specification JSON artifact."""

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.ingestion import ColumnMapping, load_samples, parse_fasta_records


def main():
    spec_data = {
        "specification_metadata": {
            "name": "Phase 5 Final Two-Tier Sealed Evaluation Protocol",
            "version": "1.0.0",
            "phase": "Phase 4.3",
            "date": "2026-11-04",
            "author": "Senior ML Evaluation Engineer & Computational Biology Researcher",
            "dataset_sha256": "eb14d47f3eee44cb2e2fc1ef75b281f6966dd2ee9957d079379685a53e9a567c",
            "total_dataset_samples": 1200,
        },
        "evaluation_architecture": {
            "tier_1_development_cohort": {
                "sample_count": 960,
                "percentage_of_total": 80.0,
                "purpose": "All exploratory modeling, k-mer vocabulary sizing, regularizer optimization, architecture search, and model selection.",
                "class_distribution": {
                    "Cervical Cancer": 240,
                    "Colorectal Cancer": 240,
                    "Endometrial Cancer": 240,
                    "Liver Cancer": 240
                },
                "cervical_gene_composition": {
                    "gene_family_count": 23,
                    "total_cervical_samples": 240,
                    "included_genes": [
                        "SNRPN (112)", "NIPA2 (51)", "UBE3A (48)", "POTEB3 (4)", "GOLGA6L2 (3)",
                        "GOLGA6L22 (2)", "NIPA1 (2)", "GOLGA6L26 (2)", "GOLGA8S (2)", "LOC124903442 (1)",
                        "GOLGA6L6 (1)", "OR4M2B (1)", "OR4N4C (1)", "LOC102724737 (1)", "LOC102724971 (1)",
                        "OR4M2 (1)", "OR4N4 (1)", "LOC107983983 (1)", "GOLGA6L1 (1)", "MKRN3 (1)",
                        "MAGEL2 (1)", "NDN (1)", "NPAP1 (1)"
                    ]
                },
                "model_selection_strategy": {
                    "primary_scheme": "Group-Balanced 3-Fold Cross-Validation",
                    "grouping_mode": "within_organism",
                    "selection_metric": "Out-of-fold Validation Macro F1 Score",
                    "secondary_verification": "10-Iteration Group Stratified ShuffleSplit (80% Train, 20% Val)",
                    "infeasibility_of_5_fold": "5-Fold Grouped CV is statistically barred because SNRPN (112 samples) constitutes 46.7% of the Cervical development cohort, which would leave other validation folds with fewer than 32 samples."
                }
            },
            "tier_2_sealed_test_vault": {
                "sample_count": 240,
                "percentage_of_total": 20.0,
                "purpose": "Strict one-shot final generalization verification of the single winning frozen model selected in Tier 1.",
                "class_distribution": {
                    "Cervical Cancer": 60,
                    "Colorectal Cancer": 60,
                    "Endometrial Cancer": 60,
                    "Liver Cancer": 60
                },
                "cervical_gene_composition": {
                    "gene_family_count": 8,
                    "total_cervical_samples": 60,
                    "quarantined_genes": [
                        {"gene": "ATP10A", "sample_count": 9},
                        {"gene": "TUBGCP5", "sample_count": 17},
                        {"gene": "CYFIP1", "sample_count": 10},
                        {"gene": "GABRA5", "sample_count": 7},
                        {"gene": "GABRB3", "sample_count": 6},
                        {"gene": "POTEB", "sample_count": 4},
                        {"gene": "POTEB2", "sample_count": 4},
                        {"gene": "SNURF", "sample_count": 3}
                    ],
                    "total_held_out_samples": 60
                },
                "leakage_guarantee": {
                    "sequence_overlap_with_dev": 0,
                    "gene_overlap_with_dev": 0,
                    "is_strictly_disjoint": True
                },
                "operational_protocol": {
                    "vault_status": "SEALED",
                    "lockout_enforcement": "The test vault must NEVER be passed to feature vectorizers, trainers, or evaluators during Phase 5 model tuning.",
                    "unsealing_condition": "Exactly once, only after model architecture, hyperparameters, and feature representations are completely frozen.",
                    "post_unseal_rule": "No re-tuning, parameter adjustment, or selective reporting is permitted after unsealing."
                }
            }
        },
        "grouping_definitions": {
            "exact_sequence_grouping": {
                "key_format": "seq_{sha256_hash}",
                "description": "Groups identical nucleotide sequences. Prevents exact duplicate leakage across splits. Does not account for splice variant homology.",
                "biological_validity": "Purely computational deduplication; not biological gene-family grouping."
            },
            "within_organism_gene_family_grouping": {
                "key_format": "gene_{organism_or_label}_{gene_symbol}",
                "description": "Groups all splice variants and transcripts of a gene family within a specific biological taxon. Isolates related isoforms while preventing spurious cross-organism grouping.",
                "biological_validity": "Scientifically verified for Homo sapiens chromosome 15 transcripts."
            },
            "cross_organism_ortholog_grouping": {
                "key_format": "gene_{gene_symbol}",
                "description": "Groups identical gene symbols across different organisms. Rejected as default because shared bacterial symbols (e.g. coxA) between S. amylolyticus and A. pallidiroseus exhibit only ~26% nucleotide identity and distinct UniProt IDs.",
                "biological_validity": "Unsupported by sequence alignment metadata."
            }
        },
        "metric_reporting_standards": {
            "required_primary_metrics": ["accuracy", "macro_precision", "macro_recall", "macro_f1"],
            "required_per_class_metrics": ["precision", "recall", "f1_score", "support"],
            "required_structural_metrics": ["confusion_matrix", "fold_group_counts", "fold_class_counts"],
            "uncertainty_quantification": {
                "accuracy_and_recall": "95% Wilson Score Interval",
                "macro_f1": "1,000-iteration non-parametric Bootstrap Confidence Interval"
            },
            "sample_size_limitations": (
                "With N=60 test samples per class, each individual misclassification alters class recall by 1.67%. "
                "Confidence intervals will naturally be broader (+/- 4-6%) than historical inflated benchmarks. "
                "This uncertainty must be reported transparently rather than claiming artificial precision."
            )
        },
        "historical_test_status": {
            "phase_4_test_status": "CONTAMINATED / OPTIMISTIC SELECTION BIAS",
            "historical_reported_accuracy": 0.9677,
            "historical_reported_cervical_recall": 1.0000,
            "audit_disqualification_reason": "Evaluated 24 times in inner experiment loop and used to rank configurations without validation blind holdout; additionally confounded by splice isoform leakage."
        }
    }

    out_path = Path("artifacts/evaluation/final_protocol_specification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec_data, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

