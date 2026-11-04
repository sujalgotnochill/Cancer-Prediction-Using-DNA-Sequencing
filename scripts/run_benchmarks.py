"""Script to execute Phase 4 Baseline Machine Learning Benchmarks."""

import json
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.training.benchmark_runner import BaselineBenchmarkRunner


def main():
    print("=" * 80)
    print("  PHASE 4: BASELINE MACHINE LEARNING BENCHMARK SYSTEM")
    print("  Dataset: Historical Reconstructed 1,200 DNA Samples")
    print("  Notice: Educational sequence-classification benchmark.")
    print("          Labels are not clinical cancer diagnoses.")
    print("=" * 80)

    runner = BaselineBenchmarkRunner(
        dataset_path="data/interim/historical_reconstruction/DNAdata_reconstructed.csv",
        output_dir="artifacts/evaluation",
        random_seed=42,
    )

    print("\n[1/3] Running full benchmark matrix (k in [3, 4, 5, 6], duplicate-grouped vs stratified)...")
    benchmark_data = runner.run_benchmark_matrix(
        k_values=[3, 4, 5, 6],
        strategies=["duplicate_grouped", "stratified"],
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
    )

    results = benchmark_data["results"]
    plots = benchmark_data["plots"]

    print("\n[2/3] Benchmark Results Summary:")
    print("-" * 105)
    print(f"{'Strategy':<20} {'k':<3} {'Model':<20} {'Features':<9} {'Accuracy':<10} {'Macro F1':<10} {'Train(s)':<9} {'Leakage-Free':<12}")
    print("-" * 105)
    for r in results:
        leak_str = "YES" if r["is_leakage_free"] else "LEAKED"
        print(
            f"{r['strategy']:<20} {r['k']:<3} {r['model']:<20} {r['n_features']:<9} "
            f"{r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {r['train_seconds']:<9.3f} {leak_str:<12}"
        )
    print("-" * 105)

    print("\n[3/3] Generated Evaluation Plots:")
    for plot_name, plot_path in plots.items():
        print(f"  - {plot_name}: {plot_path}")

    # Save aggregated benchmark summary
    summary_path = Path("artifacts/evaluation/baseline_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved full benchmark summary to {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

