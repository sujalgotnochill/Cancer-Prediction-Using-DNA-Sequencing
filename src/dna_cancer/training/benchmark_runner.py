"""Baseline benchmark experiment orchestrator for DNA sequence classification.

Executes comprehensive benchmarks across:
- Splitting strategies (Duplicate-Grouped vs. Conventional Stratified)
- k-mer sizes (k=3, 4, 5, 6)
- Baseline models (Multinomial Naive Bayes, Logistic Regression, Random Forest)

Strictly enforces:
- Training-only feature fitting (zero vectorizer data leakage)
- Untouched held-out test evaluation
- Telemetry logging via ExperimentTracker
- Visualization generation saved to artifacts/evaluation/
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dna_cancer.data.ingestion import ColumnMapping, load_samples
from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.schema import DNASample
from dna_cancer.data.splitter import DatasetSplitter, verify_split_leakage
from dna_cancer.evaluation.evaluator import Evaluator
from dna_cancer.evaluation.visualization import (
    plot_confusion_matrix,
    plot_kmer_ablation,
    plot_model_comparison,
    plot_per_class_f1,
)
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models.baseline_models import (
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
    RandomForestClassifierWrapper,
)
from dna_cancer.tracking.tracker import ExperimentTracker
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineBenchmarkRunner:
    """Orchestrates comprehensive baseline machine learning benchmarks."""

    def __init__(
        self,
        dataset_path: Union[str, Path] = "data/interim/historical_reconstruction/DNAdata_reconstructed.csv",
        output_dir: Union[str, Path] = "artifacts/evaluation",
        random_seed: int = 42,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.tracker = ExperimentTracker(
            experiment_name="dna_baseline_benchmarks",
            artifact_dir="artifacts/experiments",
        )
        self.dataset_checksum = calculate_file_checksum(self.dataset_path) if self.dataset_path.exists() else "unknown"

    def load_dataset(self) -> List[DNASample]:
        """Load samples from the reconstructed historical dataset CSV."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Reconstructed dataset not found at {self.dataset_path}")

        mapping = ColumnMapping(sequence_col="sequence", label_col="cancer", id_col="Unnamed: 0")
        samples = load_samples(self.dataset_path, file_format="csv", column_mapping=mapping)
        logger.info("Loaded %d samples from %s (SHA-256: %s)", len(samples), self.dataset_path, self.dataset_checksum[:12])
        return samples

    def run_benchmark_matrix(
        self,
        k_values: Optional[List[int]] = None,
        strategies: Optional[List[str]] = None,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Dict[str, Any]:
        """Run full benchmark matrix across k values and splitting strategies.

        Args:
            k_values: List of k sizes to evaluate (default [3, 4, 5, 6]).
            strategies: List of splitting strategies (default ['duplicate_grouped', 'stratified']).
            train_size: Proportion of dataset for training.
            val_size: Proportion of dataset for validation.
            test_size: Proportion of dataset for test.

        Returns:
            Dictionary containing aggregated results, rankings, and generated artifact paths.
        """
        k_list = k_values or [3, 4, 5, 6]
        strat_list = strategies or ["duplicate_grouped", "stratified"]
        samples = self.load_dataset()

        all_results: List[Dict[str, Any]] = []
        kmer_ablation_data: Dict[str, Dict[str, List[float]]] = {}

        for strat in strat_list:
            logger.info("=== Running Benchmark for Splitting Strategy: '%s' ===", strat)
            splitter = DatasetSplitter(
                strategy=strat,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=self.random_seed,
            )
            train_samples, val_samples, test_samples = splitter.split(samples)
            leakage_audit = verify_split_leakage(train_samples, val_samples, test_samples)
            split_summary = splitter.summarize_split(train_samples, val_samples, test_samples)

            logger.info(
                "Split sizes: Train=%d, Val=%d, Test=%d. Leakage-free: %s",
                len(train_samples),
                len(val_samples),
                len(test_samples),
                leakage_audit["is_leakage_free"],
            )

            strat_kmer_accs: Dict[str, List[float]] = {
                "MultinomialNB": [],
                "LogisticRegression": [],
                "RandomForest": [],
            }

            for k in k_list:
                logger.info("--- Evaluating k=%d (strategy: %s) ---", k, strat)
                # 1. Feature extraction strictly fitted on training data
                fb = FeatureBuilder(
                    k=k,
                    stride=1,
                    vectorizer_type="count",
                    ngram_range=(1, 1),
                    max_features=None,
                    return_sparse=True,
                )

                train_seqs = [s.sequence for s in train_samples]
                train_labels = [s.label for s in train_samples]
                val_seqs = [s.sequence for s in val_samples]
                val_labels = [s.label for s in val_samples]
                test_seqs = [s.sequence for s in test_samples]
                test_labels = [s.label for s in test_samples]

                fb.fit(train_seqs, train_labels)

                X_train = fb.transform_sequences(train_seqs)
                y_train = fb.transform_labels(train_labels)

                X_val = fb.transform_sequences(val_seqs)
                y_val = fb.transform_labels(val_labels)

                X_test = fb.transform_sequences(test_seqs)
                y_test = fb.transform_labels(test_labels)

                target_names = list(fb.label_encoder.classes_)
                evaluator = Evaluator(target_names=target_names)

                # Models to benchmark
                models_to_test = [
                    ("MultinomialNB", MultinomialNBClassifier(alpha=1.0)),
                    (
                        "LogisticRegression",
                        LogisticRegressionClassifier(C=1.0, max_iter=1000, random_state=self.random_seed),
                    ),
                    (
                        "RandomForest",
                        RandomForestClassifierWrapper(n_estimators=100, random_state=self.random_seed),
                    ),
                ]

                for model_label, model_inst in models_to_test:
                    # Measure training time
                    t0 = time.perf_counter()
                    model_inst.fit(X_train, y_train, validation_data=(X_val, y_val))
                    train_duration = time.perf_counter() - t0

                    # Measure prediction time
                    t1 = time.perf_counter()
                    val_metrics = evaluator.evaluate(model_inst, X_val, y_val)
                    test_metrics = evaluator.evaluate(model_inst, X_test, y_test)
                    pred_duration = time.perf_counter() - t1

                    strat_kmer_accs[model_label].append(test_metrics["accuracy"])

                    # Save model and feature artifacts
                    model_save_dir = Path("artifacts/models") / f"{model_label.lower()}_{strat}_k{k}"
                    fb_save_dir = Path("artifacts/vectorizers") / f"kmer_{strat}_k{k}"
                    model_inst.save(model_save_dir / "model.joblib")
                    fb.save(fb_save_dir / "vectorizer.joblib", fb_save_dir / "encoder.joblib")

                    # Log run via tracker
                    run_record = self.tracker.log_run(
                        run_name=f"{model_label}_{strat}_k{k}",
                        dataset_id="DNAdata_historical_reconstruction",
                        dataset_checksum=self.dataset_checksum,
                        split_config={
                            "strategy": strat,
                            "train_size": train_size,
                            "val_size": val_size,
                            "test_size": test_size,
                            "train_count": len(train_samples),
                            "val_count": len(val_samples),
                            "test_count": len(test_samples),
                            "is_leakage_free": leakage_audit["is_leakage_free"],
                            "sequence_leakage": leakage_audit["sequence_leakage"],
                        },
                        feature_config={
                            "k": k,
                            "stride": 1,
                            "vectorizer_type": "count",
                            "n_features": fb.n_features,
                        },
                        model_name=model_label,
                        model_params=model_inst.get_params(),
                        random_seed=self.random_seed,
                        metrics=test_metrics,
                        training_duration=train_duration,
                        prediction_duration=pred_duration,
                        artifact_paths={
                            "model_checkpoint": str(model_save_dir / "model.joblib"),
                            "vectorizer": str(fb_save_dir / "vectorizer.joblib"),
                        },
                    )

                    all_results.append({
                        "strategy": strat,
                        "k": k,
                        "model": model_label,
                        "n_features": fb.n_features,
                        "accuracy": test_metrics["accuracy"],
                        "precision_macro": test_metrics["precision_macro"],
                        "recall_macro": test_metrics["recall_macro"],
                        "f1_macro": test_metrics["f1_macro"],
                        "train_seconds": train_duration,
                        "pred_seconds": pred_duration,
                        "test_confusion_matrix": test_metrics["confusion_matrix"],
                        "classification_report": test_metrics["classification_report"],
                        "is_leakage_free": leakage_audit["is_leakage_free"],
                    })

            kmer_ablation_data[strat] = strat_kmer_accs

        # Generate Visualizations
        plots_generated = self._generate_visualizations(all_results, kmer_ablation_data, k_list)

        return {
            "results": all_results,
            "kmer_ablation": kmer_ablation_data,
            "plots": plots_generated,
        }

    def _generate_visualizations(
        self,
        results: List[Dict[str, Any]],
        kmer_ablation: Dict[str, Dict[str, List[float]]],
        k_list: List[int],
    ) -> Dict[str, str]:
        """Generate and save comparison plots under artifacts/evaluation/."""
        plots: Dict[str, str] = {}

        # 1. k-mer ablation curves
        for strat, model_accs in kmer_ablation.items():
            p_path = self.output_dir / f"kmer_ablation_{strat}.png"
            plot_kmer_ablation(
                k_values=k_list,
                models_accuracies=model_accs,
                output_path=p_path,
                title=f"k-mer Size Ablation ({strat.replace('_', ' ').title()})",
            )
            plots[f"kmer_ablation_{strat}"] = str(p_path)

        # 2. Duplicate-grouped benchmark model comparison at k=6
        dg_k6_results = [
            {
                "model_name": r["model"],
                "accuracy": r["accuracy"],
                "precision_macro": r["precision_macro"],
                "recall_macro": r["recall_macro"],
                "f1_macro": r["f1_macro"],
            }
            for r in results
            if r["strategy"] == "duplicate_grouped" and r["k"] == 6
        ]
        if dg_k6_results:
            p_comp = self.output_dir / "baseline_model_comparison_duplicate_grouped_k6.png"
            plot_model_comparison(
                dg_k6_results,
                p_comp,
                title="Model Performance (Duplicate-Grouped Benchmark, k=6)",
            )
            plots["model_comparison_duplicate_grouped_k6"] = str(p_comp)

        # 3. Stratified benchmark model comparison at k=6
        strat_k6_results = [
            {
                "model_name": r["model"],
                "accuracy": r["accuracy"],
                "precision_macro": r["precision_macro"],
                "recall_macro": r["recall_macro"],
                "f1_macro": r["f1_macro"],
            }
            for r in results
            if r["strategy"] == "stratified" and r["k"] == 6
        ]
        if strat_k6_results:
            p_strat = self.output_dir / "baseline_model_comparison_stratified_k6.png"
            plot_model_comparison(
                strat_k6_results,
                p_strat,
                title="Model Performance (Stratified Benchmark with Leakage, k=6)",
            )
            plots["model_comparison_stratified_k6"] = str(p_strat)

        # 4. Confusion matrices for duplicate-grouped k=6
        class_names = ["Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"]
        for r in results:
            if r["strategy"] == "duplicate_grouped" and r["k"] == 6:
                cm_path = self.output_dir / f"confusion_matrix_{r['model'].lower()}_duplicate_grouped_k6.png"
                plot_confusion_matrix(
                    cm=r["test_confusion_matrix"],
                    class_names=class_names,
                    output_path=cm_path,
                    title=f"Confusion Matrix: {r['model']} (k=6, Duplicate-Grouped)",
                )
                plots[f"cm_{r['model'].lower()}_dg_k6"] = str(cm_path)

        # 5. Per-class F1 comparison for duplicate-grouped k=6
        model_f1s: Dict[str, List[float]] = {}
        for r in results:
            if r["strategy"] == "duplicate_grouped" and r["k"] == 6:
                rep = r["classification_report"]
                f1_list = [rep.get(c, {}).get("f1-score", 0.0) for c in class_names]
                model_f1s[r["model"]] = f1_list

        if model_f1s:
            p_f1 = self.output_dir / "per_class_f1_duplicate_grouped_k6.png"
            plot_per_class_f1(
                class_names=class_names,
                model_class_f1s=model_f1s,
                output_path=p_f1,
                title="Per-Class F1 Score (Duplicate-Grouped Benchmark, k=6)",
            )
            plots["per_class_f1_dg_k6"] = str(p_f1)

        return plots
