"""Phase 5: Model Optimization, Feature Engineering, and Robust Sequence Classification.

Orchestrates:
1. Feature Engineering experiments: k-mer counts, TF-IDF, Normalized Frequencies (k=3, 4, 5, 6).
2. Reverse-complement canonicalization investigation.
3. Baseline & Advanced Model benchmarking: MultinomialNB, LogisticRegression, LinearSVC, RandomForest, GradientBoosting.
4. Bounded hyperparameter optimization on Primary 3-fold within-organism grouped CV.
5. Secondary 10-iteration Group-Stratified ShuffleSplit evaluation.
6. In-depth error analysis: per-gene Cervical failure breakdown, bacterial GC content/length correlation, k=1 nucleotide composition reliance.
7. Model selection, freezing, and persistence of the winning development candidate.

NOTE: All preprocessing is fitted strictly inside each training fold.
      The sealed holdout vault is NOT read, unsealed, or evaluated.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.development import get_development_cv_folds, load_development_cohort
from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.schema import DNASample
from dna_cancer.data.splitter import generate_group_shufflesplit_folds
from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models import (
    GradientBoostingClassifierWrapper,
    LinearSVCClassifier,
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
    RandomForestClassifierWrapper,
)
from dna_cancer.utils.logger import get_logger

logger = get_logger("phase5_optimization")


def compute_gc_content(seq: str) -> float:
    """Calculate GC nucleotide content of a sequence string."""
    seq_u = seq.upper()
    g = seq_u.count("G")
    c = seq_u.count("C")
    return (g + c) / max(len(seq_u), 1)


def evaluate_fold(
    train_samples: List[DNASample],
    val_samples: List[DNASample],
    k: int,
    vectorizer_type: str,
    canonical: bool,
    model_cls: Any,
    model_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Train and evaluate on a single cross-validation fold with strict fold isolation."""
    # 1. Fit FeatureBuilder exclusively on training fold
    fb = FeatureBuilder(
        k=k,
        vectorizer_type=vectorizer_type,
        canonical=canonical,
        return_sparse=True,
    )
    train_seqs = [s.sequence for s in train_samples]
    train_labels = [s.label for s in train_samples]

    t_feat_start = time.time()
    fb.fit(train_seqs, train_labels)
    X_train = fb.transform_sequences(train_seqs)
    y_train = fb.transform_labels(train_labels)

    # 2. Transform validation fold without data leakage
    val_seqs = [s.sequence for s in val_samples]
    val_labels = [s.label for s in val_samples]
    X_val = fb.transform_sequences(val_seqs)
    y_val = fb.transform_labels(val_labels)
    t_feat = time.time() - t_feat_start

    # 3. Fit classifier
    clf = model_cls(**model_kwargs)
    t_train_start = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t_train_start

    # 4. Predict
    t_inf_start = time.time()
    preds_encoded = clf.predict(X_val)
    t_inf = time.time() - t_inf_start

    preds = fb.inverse_transform_labels(preds_encoded)

    # 5. Metrics
    metrics = compute_classification_metrics(y_true=val_labels, y_pred=preds)

    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "metrics": metrics,
        "train_time": t_train,
        "inference_time": t_inf,
        "feature_prep_time": t_feat,
        "n_features": fb.n_features,
        "val_labels": val_labels,
        "val_preds": preds,
        "val_sample_ids": [s.sample_id for s in val_samples],
        "val_genes": [s.metadata.gene or "UNKNOWN" for s in val_samples],
        "val_seq_lens": [len(s.sequence) for s in val_samples],
        "val_gc": [compute_gc_content(s.sequence) for s in val_samples],
    }


def run_experiment(
    exp_id: str,
    name: str,
    k: int,
    vectorizer_type: str,
    canonical: bool,
    model_name: str,
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    folds: List[Tuple[List[DNASample], List[DNASample]]],
) -> Dict[str, Any]:
    """Execute 3-fold grouped cross-validation for a specific configuration."""
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_sample_ids = []
    all_genes = []
    all_seq_lens = []
    all_gc = []

    for fold_idx, (train_set, val_set) in enumerate(folds):
        res = evaluate_fold(
            train_samples=train_set,
            val_samples=val_set,
            k=k,
            vectorizer_type=vectorizer_type,
            canonical=canonical,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
        )
        fold_results.append({
            "fold": fold_idx,
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "train_time": res["train_time"],
            "inference_time": res["inference_time"],
            "n_features": res["n_features"],
            "cervical_recall": res["metrics"]["per_class"].get("Cervical Cancer", {}).get("recall", 0.0),
            "colorectal_recall": res["metrics"]["per_class"].get("Colorectal Cancer", {}).get("recall", 0.0),
            "endometrial_recall": res["metrics"]["per_class"].get("Endometrial Cancer", {}).get("recall", 0.0),
            "liver_recall": res["metrics"]["per_class"].get("Liver Cancer", {}).get("recall", 0.0),
        })
        all_y_true.extend(res["val_labels"])
        all_y_pred.extend(res["val_preds"])
        all_sample_ids.extend(res["val_sample_ids"])
        all_genes.extend(res["val_genes"])
        all_seq_lens.extend(res["val_seq_lens"])
        all_gc.extend(res["val_gc"])

    # Aggregate out-of-fold metrics
    agg_metrics = compute_classification_metrics(y_true=all_y_true, y_pred=all_y_pred)
    f1_list = [f["f1_macro"] for f in fold_results]
    acc_list = [f["accuracy"] for f in fold_results]

    mean_f1 = float(np.mean(f1_list))
    std_f1 = float(np.std(f1_list))
    mean_acc = float(np.mean(acc_list))
    std_acc = float(np.std(acc_list))
    total_train_time = sum(f["train_time"] for f in fold_results)
    total_inf_time = sum(f["inference_time"] for f in fold_results)
    n_features = fold_results[0]["n_features"]

    # Error analysis record
    misclassified_records = []
    for sid, true_lbl, pred_lbl, gene, slen, gc in zip(
        all_sample_ids, all_y_true, all_y_pred, all_genes, all_seq_lens, all_gc
    ):
        if true_lbl != pred_lbl:
            misclassified_records.append({
                "sample_id": sid,
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "gene": gene,
                "sequence_length": slen,
                "gc_content": round(gc, 4),
            })

    return {
        "exp_id": exp_id,
        "name": name,
        "k": k,
        "vectorizer_type": vectorizer_type,
        "canonical": canonical,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "n_features": n_features,
        "total_train_time_sec": round(total_train_time, 4),
        "total_inference_time_sec": round(total_inf_time, 4),
        "fold_results": fold_results,
        "oof_aggregate_metrics": agg_metrics,
        "total_misclassifications": len(misclassified_records),
        "misclassified_records": misclassified_records,
    }


def main():
    print("=" * 80)
    print("PHASE 5: MODEL OPTIMIZATION AND ROBUST SEQUENCE CLASSIFICATION")
    print("=" * 80)

    # 1. Load Development Cohort strictly
    dev_cohort_path = "data/processed/development_cohort.csv"
    manifest_path = "artifacts/vault/test_sealed_manifest.json"
    samples = load_development_cohort(path=dev_cohort_path, verify_checksum=True, manifest_path=manifest_path)
    print(f"Loaded {len(samples)} samples from {dev_cohort_path} (verified against manifest).")

    # 2. Generate Primary 3-Fold within-organism grouped CV
    cv_folds = get_development_cv_folds(samples, n_splits=3, seed=42)
    print("Generated 3-Fold within-organism grouped CV splits.")
    for f_idx, (tr, va) in enumerate(cv_folds):
        cervical_genes = {s.metadata.gene for s in va if s.label == "Cervical Cancer"}
        print(f"  Fold {f_idx}: Train={len(tr)}, Val={len(va)}, Cervical Val Genes={len(cervical_genes)} ({list(cervical_genes)[:3]}...)")

    results = []

    # -------------------------------------------------------------
    # SUITE 1: Baseline Architecture Comparison (k=5, TF-IDF)
    # -------------------------------------------------------------
    print("\n--- Suite 1: Baseline Architecture Comparison (k=5, TF-IDF) ---")
    architectures = [
        ("multinomial_nb", MultinomialNBClassifier, {"alpha": 1.0}),
        ("logistic_regression", LogisticRegressionClassifier, {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}),
        ("linear_svc", LinearSVCClassifier, {"C": 1.0, "dual": "auto", "max_iter": 2000}),
        ("random_forest", RandomForestClassifierWrapper, {"n_estimators": 100, "random_state": 42}),
        ("gradient_boosting", GradientBoostingClassifierWrapper, {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}),
    ]

    for model_name, clf_cls, kwargs in architectures:
        exp_id = f"baseline_{model_name}_k5_tfidf"
        res = run_experiment(
            exp_id=exp_id,
            name=f"Baseline {model_name} (k=5, TF-IDF)",
            k=5,
            vectorizer_type="tfidf",
            canonical=False,
            model_name=model_name,
            model_cls=clf_cls,
            model_kwargs=kwargs,
            folds=cv_folds,
        )
        results.append(res)
        print(f"  [{model_name:20}] Mean Macro F1: {res['mean_macro_f1']:.4f} (+/- {res['std_macro_f1']:.4f}), Acc: {res['mean_accuracy']:.4f}, Time: {res['total_train_time_sec']:.2f}s, Cervical Fold F1s: {[round(f['cervical_recall'], 3) for f in res['fold_results']]}")

    # -------------------------------------------------------------
    # SUITE 2: Feature Representation & K-mer Ablation (LinearSVC & LogisticRegression)
    # -------------------------------------------------------------
    print("\n--- Suite 2: Feature Representation & K-mer Ablation (k in {3, 4, 5, 6}) ---")
    for k in [3, 4, 5, 6]:
        for rep in ["count", "tfidf", "frequency"]:
            for model_name, clf_cls, kwargs in [
                ("linear_svc", LinearSVCClassifier, {"C": 1.0, "dual": "auto", "max_iter": 2000}),
                ("logistic_regression", LogisticRegressionClassifier, {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}),
                ("multinomial_nb", MultinomialNBClassifier, {"alpha": 1.0}),
            ]:
                exp_id = f"repr_{model_name}_k{k}_{rep}"
                res = run_experiment(
                    exp_id=exp_id,
                    name=f"{model_name} (k={k}, {rep})",
                    k=k,
                    vectorizer_type=rep,
                    canonical=False,
                    model_name=model_name,
                    model_cls=clf_cls,
                    model_kwargs=kwargs,
                    folds=cv_folds,
                )
                results.append(res)
                print(f"  [k={k} {rep:10} {model_name:18}] F1: {res['mean_macro_f1']:.4f} (+/- {res['std_macro_f1']:.4f}), Acc: {res['mean_accuracy']:.4f}, Feats: {res['n_features']}")

    # -------------------------------------------------------------
    # SUITE 3: Reverse-Complement Canonicalization Investigation
    # -------------------------------------------------------------
    print("\n--- Suite 3: Reverse-Complement Canonicalization Investigation ---")
    for k in [4, 5, 6]:
        for model_name, clf_cls, kwargs in [
            ("linear_svc", LinearSVCClassifier, {"C": 1.0, "dual": "auto", "max_iter": 2000}),
            ("logistic_regression", LogisticRegressionClassifier, {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}),
        ]:
            # Standard (non-canonical)
            exp_id_std = f"canonical_comp_{model_name}_k{k}_standard"
            res_std = run_experiment(
                exp_id=exp_id_std,
                name=f"{model_name} (k={k}, TF-IDF, standard)",
                k=k,
                vectorizer_type="tfidf",
                canonical=False,
                model_name=model_name,
                model_cls=clf_cls,
                model_kwargs=kwargs,
                folds=cv_folds,
            )
            # Canonical
            exp_id_can = f"canonical_comp_{model_name}_k{k}_canonical"
            res_can = run_experiment(
                exp_id=exp_id_can,
                name=f"{model_name} (k={k}, TF-IDF, canonical)",
                k=k,
                vectorizer_type="tfidf",
                canonical=True,
                model_name=model_name,
                model_cls=clf_cls,
                model_kwargs=kwargs,
                folds=cv_folds,
            )
            results.extend([res_std, res_can])
            diff = res_can["mean_macro_f1"] - res_std["mean_macro_f1"]
            print(f"  [k={k} {model_name:18}] Standard F1: {res_std['mean_macro_f1']:.4f} vs Canonical F1: {res_can['mean_macro_f1']:.4f} (Delta: {diff:+.4f})")

    # -------------------------------------------------------------
    # SUITE 4: Hyperparameter Optimization Grid Search
    # -------------------------------------------------------------
    print("\n--- Suite 4: Hyperparameter Optimization Grid Search ---")

    # 4A. Linear SVM Tuning (k=5 & k=6, TF-IDF)
    print("  Tuning LinearSVC...")
    for k_val in [5, 6]:
        for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            for cw in [None, "balanced"]:
                exp_id = f"tune_linear_svc_k{k_val}_C{c_val}_cw_{'bal' if cw else 'none'}"
                res = run_experiment(
                    exp_id=exp_id,
                    name=f"LinearSVC (k={k_val}, C={c_val}, cw={cw})",
                    k=k_val,
                    vectorizer_type="tfidf",
                    canonical=False,
                    model_name="linear_svc",
                    model_cls=LinearSVCClassifier,
                    model_kwargs={"C": c_val, "class_weight": cw, "dual": "auto", "max_iter": 3000},
                    folds=cv_folds,
                )
                results.append(res)

    # 4B. Logistic Regression Tuning (k=5 & k=6, TF-IDF)
    print("  Tuning LogisticRegression...")
    for k_val in [5, 6]:
        for c_val in [0.1, 1.0, 10.0, 50.0]:
            for cw in [None, "balanced"]:
                exp_id = f"tune_logreg_k{k_val}_C{c_val}_cw_{'bal' if cw else 'none'}"
                res = run_experiment(
                    exp_id=exp_id,
                    name=f"LogisticRegression (k={k_val}, C={c_val}, cw={cw})",
                    k=k_val,
                    vectorizer_type="tfidf",
                    canonical=False,
                    model_name="logistic_regression",
                    model_cls=LogisticRegressionClassifier,
                    model_kwargs={"C": c_val, "class_weight": cw, "max_iter": 1500, "solver": "lbfgs"},
                    folds=cv_folds,
                )
                results.append(res)

    # 4C. MultinomialNB Tuning (k=5, count & frequency)
    print("  Tuning MultinomialNB...")
    for alpha_val in [0.01, 0.1, 0.5, 1.0, 2.0]:
        for rep in ["count", "frequency"]:
            exp_id = f"tune_mnb_alpha{alpha_val}_{rep}"
            res = run_experiment(
                exp_id=exp_id,
                name=f"MultinomialNB (k=5, alpha={alpha_val}, {rep})",
                k=5,
                vectorizer_type=rep,
                canonical=False,
                model_name="multinomial_nb",
                model_cls=MultinomialNBClassifier,
                model_kwargs={"alpha": alpha_val},
                folds=cv_folds,
            )
            results.append(res)

    # 4D. Random Forest Tuning (k=5, TF-IDF)
    print("  Tuning RandomForest...")
    for n_est in [50, 100, 200]:
        for max_d in [10, 20, None]:
            for min_leaf in [1, 2]:
                exp_id = f"tune_rf_n{n_est}_d{max_d}_leaf{min_leaf}"
                res = run_experiment(
                    exp_id=exp_id,
                    name=f"RandomForest (n={n_est}, d={max_d}, leaf={min_leaf})",
                    k=5,
                    vectorizer_type="tfidf",
                    canonical=False,
                    model_name="random_forest",
                    model_cls=RandomForestClassifierWrapper,
                    model_kwargs={
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "min_samples_leaf": min_leaf,
                        "random_state": 42,
                    },
                    folds=cv_folds,
                )
                results.append(res)

    # -------------------------------------------------------------
    # SUITE 5: Base Composition Baseline (k=1)
    # -------------------------------------------------------------
    print("\n--- Suite 5: Nucleotide Composition Baseline (k=1) ---")
    for model_name, clf_cls, kwargs in [
        ("logistic_regression", LogisticRegressionClassifier, {"C": 1.0, "max_iter": 3000, "solver": "lbfgs"}),
        ("linear_svc", LinearSVCClassifier, {"C": 1.0, "dual": "auto", "max_iter": 3000}),
        ("multinomial_nb", MultinomialNBClassifier, {"alpha": 1.0}),
    ]:
        exp_id = f"k1_composition_baseline_{model_name}"
        res = run_experiment(
            exp_id=exp_id,
            name=f"k=1 Composition Baseline {model_name}",
            k=1,
            vectorizer_type="frequency",
            canonical=False,
            model_name=model_name,
            model_cls=clf_cls,
            model_kwargs=kwargs,
            folds=cv_folds,
        )
        results.append(res)
        print(f"  [k=1 {model_name:18}] F1: {res['mean_macro_f1']:.4f}, Acc: {res['mean_accuracy']:.4f}")

    # -------------------------------------------------------------
    # Identify Winning Model from Primary 3-Fold Grouped CV
    # -------------------------------------------------------------
    # Sort results by mean_macro_f1 descending
    results.sort(key=lambda r: (-r["mean_macro_f1"], r["std_macro_f1"], r["total_inference_time_sec"]))
    top_candidates = results[:10]

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (Primary 3-Fold Grouped CV):")
    print("=" * 80)
    for rank, cand in enumerate(top_candidates, start=1):
        fold_f1s = [round(f["f1_macro"], 4) for f in cand["fold_results"]]
        cervical_recalls = [round(f["cervical_recall"], 4) for f in cand["fold_results"]]
        print(f"Rank {rank:2d}: {cand['name']:45} | Macro F1: {cand['mean_macro_f1']:.4f} (+/- {cand['std_macro_f1']:.4f}) | Acc: {cand['mean_accuracy']:.4f} | Folds F1: {fold_f1s} | Cervical Recalls: {cervical_recalls} | Inf Time: {cand['total_inference_time_sec']:.3f}s")

    winning_cand = top_candidates[0]
    print(f"\nWinning Candidate Selected: {winning_cand['name']} (Exp ID: {winning_cand['exp_id']})")
    print(f"  Mean Macro F1: {winning_cand['mean_macro_f1']:.4f} (+/- {winning_cand['std_macro_f1']:.4f})")
    print(f"  Mean Accuracy: {winning_cand['mean_accuracy']:.4f} (+/- {winning_cand['std_accuracy']:.4f})")

    # -------------------------------------------------------------
    # SUITE 6: Secondary Protocol Verification (10-Iteration GroupShuffleSplit)
    # -------------------------------------------------------------
    print("\n--- Suite 6: Secondary 10-Iteration GroupShuffleSplit Verification ---")
    shuffle_folds = generate_group_shufflesplit_folds(samples, n_splits=10, val_size=0.25, seed=42)

    top_3_models = [top_candidates[0]]
    seen_models = {top_candidates[0]["model_name"]}
    for c in results:
        if c["model_name"] not in seen_models:
            top_3_models.append(c)
            seen_models.add(c["model_name"])
            if len(top_3_models) >= 3:
                break

    shuffle_results = {}
    for cand in top_3_models:
        model_name = cand["model_name"]
        clf_cls = (
            LinearSVCClassifier
            if model_name == "linear_svc"
            else (
                LogisticRegressionClassifier
                if model_name == "logistic_regression"
                else (MultinomialNBClassifier if model_name == "multinomial_nb" else RandomForestClassifierWrapper)
            )
        )
        sh_f1s = []
        sh_accs = []
        for sh_idx, (tr_set, va_set) in enumerate(shuffle_folds):
            r = evaluate_fold(
                train_samples=tr_set,
                val_samples=va_set,
                k=cand["k"],
                vectorizer_type=cand["vectorizer_type"],
                canonical=cand["canonical"],
                model_cls=clf_cls,
                model_kwargs=cand["model_kwargs"],
            )
            sh_f1s.append(r["f1_macro"])
            sh_accs.append(r["accuracy"])

        med_f1 = float(np.median(sh_f1s))
        iqr_f1 = float(np.percentile(sh_f1s, 75) - np.percentile(sh_f1s, 25))
        med_acc = float(np.median(sh_accs))
        iqr_acc = float(np.percentile(sh_accs, 75) - np.percentile(sh_accs, 25))

        shuffle_results[cand["exp_id"]] = {
            "name": cand["name"],
            "model_name": cand["model_name"],
            "iterations": 10,
            "f1_scores": [round(x, 4) for x in sh_f1s],
            "median_macro_f1": round(med_f1, 4),
            "iqr_macro_f1": round(iqr_f1, 4),
            "median_accuracy": round(med_acc, 4),
            "iqr_accuracy": round(iqr_acc, 4),
            "independence_disclosure": "Resamplings are non-independent correlated draws from the 23 development gene families. Reported as Median and IQR.",
        }
        print(f"  {cand['name']:40}: Median Macro F1: {med_f1:.4f} (IQR: {iqr_f1:.4f}) | Median Acc: {med_acc:.4f} (IQR: {iqr_acc:.4f})")

    # -------------------------------------------------------------
    # SUITE 7: In-Depth Error Analysis on Winning Candidate
    # -------------------------------------------------------------
    print("\n--- Suite 7: In-Depth Error Analysis on Winning Candidate ---")
    cervical_errors = [m for m in winning_cand["misclassified_records"] if m["true_label"] == "Cervical Cancer"]
    bacterial_errors = [m for m in winning_cand["misclassified_records"] if m["true_label"] != "Cervical Cancer"]

    gene_fail_counts = {}
    for err in cervical_errors:
        gene_fail_counts[err["gene"]] = gene_fail_counts.get(err["gene"], 0) + 1

    print(f"  Total Out-of-Fold Misclassifications: {len(winning_cand['misclassified_records'])} / 960 ({len(winning_cand['misclassified_records'])/960*100:.2f}%)")
    print(f"  Cervical Misclassifications: {len(cervical_errors)} / 240 ({len(cervical_errors)/240*100:.2f}%)")
    print(f"  Cervical Failed Genes: {gene_fail_counts}")
    print(f"  Bacterial Misclassifications: {len(bacterial_errors)} / 720 ({len(bacterial_errors)/720*100:.2f}%)")

    # Bacterial length and GC correlation
    if bacterial_errors:
        bact_lens = [m["sequence_length"] for m in bacterial_errors]
        bact_gcs = [m["gc_content"] for m in bacterial_errors]
        print(f"  Bacterial Error Avg Seq Length: {np.mean(bact_lens):.1f} bp (vs all dev avg)")
        print(f"  Bacterial Error Avg GC Content: {np.mean(bact_gcs):.4f}")

    # -------------------------------------------------------------
    # SUITE 8: Model Serialization & Candidate Freezing
    # -------------------------------------------------------------
    print("\n--- Suite 8: Model Serialization & Candidate Freezing ---")
    model_export_dir = Path("artifacts/models/phase5_best_candidate")
    model_export_dir.mkdir(parents=True, exist_ok=True)

    # Train winning architecture on FULL development cohort (N=960)
    dev_seqs = [s.sequence for s in samples]
    dev_labels = [s.label for s in samples]

    winning_fb = FeatureBuilder(
        k=winning_cand["k"],
        vectorizer_type=winning_cand["vectorizer_type"],
        canonical=winning_cand["canonical"],
        return_sparse=True,
    )
    winning_fb.fit(dev_seqs, dev_labels)
    X_dev_full = winning_fb.transform_sequences(dev_seqs)
    y_dev_full = winning_fb.transform_labels(dev_labels)

    winning_clf_cls = (
        LinearSVCClassifier
        if winning_cand["model_name"] == "linear_svc"
        else (
            LogisticRegressionClassifier
            if winning_cand["model_name"] == "logistic_regression"
            else (MultinomialNBClassifier if winning_cand["model_name"] == "multinomial_nb" else RandomForestClassifierWrapper)
        )
    )
    winning_model = winning_clf_cls(**winning_cand["model_kwargs"])
    winning_model.fit(X_dev_full, y_dev_full)

    # Save artifacts
    winning_model.save(model_export_dir / "model.joblib")
    winning_fb.save(
        vectorizer_path=model_export_dir / "vectorizer.joblib",
        encoder_path=model_export_dir / "encoder.joblib",
    )

    dev_cohort_sha = calculate_file_checksum(dev_cohort_path)

    frozen_config = {
        "model_candidate_id": winning_cand["exp_id"],
        "model_name": winning_cand["model_name"],
        "architecture_class": winning_clf_cls.__name__,
        "is_frozen": True,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_cohort": {
            "path": dev_cohort_path,
            "sample_count": len(samples),
            "sha256": dev_cohort_sha,
        },
        "feature_configuration": {
            "k": winning_cand["k"],
            "vectorizer_type": winning_cand["vectorizer_type"],
            "canonical": winning_cand["canonical"],
            "n_features": winning_fb.n_features,
            "sparse_representation": True,
        },
        "hyperparameters": winning_cand["model_kwargs"],
        "primary_cv_performance": {
            "mean_macro_f1": winning_cand["mean_macro_f1"],
            "std_macro_f1": winning_cand["std_macro_f1"],
            "mean_accuracy": winning_cand["mean_accuracy"],
            "std_accuracy": winning_cand["std_accuracy"],
            "fold_results": winning_cand["fold_results"],
        },
        "secondary_shufflesplit_performance": shuffle_results.get(winning_cand["exp_id"]),
        "safeguard_attestation": "Model configuration frozen for Phase 5. The Tier 2 Sealed Test Vault remains locked.",
    }

    with open(model_export_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(frozen_config, f, indent=2)
    print(f"Saved frozen model candidate and configuration to {model_export_dir}")

    # -------------------------------------------------------------
    # SUITE 9: Save JSON Reports
    # -------------------------------------------------------------
    print("\n--- Suite 9: Persisting Experiment & Evaluation Artifacts ---")
    experiments_dir = Path("artifacts/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    results_export = {
        "metadata": {
            "phase": "Phase 5: Model Optimization and Robust Sequence Classification",
            "executed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dev_cohort_path,
            "dataset_sha256": dev_cohort_sha,
            "total_samples": len(samples),
            "total_experiments_evaluated": len(results),
            "primary_protocol": "3-Fold within-organism gene-family grouped CV",
            "secondary_protocol": "10-Iteration Group-Stratified ShuffleSplit (val_size=0.25)",
        },
        "winning_candidate": frozen_config,
        "experiments": results,
    }

    with open(experiments_dir / "phase5_results.json", "w", encoding="utf-8") as f:
        json.dump(results_export, f, indent=2)
    print(f"Saved {experiments_dir / 'phase5_results.json'} ({len(results)} experiments recorded)")

    cv_summary = {
        "metadata": {
            "phase": "Phase 5",
            "development_cohort_samples": len(samples),
            "sealed_vault_status": "LOCKED (unseal_count: 0)",
        },
        "primary_grouped_cv_summary": {
            "winning_model": winning_cand["name"],
            "mean_macro_f1": winning_cand["mean_macro_f1"],
            "std_macro_f1": winning_cand["std_macro_f1"],
            "mean_accuracy": winning_cand["mean_accuracy"],
            "std_accuracy": winning_cand["std_accuracy"],
            "per_fold_breakdown": winning_cand["fold_results"],
            "oof_per_class": winning_cand["oof_aggregate_metrics"]["per_class"],
            "confusion_matrix": winning_cand["oof_aggregate_metrics"]["confusion_matrix"],
            "labels": winning_cand["oof_aggregate_metrics"]["labels"],
        },
        "secondary_shufflesplit_summary": shuffle_results,
        "error_analysis_summary": {
            "total_misclassifications": winning_cand["total_misclassifications"],
            "cervical_misclassifications": len(cervical_errors),
            "cervical_gene_failure_distribution": gene_fail_counts,
            "bacterial_misclassifications": len(bacterial_errors),
        },
        "top_10_candidates": [
            {
                "rank": rank + 1,
                "name": c["name"],
                "model": c["model_name"],
                "k": c["k"],
                "repr": c["vectorizer_type"],
                "canonical": c["canonical"],
                "mean_macro_f1": c["mean_macro_f1"],
                "std_macro_f1": c["std_macro_f1"],
                "mean_acc": c["mean_accuracy"],
            }
            for rank, c in enumerate(top_candidates)
        ],
    }

    with open(eval_dir / "phase5_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"Saved {eval_dir / 'phase5_cv_summary.json'}")

    print("\n" + "=" * 80)
    print("PHASE 5 OPTIMIZATION EXPERIMENTS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

