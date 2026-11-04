"""Official Phase 6 Final One-Shot Holdout Evaluation Script."""

import csv
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.vault import AUTHORIZED_UNSEAL_TOKEN, SealedVault
from dna_cancer.evaluation.final_evaluator import FinalHoldoutEvaluator
from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.features.kmer_extractor import extract_kmers
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.utils.exceptions import DataValidationError


def compute_sha256(file_path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return {"proportion": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "confidence_level": confidence}
    z = 1.959963984540054  # 95% two-sided normal quantile
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2)))
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return {
        "successes": successes,
        "total": total,
        "proportion": float(p_hat),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence,
    }


class FrozenHoldoutPipeline(BaseDNAClassifier):
    """End-to-end inference wrapper for the frozen Phase 5 candidate."""

    def __init__(
        self,
        model: Any,
        vectorizer: Any,
        encoder: Any,
        k: int = 4,
        canonical: bool = False,
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.encoder = encoder
        self.k = k
        self.canonical = canonical

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "FrozenHoldoutPipeline":
        raise RuntimeError("Frozen candidate model cannot be refit.")

    def transform_sequences(self, sequences: List[str]) -> Any:
        kmers = [" ".join(extract_kmers(s, k=self.k, canonical=self.canonical)) for s in sequences]
        return self.vectorizer.transform(kmers)

    def predict(self, sequences: List[str]) -> Any:
        X = self.transform_sequences(sequences)
        pred_idx = self.model.predict(X)
        return self.encoder.inverse_transform(pred_idx)

    def predict_proba(self, sequences: List[str]) -> Any:
        # Note: Uncalibrated softmax on decision function; not empirical probabilities.
        X = self.transform_sequences(sequences)
        return self.model.predict_proba(X)

    def decision_function(self, sequences: List[str]) -> Any:
        X = self.transform_sequences(sequences)
        return self.model.model.decision_function(X)

    def save(self, path: Union[str, Path]) -> None:
        raise RuntimeError("Pipeline wrapper does not overwrite frozen candidate.")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "FrozenHoldoutPipeline":
        raise NotImplementedError


def run_pre_unseal_readiness_checks() -> Dict[str, Any]:
    """Execute strict pre-unseal readiness checks. If any fails, raises an exception."""
    print("[1/7] Executing pre-unseal readiness checks...")
    cand_dir = Path("artifacts/models/phase5_best_candidate")
    vault_dir = Path("artifacts/vault")
    manifest_path = vault_dir / "test_sealed_manifest.json"
    holdout_path = vault_dir / "test_holdout_sealed.json"
    vault_state_path = vault_dir / "vault_state.json"
    dev_cohort_path = Path("data/processed/development_cohort.csv")
    events_log_path = vault_dir / "evaluation_events.jsonl"

    # 1. Config check
    config_path = cand_dir / "model_config.json"
    assert config_path.exists(), f"Missing config: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config.get("is_frozen") is True, "Model configuration is not frozen ('is_frozen': True)"
    print("  [OK] Frozen model configuration verified ('is_frozen': True)")

    # 2. Checksums against checksums.sha256
    checksums_file = cand_dir / "checksums.sha256"
    assert checksums_file.exists(), f"Missing checksums file: {checksums_file}"
    with open(checksums_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                expected_sha, fname = line.strip().split()
                actual_sha = compute_sha256(cand_dir / fname)
                assert actual_sha == expected_sha, f"Artifact checksum mismatch for {fname}: {actual_sha} vs {expected_sha}"
    print("  [OK] All candidate artifact checksums verified against checksums.sha256")

    # 3. Model/vectorizer/encoder compatibility
    model = joblib.load(cand_dir / "model.joblib")
    vec = joblib.load(cand_dir / "vectorizer.joblib")
    enc = joblib.load(cand_dir / "encoder.joblib")
    assert len(vec.vocabulary_) == 256, f"Expected 256 vocabulary features, got {len(vec.vocabulary_)}"
    assert model.model.coef_.shape == (4, 256), f"Unexpected model coef shape: {model.model.coef_.shape}"
    expected_classes = ["Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"]
    assert list(enc.classes_) == expected_classes, f"Encoder classes mismatch: {enc.classes_}"
    print("  [OK] Model/vectorizer/encoder compatibility verified (256-dim input/output, 4 target classes)")

    # 4. Manifest checks
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Dev cohort checksum
    assert dev_cohort_path.exists(), f"Missing dev cohort: {dev_cohort_path}"
    actual_dev_sha = compute_sha256(dev_cohort_path)
    expected_dev_sha = manifest["development_cohort"]["sha256"]
    assert actual_dev_sha == expected_dev_sha, f"Dev cohort checksum mismatch: {actual_dev_sha} vs {expected_dev_sha}"
    print("  [OK] Development dataset checksum verified against manifest")

    # Holdout checksum
    assert holdout_path.exists(), f"Missing holdout: {holdout_path}"
    actual_holdout_sha = compute_sha256(holdout_path)
    expected_holdout_sha = manifest["sealed_holdout"]["sha256"]
    assert actual_holdout_sha == expected_holdout_sha, f"Holdout checksum mismatch: {actual_holdout_sha} vs {expected_holdout_sha}"
    print("  [OK] Sealed holdout checksum verified against manifest without unsealing")

    # 5. Vault state check
    assert vault_state_path.exists(), f"Missing vault state: {vault_state_path}"
    with open(vault_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    assert state.get("status") == "LOCKED", f"Vault is not LOCKED: {state.get('status')}"
    assert state.get("unsealed") is False, "Vault unsealed flag is True"
    assert state.get("unseal_count") == 0, f"Vault unseal_count is not 0: {state.get('unseal_count')}"
    print("  [OK] Vault status confirmed LOCKED with unseal_count == 0")

    # 6. Evaluation events ledger
    events_exist = events_log_path.exists()
    events_bytes = events_log_path.stat().st_size if events_exist else 0
    assert events_bytes == 0, f"Prior evaluation events detected in ledger ({events_bytes} bytes)"
    print("  [OK] Zero prior evaluation events exist in audit ledger")

    # 7. Probability reporting inspection
    # Check whether genuine probability calibration exists
    has_genuine_calibration = False
    calibration_method = "None (Standard LinearSVC decision_function reported; uncalibrated softmax available for heuristic ranking)"
    print(f"  [OK] Probability reporting mode: {calibration_method}")

    return {
        "model_config": config,
        "dev_sha256": actual_dev_sha,
        "holdout_sha256": actual_holdout_sha,
        "calibration_method": calibration_method,
        "readiness_passed": True,
    }


def main():
    print("=" * 70)
    print("PHASE 6: FINAL ONE-SHOT HOLDOUT EVALUATION")
    print("=" * 70)

    # 1. Pre-unseal readiness checks
    readiness = run_pre_unseal_readiness_checks()
    print("\n>>> ALL PRE-UNSEAL READINESS CHECKS PASSED. PROCEEDING TO UNSEAL. <<<\n")

    # 2. Initialize Evaluator & Pipeline
    cand_dir = Path("artifacts/models/phase5_best_candidate")
    model = joblib.load(cand_dir / "model.joblib")
    vec = joblib.load(cand_dir / "vectorizer.joblib")
    enc = joblib.load(cand_dir / "encoder.joblib")
    config = readiness["model_config"]

    pipeline = FrozenHoldoutPipeline(
        model=model,
        vectorizer=vec,
        encoder=enc,
        k=config["feature_configuration"]["k"],
        canonical=config["feature_configuration"]["canonical"],
    )

    vault = SealedVault()
    eval_events_path = Path("artifacts/vault/evaluation_events.jsonl")
    evaluator = FinalHoldoutEvaluator(vault=vault, event_log_path=eval_events_path)

    # 3. Execute final evaluation exactly once
    print("[2/7] Executing one-shot holdout evaluation via FinalHoldoutEvaluator...")
    eval_reason = (
        "Official Phase 6 Final One-Shot Holdout Evaluation of frozen candidate "
        f"'{config['model_candidate_id']}' on unseen Tier 2 holdout (N=240)."
    )
    unseal_timestamp = datetime.datetime.now().isoformat()

    eval_result = evaluator.evaluate_frozen_model(
        model=pipeline,
        model_config=config,
        unseal_token=AUTHORIZED_UNSEAL_TOKEN,
        reason=eval_reason,
    )
    print("  [OK] Vault successfully unsealed and evaluated.")

    # 4. Extract detailed predictions and decision scores
    print("[3/7] Computing comprehensive final metrics and Wilson confidence intervals...")
    # Access unsealed samples through the approved one-shot result or vault data path now that it is legally unsealed
    with open(vault.vault_data_path, "r", encoding="utf-8") as f:
        raw_holdout = json.load(f)

    holdout_seqs = [r["sequence"] for r in raw_holdout]
    y_true = [r["label"] for r in raw_holdout]
    sample_ids = [r["sample_id"] for r in raw_holdout]

    y_pred = pipeline.predict(holdout_seqs)
    dec_scores = pipeline.decision_function(holdout_seqs)

    # Standard metrics
    metrics = eval_result["metrics"]
    target_classes = ["Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"]

    # Compute Wilson CIs
    total_samples = len(y_true)
    correct_total = int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp))
    accuracy_wilson = compute_wilson_ci(correct_total, total_samples, confidence=0.95)

    per_class_metrics = {}
    for cls_name in target_classes:
        cls_total = sum(1 for yt in y_true if yt == cls_name)
        cls_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls_name and yp == cls_name)
        recall_wilson = compute_wilson_ci(cls_correct, cls_total, confidence=0.95)
        cls_metrics = metrics["per_class"][cls_name]
        per_class_metrics[cls_name] = {
            "precision": float(cls_metrics["precision"]),
            "recall": float(cls_metrics["recall"]),
            "f1_score": float(cls_metrics["f1-score"]),
            "support": int(cls_metrics["support"]),
            "correct_count": cls_correct,
            "error_count": cls_total - cls_correct,
            "recall_95ci": {
                "ci_lower": recall_wilson["ci_lower"],
                "ci_upper": recall_wilson["ci_upper"],
            },
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=target_classes).tolist()

    # Detailed misclassifications
    misclassified = []
    for i, (sid, yt, yp) in enumerate(zip(sample_ids, y_true, y_pred)):
        if yt != yp:
            misclassified.append({
                "sample_id": sid,
                "true_label": yt,
                "predicted_label": yp,
                "gene": raw_holdout[i].get("metadata", {}).get("gene"),
                "sequence_length": len(holdout_seqs[i]),
            })

    # 5. Development vs Holdout Comparison
    print("[4/7] Generating development vs holdout comparative analysis...")
    dev_f1 = float(config["primary_cv_performance"]["mean_macro_f1"])
    dev_f1_std = float(config["primary_cv_performance"]["std_macro_f1"])
    dev_acc = float(config["primary_cv_performance"]["mean_accuracy"])

    holdout_f1 = float(metrics["f1_macro"])
    holdout_acc = float(metrics["accuracy"])

    comparison = {
        "development_grouped_cv": {
            "mean_macro_f1": dev_f1,
            "std_macro_f1": dev_f1_std,
            "mean_accuracy": dev_acc,
            "cervical_recall_mean": float(np.mean([f["cervical_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
            "colorectal_recall_mean": float(np.mean([f["colorectal_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
            "endometrial_recall_mean": float(np.mean([f["endometrial_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
            "liver_recall_mean": float(np.mean([f["liver_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
        },
        "final_holdout_evaluation": {
            "macro_f1": holdout_f1,
            "accuracy": holdout_acc,
            "accuracy_95ci": {
                "ci_lower": accuracy_wilson["ci_lower"],
                "ci_upper": accuracy_wilson["ci_upper"],
            },
            "cervical_recall": per_class_metrics["Cervical Cancer"]["recall"],
            "cervical_recall_95ci": per_class_metrics["Cervical Cancer"]["recall_95ci"],
            "colorectal_recall": per_class_metrics["Colorectal Cancer"]["recall"],
            "colorectal_recall_95ci": per_class_metrics["Colorectal Cancer"]["recall_95ci"],
            "endometrial_recall": per_class_metrics["Endometrial Cancer"]["recall"],
            "endometrial_recall_95ci": per_class_metrics["Endometrial Cancer"]["recall_95ci"],
            "liver_recall": per_class_metrics["Liver Cancer"]["recall"],
            "liver_recall_95ci": per_class_metrics["Liver Cancer"]["recall_95ci"],
        },
        "differences": {
            "delta_macro_f1": float(holdout_f1 - dev_f1),
            "delta_accuracy": float(holdout_acc - dev_acc),
            "delta_cervical_recall": float(per_class_metrics["Cervical Cancer"]["recall"] - np.mean([f["cervical_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
            "delta_colorectal_recall": float(per_class_metrics["Colorectal Cancer"]["recall"] - np.mean([f["colorectal_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
            "delta_endometrial_recall": float(per_class_metrics["Endometrial Cancer"]["recall"] - np.mean([f["endometrial_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
            "delta_liver_recall": float(per_class_metrics["Liver Cancer"]["recall"] - np.mean([f["liver_recall"] for f in config["primary_cv_performance"]["fold_results"]])),
        },
    }

    # 6. Plot Confusion Matrix
    print("[5/7] Rendering confusion matrix visualization...")
    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_path = eval_dir / "final_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(target_classes)),
        yticks=np.arange(len(target_classes)),
        xticklabels=["Cervical", "Colorectal", "Endometrial", "Liver"],
        yticklabels=["Cervical", "Colorectal", "Endometrial", "Liver"],
        title="Final Holdout Confusion Matrix (N=240)",
        ylabel="True Sequence Source Label",
        xlabel="Predicted Sequence Source Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = np.max(cm) / 2.0
    for i in range(len(target_classes)):
        for j in range(len(target_classes)):
            ax.text(
                j, i, format(cm[i][j], "d"),
                ha="center", va="center",
                color="white" if cm[i][j] > thresh else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved confusion matrix plot to {fig_path}")

    # 7. Generate Deliverable Files
    print("[6/7] Writing final evaluation result files and receipt...")

    # A. final_holdout_results.json
    final_results = {
        "evaluation_event": "FINAL_HOLDOUT_EVALUATION",
        "timestamp": unseal_timestamp,
        "sample_count": total_samples,
        "summary_metrics": {
            "accuracy": holdout_acc,
            "accuracy_95ci": accuracy_wilson,
            "macro_precision": float(metrics["precision_macro"]),
            "macro_recall": float(metrics["recall_macro"]),
            "macro_f1": holdout_f1,
            "weighted_f1": float(metrics["f1_weighted"]),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": target_classes,
            "matrix": cm,
        },
        "development_comparison": comparison,
        "total_misclassifications": len(misclassified),
        "misclassifications": misclassified,
        "decision_scores_summary": {
            "mean_max_margin": float(np.mean(np.max(dec_scores, axis=1))),
            "min_max_margin": float(np.min(np.max(dec_scores, axis=1))),
            "probability_calibration_disclosure": readiness["calibration_method"],
        },
    }
    results_path = eval_dir / "final_holdout_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"  [OK] Saved final holdout results to {results_path}")

    # B. final_classification_report.json
    report_path = eval_dir / "final_classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics["classification_report"], f, indent=2)
    print(f"  [OK] Saved final classification report to {report_path}")

    # C. final_evaluation_receipt.json in artifacts/vault/
    receipt = {
        "receipt_version": "1.0.0",
        "event_identifier": "FINAL_HOLDOUT_EVALUATION",
        "evaluation_timestamp": unseal_timestamp,
        "evaluator_version": "FinalHoldoutEvaluator v1.0.0",
        "unseal_count": 1,
        "authorization_token_hash": hashlib.sha256(AUTHORIZED_UNSEAL_TOKEN.encode("utf-8")).hexdigest()[:16],
        "cryptographic_verification": {
            "model_joblib_sha256": compute_sha256(cand_dir / "model.joblib"),
            "vectorizer_joblib_sha256": compute_sha256(cand_dir / "vectorizer.joblib"),
            "encoder_joblib_sha256": compute_sha256(cand_dir / "encoder.joblib"),
            "model_config_sha256": compute_sha256(cand_dir / "model_config.json"),
            "development_cohort_sha256": readiness["dev_sha256"],
            "sealed_holdout_sha256": readiness["holdout_sha256"],
        },
        "final_metric_summary": {
            "accuracy": holdout_acc,
            "accuracy_95ci": [accuracy_wilson["ci_lower"], accuracy_wilson["ci_upper"]],
            "macro_f1": holdout_f1,
            "macro_precision": float(metrics["precision_macro"]),
            "macro_recall": float(metrics["recall_macro"]),
            "cervical_recall": per_class_metrics["Cervical Cancer"]["recall"],
            "colorectal_recall": per_class_metrics["Colorectal Cancer"]["recall"],
            "endometrial_recall": per_class_metrics["Endometrial Cancer"]["recall"],
            "liver_recall": per_class_metrics["Liver Cancer"]["recall"],
        },
        "scientific_disclaimer": (
            "Educational DNA sequence-source classification benchmark. Sequences originate from human "
            "chromosome 15 transcripts and bacterial genomic assemblies. Not validated for clinical cancer diagnosis."
        ),
    }
    receipt_path = Path("artifacts/vault/final_evaluation_receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"  [OK] Saved final evaluation receipt to {receipt_path}")

    # 8. Post-Evaluation Vault Verification
    print("[7/7] Verifying post-evaluation vault lock state...")
    post_state = vault.get_vault_status()
    assert post_state["status"] == "UNSEALED", f"Expected UNSEALED, got {post_state['status']}"
    assert post_state["unseal_count"] == 1, f"Expected unseal_count 1, got {post_state['unseal_count']}"
    assert post_state["unsealed"] is True, "Expected unsealed True"
    assert post_state["is_locked"] is False, "Expected is_locked False"

    # Confirm repeated evaluation is blocked
    try:
        evaluator.evaluate_frozen_model(
            model=pipeline,
            model_config=config,
            unseal_token=AUTHORIZED_UNSEAL_TOKEN,
            reason="Blocked repeated evaluation attempt",
        )
        raise RuntimeError("CRITICAL ERROR: Evaluator permitted repeated evaluation!")
    except PermissionError:
        print("  [OK] Second evaluation attempt was strictly blocked with PermissionError.")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION METRIC SUMMARY:")
    print(f"  Accuracy:         {holdout_acc:.4f} (95% CI: [{accuracy_wilson['ci_lower']:.4f}, {accuracy_wilson['ci_upper']:.4f}])")
    print(f"  Macro F1:         {holdout_f1:.4f}")
    print(f"  Macro Precision:  {metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:     {metrics['recall_macro']:.4f}")
    print(f"  Weighted F1:      {metrics['f1_weighted']:.4f}")
    print(f"  Total Errors:     {len(misclassified)} / {total_samples}")
    print("-" * 70)
    print("  Per-Class Recall & Wilson 95% CIs:")
    for cls_name in target_classes:
        cm_data = per_class_metrics[cls_name]
        print(f"    - {cls_name:18s}: Recall={cm_data['recall']:.4f} "
              f"[{cm_data['recall_95ci']['ci_lower']:.4f}, {cm_data['recall_95ci']['ci_upper']:.4f}], "
              f"Precision={cm_data['precision']:.4f}, F1={cm_data['f1_score']:.4f}, "
              f"Errors={cm_data['error_count']}/{cm_data['support']}")
    print("-" * 70)
    print(f"  Development CV Macro F1: {dev_f1:.6f} vs Final Holdout: {holdout_f1:.6f} (Delta: {comparison['differences']['delta_macro_f1']:+.6f})")
    print(f"  Development CV Accuracy: {dev_acc:.6f} vs Final Holdout: {holdout_acc:.6f} (Delta: {comparison['differences']['delta_accuracy']:+.6f})")
    print("=" * 70)
    print("PHASE 6 ONE-SHOT EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
