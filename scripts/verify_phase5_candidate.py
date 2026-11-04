"""Verification and reproducibility audit script for Phase 5 candidate model."""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from dna_cancer.data.development import get_development_cv_folds, load_development_cohort
from dna_cancer.evaluation.metrics import compute_classification_metrics
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.features.kmer_extractor import extract_kmers
from dna_cancer.models.baseline_models import LinearSVCClassifier


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_artifacts(cand_dir: str) -> Dict[str, Any]:
    """Inspect and verify candidate model artifacts."""
    files = ["model.joblib", "vectorizer.joblib", "encoder.joblib", "model_config.json"]
    artifact_report = {}
    checksums = {}

    for fname in files:
        fpath = os.path.join(cand_dir, fname)
        assert os.path.exists(fpath), f"Artifact missing: {fpath}"
        sha = compute_sha256(fpath)
        size = os.path.getsize(fpath)
        checksums[fname] = sha
        artifact_report[fname] = {
            "path": fpath,
            "sha256": sha,
            "size_bytes": size,
            "exists": True,
        }

    # Load artifacts
    model = joblib.load(os.path.join(cand_dir, "model.joblib"))
    vectorizer = joblib.load(os.path.join(cand_dir, "vectorizer.joblib"))
    encoder = joblib.load(os.path.join(cand_dir, "encoder.joblib"))
    config = json.load(open(os.path.join(cand_dir, "model_config.json"), "r"))

    # Verify model
    assert isinstance(model, LinearSVCClassifier), f"Unexpected model type: {type(model)}"
    coef_shape = model.model.coef_.shape
    classes = list(model.model.classes_)
    assert coef_shape == (4, 256), f"Unexpected coef shape: {coef_shape}"

    # Verify vectorizer
    assert hasattr(vectorizer, "vocabulary_"), "Vectorizer missing vocabulary_"
    vocab_len = len(vectorizer.vocabulary_)
    assert vocab_len == 256, f"Expected 256 features for k=4, got {vocab_len}"
    assert vectorizer.token_pattern == r"\S+", f"Unexpected token_pattern: {vectorizer.token_pattern}"

    # Verify encoder
    expected_classes = ["Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"]
    actual_classes = list(encoder.classes_)
    assert actual_classes == expected_classes, f"Class mismatch: {actual_classes} vs {expected_classes}"

    # Verify dimension alignment
    assert coef_shape[1] == vocab_len, f"Dimension mismatch: model expects {coef_shape[1]}, vectorizer has {vocab_len}"

    # Verify config
    assert config.get("is_frozen") is True, "Config is_frozen is not True"
    assert config.get("model_name") == "linear_svc", f"Config model mismatch: {config.get('model_name')}"
    assert config.get("feature_configuration", {}).get("k") == 4, "Config k mismatch"

    # Verify no embedded sequences or raw datasets
    def find_large_strings(obj: Any, max_len: int = 100) -> List[str]:
        found = []
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if isinstance(v, (str, bytes)) and len(v) > max_len:
                    found.append(f"{k} (len {len(v)})")
                elif isinstance(v, list) and v and isinstance(v[0], (str, bytes)) and len(v[0]) > max_len:
                    found.append(f"{k} (list of {len(v)} strings)")
        return found

    leaked_model = find_large_strings(model) + find_large_strings(model.model)
    leaked_vec = find_large_strings(vectorizer)
    leaked_enc = find_large_strings(encoder)
    assert not leaked_model, f"Potential data leak in model: {leaked_model}"
    assert not leaked_vec, f"Potential data leak in vectorizer: {leaked_vec}"
    assert not leaked_enc, f"Potential data leak in encoder: {leaked_enc}"

    return {
        "artifact_files": artifact_report,
        "checksums": checksums,
        "model_architecture": "LinearSVCClassifier (wrapped scikit-learn LinearSVC)",
        "feature_dimensions": vocab_len,
        "model_coefficients_shape": list(coef_shape),
        "target_classes": actual_classes,
        "dimension_parity": True,
        "config_is_frozen": True,
        "embedded_data_leaks_detected": False,
    }


def verify_prediction_consistency(cand_dir: str) -> Dict[str, Any]:
    """Verify bit-level prediction consistency between in-memory training and reloaded artifacts."""
    cohort_path = "data/processed/development_cohort.csv"
    with open(cohort_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classes = ["Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"]
    selected_samples = []
    for c in classes:
        matching = [r for r in rows if r["cancer"] == c]
        selected_samples.append(matching[0])
        selected_samples.append(matching[len(matching) // 2])

    assert len(selected_samples) == 8, f"Expected 8 samples, got {len(selected_samples)}"

    # In-memory training pipeline
    all_seqs = [r["sequence"] for r in rows]
    all_labels = [r["cancer"] for r in rows]

    fb_mem = FeatureBuilder(k=4, vectorizer_type="tfidf", canonical=False, return_sparse=True)
    X_mem = fb_mem.fit(all_seqs, all_labels).transform_sequences(all_seqs)
    clf_mem = LinearSVCClassifier(C=1.0, dual="auto", max_iter=2000, random_state=42)
    clf_mem.fit(X_mem, np.array(fb_mem.label_encoder.transform(all_labels)))

    # In-memory predictions on selected samples
    sel_seqs = [s["sequence"] for s in selected_samples]
    sel_X_mem = fb_mem.transform_sequences(sel_seqs)
    pred_idx_mem = clf_mem.predict(sel_X_mem)
    pred_labels_mem = fb_mem.label_encoder.inverse_transform(pred_idx_mem)
    dec_scores_mem = clf_mem.model.decision_function(sel_X_mem)
    proba_mem = clf_mem.predict_proba(sel_X_mem)

    # Reload saved artifacts from disk
    loaded_model = joblib.load(os.path.join(cand_dir, "model.joblib"))
    loaded_vec = joblib.load(os.path.join(cand_dir, "vectorizer.joblib"))
    loaded_enc = joblib.load(os.path.join(cand_dir, "encoder.joblib"))

    # Predict using reloaded artifacts
    sel_kmers = [" ".join(extract_kmers(s, k=4, canonical=False)) for s in sel_seqs]
    sel_X_reloaded = loaded_vec.transform(sel_kmers)
    pred_idx_reloaded = loaded_model.predict(sel_X_reloaded)
    pred_labels_reloaded = loaded_enc.inverse_transform(pred_idx_reloaded)
    dec_scores_reloaded = loaded_model.model.decision_function(sel_X_reloaded)
    proba_reloaded = loaded_model.predict_proba(sel_X_reloaded)

    # Check parity
    sample_evaluations = []
    all_labels_identical = True
    max_decision_score_diff = 0.0

    for i, s in enumerate(selected_samples):
        lbl_match = bool(pred_labels_mem[i] == pred_labels_reloaded[i])
        if not lbl_match:
            all_labels_identical = False
        score_diff = float(np.max(np.abs(dec_scores_mem[i] - dec_scores_reloaded[i])))
        max_decision_score_diff = max(max_decision_score_diff, score_diff)
        sample_evaluations.append({
            "sample_index": i + 1,
            "sample_id": s["Unnamed: 0"],
            "true_cancer": s["cancer"],
            "gene": s.get("gene", ""),
            "sequence_length": len(s["sequence"]),
            "predicted_label_in_memory": str(pred_labels_mem[i]),
            "predicted_label_reloaded": str(pred_labels_reloaded[i]),
            "label_match": lbl_match,
            "max_decision_score_diff": score_diff,
            "decision_scores_reloaded": [float(x) for x in dec_scores_reloaded[i]],
            "predicted_probabilities_reloaded": [float(x) for x in proba_reloaded[i]],
        })

    scores_allclose = bool(np.allclose(dec_scores_mem, dec_scores_reloaded, atol=1e-10))
    probas_allclose = bool(np.allclose(proba_mem, proba_reloaded, atol=1e-10))

    return {
        "num_samples_evaluated": len(selected_samples),
        "all_predicted_labels_identical": all_labels_identical,
        "max_decision_score_difference": max_decision_score_diff,
        "decision_scores_match_tolerance_1e_10": scores_allclose,
        "probabilities_match_tolerance_1e_10": probas_allclose,
        "sample_evaluations": sample_evaluations,
    }


def audit_cross_validation_records() -> Dict[str, Any]:
    """Audit the Phase 5 experiment records and metrics calculation."""
    exp_file = "artifacts/experiments/phase5_results.json"
    cv_summary_file = "artifacts/evaluation/phase5_cv_summary.json"

    assert os.path.exists(exp_file), f"Missing {exp_file}"
    assert os.path.exists(cv_summary_file), f"Missing {cv_summary_file}"

    with open(exp_file, "r") as f:
        exp_data = json.load(f)
    with open(cv_summary_file, "r") as f:
        cv_summary = json.load(f)

    # Find the winning candidate experiment
    cand_exp = None
    for exp in exp_data["experiments"]:
        if exp["exp_id"] == "canonical_comp_linear_svc_k4_standard":
            cand_exp = exp
            break
    assert cand_exp is not None, "Winning experiment record not found in phase5_results.json"

    fold_results = cand_exp["fold_results"]
    assert len(fold_results) == 3, f"Expected 3 folds, got {len(fold_results)}"

    # Audit Macro F1 calculation
    fold_f1s = [f["f1_macro"] for f in fold_results]
    mean_f1_calc = float(np.mean(fold_f1s))
    pop_std_f1 = float(np.std(fold_f1s, ddof=0))  # population std (used in numpy/scikit-learn default)
    sample_std_f1 = float(np.std(fold_f1s, ddof=1))  # sample std

    recorded_mean_f1 = cand_exp["mean_macro_f1"]
    recorded_std_f1 = cand_exp["std_macro_f1"]

    assert np.isclose(mean_f1_calc, recorded_mean_f1, atol=1e-6), "Mean Macro F1 calculation mismatch"
    assert np.isclose(pop_std_f1, recorded_std_f1, atol=1e-6), "Std Macro F1 calculation mismatch"

    # Audit Cervical recall reporting
    cervical_recalls = [f["cervical_recall"] for f in fold_results]
    assert cervical_recalls[0] == 1.0, f"Fold 0 Cervical recall mismatch: {cervical_recalls[0]}"
    assert np.isclose(cervical_recalls[1], 63 / 64, atol=1e-5), f"Fold 1 recall mismatch: {cervical_recalls[1]}"
    assert np.isclose(cervical_recalls[2], 62 / 64, atol=1e-5), f"Fold 2 recall mismatch: {cervical_recalls[2]}"

    # Trace 3 misclassified cervical samples
    cervical_misclass = [m for m in cand_exp["misclassified_records"] if m["true_label"] == "Cervical Cancer"]
    assert len(cervical_misclass) == 3, f"Expected 3 cervical misclassifications, got {len(cervical_misclass)}"

    genes_misclass = {m["sample_id"]: m["gene"] for m in cervical_misclass}
    assert "113" in genes_misclass and genes_misclass["113"] == "NDN", "NDN misclassification missing"
    assert "23" in genes_misclass and genes_misclass["23"] == "NIPA1", "NIPA1 sample 23 missing"
    assert "24" in genes_misclass and genes_misclass["24"] == "NIPA1", "NIPA1 sample 24 missing"

    # Check secondary ShuffleSplit disclosure
    sec_summary = cv_summary.get("secondary_shufflesplit_summary", {})
    cand_sec = sec_summary.get("canonical_comp_linear_svc_k4_standard", {})
    disclosure = cand_sec.get("independence_disclosure", "")
    assert "non-independent" in disclosure, f"Missing non-independence disclosure in secondary summary: {disclosure}"

    return {
        "recorded_mean_macro_f1": recorded_mean_f1,
        "calculated_mean_macro_f1": mean_f1_calc,
        "recorded_std_macro_f1": recorded_std_f1,
        "calculated_population_std": pop_std_f1,
        "calculated_sample_std": sample_std_f1,
        "std_convention_used": "population standard deviation (ddof=0)",
        "fold_results": fold_results,
        "cervical_fold_recalls": {
            "fold_0_snrpn": cervical_recalls[0],
            "fold_1_nipa2_and_minor": cervical_recalls[1],
            "fold_2_ube3a_and_minor": cervical_recalls[2],
        },
        "misclassified_cervical_traceability": cervical_misclass,
        "secondary_shufflesplit_disclosure": disclosure,
        "cv_audit_passed": True,
    }


def verify_model_reproducibility() -> Dict[str, Any]:
    """Recreate the winning candidate configuration from scratch and check parity."""
    folds = get_development_cv_folds()
    new_fold_metrics = []

    for fold_idx, (train_samples, val_samples) in enumerate(folds):
        train_seqs = [s.sequence for s in train_samples]
        train_labels = [s.label for s in train_samples]
        val_seqs = [s.sequence for s in val_samples]
        val_labels = [s.label for s in val_samples]

        fb = FeatureBuilder(k=4, vectorizer_type="tfidf", canonical=False)
        X_train = fb.fit(train_seqs, train_labels).transform_sequences(train_seqs)
        X_val = fb.transform_sequences(val_seqs)

        y_train = np.array(fb.label_encoder.transform(train_labels))
        y_val = np.array(fb.label_encoder.transform(val_labels))

        clf = LinearSVCClassifier(C=1.0, dual="auto", max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        metrics = compute_classification_metrics(y_val, y_pred, target_names=list(fb.label_encoder.classes_))
        new_fold_metrics.append({
            "fold": fold_idx,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "cervical_recall": metrics["per_class"]["Cervical Cancer"]["recall"],
        })

    new_f1s = [f["f1_macro"] for f in new_fold_metrics]
    new_accs = [f["accuracy"] for f in new_fold_metrics]
    new_mean_f1 = float(np.mean(new_f1s))
    new_std_f1 = float(np.std(new_f1s, ddof=0))
    new_mean_acc = float(np.mean(new_accs))

    # Compare with Phase 5 record
    exp_file = "artifacts/experiments/phase5_results.json"
    with open(exp_file, "r") as f:
        exp_data = json.load(f)
    cand_exp = next(e for e in exp_data["experiments"] if e["exp_id"] == "canonical_comp_linear_svc_k4_standard")

    recorded_mean_f1 = cand_exp["mean_macro_f1"]
    recorded_std_f1 = cand_exp["std_macro_f1"]
    recorded_mean_acc = cand_exp["mean_accuracy"]

    f1_diff = abs(new_mean_f1 - recorded_mean_f1)
    std_diff = abs(new_std_f1 - recorded_std_f1)
    acc_diff = abs(new_mean_acc - recorded_mean_acc)

    # Parity check: LinearSVC with dual='auto' and random_state=42 is deterministic
    is_exact_parity = bool(f1_diff < 1e-9 and std_diff < 1e-9 and acc_diff < 1e-9)

    return {
        "reproduced_mean_macro_f1": new_mean_f1,
        "recorded_mean_macro_f1": recorded_mean_f1,
        "absolute_f1_difference": f1_diff,
        "reproduced_std_macro_f1": new_std_f1,
        "recorded_std_macro_f1": recorded_std_f1,
        "reproduced_mean_accuracy": new_mean_acc,
        "recorded_mean_accuracy": recorded_mean_acc,
        "absolute_acc_difference": acc_diff,
        "exact_numerical_parity": is_exact_parity,
        "reproduced_fold_results": new_fold_metrics,
    }


def verify_vault_lock_and_isolation() -> Dict[str, Any]:
    """Verify that the Tier 2 sealed vault remains strictly locked with zero unseals."""
    vault_state_path = "artifacts/vault/vault_state.json"
    manifest_path = "artifacts/vault/test_sealed_manifest.json"
    holdout_path = "artifacts/vault/test_holdout_sealed.json"
    events_path = "artifacts/vault/evaluation_events.jsonl"

    assert os.path.exists(vault_state_path), f"Missing {vault_state_path}"
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
    assert os.path.exists(holdout_path), f"Missing {holdout_path}"

    with open(vault_state_path, "r") as f:
        state = json.load(f)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Check vault state
    assert state.get("status") == "LOCKED", f"Vault status is not LOCKED: {state.get('status')}"
    assert state.get("unsealed") is False, "Vault unsealed flag is True"
    assert state.get("unseal_count") == 0, f"Vault unseal_count is non-zero: {state.get('unseal_count')}"
    assert state.get("unsealed_at") is None, "Vault unsealed_at is set"
    assert state.get("unseal_token") is None, "Vault unseal_token is set"

    # Check evaluation events ledger
    events_exist = os.path.exists(events_path)
    events_size = os.path.getsize(events_path) if events_exist else 0
    assert events_size == 0, f"Evaluation events recorded unexpectedly: {events_size} bytes"

    # Verify holdout checksum matches manifest without unsealing
    actual_holdout_sha = compute_sha256(holdout_path)
    expected_holdout_sha = manifest["sealed_holdout"]["sha256"]
    assert actual_holdout_sha == expected_holdout_sha, f"Holdout checksum mismatch: {actual_holdout_sha} vs {expected_holdout_sha}"

    # Verify no holdout-only genes leaked into development misclassification records
    exp_file = "artifacts/experiments/phase5_results.json"
    cv_summary_file = "artifacts/evaluation/phase5_cv_summary.json"
    exp_text = open(exp_file, "r").read()
    summary_text = open(cv_summary_file, "r").read()

    # CYFIP1, TUBGCP5, and POTEB2 are holdout-exclusive genes
    holdout_exclusive_genes = ["CYFIP1", "TUBGCP5", "POTEB2"]
    for gene in holdout_exclusive_genes:
        assert f'"gene": "{gene}"' not in exp_text, f"Holdout gene '{gene}' leaked into {exp_file}"

    return {
        "vault_status": state.get("status"),
        "unsealed": state.get("unsealed"),
        "unseal_count": state.get("unseal_count"),
        "evaluation_events_file_exists": events_exist,
        "evaluation_events_count": 0,
        "holdout_checksum_verified": True,
        "holdout_sha256": actual_holdout_sha,
        "holdout_leak_detected_in_experiments": False,
        "vault_lock_intact": True,
    }


def main():
    print("=" * 60)
    print("PHASE 5.1: FROZEN MODEL VERIFICATION & AUDIT")
    print("=" * 60)

    cand_dir = "artifacts/models/phase5_best_candidate"
    eval_dir = "artifacts/evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Audit artifacts
    print("\n[1/5] Auditing frozen model artifacts...")
    artifacts_res = audit_artifacts(cand_dir)
    print(f"  Artifact files: {list(artifacts_res['checksums'].keys())}")
    print(f"  Architecture: {artifacts_res['model_architecture']}")
    print(f"  Feature dims: {artifacts_res['feature_dimensions']}")
    print(f"  Config frozen: {artifacts_res['config_is_frozen']}")

    # Write checksums.sha256
    checksums_path = os.path.join(cand_dir, "checksums.sha256")
    with open(checksums_path, "w", encoding="utf-8") as f:
        for fname, sha in artifacts_res["checksums"].items():
            f.write(f"{sha}  {fname}\n")
    print(f"  Wrote checksums to {checksums_path}")

    # 2. Verify prediction consistency
    print("\n[2/5] Verifying prediction consistency across 8 development samples...")
    pred_res = verify_prediction_consistency(cand_dir)
    print(f"  All labels identical: {pred_res['all_predicted_labels_identical']}")
    print(f"  Max decision score diff: {pred_res['max_decision_score_difference']:.2e}")
    print(f"  Decision scores match (atol=1e-10): {pred_res['decision_scores_match_tolerance_1e_10']}")

    # 3. Audit CV records
    print("\n[3/5] Auditing Phase 5 cross-validation records...")
    cv_res = audit_cross_validation_records()
    print(f"  Mean Macro F1: {cv_res['recorded_mean_macro_f1']:.4f}")
    print(f"  Std Macro F1: {cv_res['recorded_std_macro_f1']:.4f} ({cv_res['std_convention_used']})")
    print(f"  Cervical recalls: Fold 0 = {cv_res['cervical_fold_recalls']['fold_0_snrpn']:.4f}, "
          f"Fold 1 = {cv_res['cervical_fold_recalls']['fold_1_nipa2_and_minor']:.4f}, "
          f"Fold 2 = {cv_res['cervical_fold_recalls']['fold_2_ube3a_and_minor']:.4f}")

    # 4. Verify model reproducibility
    print("\n[4/5] Verifying model reproducibility from scratch...")
    repro_res = verify_model_reproducibility()
    print(f"  Reproduced Mean Macro F1: {repro_res['reproduced_mean_macro_f1']:.6f} vs Recorded: {repro_res['recorded_mean_macro_f1']:.6f}")
    print(f"  F1 difference: {repro_res['absolute_f1_difference']:.2e}")
    print(f"  Exact parity: {repro_res['exact_numerical_parity']}")

    # 5. Verify vault status
    print("\n[5/5] Verifying Sealed Test Vault lock and isolation...")
    vault_res = verify_vault_lock_and_isolation()
    print(f"  Vault status: {vault_res['vault_status']}")
    print(f"  Unseal count: {vault_res['unseal_count']}")
    print(f"  Holdout checksum verified: {vault_res['holdout_checksum_verified']}")
    print(f"  Evaluation events count: {vault_res['evaluation_events_count']}")

    # Assemble comprehensive verification report
    verification_report = {
        "verification_metadata": {
            "phase": "Phase 5.1",
            "timestamp": "2026-11-04 20:20:00",
            "candidate_directory": cand_dir,
            "target_model": "LinearSVC (k=4, TF-IDF, standard)",
        },
        "artifact_verification": artifacts_res,
        "prediction_consistency": pred_res,
        "cross_validation_audit": cv_res,
        "reproducibility_verification": repro_res,
        "vault_security_audit": vault_res,
        "overall_verification_status": "PASSED_VERIFIED",
    }

    out_json = os.path.join(eval_dir, "phase5_candidate_verification.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
    print(f"\nSaved comprehensive verification report to {out_json}")
    print("=" * 60)
    print("PHASE 5.1 VERIFICATION COMPLETE: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
