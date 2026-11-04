"""Prediction and inference orchestration module."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dna_cancer.data.validation import SequenceValidator
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.models.model_factory import ModelFactory
from dna_cancer.preprocessing.sequence_cleaner import SequenceCleaner
from dna_cancer.utils.exceptions import PredictionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class DNAPredictor:
    """Loads trained artifacts to perform cancer type classification on new DNA sequences."""

    def __init__(
        self,
        model: Optional[BaseDNAClassifier] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        cleaner: Optional[SequenceCleaner] = None,
        validator: Optional[SequenceValidator] = None,
    ) -> None:
        self.model = model
        self.feature_builder = feature_builder or FeatureBuilder()
        # Default cleaner in predictor preserves characters to ensure validation catches invalid bases
        self.cleaner = cleaner or SequenceCleaner(to_upper=True, remove_whitespace=True, remove_non_nucleotide=False)
        self.validator = validator or SequenceValidator()

    @classmethod
    def from_artifacts(
        cls,
        model_name: str,
        model_path: Union[str, Path],
        vectorizer_path: Union[str, Path],
        encoder_path: Optional[Union[str, Path]] = None,
        k: Optional[int] = None,
        canonical: Optional[bool] = None,
    ) -> "DNAPredictor":
        """Factory method to instantiate a predictor from saved artifact files."""
        model = ModelFactory.load(model_name, model_path)

        # Inspect sibling model_config.json if available to infer k and canonical representations
        config_path = Path(model_path).parent / "model_config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                feat_cfg = cfg_data.get("feature_configuration", {})
                if k is None and "k" in feat_cfg:
                    k = int(feat_cfg["k"])
                if canonical is None and "canonical" in feat_cfg:
                    canonical = bool(feat_cfg["canonical"])
            except Exception:
                pass

        init_k = k if k is not None else 6
        init_canonical = canonical if canonical is not None else False
        feature_builder = FeatureBuilder(k=init_k, canonical=init_canonical)
        feature_builder.load(vectorizer_path, encoder_path)
        return cls(model=model, feature_builder=feature_builder)

    def predict_sequence(self, raw_sequence: str) -> Dict[str, Any]:
        """Perform cancer classification on a single raw DNA sequence.

        Args:
            raw_sequence: Nucleotide sequence string.

        Returns:
            Dictionary containing predicted_class, confidence, and probabilities per class.

        Raises:
            PredictionError: If sequence is invalid or model is not loaded.
        """
        if self.model is None:
            raise PredictionError(
                "Model has not been initialized or loaded in DNAPredictor. "
                "Complete Phase 2/3 model training or load existing artifacts."
            )

        if not isinstance(raw_sequence, str):
            raise PredictionError(f"Expected sequence string, got {type(raw_sequence).__name__}")

        if not raw_sequence.strip():
            raise PredictionError("Input sequence cannot be empty.")

        cleaned = self.cleaner.clean(raw_sequence)
        if not self.validator.is_valid(cleaned):
            raise PredictionError("Sequence failed validation: invalid nucleotide composition or length.")

        features = self.feature_builder.transform_sequences([cleaned])
        pred_idx = int(self.model.predict(features)[0])

        proba: Optional[List[float]] = None
        confidence: Optional[float] = None
        try:
            proba_arr = self.model.predict_proba(features)[0]
            proba = [float(p) for p in proba_arr]
            confidence = float(proba_arr[pred_idx])
        except Exception:
            pass

        decision_scores: Optional[Dict[str, float]] = None
        if hasattr(self.model, "decision_function"):
            try:
                df_out = self.model.decision_function(features)
                if hasattr(df_out, "ndim") and df_out.ndim == 2:
                    raw_scores = df_out[0]
                elif hasattr(df_out, "ndim") and df_out.ndim == 1:
                    raw_scores = df_out
                else:
                    raw_scores = df_out[0] if isinstance(df_out, (list, tuple)) else df_out

                classes = getattr(self.feature_builder.label_encoder, "classes_", None)
                if classes is not None and len(classes) == len(raw_scores):
                    decision_scores = {str(c): float(s) for c, s in zip(classes, raw_scores)}
                else:
                    decision_scores = {f"class_{i}": float(s) for i, s in enumerate(raw_scores)}
            except Exception:
                decision_scores = None

        predicted_class: str
        if hasattr(self.feature_builder.label_encoder, "classes_"):
            predicted_class = self.feature_builder.inverse_transform_labels([pred_idx])[0]
        else:
            predicted_class = str(pred_idx)

        return {
            "predicted_class": predicted_class,
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": proba,
            "decision_scores": decision_scores,
            "sequence_length": len(cleaned),
        }

    def predict_batch(self, raw_sequences: List[str]) -> List[Dict[str, Any]]:
        """Perform predictions on a list of raw DNA sequences."""
        return [self.predict_sequence(s) for s in raw_sequences]
