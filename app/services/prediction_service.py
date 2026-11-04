"""Prediction service for DNA sequence classification using the frozen Phase 5 model.

Integrates the verified Phase 5 frozen candidate:
- Architecture: LinearSVC
- Features: 4-mers (k=4), TF-IDF, Standard representation (256 features)
- Preprocessing: Validated A, C, G, T, N IUPAC characters

Strict Safeguards:
- Model is loaded once at application startup (never refitted).
- Zero retraining or fitting during inference.
- Never accesses or unseals holdout data or vault.
- Never logs complete DNA sequences.
- Accurately reports raw decision scores (NOT probabilities or clinical confidences).
- Enforces mandatory educational disclosure.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dna_cancer.data.validation import STANDARD_DNA_BASES, validate_dna_sequence
from dna_cancer.prediction.predictor import DNAPredictor
from dna_cancer.preprocessing.sequence_cleaner import clean_sequence
from dna_cancer.utils.exceptions import DataValidationError, PredictionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = Path("artifacts/models/phase5_best_candidate")
DEFAULT_MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", 50000))
MIN_SEQUENCE_LENGTH = 4

# Clearly identified synthetic demonstration sequence (educational, non-clinical)
SYNTHETIC_DEMO_SEQUENCE = (
    ">synthetic_demo_sequence_educational_use_only | Synthetic benchmark test construct\n"
    "ATGGCGTACGTTAGCTAGCTAGCTGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTA\n"
    "GCTAGCGATCGAATCGATCGATCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAG\n"
    "CTAAGCTAGCTAGCTAGCTAGCGATCGAATCGATCGATCGATCGATCGATCGATCGATC\n"
    "GATCGATCGTAGCTAGCTAGCTAAGCTAGCTAGCTAGCTAGCGATCGAATCGATCGATC\n"
    "GATCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAGCTAAGCTAGCTAGCTAGCT"
)

SCIENTIFIC_DISCLOSURE_TEXT = (
    "This is an educational DNA sequence-source classification project. "
    "Its training dataset has known biological provenance limitations and is not clinically validated. "
    "Predictions reflect sequence alignment with historical training cohort sources (human chromosome 15 "
    "transcripts for Cervical, bacterial genomes for Colorectal, Endometrial, and Liver) and must not be "
    "interpreted as cancer diagnoses or medical advice."
)


class PredictionServiceError(Exception):
    """Custom exception raised when prediction service encounters configuration or runtime failure."""
    pass


class PredictionService:
    """Singleton-style service managing frozen Phase 5 model inference."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.max_sequence_length = max_sequence_length
        self.predictor: Optional[DNAPredictor] = None
        self.model_config: Dict[str, Any] = {}
        self.is_ready = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load frozen model, vectorizer, and encoder artifacts from disk."""
        logger.info("Initializing PredictionService from %s", self.model_dir)

        model_path = self.model_dir / "model.joblib"
        vec_path = self.model_dir / "vectorizer.joblib"
        enc_path = self.model_dir / "encoder.joblib"
        cfg_path = self.model_dir / "model_config.json"

        missing = []
        for name, p in [
            ("model", model_path),
            ("vectorizer", vec_path),
            ("encoder", enc_path),
            ("config", cfg_path),
        ]:
            if not p.exists():
                missing.append(f"{name} ({p})")

        if missing:
            msg = f"Missing required Phase 5 model artifacts in {self.model_dir}: {', '.join(missing)}"
            logger.error(msg)
            self.is_ready = False
            raise PredictionServiceError(msg)

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

            # Instantiate DNAPredictor once
            self.predictor = DNAPredictor.from_artifacts(
                model_name="linear_svc",
                model_path=model_path,
                vectorizer_path=vec_path,
                encoder_path=enc_path,
                k=4,
                canonical=False,
            )
            self.is_ready = True
            logger.info("Frozen Phase 5 model successfully loaded and ready for inference")
        except Exception as exc:
            self.is_ready = False
            msg = f"Failed to load frozen model artifacts: {exc}"
            logger.error(msg, exc_info=True)
            raise PredictionServiceError(msg) from exc

    def parse_raw_input(self, raw_text: str) -> Tuple[Optional[str], str]:
        """Parse raw user input into an optional FASTA header and nucleotide sequence.

        Supports single-record FASTA format or raw nucleotide string.
        Rejects multi-record FASTA files with a descriptive error message.

        Args:
            raw_text: Raw text string from textarea or file upload.

        Returns:
            Tuple of (header_id_or_description, raw_sequence_string).

        Raises:
            DataValidationError: If input is empty, malformed, or contains multiple FASTA records.
        """
        if not raw_text or not raw_text.strip():
            raise DataValidationError("Input sequence is empty. Please provide a DNA sequence.")

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            raise DataValidationError("Input contains only whitespace characters.")

        # Check if FASTA format
        fasta_headers = [line for line in lines if line.startswith(">")]
        if fasta_headers:
            if len(fasta_headers) > 1:
                header_previews = [h[1:].strip().split()[0] for h in fasta_headers[:3]]
                trunc_note = f" ... and {len(fasta_headers) - 3} more" if len(fasta_headers) > 3 else ""
                raise DataValidationError(
                    f"Multiple FASTA records detected ({len(fasta_headers)} records found: {', '.join(header_previews)}{trunc_note}). "
                    f"The prediction dashboard currently processes a single sequence per submission. "
                    f"Please submit an individual FASTA record."
                )

            # Single FASTA record
            first_header = fasta_headers[0]
            header_desc = first_header[1:].strip()
            seq_parts = [line for line in lines if not line.startswith(">")]
            if not seq_parts:
                raise DataValidationError(
                    f"FASTA record '{header_desc[:50]}' contains a header but no nucleotide sequence data."
                )
            sequence = "".join(seq_parts)
            return header_desc, sequence

        # Standard non-FASTA sequence
        return None, "".join(lines)

    def validate_and_clean_sequence(self, raw_sequence: str) -> str:
        """Validate sequence against length constraints and nucleotide alphabet.

        Args:
            raw_sequence: Nucleotide sequence string before cleaning.

        Returns:
            Standardized, cleaned uppercase DNA sequence.

        Raises:
            DataValidationError: If sequence fails validation rules.
        """
        # Remove whitespace first to evaluate true sequence length
        seq_nowhitespace = re.sub(r"\s+", "", raw_sequence)
        if not seq_nowhitespace:
            raise DataValidationError("DNA sequence contains no nucleotide characters.")

        # Check for invalid characters before cleaning to provide descriptive feedback
        upper_raw = seq_nowhitespace.upper()
        invalid_chars = sorted(list(set(upper_raw) - STANDARD_DNA_BASES))
        if invalid_chars:
            chars_display = ", ".join(f"'{c}'" for c in invalid_chars[:8])
            trunc = "..." if len(invalid_chars) > 8 else ""
            raise DataValidationError(
                f"Sequence contains unsupported characters: [{chars_display}{trunc}]. "
                f"Only standard DNA nucleotides (A, C, G, T) and ambiguous base (N) are accepted."
            )

        # Enforce maximum length limit - never silently truncate
        seq_len = len(upper_raw)
        if seq_len > self.max_sequence_length:
            raise DataValidationError(
                f"Sequence length ({seq_len:,} bp) exceeds the maximum allowed length of "
                f"{self.max_sequence_length:,} bp. Sequences must not be truncated."
            )

        # Enforce minimum length for 4-mer extraction
        if seq_len < MIN_SEQUENCE_LENGTH:
            raise DataValidationError(
                f"Sequence length ({seq_len} bp) is shorter than the minimum {MIN_SEQUENCE_LENGTH} bp "
                f"required to compute 4-mer k-mer feature vectors."
            )

        # Standardize using existing sequence cleaner
        cleaned = clean_sequence(
            seq_nowhitespace,
            to_upper=True,
            remove_whitespace=True,
            remove_non_nucleotide=False,
        )
        return cleaned

    def compute_sequence_statistics(self, sequence: str) -> Dict[str, Any]:
        """Compute base counts, percentages, and GC content for cleaned sequence.

        Args:
            sequence: Validated uppercase DNA sequence string.

        Returns:
            Dictionary of metrics including length, gc_content, and base frequencies.
        """
        length = len(sequence)
        if length == 0:
            return {"length": 0, "gc_content": 0.0, "frequencies": {}}

        counts = {
            "A": sequence.count("A"),
            "C": sequence.count("C"),
            "G": sequence.count("G"),
            "T": sequence.count("T"),
            "N": sequence.count("N"),
        }

        frequencies = {
            base: {
                "count": count,
                "percentage": round((count / length) * 100.0, 2),
            }
            for base, count in counts.items()
        }

        gc_count = counts["G"] + counts["C"]
        gc_content = round((gc_count / length) * 100.0, 2)

        return {
            "length": length,
            "gc_content": gc_content,
            "frequencies": frequencies,
            "gc_count": gc_count,
            "at_count": counts["A"] + counts["T"],
            "kmer_count": max(0, length - MIN_SEQUENCE_LENGTH + 1),
        }

    def predict(self, raw_input: str) -> Dict[str, Any]:
        """Perform end-to-end validation, feature extraction, and frozen model prediction.

        Args:
            raw_input: Raw sequence or FASTA text from user.

        Returns:
            Dictionary containing prediction results, scores, statistics, and metadata.

        Raises:
            DataValidationError: If input format or sequence is invalid.
            PredictionServiceError: If model is not loaded or prediction fails.
        """
        if not self.is_ready or self.predictor is None:
            raise PredictionServiceError(
                "Prediction engine is not ready. Model artifacts could not be loaded."
            )

        # 1. Parse FASTA or raw sequence
        header, raw_seq = self.parse_raw_input(raw_input)

        # 2. Validate and standardize sequence
        cleaned_seq = self.validate_and_clean_sequence(raw_seq)

        # 3. Compute sequence composition statistics
        stats = self.compute_sequence_statistics(cleaned_seq)

        # 4. Perform model inference
        try:
            pred_out = self.predictor.predict_sequence(cleaned_seq)
        except Exception as exc:
            logger.error("Inference failed on sequence of length %d: %s", len(cleaned_seq), exc)
            raise PredictionServiceError(f"Prediction failed: {exc}") from exc

        # 5. Structure raw decision scores
        decision_scores = pred_out.get("decision_scores") or {}

        # 6. Build response object
        model_name = self.model_config.get("model_name", "LinearSVC")
        model_id = self.model_config.get("model_candidate_id", "linear_svc_k4_standard")
        feat_cfg = self.model_config.get("feature_configuration", {})

        return {
            "predicted_class": pred_out["predicted_class"],
            "sequence_length": stats["length"],
            "gc_content": stats["gc_content"],
            "base_frequencies": stats["frequencies"],
            "kmer_count": stats["kmer_count"],
            "decision_scores": decision_scores,
            "model_metadata": {
                "model_name": "Linear Support Vector Classifier (LinearSVC)",
                "candidate_id": model_id,
                "kmer_size": feat_cfg.get("k", 4),
                "vectorizer_type": "TF-IDF (Standard Representation)",
                "n_features": feat_cfg.get("n_features", 256),
                "is_frozen": self.model_config.get("is_frozen", True),
                "frozen_at": self.model_config.get("frozen_at", "2026-11-04 17:28:21"),
                "verified_holdout_accuracy": 0.9125,
                "verified_holdout_macro_f1": 0.9105,
            },
            "header": header,
            "sequence_preview": (
                cleaned_seq[:160] + ("..." if len(cleaned_seq) > 160 else "")
            ),
            "scientific_disclosure": SCIENTIFIC_DISCLOSURE_TEXT,
        }

    @staticmethod
    def get_synthetic_demo_sequence() -> str:
        """Return the synthetic demonstration FASTA sequence."""
        return SYNTHETIC_DEMO_SEQUENCE

