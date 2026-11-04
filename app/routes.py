"""Flask routing definitions for the DNA Sequence Classifier web application."""

import os
from typing import Any, Dict, Optional

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from app.services.prediction_service import (
    SCIENTIFIC_DISCLOSURE_TEXT,
    SYNTHETIC_DEMO_SEQUENCE,
)
from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {".txt", ".fasta", ".fa", ".fna"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


def allowed_file(filename: str) -> bool:
    """Validate uploaded filename against allowed extensions."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@bp.route("/", methods=["GET"])
def index():
    """Render main dashboard page with sequence input form."""
    prediction_service = current_app.prediction_service
    model_ready = prediction_service.is_ready if prediction_service else False
    model_config = prediction_service.model_config if prediction_service else {}

    return render_template(
        "index.html",
        model_ready=model_ready,
        model_config=model_config,
        demo_sequence=SYNTHETIC_DEMO_SEQUENCE,
        scientific_disclosure=SCIENTIFIC_DISCLOSURE_TEXT,
    )


@bp.route("/predict", methods=["POST"])
def predict():
    """Handle sequence classification submissions from textarea or file upload."""
    prediction_service = current_app.prediction_service
    history_service = current_app.history_service

    if not prediction_service or not prediction_service.is_ready:
        flash(
            "Prediction model is currently unavailable. Please verify model artifacts.",
            "error",
        )
        return redirect(url_for("main.index"))

    raw_input_text: str = ""
    source_filename: Optional[str] = None

    # 1. Check for file upload first
    uploaded_file = request.files.get("sequence_file")
    if uploaded_file and uploaded_file.filename:
        safe_name = secure_filename(uploaded_file.filename)
        if not allowed_file(safe_name):
            msg = (
                f"Unsupported file format '{safe_name}'. "
                f"Supported file extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
            if request.headers.get("Accept") == "application/json":
                return jsonify({"error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("main.index"))

        # Read file with size safeguard
        file_bytes = uploaded_file.read(MAX_FILE_BYTES + 1)
        if len(file_bytes) > MAX_FILE_BYTES:
            msg = f"Uploaded file exceeds maximum allowed size of {MAX_FILE_BYTES // (1024 * 1024)} MB."
            if request.headers.get("Accept") == "application/json":
                return jsonify({"error": msg}), 413
            flash(msg, "error")
            return redirect(url_for("main.index"))

        try:
            raw_input_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_input_text = file_bytes.decode("latin-1")
            except Exception as dec_err:
                msg = f"Failed to decode uploaded text file: {dec_err}"
                if request.headers.get("Accept") == "application/json":
                    return jsonify({"error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("main.index"))

        source_filename = safe_name

    # 2. Check textarea input if no file provided
    if not raw_input_text:
        raw_input_text = request.form.get("sequence_text", "")

    # 3. Validate presence of input
    if not raw_input_text or not raw_input_text.strip():
        msg = "No sequence provided. Please paste a DNA sequence or upload a FASTA file."
        if request.headers.get("Accept") == "application/json":
            return jsonify({"error": msg}), 400
        flash(msg, "error")
        return redirect(url_for("main.index"))

    # 4. Perform prediction through prediction service
    try:
        result = prediction_service.predict(raw_input_text)
    except DataValidationError as val_err:
        logger.warning("Validation error on user input: %s", val_err)
        if request.headers.get("Accept") == "application/json":
            return jsonify({"error": str(val_err)}), 400
        flash(str(val_err), "error")
        return redirect(url_for("main.index"))
    except Exception as exc:
        logger.error("Internal prediction error: %s", exc, exc_info=True)
        msg = f"Prediction processing error: {exc}"
        if request.headers.get("Accept") == "application/json":
            return jsonify({"error": msg}), 500
        flash(msg, "error")
        return redirect(url_for("main.index"))

    # 5. Record to local SQLite history (never stores the sequence itself)
    if history_service:
        try:
            label = result.get("header") or source_filename or "Direct Input"
            history_service.record_prediction(
                predicted_class=result["predicted_class"],
                sequence_length=result["sequence_length"],
                gc_content=result["gc_content"],
                model_identifier=result["model_metadata"]["candidate_id"],
                record_label=label,
            )
        except Exception as hist_err:
            logger.error("Failed to record prediction history: %s", hist_err)

    # 6. Return response
    if request.headers.get("Accept") == "application/json":
        return jsonify(result), 200

    return render_template(
        "results.html",
        result=result,
        source_filename=source_filename,
        scientific_disclosure=SCIENTIFIC_DISCLOSURE_TEXT,
    )


@bp.route("/history", methods=["GET"])
def history():
    """Render local prediction history page."""
    history_service = current_app.history_service
    records = history_service.get_history(limit=100) if history_service else []
    total_count = history_service.count_predictions() if history_service else 0

    return render_template(
        "history.html",
        records=records,
        total_count=total_count,
        scientific_disclosure=SCIENTIFIC_DISCLOSURE_TEXT,
    )


@bp.route("/history/clear", methods=["POST"])
def clear_history():
    """Clear local prediction history upon user confirmation."""
    history_service = current_app.history_service
    if history_service:
        deleted = history_service.clear_history()
        flash(f"Cleared {deleted} prediction records from local history.", "info")
    else:
        flash("History service is not configured.", "warning")

    return redirect(url_for("main.history"))


@bp.route("/about", methods=["GET"])
def about():
    """Render detailed project architecture and scientific disclosure page."""
    prediction_service = current_app.prediction_service
    model_config = prediction_service.model_config if prediction_service else {}

    return render_template(
        "about.html",
        model_config=model_config,
        scientific_disclosure=SCIENTIFIC_DISCLOSURE_TEXT,
    )


@bp.route("/api/sample", methods=["GET"])
def api_sample():
    """Return synthetic demonstration sequence as JSON."""
    return jsonify({
        "fasta": SYNTHETIC_DEMO_SEQUENCE,
        "description": "Synthetic benchmark sequence for educational demonstration",
        "length_approx": 295,
    })


@bp.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint for service status monitoring."""
    prediction_service = current_app.prediction_service
    model_ready = bool(prediction_service and prediction_service.is_ready)
    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_id": (
            prediction_service.model_config.get("model_candidate_id")
            if model_ready
            else None
        ),
    })

