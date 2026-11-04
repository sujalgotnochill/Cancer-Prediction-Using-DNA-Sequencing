"""Flask application factory for the DNA Sequence Classifier web application."""

import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, abort, flash, render_template, request, session

from app.services.history_service import HistoryService
from app.services.prediction_service import (
    PredictionService,
    PredictionServiceError,
)
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Base configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32)),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max payload
        MODEL_DIR=Path(os.environ.get("MODEL_DIR", "artifacts/models/phase5_best_candidate")),
        DATABASE_PATH=Path(os.environ.get("DATABASE_PATH", "artifacts/history/prediction_history.db")),
        WTF_CSRF_ENABLED=True,
    )

    if test_config:
        app.config.update(test_config)

    # Initialize Services
    prediction_service: Optional[PredictionService] = None
    try:
        prediction_service = PredictionService(
            model_dir=app.config["MODEL_DIR"],
        )
        logger.info("PredictionService initialized successfully in application factory")
    except (PredictionServiceError, Exception) as exc:
        logger.warning(
            "PredictionService initialization deferred or failed: %s. "
            "App will run with degraded prediction capability.",
            exc,
        )

    history_service: Optional[HistoryService] = None
    try:
        history_service = HistoryService(db_path=app.config["DATABASE_PATH"])
        logger.info("HistoryService initialized successfully in application factory")
    except Exception as exc:
        logger.error("Failed to initialize HistoryService: %s", exc)

    app.prediction_service = prediction_service  # type: ignore[attr-defined]
    app.history_service = history_service  # type: ignore[attr-defined]

    # CSRF Protection & Session initialization
    @app.before_request
    def handle_csrf():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_hex(32)

        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            # Allow tests to disable CSRF or explicitly enforce it
            if not app.config.get("WTF_CSRF_ENABLED", True):
                return None
            if app.config.get("TESTING") and not app.config.get("ENFORCE_CSRF", False):
                return None

            submitted_token = (
                request.form.get("csrf_token")
                or request.headers.get("X-CSRF-Token")
            )
            session_token = session.get("csrf_token")

            if not submitted_token or submitted_token != session_token:
                logger.warning(
                    "CSRF token validation failed for %s request to %s",
                    request.method,
                    request.path,
                )
                if request.headers.get("Accept") == "application/json":
                    return {"error": "Invalid or missing CSRF token"}, 400
                abort(400, description="CSRF token validation failed. Please refresh the page and try again.")

    # Context Processors
    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": session.get("csrf_token", ""),
            "app_name": "DNA Sequence Classifier",
            "model_ready": bool(app.prediction_service and app.prediction_service.is_ready),
        }

    # Register Blueprints
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html", error_message=str(e)), 404

    @app.errorhandler(400)
    def bad_request(e):
        desc = getattr(e, "description", str(e))
        return render_template("404.html", error_title="400 — Bad Request", error_message=desc), 400

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return render_template(
            "404.html",
            error_title="413 — Payload Too Large",
            error_message="The uploaded file or sequence payload exceeds the maximum permitted server size.",
        ), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template(
            "404.html",
            error_title="500 — Server Error",
            error_message="An internal server error occurred while processing your request. Please try again later.",
        ), 500

    return app

