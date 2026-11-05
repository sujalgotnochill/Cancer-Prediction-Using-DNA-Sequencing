"""Unit and integration test suite for the local Flask web application.

Verifies:
- Application factory and startup
- Route rendering (Home, About, History, Results, 404)
- Sequence validation (empty, invalid bases, oversized, min-length)
- FASTA handling (single record parsing, multi-record rejection, file uploads)
- Model prediction service integration (frozen Phase 5 LinearSVC candidate)
- Raw decision scores reporting
- Local SQLite history persistence and clearing
- Scientific disclosure visibility
- Graceful handling of missing or corrupted model artifacts
- Session CSRF protection
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add src and workspace root to path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from app import create_app
from app.services.history_service import HistoryService
from app.services.prediction_service import (
    SCIENTIFIC_DISCLOSURE_TEXT,
    SYNTHETIC_DEMO_SEQUENCE,
    PredictionService,
    PredictionServiceError,
)
from dna_cancer.utils.exceptions import DataValidationError


class TestWebApplication(unittest.TestCase):
    """Comprehensive test suite for Phase 7 web application."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp(prefix="dna_app_test_")
        cls.db_path = Path(cls.temp_dir) / "test_history.db"

        cls.app = create_app({
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "DATABASE_PATH": cls.db_path,
        })
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self) -> None:
        # Clear database records before each test
        if self.app.history_service:
            self.app.history_service.clear_history()

    # ----------------------------------------------------------------------
    # 1. Startup & Page Rendering
    # ----------------------------------------------------------------------
    def test_application_startup_and_context(self) -> None:
        """Verify application starts, binds services, and injects context globals."""
        self.assertIsNotNone(self.app.prediction_service)
        self.assertTrue(self.app.prediction_service.is_ready)
        self.assertIsNotNone(self.app.history_service)

    def test_home_page_rendering(self) -> None:
        """Verify home page loads with 200 OK and expected form elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("DNA Sequence Classification", html)
        self.assertIn("sequence_text", html)
        self.assertIn("sequence_file", html)
        self.assertIn("Classify Sequence", html)
        self.assertIn("Load Demo Sequence", html)

    def test_about_page_rendering_and_holdout_metrics(self) -> None:
        """Verify about page renders all verified Phase 5 holdout metrics accurately."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        # Check holdout results
        self.assertIn("91.25%", html)
        self.assertIn("0.9105", html)
        self.assertIn("219 / 240", html)
        self.assertIn("LinearSVC", html)
        self.assertIn("k = 4", html)
        self.assertIn("TF-IDF", html)
        # Check Cervical discrepancy explanation
        self.assertIn("Cervical recall", html)

    def test_scientific_disclaimer_visibility(self) -> None:
        """Verify scientific disclosure is prominently displayed on Home, About, and Results."""
        # 1. Home page
        res_home = self.client.get("/")
        self.assertIn("educational DNA sequence-source classification project", res_home.get_data(as_text=True))

        # 2. About page
        res_about = self.client.get("/about")
        self.assertIn("educational DNA sequence-source classification project", res_about.get_data(as_text=True))

        # 3. Results page
        res_pred = self.client.post("/predict", data={"sequence_text": SYNTHETIC_DEMO_SEQUENCE})
        self.assertEqual(res_pred.status_code, 200)
        self.assertIn("educational DNA sequence-source classification project", res_pred.get_data(as_text=True))

    # ----------------------------------------------------------------------
    # 2. Sequence Prediction & Validation
    # ----------------------------------------------------------------------
    def test_valid_sequence_submission_html(self) -> None:
        """Verify valid sequence submission renders results page with decision scores."""
        seq = "ATGC" * 30
        response = self.client.post("/predict", data={"sequence_text": seq})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Classification Results", html)
        self.assertIn("Raw Decision Hyperplane Scores", html)
        self.assertIn("Sequence Length", html)
        self.assertIn("GC Content", html)

    def test_valid_sequence_submission_json(self) -> None:
        """Verify JSON prediction request returns structured response with raw decision scores."""
        seq = "ATGC" * 30
        response = self.client.post(
            "/predict",
            data={"sequence_text": seq},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_class", data)
        self.assertEqual(data["sequence_length"], 120)
        self.assertAlmostEqual(data["gc_content"], 50.0, places=1)
        self.assertIn("decision_scores", data)
        # Verify 4 classes in decision scores
        self.assertEqual(len(data["decision_scores"]), 4)
        for cls_name in ["Cervical Cancer", "Colorectal Cancer", "Endometrial Cancer", "Liver Cancer"]:
            self.assertIn(cls_name, data["decision_scores"])

    def test_invalid_nucleotide_input(self) -> None:
        """Verify sequence with invalid bases (e.g. numbers, non-IUPAC letters) is rejected."""
        response = self.client.post(
            "/predict",
            data={"sequence_text": "ATGCATGCXYZ123"},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIn("unsupported characters", data["error"])

    def test_empty_input(self) -> None:
        """Verify empty sequence is rejected."""
        response = self.client.post(
            "/predict",
            data={"sequence_text": "   \n\t  "},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_oversized_input(self) -> None:
        """Verify sequence exceeding maximum length is rejected without truncation."""
        long_seq = "ATGC" * 15000  # 60,000 bp > 50,000 bp limit
        response = self.client.post(
            "/predict",
            data={"sequence_text": long_seq},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("exceeds the maximum allowed length", data["error"])

    def test_too_short_sequence(self) -> None:
        """Verify sequence shorter than k=4 is rejected with informative error."""
        response = self.client.post(
            "/predict",
            data={"sequence_text": "ATG"},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("shorter than the minimum 4 bp", data["error"])

    # ----------------------------------------------------------------------
    # 3. FASTA Formats & File Uploads
    # ----------------------------------------------------------------------
    def test_single_fasta_record(self) -> None:
        """Verify single FASTA record parses header and sequence correctly."""
        fasta = ">sample_gene_123 | Synthetic test\nATGCATGCATGCATGCATGC\nATGCATGCATGCATGCATGC\n"
        response = self.client.post(
            "/predict",
            data={"sequence_text": fasta},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["header"], "sample_gene_123 | Synthetic test")
        self.assertEqual(data["sequence_length"], 40)

    def test_multiple_fasta_records_rejection(self) -> None:
        """Verify multi-record FASTA submissions are rejected with a clear message."""
        multi_fasta = ">rec1\nATGCATGCATGC\n>rec2\nCGTACGTACGTA\n"
        response = self.client.post(
            "/predict",
            data={"sequence_text": multi_fasta},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Multiple FASTA records detected", data["error"])

    def test_fasta_file_upload(self) -> None:
        """Verify valid FASTA file upload produces successful prediction."""
        file_content = b">uploaded_construct\nATGCAATTGGCCATGCATGCATGCATGC\n"
        data = {
            "sequence_file": (io.BytesIO(file_content), "test_gene.fasta"),
        }
        response = self.client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        self.assertEqual(res_data["header"], "uploaded_construct")
        self.assertEqual(res_data["sequence_length"], 28)

    def test_unsupported_file_extension(self) -> None:
        """Verify uploaded file with invalid extension is rejected."""
        data = {
            "sequence_file": (io.BytesIO(b"ATGCATGC"), "test_virus.exe"),
        }
        response = self.client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        res_data = response.get_json()
        self.assertIn("Unsupported file format", res_data["error"])

    # ----------------------------------------------------------------------
    # 4. History Storage & Clearing
    # ----------------------------------------------------------------------
    def test_history_storage_and_retrieval(self) -> None:
        """Verify predictions are persisted in SQLite without saving complete sequences."""
        seq = "ATGC" * 25
        res = self.client.post("/predict", data={"sequence_text": seq})
        self.assertEqual(res.status_code, 200)

        history = self.app.history_service.get_history()
        self.assertEqual(len(history), 1)
        record = history[0]
        self.assertEqual(record["sequence_length"], 100)
        self.assertAlmostEqual(record["gc_content"], 50.0, places=1)
        self.assertIn("predicted_class", record)
        # Ensure sequence text is NOT stored in row keys
        self.assertNotIn("sequence", record)
        self.assertNotIn("raw_sequence", record)

    def test_history_clearing(self) -> None:
        """Verify clearing history removes all entries."""
        # Insert test records
        self.app.history_service.record_prediction(
            predicted_class="Colorectal Cancer",
            sequence_length=300,
            gc_content=48.5,
            model_identifier="test_model",
        )
        self.assertEqual(self.app.history_service.count_predictions(), 1)

        # Clear via route
        res = self.client.post("/history/clear", follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.app.history_service.count_predictions(), 0)

    # ----------------------------------------------------------------------
    # 5. Missing / Corrupted Artifacts Handling & Integrity Verification
    # ----------------------------------------------------------------------
    def test_missing_artifact_handling(self) -> None:
        """Verify PredictionService raises PredictionServiceError gracefully on missing artifacts."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(PredictionServiceError) as ctx:
                PredictionService(model_dir=Path(empty_dir))
            self.assertIn("Missing required Phase 5 model artifacts", str(ctx.exception))

    def test_artifact_checksum_mismatch_detection(self) -> None:
        """Verify PredictionService detects corrupted model artifacts and rejects loading."""
        src_dir = Path("artifacts/models/phase5_best_candidate")
        if not src_dir.exists():
            self.skipTest("Base artifacts directory not available for tampering test")

        with tempfile.TemporaryDirectory() as temp_model_dir:
            temp_path = Path(temp_model_dir)
            # Copy valid artifacts
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_path / item.name)

            # Corrupt model.joblib by appending a byte
            with open(temp_path / "model.joblib", "ab") as f:
                f.write(b"CORRUPTION_BYTES")

            with self.assertRaises(PredictionServiceError) as ctx:
                PredictionService(model_dir=temp_path)
            self.assertIn("Integrity check failed", str(ctx.exception))
            self.assertIn("Checksum mismatch", str(ctx.exception))

    def test_degraded_application_when_model_missing(self) -> None:
        """Verify web application runs gracefully in degraded mode when artifacts are absent."""
        with tempfile.TemporaryDirectory() as empty_dir:
            degraded_app = create_app({
                "TESTING": True,
                "WTF_CSRF_ENABLED": False,
                "MODEL_DIR": Path(empty_dir),
                "DATABASE_PATH": Path(self.temp_dir) / "degraded_history.db",
            })
            deg_client = degraded_app.test_client()

            # 1. Health endpoint reports degraded status
            res_health = deg_client.get("/api/health")
            self.assertEqual(res_health.status_code, 200)
            health_data = res_health.get_json()
            self.assertEqual(health_data["status"], "degraded")
            self.assertFalse(health_data["model_ready"])
            self.assertTrue(health_data["setup_required"])

            # 2. Home page shows setup warning banner
            res_home = deg_client.get("/")
            self.assertEqual(res_home.status_code, 200)
            home_html = res_home.get_data(as_text=True)
            self.assertIn("Model Artifacts Missing", home_html)
            self.assertIn("download_model_artifacts.py", home_html)

            # 3. Predict endpoint returns 503 on JSON request with setup instructions
            res_predict_json = deg_client.post(
                "/predict",
                data={"sequence_text": "ATGC" * 30},
                headers={"Accept": "application/json"},
            )
            self.assertEqual(res_predict_json.status_code, 503)
            self.assertIn("download_model_artifacts.py", res_predict_json.get_json()["error"])

    def test_predict_with_json_payload(self) -> None:
        """Verify predict endpoint handles native application/json request payloads."""
        seq = "ATGC" * 30
        response = self.client.post(
            "/predict",
            json={"sequence_text": seq},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_class", data)
        self.assertEqual(data["sequence_length"], 120)

    # ----------------------------------------------------------------------
    # 6. CSRF Protection Enforcement
    # ----------------------------------------------------------------------
    def test_csrf_protection_enforcement(self) -> None:
        """Verify that state-changing requests without CSRF token fail when CSRF is enabled."""
        csrf_app = create_app({
            "TESTING": True,
            "WTF_CSRF_ENABLED": True,
            "ENFORCE_CSRF": True,
            "DATABASE_PATH": Path(self.temp_dir) / "csrf_test.db",
        })
        csrf_client = csrf_app.test_client()

        # Attempt POST without token
        res = csrf_client.post("/predict", data={"sequence_text": "ATGC" * 10})
        self.assertEqual(res.status_code, 400)
        self.assertIn("CSRF token validation failed", res.get_data(as_text=True))

    # ----------------------------------------------------------------------
    # 7. Helper APIs & 404 Handling
    # ----------------------------------------------------------------------
    def test_api_sample_endpoint(self) -> None:
        """Verify GET /api/sample returns valid demonstration FASTA."""
        response = self.client.get("/api/sample")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("fasta", data)
        self.assertIn("synthetic_demo_sequence", data["fasta"])

    def test_api_health_endpoint(self) -> None:
        """Verify GET /api/health reports healthy model status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_ready"])

    def test_404_error_page(self) -> None:
        """Verify 404 page renders custom template."""
        response = self.client.get("/non_existent_page_12345")
        self.assertEqual(response.status_code, 404)
        self.assertIn("404", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()

