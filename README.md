# DNA Sequence Classification — Machine Learning & Bioinformatics Framework

[![CI Tests](https://github.com/sujalgotnochill/Cancer-Prediction-Using-DNA-Sequencing/actions/workflows/tests.yml/badge.svg)](https://github.com/sujalgotnochill/Cancer-Prediction-Using-DNA-Sequencing/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/Code%20Style-Production%20Grade-success.svg)]()

> [!CAUTION]
> **Mandatory Medical & Scientific Disclaimer:**
> This repository is an educational computational genomics and machine learning project. It is **NOT** a clinically validated diagnostic tool, medical device, or clinical decision-support system. It must **never** be used for clinical diagnosis, patient screening, tumor detection, or medical treatment planning.
>
> Forensic bioinformatic inspection revealed that the historical public benchmark dataset consists of human chromosome 15 transcripts for Cervical samples and bacterial genomic sequences for Colorectal, Endometrial, and Liver samples. Consequently, the model operates strictly as an educational sequence-source classifier, not a clinical cancer detector.

---

## 1. Project Overview

This project provides an end-to-end, reproducible machine learning and bioinformatics framework for sequence-based classification of DNA nucleotide sequences. By decomposing raw nucleotide strings into overlapping $k$-mer oligonucleotide frequencies and applying statistical text vectorization techniques, the framework maps genomic sequence patterns to source classification categories.

Originally conceived as an exploratory college project using monolithic Google Colab notebooks, this repository was completely audited, forensically investigated, and re-architected into a modular, production-ready Python codebase. The engineering process emphasized rigorous software design, cryptographic artifact versioning, group-aware cross-validation, data minimization, and automated continuous integration.

---

## 2. Current Verified Development Status

The codebase represents the completion of all planned development phases (Phases 1 through 7):

| Phase | Milestone | Verified Status |
| :--- | :--- | :--- |
| **Phase 1** | Architectural Foundation & Modularization | **Complete** &bull; Modular packages for data, preprocessing, features, models, training, evaluation, tracking, and prediction. |
| **Phase 2** | Data Ingestion, Profiling & Forensic FASTA Audit | **Complete** &bull; Multi-format ingest, nucleotide validation, and forensic provenance inspection across 611 MB of source FASTA files. |
| **Phase 3** | Baseline Benchmarking & Leakage Forensics | **Complete** &bull; Benchmarked MultinomialNB, LogisticRegression, and RandomForest across $k \in \{3,4,5,6\}$. Quantified stratified split duplicate leakage. |
| **Phase 4** | Gene Family Auditing & Group-Aware Validation | **Complete** &bull; Mapped chromosome 15 transcript families; designed and executed 3-Fold Gene-Family-Grouped Cross-Validation. |
| **Phase 5** | Model Optimization & Candidate Selection | **Complete** &bull; Evaluated linear, tree-based, and boosting models. Selected and froze the canonical `LinearSVC` candidate ($k=4$, TF-IDF, 256 features). |
| **Phase 6** | Final Holdout Evaluation & Audit Receipts | **Complete** &bull; Evaluated frozen candidate against the 240-sample holdout cohort. Cryptographic receipts recorded at `artifacts/vault/final_evaluation_receipt.json`. |
| **Phase 7** | Production Flask Web Application & Serving Layer | **Complete** &bull; Full-stack dashboard, real-time nucleotide validation, parameterized SQLite history, CSRF protection, and health API. |

- **Test Suite Status:** **170 automated tests** passing with 100% pass rate across unit, boundary, security, and integration suites.
- **Model Checkpoint Status:** Canonical Phase 5 candidate frozen with SHA-256 integrity verification.

---

## 3. Key Technical Features

- **Multi-Format Genomic Data Ingestion:** Reads FASTA, CSV, JSON, and JSONL formats with configurable column mapping and metadata preservation.
- **Nucleotide Validation & Standardized Cleaning:** Enforces strict IUPAC nucleotide alphabets (`A`, `C`, `G`, `T`, `N`), casing normalization, whitespace stripping, and length constraints without silent truncation.
- **Vectorized $k$-mer Feature Extraction:** Sliding-window tokenization supporting arbitrary $k$-mer sizes (standard $k=4$, yielding a 256-token vocabulary) with Count and TF-IDF representations.
- **Forensic Leakage Detection & Group-Aware Splitting:** Detects identical duplicate sequences, reverse complements, and conflicting labels. Enforces gene-family group boundaries to prevent transcript isoform leakage between training and validation folds.
- **Cryptographic Model Artifact Verification:** Verifies SHA-256 hashes of serialized model weights, vectorizer tokens, encoder states, and configurations before loading into memory.
- **Interactive Local Web Dashboard:** High-contrast scientific UI with real-time nucleotide counters, composition statistics (GC content, base frequencies), FASTA file drag-and-drop, and a synthetic non-clinical demonstration sequence loader.
- **Raw Decision Hyperplane Reporting:** Reports signed orthogonal decision distances ($w_k^T x + b_k$) rather than misrepresenting uncalibrated distances as clinical probabilities.
- **Data Minimization & Local Persistence:** SQLite database records operational inference metadata while strictly excluding complete nucleotide sequences and patient identifiers.
- **Enterprise Security Controls:** Session-based CSRF protection, 5 MB file upload ceiling, 6 MB HTTP payload cap, 50,000 bp sequence limit, and localhost binding (`127.0.0.1`).

---

## 4. Architecture and Directory Structure

The system is organized into a clean layered architecture with decoupled components:

```text
Cancer-Prediction-Using-DNA-Sequencing/
├── .github/
│   └── workflows/
│       └── tests.yml                 # Multi-OS (Ubuntu, Windows) CI matrix workflow
├── app/                              # Production Flask web application layer
│   ├── routes.py                     # HTTP routing, file handling, and API endpoints
│   ├── services/
│   │   ├── history_service.py        # Parameterized SQLite metadata persistence
│   │   └── prediction_service.py     # Frozen model inference, validation, stats
│   ├── static/
│   │   ├── css/style.css             # Biotech stylesheet
│   │   └── js/app.js                 # Vanilla JS client-side validation & UI
│   └── templates/                    # Jinja2 server-rendered templates
│       ├── base.html                 # Master layout with universal disclaimer banner
│       ├── index.html                # Main sequence input dashboard
│       ├── results.html              # Prediction results & raw decision scores
│       ├── history.html              # Local inference history view
│       └── 404.html                  # Error view
├── artifacts/
│   ├── datasets/                     # Forensic inspection & reconstruction metadata
│   ├── evaluation/                   # Benchmarks, ROC curves, confusion matrices
│   ├── history/                      # Prediction history SQLite database
│   ├── models/
│   │   └── phase5_best_candidate/    # Frozen canonical Phase 5 model (~20 KB total)
│   │       ├── model.joblib          # Frozen LinearSVC weights
│   │       ├── vectorizer.joblib     # TF-IDF (k=4, 256 vocabulary tokens)
│   │       ├── encoder.joblib        # 4-class label encoder
│   │       ├── model_config.json     # Hyperparameters and CV benchmarks
│   │       └── checksums.sha256      # SHA-256 cryptographic hashes
│   └── vault/                        # Evaluation receipts and manifest
│       ├── final_evaluation_receipt.json  # Phase 6 official audit receipt
│       ├── evaluation_events.jsonl        # Cryptographic unseal event log
│       ├── test_sealed_manifest.json      # Dataset manifest and checksums
│       └── vault_state.json               # Current vault lifecycle state
├── configs/                          # Declarative YAML project configurations
├── data/                             # Data directory structure (raw, interim, processed)
├── docs/                             # Engineering specifications and forensic audit logs
│   ├── architecture.md               # System architectural specification
│   ├── baseline_results.md           # Baseline model benchmarks
│   ├── local_web_application.md      # Web application user and engineering guide
│   └── phase5_model_optimization.md  # Optimization and candidate selection
├── legacy/                           # Preserved college project notebooks and data links
├── scripts/                          # Automation scripts
│   ├── download_model_artifacts.py   # Model artifact verification and setup
│   ├── execute_final_holdout_evaluation.py # Phase 6 evaluation runner
│   └── verify_phase5_candidate.py    # Candidate artifact verification
├── src/
│   └── dna_cancer/                   # Core Python library
│       ├── config/                   # Settings management
│       ├── data/                     # Ingestion, schema, splitter, vault
│       ├── evaluation/               # Metrics and evaluation orchestration
│       ├── features/                 # k-mer extraction and feature builder
│       ├── models/                   # Base models, model factory, registry
│       ├── prediction/               # DNAPredictor inference engine
│       ├── preprocessing/            # Sequence cleaner and normalizer
│       ├── tracking/                 # Experiment metrics tracker
│       ├── training/                 # Pipeline orchestration and trainers
│       └── utils/                    # Centralized logging and custom exceptions
├── tests/                            # Comprehensive automated test suite (170 tests)
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore rules
├── pyproject.toml                    # Package specifications & aligned dependencies
├── requirements.txt                  # Pinned core dependencies for reproducibility
├── run.py                            # Web application local entrypoint
└── README.md                         # Project documentation
```

---

## 5. Technology Stack

- **Runtime:** Python 3.9, 3.10, 3.11, 3.12 (CPython / Conda / venv)
- **Machine Learning & Numerics:** `scikit-learn==1.3.2`, `numpy==1.26.4`, `pandas==2.2.3`, `scipy==1.17.1`, `joblib>=1.2.0`
- **Web Serving & Templating:** `Flask>=2.2.0,<4.0.0`, `Werkzeug>=2.2.0,<4.0.0`, Jinja2, Vanilla JavaScript, HTML5, CSS3
- **Configuration & Utilities:** `PyYAML>=5.4.0`, `python-dotenv>=0.19.0`
- **Testing & Continuous Integration:** `pytest>=7.0.0`, `pytest-cov>=3.0.0`, GitHub Actions (`ubuntu-latest`, `windows-latest`)

---

## 6. Installation Instructions for Windows PowerShell

### Prerequisites

- Windows 10/11 with Windows PowerShell
- Python 3.9+ (Python 3.12 recommended)
- Git for Windows

```powershell
# 1. Clone the repository
git clone https://github.com/sujalgotnochill/Cancer-Prediction-Using-DNA-Sequencing.git
cd Cancer-Prediction-Using-DNA-Sequencing
```

---

## 7. Virtual Environment Setup

```powershell
# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional: If script execution is restricted on Windows, enable current user execution)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

*For Linux / macOS bash users:*
```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 8. Dependency Installation

Install dependencies using the aligned requirements file and install the project in editable mode:

```powershell
# 4. Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install pinned core dependencies
python -m pip install -r requirements.txt

# 6. Install the package in editable mode with development extras
python -m pip install -e ".[dev,app]"
```

---

## 9. Model Artifact Setup & Verification

The frozen Phase 5 model candidate (~20 KB total) is managed in `artifacts/models/phase5_best_candidate/`. To verify that all model artifacts are present and cryptographically intact:

```powershell
# 7. Verify model artifacts against SHA-256 checksums
python scripts/download_model_artifacts.py
```

Expected output:
```text
All Phase 5 model artifacts in 'artifacts\models\phase5_best_candidate' are verified and intact:
  [OK] model.joblib         (  9061 bytes) SHA-256: 1422d3163b1d33bb...
  [OK] vectorizer.joblib    (  7471 bytes) SHA-256: a034de878fac4ba7...
  [OK] encoder.joblib       (   615 bytes) SHA-256: 3c78f7eee9e9be78...
  [OK] model_config.json    (  2710 bytes) SHA-256: 35fe8ec228049931...
```

If setting up the repository from an archive without model artifacts, the utility can download them from a release URL or custom endpoint:
```powershell
python scripts/download_model_artifacts.py --download-url <URL_TO_RELEASE_ZIP>
```

---

## 10. Running the Flask Application

Start the local server using the standard entry point:

```powershell
# 8. Start local Flask web application
python run.py
```

Console output:
```text
=================================================================
  DNA Sequence Classification — Local Web Application
=================================================================
  Local URL:        http://127.0.0.1:5000
  Binding:          Localhost only (isolated from external network)
  Debug Mode:       False
  Model Candidate:  LinearSVC (k=4, TF-IDF standard representation)
  Dataset Notice:   Educational sequence-source classification only
=================================================================
```

Open your browser to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

*Optional Environment Variables:*
- `FLASK_HOST` (default: `127.0.0.1`)
- `FLASK_PORT` (default: `5000`)
- `FLASK_DEBUG` (default: `0`)
- `MODEL_DIR` (default: `artifacts/models/phase5_best_candidate`)
- `MAX_FILE_SIZE_BYTES` (default: `5242880` bytes, 5 MB)

---

## 11. Running Tests

The test suite runs with zero network access and verifies components across all engineering phases:

```powershell
# Run the complete test suite
pytest

# Run tests with verbose output and summary
pytest -v

# Run web application tests specifically
pytest tests/test_web_application.py -v

# Run Phase 5 frozen candidate verification tests
pytest tests/test_phase5_verification.py -v
```

**Verified Test Summary:** 170 passed, 0 failed, 0 errors.

---

## 12. API Endpoints

The web application exposes structured HTTP endpoints for interactive usage and automated health monitoring:

### 1. `POST /predict`
Classifies a sequence submitted via web form, multipart file upload, or raw JSON body.

- **Request Headers:** `Content-Type: application/json`, `Accept: application/json`
- **Request Body (JSON):**
  ```json
  {
    "sequence_text": "ATGGCGTACGTTAGCTAGCTAGCTGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTA..."
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "predicted_class": "Colorectal Cancer",
    "sequence_length": 120,
    "gc_content": 50.0,
    "kmer_count": 117,
    "base_frequencies": {
      "A": { "count": 30, "percentage": 25.0 },
      "C": { "count": 30, "percentage": 25.0 },
      "G": { "count": 30, "percentage": 25.0 },
      "T": { "count": 30, "percentage": 25.0 },
      "N": { "count": 0, "percentage": 0.0 }
    },
    "decision_scores": {
      "Cervical Cancer": -0.8523,
      "Colorectal Cancer": 1.4215,
      "Endometrial Cancer": -0.6120,
      "Liver Cancer": -0.7412
    },
    "model_metadata": {
      "model_name": "Linear Support Vector Classifier (LinearSVC)",
      "candidate_id": "canonical_comp_linear_svc_k4_standard",
      "kmer_size": 4,
      "n_features": 256,
      "is_frozen": true,
      "verified_holdout_accuracy": 0.9125,
      "verified_holdout_macro_f1": 0.9105
    },
    "scientific_disclosure": "This is an educational DNA sequence-source classification project..."
  }
  ```
- **Degraded Response (503 Service Unavailable):** Returned if model artifacts are uninstalled or corrupted.

### 2. `GET /api/health`
Health check endpoint reporting model availability and candidate ID.
```json
{
  "status": "healthy",
  "model_ready": true,
  "model_id": "canonical_comp_linear_svc_k4_standard",
  "setup_required": false
}
```

### 3. `GET /api/sample`
Returns the synthetic non-clinical demonstration FASTA sequence for UI testing.

### 4. Page Routes:
- `GET /`: Interactive classification dashboard.
- `GET /about`: Architecture details, provenance disclosures, and holdout benchmarks.
- `GET /history`: Local SQLite prediction history.
- `POST /history/clear`: Clears local SQLite prediction records.

---

## 13. Model Architecture & Feature Extraction

### Feature Extraction Pipeline
1. **Validation & Cleaning:** Sequences are validated against the standard DNA base alphabet (`A`, `C`, `G`, `T`, `N`). Whitespace and carriage returns are stripped; characters are uppercased.
2. **$k$-mer Decomposition:** Sliding window with $k=4$ and stride $s=1$. For a sequence of length $L$, exactly $L - k + 1$ overlapping 4-mers are generated.
3. **TF-IDF Vectorization:** The 256 possible 4-mers ($4^4 = 256$) form a fixed vocabulary. Term Frequency-Inverse Document Frequency (TF-IDF) weighting normalizes differences in sequence lengths.

### Classifier Architecture
- **Model:** `LinearSVC` (Linear Support Vector Classifier).
- **Hyperparameters:** $C = 1.0$, `loss='squared_hinge'`, `max_iter=2000`, `dual='auto'`.
- **Decision Formulation:** Multi-class classification via One-vs-Rest (OvR) formulating 4 hyperplanes:
  $$f_k(x) = w_k^T x + b_k$$
  The predicted class corresponds to $\arg\max_{k} f_k(x)$.
- **Raw Decision Scores:** Output scores represent signed perpendicular distances to each decision hyperplane. Negative values indicate position on the opposing side of the decision boundary; positive values indicate position on the class-aligned side.

---

## 14. Evaluation Methodology & Verified Results

### Development Cohort Cross-Validation ($N=960$)
To prevent transcript isoform leakage between split folds, a 3-Fold Gene-Family-Grouped Cross-Validation protocol was implemented across the 23 development gene families:

- **Mean Macro F1:** $0.9671 \pm 0.0063$
- **Mean Accuracy:** $0.9674 \pm 0.0073$
- **Inference Time:** $< 1.0$ ms per sequence

### Historical Holdout Evaluation Benchmark ($N=240$)
The frozen candidate was evaluated against the 240-sample holdout cohort. Official evaluation receipt (`artifacts/vault/final_evaluation_receipt.json`):

- **Overall Accuracy:** 91.25% (219 / 240 correct, 95% CI: [87.0%, 94.2%])
- **Macro F1 Score:** 0.9105
- **Per-Class Recall:**
  - Cervical: 76.67% (46 / 60 correct, 14 misclassifications)
  - Colorectal: 90.00% (54 / 60 correct, 6 misclassifications)
  - Endometrial: 98.33% (59 / 60 correct, 1 misclassification)
  - Liver: 100.00% (60 / 60 correct, 0 misclassifications)

*Forensic Observation:* While Colorectal, Endometrial, and Liver achieved 90%–100% recall, Cervical dropped to 76.67% due to novel gene families in the holdout (`ATP10A`, `TUBGCP5`). This discrepancy demonstrated that $k$-mer models partially memorize gene-family-specific sequence motifs rather than general cancer signatures.

---

## 15. Scientific Limitations & Mandatory Medical Disclaimer

1. **Biological Provenance Limitation:** Detailed header analysis of the source FASTA files revealed that the classes represent human chromosome 15 transcripts for Cervical and bacterial genomic sequences for Colorectal, Endometrial, and Liver.
2. **Not a Clinical Diagnostic Tool:** The classifier distinguishes sequence origin within this specific historical dataset; it **does not** detect clinical cancer, somatic oncogenic mutations, or biopsy malignancy.
3. **Decision Scores vs. Probabilities:** Decision scores reflect hyperplane distances, not calibrated biological probabilities.

---

## 16. Security & Data Privacy Considerations

- **Data Minimization:** The SQLite database stores only operational metadata (timestamp, length, GC%, predicted class). Complete DNA sequences and patient information are strictly never saved.
- **CSRF Defense:** Session tokens are generated on session initialization and validated on all state-changing POST requests.
- **Payload Restrictions:** Strict 5 MB file size limit and 6 MB HTTP envelope prevent memory exhaustion attacks. Input sequences are limited to 50,000 bp.
- **Host Isolation:** Server binds to `127.0.0.1` by default; debug mode is disabled.
- **Holdout Evaluation Integrity Disclosure:** The historical 240-sample holdout data was unsealed during Phase 6 and subsequently committed to version control in commit `864e758`. Consequently, it is no longer sealed or blind. Any future candidate models require a newly constructed holdout to maintain scientific validity.

---

## 17. Future Improvements

- Ingestion of clinically verified, somatic human genomic datasets (such as TCGA/GDC somatic mutation or RNA-Seq profiles).
- Establishment of an independent, newly accessioned sealed holdout vault for future candidate evaluation.
- Exploration of genomic foundation models and 1D-CNN architectures on true somatic variant benchmarks.

---

## 18. Author & GitHub Information

- **Author:** Sujal Vishwakarma
  - GitHub: [@sujalgotnochill](https://github.com/sujalgotnochill)
  - Repository: [Cancer-Prediction-Using-DNA-Sequencing](https://github.com/sujalgotnochill/Cancer-Prediction-Using-DNA-Sequencing)
  - Email: sujalvishwakarma5@gmail.com
  - Instagram: [@sujalgotnochill](https://www.instagram.com/sujalgotnochill/)
  - Twitter/X: [@Sujalgotnochill](https://twitter.com/Sujalgotnochill)
