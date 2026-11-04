# Cancer Prediction Using DNA Sequencing

> [!CAUTION]
> **Medical & Diagnostic Disclaimer:** This repository is an experimental research and educational classification project. It is **NOT** a clinically validated diagnostic system, medical device, or medical decision-support tool. It must never be used for clinical diagnosis, patient management, or medical decision-making.

---

## 1. Project Overview

Deoxyribonucleic acid (DNA) serves as the fundamental repository of genetic information within living organisms. In recent years, advancements in high-throughput sequencing technologies have revolutionized our ability to read DNA sequences rapidly and cost-effectively.

This project investigates machine learning and bioinformatics approaches for multi-class cancer classification from DNA nucleotide sequences. By transforming biological sequences into overlapping $k$-mer representations and leveraging statistical text-vectorization techniques, the framework maps genomic sequence patterns to cancer type categories.

Originally developed as a second-year college exploration project using Google Colab notebooks, this repository has been re-architected into a clean, modular, extensible Python machine learning system adhering to modern software engineering standards.

---

## 2. Current Development Status

- **Phase:** Phase 1 – Architectural Foundation & Modularization (Completed).
- **Core Modules:** Fully implemented clean interfaces for configuration, data ingestion, sequence validation, cleaning, $k$-mer extraction, feature building, model registry, training orchestration, evaluation, and inference prediction.
- **Unit Testing:** 34 unit tests implemented with synthetic DNA sequences covering validation, cleaning, $k$-mer extraction, and module imports; 100% passing.
- **Model Status:** Model selection, tuning, and re-training are intentionally deferred to Phase 2/3 after finalizing data provenance and verification.

---

## 3. System Architecture & Data Flow

The project follows a layered architecture ensuring clear separation of concerns:

```text
Raw Sequences (FASTA / CSV)
       │
       ▼
Data Ingestion (load_csv, parse_fasta_records)
       │
       ▼
Validation & Normalization (SequenceValidator, SequenceCleaner)
       │
       ▼
Feature Extraction (KMerExtractor, FeatureBuilder)
       │
       ▼
Model Abstraction & Factory (BaseDNAClassifier, ModelFactory)
       │
       ▼
Training & Evaluation (Trainer, TrainingPipeline, Evaluator)
       │
       ▼
Inference & Prediction (DNAPredictor)
```

For complete architectural details, see [System Architecture](docs/architecture.md).

---

## 4. Project Directory Structure

```text
Cancer-Prediction-Using-DNA-Sequencing/
├── configs/                      # Declarative YAML configurations
│   ├── project_config.yaml       # Project metadata, paths, logging
│   ├── data_config.yaml          # Dataset schema, paths, split ratios
│   ├── feature_config.yaml       # K-mer parameters, vectorizer settings
│   └── model_config.yaml         # Model architecture & hyperparameters
├── data/                         # Data directory (managed by data pipeline)
│   ├── raw/                      # Immutable raw datasets (FASTA, CSV)
│   ├── interim/                  # Cleaned & standardized sequences
│   ├── processed/                # Tokenized & vectorized feature matrices
│   └── README.md                 # Data lifecycle documentation
├── artifacts/                    # Saved models and experiment outputs
│   ├── models/                   # Serialized model checkpoints
│   ├── vectorizers/              # Fitted k-mer vectorizer artifacts
│   ├── encoders/                 # Fitted label encoder artifacts
│   └── evaluation/               # Metrics JSON reports & evaluation curves
├── legacy/                       # Preserved college project experiments
│   ├── old_notebooks/            # Original Colab research notebooks
│   │   ├── data_preprocessing.ipynb
│   │   ├── DNA_model.ipynb
│   │   └── fasta2csv.ipynb
│   └── Data.txt                  # Original dataset Google Drive reference
├── notebooks/                    # Interactive Jupyter notebooks
│   └── exploration/              # Exploratory data analysis notebooks
├── src/                          # Production Python source package
│   └── dna_cancer/
│       ├── __init__.py
│       ├── config/               # Settings loading & dataclass schemas
│       ├── data/                 # Ingestion, validation, & dataset classes
│       ├── preprocessing/        # DNA sequence cleaning & normalization
│       ├── features/             # K-mer extraction & feature builder
│       ├── models/               # Abstract base model, registry, & factory
│       ├── training/             # Trainer & end-to-end training pipeline
│       ├── evaluation/           # Metrics computation & evaluator
│       ├── prediction/           # Inference predictor for new sequences
│       └── utils/                # Centralized logger & custom exceptions
├── app/                          # Web application serving layer (Phase 4)
│   ├── templates/                # HTML templates
│   └── static/                   # Static assets (CSS/JS)
├── scripts/                      # CLI entrypoint scripts (train, predict)
├── tests/                        # Comprehensive unit test suite
├── docs/                         # Detailed project documentation
│   ├── architecture.md           # Architecture design & data flow
│   ├── data_dictionary.md        # Dataset schema & unverified provenance notes
│   └── development_roadmap.md    # Multi-phase development roadmap
├── figs/                         # Preserved original experiment figures
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore configuration
├── pyproject.toml                # Project packaging configuration
├── requirements.txt              # Project dependencies
└── README.md                     # Root project documentation
```

---

## 5. Setup & Installation Instructions

### Prerequisites
- Python 3.8+ (tested on Python 3.8 to 3.12, Windows compatible)
- `pip` or virtual environment manager (`venv`)

### 1. Clone repository
```bash
git clone https://github.com/sujalgotnochill/Cancer-Prediction-Using-DNA-Sequencing.git
cd Cancer-Prediction-Using-DNA-Sequencing
```

### 2. Set up virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

To install optional deep learning (TensorFlow), bioinformatics (BioPython), or web serving dependencies:
```bash
# Editable package installation
pip install -e .

# Or with optional extras
pip install -e ".[deep-learning,bio,app,dev]"
```

### 4. Configure environment
```bash
cp .env.example .env
```

---

## Run the Web Application Locally

The project includes a complete local scientific dashboard built with Flask, Jinja2, HTML5, CSS3, and Vanilla JavaScript to inspect DNA sequences and perform inference against the frozen Phase 5 LinearSVC model (\(k=4\), TF-IDF).

### Quick Start (Windows PowerShell)

1. **Open PowerShell in the project directory:**
   ```powershell
   cd D:\Cancer-Prediction-Using-DNA-Sequencing
   ```

2. **Ensure dependencies are installed:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. **Start the local server:**
   ```powershell
   python run.py
   ```

4. **Access the web application in your browser:**
   Open:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Key Web Application Features

- **Interactive Scientific Dashboard:** Dark biotechnology aesthetic with live nucleotide character counter and composition metrics.
- **Flexible Sequence Ingestion:** Direct sequence textarea, FASTA file upload, and drag-and-drop zone.
- **Demonstration Sequence:** One-click synthetic non-clinical demo sequence loader.
- **Frozen Phase 5 Model:** Real-time inference using the frozen LinearSVC candidate (91.25% verified holdout accuracy, 0.9105 Macro F1).
- **Raw Decision Scores:** Reports true signed distances from decision hyperplanes rather than misrepresenting them as probabilities.
- **Local Prediction History:** Persistent SQLite history tracking metadata and sequence statistics (raw DNA sequences and patient information are strictly never saved).
- **Comprehensive Documentation:** Full architectural and usage details in [`docs/local_web_application.md`](docs/local_web_application.md).

> [!WARNING]
> **Mandatory Scientific Disclosure:** This application performs educational sequence-source classification on a historically labeled benchmark dataset. It does not perform clinical cancer diagnosis, patient risk prediction, or medical advice.

---

## 6. Running Tests

The test suite runs with zero external dependencies using Python's standard library `unittest` runner, or with `pytest` if installed:

```bash
# Using Python standard library unittest
python -m unittest discover -s tests -p "test_*.py" -v

# Or using pytest (if installed)
pytest tests/ -v
```

All 34 tests validate sequence cleaning, validation, k-mer extraction, and modular imports using small synthetic DNA sequences.

---

## 7. Preserved Legacy Work

The original experimental work developed during college has been safely preserved:
- Original notebooks are preserved under [`legacy/old_notebooks/`](legacy/old_notebooks/).
- Original figures are preserved in [`figs/`](figs/).
- Google Drive dataset link is preserved in [`legacy/Data.txt`](legacy/Data.txt).

---

## 8. Planned Features & Development Roadmap

See [`docs/development_roadmap.md`](docs/development_roadmap.md) for full phase-by-phase details:
- **Phase 2:** Data provenance verification, public genomic dataset ingestion, bioinformatic quality control.
- **Phase 3:** Machine learning benchmarks (Naive Bayes, Random Forest) and deep learning architecture refinement (MLP, 1D-CNN).
- **Phase 4:** Interactive web application for sequence upload and probability inspection.
- **Phase 5:** Biological motif interpretability and genomic alignment.

---

## 9. Contact & Attribution

- **Original Author:** Sujal Vishwakarma
  - [Instagram](https://www.instagram.com/sujalgotnochill/)
  - [Email](mailto:sujalvishwakarma5@gmail.com)
  - [Twitter](https://twitter.com/Sujalgotnochill)
