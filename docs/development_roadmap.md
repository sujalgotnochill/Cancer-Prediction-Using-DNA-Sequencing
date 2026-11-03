# Development Roadmap: DNA Cancer Classification

This roadmap outlines the planned evolution of the project from the initial college prototype into a modular, production-ready machine learning framework.

---

## Phase 1: Foundational Architecture & Modular Refactoring (Completed)
- [x] Inspect and catalog legacy codebase and Colab experiments.
- [x] Preserve original notebooks (`legacy/old_notebooks/`) and dataset links (`legacy/Data.txt`).
- [x] Establish modular Python package structure under `src/dna_cancer/`.
- [x] Implement centralized configuration loading (`configs/`, `settings.py`) with YAML support and safe standard-library fallback.
- [x] Implement defensive sequence validation (`data/validation.py`) and schema checking.
- [x] Implement sequence normalization and cleaning (`preprocessing/sequence_cleaner.py`).
- [x] Implement configurable $k$-mer extraction (`features/kmer_extractor.py`) and vectorizer builder (`features/feature_builder.py`).
- [x] Design abstract model interface (`models/base_model.py`), registry (`models/registry.py`), and factory (`models/model_factory.py`).
- [x] Create training orchestration (`training/trainer.py`, `training/pipeline.py`) and evaluation modules (`evaluation/metrics.py`, `evaluation/evaluator.py`).
- [x] Build inference prediction interface (`prediction/predictor.py`).
- [x] Establish initial unit test suite (`tests/`) using synthetic DNA sequences (all 34 tests passing).
- [x] Author core documentation and data dictionary.

---

## Phase 2: Data Ingestion, Verification & Bioinformatic Quality Control (Next)
- [ ] Ingest and verify raw genomic datasets from public archives (e.g., NCBI GenBank, TCGA) or legacy drive repository.
- [ ] Implement data versioning and checksum verification for raw FASTA/CSV files.
- [ ] Perform bioinformatic exploratory data analysis (EDA):
  - Nucleotide composition and GC content analysis.
  - Sequence length distribution and outlier detection.
  - Ambiguous base ($N$) frequency profiling across cancer classes.
- [ ] Add automated data validation checks into pipeline prior to preprocessing.

---

## Phase 3: Model Implementation & Benchmark Evaluation
- [ ] Implement baseline machine learning classifiers:
  - Multinomial Naive Bayes (classic k-mer bag-of-words baseline).
  - Logistic Regression (L2-regularized).
  - Random Forest / Gradient Boosting (XGBoost / LightGBM).
- [ ] Refactor and improve Deep Learning classifier:
  - Multi-layer Perceptron (MLP) with proper dropout and batch normalization.
  - 1D Convolutional Neural Network (1D-CNN) for motif and k-mer pattern discovery.
- [ ] Multi-metric model comparison:
  - Stratified 5-fold cross-validation.
  - Precision-Recall and ROC-AUC curves for multi-class evaluation.
  - Macro and weighted F1-score comparisons.
- [ ] Persist best-performing model artifacts and serialization verification.

---

## Phase 4: Application Layer & Serving
- [ ] Implement lightweight backend API (Flask or FastAPI) in `app/`.
- [ ] Build user-facing interface (`app/templates/`, `app/static/`):
  - FASTA sequence upload and text-area sequence input.
  - Real-time sequence validation feedback.
  - Multi-class cancer probability breakdown and bar charts.
- [ ] Containerize application with Docker for cross-platform reproducibility.

---

## Phase 5: Interpretability & Advanced Bioinformatics
- [ ] $k$-mer attribution and importance analysis:
  - Identify top informative $k$-mers per cancer class.
  - Map informative $k$-mers back to known genomic loci or oncogenes.
- [ ] Integrate optional biological sequence alignment validation (e.g., BLAST / BioPython).
- [ ] CI/CD automation with GitHub Actions (linting, type-checking, automated unit test runs).

