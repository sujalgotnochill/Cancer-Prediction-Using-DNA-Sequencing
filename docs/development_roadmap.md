# Development Roadmap: DNA Cancer Classification

This roadmap outlines the systematic evolution of the project from the initial second-year college prototype into a modular, production-grade machine learning and bioinformatics system.

---

## Phase 1: Foundational Architecture & Modular Restructuring (Completed)
- [x] Catalog legacy codebase, notebooks, and Google Drive references (`legacy/old_notebooks/`, `legacy/Data.txt`).
- [x] Establish modular Python package structure under `src/dna_cancer/`.
- [x] Create centralized configuration subsystem (`configs/`, `settings.py`) with YAML support and safe standard-library parser fallback.
- [x] Implement initial defensive sequence validation (`data/validation.py`) and schema verification.
- [x] Implement sequence normalization and cleaning (`preprocessing/sequence_cleaner.py`, `sequence_processor.py`).
- [x] Implement configurable $k$-mer extraction (`features/kmer_extractor.py`) and feature vectorization (`features/feature_builder.py`).
- [x] Establish abstract model interface (`models/base_model.py`), registry (`models/registry.py`), and factory (`models/model_factory.py`).
- [x] Create training orchestration (`training/trainer.py`, `training/pipeline.py`) and evaluation modules (`evaluation/metrics.py`, `evaluation/evaluator.py`).
- [x] Build inference prediction interface (`prediction/predictor.py`).
- [x] Author core documentation and architecture definitions.

---

## Phase 2: Foundation Review & Integration Testing (Completed)
- [x] Review every module for import errors, circular dependencies, and Windows path resolution.
- [x] Add synthetic-data integration tests verifying end-to-end processing pipelines.
- [x] Identify and fix configuration parsing bugs (simple YAML parser dictionary bug).
- [x] Fix silent nucleotide dropping in prediction module.
- [x] Fix vectorizer error order check in feature builder.
- [x] Verify complete test suite execution (76 tests passing).
- [x] Author comprehensive audit report (`docs/foundation_review.md`).

---

## Phase 3: Dataset Engineering & Management System (Completed)
- [x] Design unified dataset schema (`DNASample`, `SampleMetadata`) strictly separating biological metadata from labels.
- [x] Build multi-format ingestion module (`data/ingestion.py`) supporting CSV, FASTA, JSON, and JSONL.
- [x] Implement configurable column mapping (`ColumnMapping`) for heterogeneous dataset formats.
- [x] Implement explicit FASTA record label mapping (`FastaLabelMapper`) from dictionaries, CSV, and JSON files without inferring labels from headers.
- [x] Implement quality assurance and anomaly detection (`data/quality.py`):
  - Sequence duplicate detection and configurable handling policies (`DuplicatePolicy`: `REPORT_ONLY`, `KEEP_FIRST`, `REMOVE_ALL`, `ERROR`).
  - Conflicting label detection for identical sequences.
  - Reverse-complement duplicate detection for strandedness analysis.
  - Duplicate sample ID and missing/unexpected label detection.
  - Severe class imbalance warnings and metadata completeness tracking.
- [x] Build dataset profiler (`data/profiler.py`):
  - Summary sequence length statistics (min, max, mean, median, standard deviation, quartiles).
  - Nucleotide composition frequencies (A, C, G, T, N) and overall GC-content.
  - Dual report export (JSON and GitHub Flavored Markdown in `artifacts/datasets/`).
- [x] Implement persistent dataset registry (`data/registry.py`):
  - File integrity tracking with SHA-256 checksums.
  - Semantic dataset versioning and duplicate file detection.
  - Persistent JSON storage at `artifacts/datasets/registry.json`.
- [x] Implement dataset splitting module (`data/splitter.py`):
  - Automated strategy recommendation (`recommend_splitting_strategy`).
  - Group-based splitting (`split_grouped`) strictly preventing patient/group leakage across partitions.
  - Stratified splitting (`split_stratified`) preserving cancer class distributions.
  - Deterministic random splitting (`split_random`).
- [x] Create command-line dataset management utilities in `scripts/`:
  - `import_dataset.py`, `validate_dataset.py`, `profile_dataset.py`, `list_datasets.py`.
  - Full Windows console compatibility (ASCII-safe CLI indicators).
- [x] Comprehensive test coverage (104 passing tests).
- [x] Author dataset management documentation (`docs/dataset_management.md`, `docs/data_dictionary.md`, `docs/dataset_engineering_review.md`).

---

## Phase 4: Baseline Models & Benchmark Evaluation (Next)
- [ ] Implement baseline machine learning classifiers:
  - Multinomial Naive Bayes (classic $k$-mer bag-of-words baseline).
  - Logistic Regression (L2-regularized multinomial).
  - Random Forest & Gradient Boosted Decision Trees.
- [ ] Implement cross-validation workflows:
  - Stratified $K$-fold cross-validation.
  - Group $K$-fold cross-validation for patient-partitioned datasets.
- [ ] Benchmark baseline classifiers on verified datasets:
  - Macro and weighted F1-score comparisons.
  - Multi-class ROC-AUC and Precision-Recall curves.
  - Confusion matrix generation and error analysis.
- [ ] Model serialization and artifact persistence under `artifacts/models/`.

---

## Phase 5: Deep Learning & Advanced Bioinformatic Features
- [ ] Refactor and modernise deep learning architectures:
  - Multi-Layer Perceptron (MLP) with dropout and batch normalization.
  - 1D Convolutional Neural Network (1D-CNN) for sequence motif discovery.
- [ ] Feature interpretability:
  - Attribution analysis to identify top cancer-discriminative $k$-mers.
  - Genomic locus mapping of salient motifs.
- [ ] Evaluation on multi-cohort external test sets to test cross-study generalizability.

---

## Phase 6: Web Application & Serving Interface
- [ ] Implement lightweight web application in `app/` (FastAPI or Flask).
- [ ] Interactive interface for FASTA/raw sequence input and validation.
- [ ] Visual probability distribution plots for predicted cancer types.
- [ ] Exportable prediction reports for exploratory research.
