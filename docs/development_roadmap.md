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

## Phase 3.5: Real Dataset Acceptance Test (Completed)
- [x] Locate original project dataset references in `legacy/Data.txt` and legacy notebooks.
- [x] Identify raw FASTA collections in `data/raw/FASTA/` (Cervical, Colorectal, Endometrial, Gastric, Liver).
- [x] Perform acceptance audit documented in `docs/real_dataset_acceptance_review.md`.

---

## Phase 3.6 & 3.6.1: FASTA Inspection & Discrepancy Audit (Completed)
- [x] Stream-inspect multi-gigabyte FASTA collections without memory overflow.
- [x] Audit Gastric cancer 4.8 GB vs 491 MB discrepancy (resolved: Windows cluster allocation vs raw byte size).
- [x] Measure exact record counts, sequence length profiles, and nucleotide alphabet distributions.
- [x] Document findings in `docs/fasta_dataset_inspection.md` and `docs/fasta_inspection_audit.md`.

---

## Phase 3.7: Historical Dataset Reconstruction (Completed)
- [x] Analyze legacy conversion notebooks (`fasta2csv.ipynb`, `data_preprocessing.ipynb`).
- [x] Reverse-engineer exact historical sampling mechanism (first 300 records per FASTA file).
- [x] Identify biological origin of sequences:
  - Cervical: *Homo sapiens* Chromosome 15 mRNA transcripts.
  - Colorectal: *Ktedonobacter racemifer* bacterial genomic DNA.
  - Endometrial: *Saliniradius amylolyticus* bacterial genomic DNA.
  - Liver: *Adhaeribacter pallidiroseus* bacterial genomic DNA.
- [x] Generate reproducible reconstructed dataset with verified SHA-256 checksum (`data/interim/historical_reconstruction/DNAdata_reconstructed.csv`).
- [x] Document limitations in `docs/historical_dataset_reconstruction.md`.

---

## Phase 4: Baseline Models & Benchmark Evaluation (Completed)
- [x] Implement baseline machine learning classifiers in `src/dna_cancer/models/baseline_models.py`:
  - Multinomial Naive Bayes (Laplace smoothing).
  - Logistic Regression (L-BFGS, L2 regularized).
  - Random Forest Classifier (100 estimators).
- [x] Build experiment runner (`src/dna_cancer/training/benchmark_runner.py`) and tracker (`src/dna_cancer/tracking/tracker.py`).
- [x] Execute 24 baseline experiments covering $k \in \{3, 4, 5, 6\}$ across Stratified and Duplicate-Grouped splits.
- [x] Generate publication-quality visual artifacts in `artifacts/evaluation/`:
  - Ablation curves, confusion matrices, per-class F1 comparisons.
- [x] Document methodology and findings in `docs/baseline_experiment_methodology.md` and `docs/baseline_results.md`.

---

## Phase 4.1: Baseline Benchmark Integrity Audit (Completed)
- [x] Verify exact dataset provenance and SHA-256 checksum.
- [x] Audit sequence duplicate quarantine in Duplicate-Grouped splits (0 identical sequences overlap).
- [x] Uncover splice-isoform gene family leakage in duplicate splits (8 human gene families overlap).
- [x] Document findings in `docs/benchmark_integrity_audit.md`.

---

## Phase 4.2: Gene-Family Generalization & Evaluation Protocol Audit (Completed)
- [x] Forensic audit of test-set isolation in `benchmark_runner.py`:
  - Confirmed 24 test evaluations inside inner experiment loop.
  - Formally documented evaluation contamination and test-set peeking.
- [x] Implement gene-family-grouped splitting algorithm (`split_gene_family_grouped`) and leakage auditor (`verify_gene_family_leakage`) in `src/dna_cancer/data/splitter.py`.
- [x] Conduct 4-protocol comparison (Stratified vs Duplicate-Grouped vs Gene-Family Full vs Gene-Family Strict $N=448$).
- [x] Quantify isoform memorization: holding out gene families dropped Cervical recall from 100% to 77.97%–85.71%.
- [x] Author comprehensive reports:
  - `docs/gene_family_generalization_audit.md`
  - `docs/evaluation_protocol_audit.md`
  - `artifacts/evaluation/gene_family_generalization.json`
  - `artifacts/evaluation/evaluation_protocol_audit.json`
- [x] Establish Two-Tier Sealed Evaluation Protocol for Phase 5.
- [x] 124 passing unit tests.

---

## Phase 4.3: Final Evaluation Protocol Design & Cross-Validation Feasibility (Completed)
- [x] Audit `split_gene_family_grouped` and enhance with `grouping_mode`:
  - `within_organism` (default): Prevents cross-organism conflation of identical gene symbols.
  - `cross_organism`: Explicit global symbol grouping with orthology disclaimer.
  - `exact_sequence`: Strict nucleotide sequence deduplication.
  - Transparent fallback reporting: sequence hashes are never misrepresented as biological genes.
- [x] Forensic audit of five shared bacterial gene symbols (`coxA`, `coxB`, `coxC`, `cyoE`, `priA`):
  - Confirmed sequence identities at chance background levels (26.1%–32.1%) with distinct UniProt IDs.
  - Verified they are functional enzyme designations across distinct phyla, not identical biological genes.
- [x] Prove mathematical and statistical infeasibility of 5-Fold Grouped CV:
  - `SNRPN` alone accounts for 112 / 300 Cervical samples (37.33%).
  - 3 of 5 folds (60%) would contain transcripts from only a single gene family, inducing extreme bimodal evaluation variance.
- [x] Specify binding Two-Tier Sealed Evaluation Protocol for Phase 5:
  - Tier 1: Development Cohort ($N=960$, 80%) evaluated via Group-Balanced 3-Fold Cross-Validation.
  - Tier 2: Sealed Test Vault ($N=240$, 20%): exactly 60 samples per class across 8 unseen human genes, zero leakage, locked out until single final model evaluation.
- [x] Establish rigorous metric battery with Wilson Score and Bootstrap confidence intervals.
- [x] Author deliverables:
  - `docs/grouped_cross_validation_feasibility.md`
  - `docs/final_evaluation_protocol.md`
  - `artifacts/evaluation/cv_feasibility.json`
  - `artifacts/evaluation/final_protocol_specification.json`
- [x] 125 passing unit tests.

---

## Phase 5: Sealed Cross-Validation & Model Optimization (Next)
- [ ] Implement Two-Tier Evaluation Architecture:
  - Generate Tier 1 Development Cohort ($N=960$) and Tier 2 Sealed Test Vault ($N=240$).
  - Lock out Tier 2 Test Vault from all feature extraction, training, and tuning pipelines.
- [ ] Implement Group-Balanced 3-Fold Cross-Validation on Development Set:
  - Prevent splice-isoform memorization across folds.
  - Secondary verification via 10-Iteration Group Stratified ShuffleSplit.
- [ ] Feature engineering and representation exploration:
  - TF-IDF $k$-mer representations.
  - Subsequence motif embeddings.
- [ ] Model architecture optimization:
  - Regularization tuning on linear baselines ($C$ ablation).
  - Multi-Layer Perceptrons (MLP) with dropout and batch normalization.
  - 1D Convolutional Neural Networks (1D-CNN) for sequence motif detection.
- [ ] Final One-Shot Evaluation on Sealed Test Vault:
  - Unseal test set strictly once for the single winning architecture.
  - Report unbiased generalization performance with 95% confidence intervals and confusion matrix.

---

## Phase 6: Web Application & Serving Interface
- [ ] Implement lightweight web application in `app/` (FastAPI or Flask).
- [ ] Interactive interface for FASTA/raw sequence input and validation.
- [ ] Visual probability distribution plots for predicted cancer types.
- [ ] Exportable prediction reports for exploratory research.
