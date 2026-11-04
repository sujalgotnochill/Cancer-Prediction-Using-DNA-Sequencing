"""End-to-end training pipeline workflow orchestration."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from dna_cancer.config.settings import AppConfig, load_config
from dna_cancer.data.dataset import DNADataset
from dna_cancer.data.ingestion import load_csv
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.preprocessing.sequence_processor import SequenceProcessor
from dna_cancer.utils.exceptions import TrainingError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the complete lifecycle: Ingestion -> Preprocessing -> Features -> Training -> Artifact Export."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.processor = SequenceProcessor()
        self.feature_builder = FeatureBuilder(
            k=self.config.features.k,
            stride=self.config.features.stride,
            vectorizer_type=self.config.features.vectorizer_type,
            ngram_range=self.config.features.ngram_range,
            max_features=self.config.features.max_features,
        )

    def run(self, raw_data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Execute the end-to-end training pipeline.

        Args:
            raw_data_path: Optional override for raw dataset CSV path.

        Returns:
            Dictionary containing pipeline execution status and metadata.

        Raises:
            TrainingError: If execution fails at any stage.
        """
        data_path = Path(raw_data_path) if raw_data_path else self.config.data.get_raw_file_path()
        if not data_path.exists():
            raise TrainingError(
                f"Raw dataset file not found at: {data_path}. "
                "Please place the dataset in data/raw/ or update configs/data_config.yaml."
            )

        logger.info("Step 1: Ingesting dataset from %s", data_path)
        df_raw = load_csv(data_path)

        logger.info("Step 2: Preprocessing and validating sequences...")
        df_clean = self.processor.process_dataframe(
            df_raw,
            sequence_column=self.config.data.sequence_column,
        )

        logger.info("Step 3: Creating DNADataset and splitting train/val/test...")
        dataset = DNADataset.from_dataframe(
            df_clean,
            sequence_column=self.config.data.sequence_column,
            target_column=self.config.data.target_column,
        )

        train_ds, val_ds, test_ds = dataset.split(
            train_size=self.config.data.train_size,
            val_size=self.config.data.val_size,
            test_size=self.config.data.test_size,
            random_state=self.config.project.seed,
        )

        logger.info("Step 4: Building k-mer feature representations...")
        train_seqs = train_ds.get_sequences()
        train_labels = [str(l) for l in train_ds.get_labels()]
        self.feature_builder.fit(train_seqs, train_labels)

        # Persist vectorizer and encoder artifacts
        v_path = self.config.paths.get_vectorizers_dir() / "kmer_vectorizer.joblib"
        e_path = self.config.paths.get_encoders_dir() / "label_encoder.joblib"
        self.feature_builder.save(v_path, e_path)

        # In this architectural review phase, model training is deferred to Phase 3.
        logger.info("Step 5: Model training deferred to Phase 3 (Benchmark and Training Phase).")

        return {
            "status": "pipeline_ready",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "vectorizer_path": str(v_path),
            "encoder_path": str(e_path),
        }
