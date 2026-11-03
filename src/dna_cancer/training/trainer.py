"""Training orchestration module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.utils.exceptions import TrainingError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Hyperparameters and orchestration parameters for training."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    checkpoint_dir: Optional[str] = "artifacts/models"
    early_stopping_patience: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Orchestrates model training, validation checkpoints, and progress logging."""

    def __init__(
        self,
        model: BaseDNAClassifier,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute training of the model on provided features.

        Args:
            X_train: Training feature matrix.
            y_train: Training targets.
            X_val: Optional validation feature matrix.
            y_val: Optional validation targets.

        Returns:
            Dictionary containing training summary and metrics history.

        Raises:
            TrainingError: If training data is invalid or training fails.
        """
        if len(X_train) == 0 or len(y_train) == 0:
            raise TrainingError("Cannot train on empty dataset.")

        if len(X_train) != len(y_train):
            raise TrainingError(f"Features and targets length mismatch: {len(X_train)} vs {len(y_train)}")

        logger.info(
            "Starting model training: %d train samples, %d val samples, %d epochs, batch_size=%d",
            len(X_train),
            len(X_val) if X_val is not None else 0,
            self.config.epochs,
            self.config.batch_size,
        )

        validation_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        # Train model via abstract interface
        self.model.fit(
            X=X_train,
            y=y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            **self.config.extra_params,
        )

        logger.info("Model training completed successfully.")
        return {
            "status": "completed",
            "epochs": self.config.epochs,
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist current model weights and state."""
        self.model.save(path)
        logger.info("Saved training checkpoint to %s", path)
