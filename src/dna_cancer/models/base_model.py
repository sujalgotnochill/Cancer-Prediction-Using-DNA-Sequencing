"""Common interface for all DNA sequence classification models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class BaseDNAClassifier(ABC):
    """Abstract base class establishing the contract for DNA sequence classifiers."""

    @abstractmethod
    def fit(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[tuple] = None,
        **kwargs: Any,
    ) -> "BaseDNAClassifier":
        """Train the classifier on feature matrix X and target labels y.

        Args:
            X: 2D array of input features (samples x features).
            y: 1D array of target class indices.
            validation_data: Optional tuple of (X_val, y_val).
            **kwargs: Additional model-specific training hyperparameters.

        Returns:
            Self instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict class labels for feature matrix X.

        Args:
            X: 2D array of input features.

        Returns:
            1D array of predicted class indices.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Predict class probability distributions for feature matrix X.

        Args:
            X: 2D array of input features.

        Returns:
            2D array of predicted class probabilities (samples x classes).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist model architecture and learned weights to disk.

        Args:
            path: Destination file or directory path.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "BaseDNAClassifier":
        """Load a persisted model from disk.

        Args:
            path: Source file or directory path.
            **kwargs: Additional configuration parameters.

        Returns:
            Loaded BaseDNAClassifier instance.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters dictionary."""
        return {}
