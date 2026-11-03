"""Model creation and instantiation factory."""

from pathlib import Path
from typing import Any, Union

from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.models.registry import get_model_class
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory to instantiate or load registered DNA sequence classifiers."""

    @staticmethod
    def create(model_name: str, **kwargs: Any) -> BaseDNAClassifier:
        """Instantiate a registered model by name.

        Args:
            model_name: Registered identifier of the model.
            **kwargs: Hyperparameters and initialization arguments.

        Returns:
            An instance of BaseDNAClassifier.
        """
        model_cls = get_model_class(model_name)
        logger.info("Instantiating model '%s' via ModelFactory", model_name)
        return model_cls(**kwargs)

    @staticmethod
    def load(model_name: str, path: Union[str, Path], **kwargs: Any) -> BaseDNAClassifier:
        """Load a persisted model using its registered class loader.

        Args:
            model_name: Registered identifier of the model.
            path: Path to persisted model file or directory.
            **kwargs: Additional loader arguments.

        Returns:
            Loaded BaseDNAClassifier instance.
        """
        model_cls = get_model_class(model_name)
        logger.info("Loading model '%s' from %s via ModelFactory", model_name, path)
        return model_cls.load(path, **kwargs)

