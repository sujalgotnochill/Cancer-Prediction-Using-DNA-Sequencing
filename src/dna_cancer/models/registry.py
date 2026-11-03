"""Model registration and discovery mechanism."""

from typing import Callable, Dict, List, Type

from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.utils.exceptions import ModelNotFoundError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

_REGISTRY: Dict[str, Type[BaseDNAClassifier]] = {}


def register_model(name: str) -> Callable[[Type[BaseDNAClassifier]], Type[BaseDNAClassifier]]:
    """Decorator to register a new model class by unique name.

    Args:
        name: Unique string identifier for the model architecture.
    """
    def decorator(cls: Type[BaseDNAClassifier]) -> Type[BaseDNAClassifier]:
        if not issubclass(cls, BaseDNAClassifier):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseDNAClassifier.")
        if name in _REGISTRY:
            logger.warning("Overwriting existing model registration for '%s'.", name)
        _REGISTRY[name] = cls
        logger.debug("Registered model architecture '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def get_model_class(name: str) -> Type[BaseDNAClassifier]:
    """Retrieve a registered model class by name.

    Args:
        name: Name of the registered model.

    Returns:
        The model class.

    Raises:
        ModelNotFoundError: If no model is registered with the given name.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ModelNotFoundError(f"Model '{name}' not found in registry. Available models: {available}")
    return _REGISTRY[name]


def list_registered_models() -> List[str]:
    """List all currently registered model architectures."""
    return sorted(list(_REGISTRY.keys()))

