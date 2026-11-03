"""Custom exceptions for the DNA Cancer Classification framework."""


class DNACancerError(Exception):
    """Base exception for all DNA cancer classification errors."""


class ConfigurationError(DNACancerError):
    """Raised when configuration validation or loading fails."""


class DataValidationError(DNACancerError):
    """Raised when a dataset or DNA sequence fails validation rules."""


class SequenceFormatError(DataValidationError):
    """Raised when a DNA sequence contains invalid characters or bad structure."""


class DataIngestionError(DNACancerError):
    """Raised when reading or parsing raw data fails."""


class PreprocessingError(DNACancerError):
    """Raised when an error occurs during sequence preprocessing."""


class FeatureExtractionError(DNACancerError):
    """Raised when k-mer extraction or feature vector generation fails."""


class ModelError(DNACancerError):
    """Base class for model-related errors."""


class ModelNotFoundError(ModelError):
    """Raised when a requested model architecture is not registered or found."""


class TrainingError(DNACancerError):
    """Raised when an error occurs during model training orchestration."""


class PredictionError(DNACancerError):
    """Raised during model prediction or batch inference."""

