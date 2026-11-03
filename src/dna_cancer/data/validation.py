"""Dataset and DNA sequence validation module."""

from typing import Any, Iterable, List, Optional, Set

from dna_cancer.utils.exceptions import DataValidationError, SequenceFormatError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

STANDARD_DNA_BASES = frozenset("ACGTN")


def validate_dna_sequence(
    sequence: str,
    allowed_bases: Iterable[str] = STANDARD_DNA_BASES,
    min_length: int = 1,
    max_length: Optional[int] = None,
    allow_empty: bool = False,
    raise_exception: bool = False,
) -> bool:
    """Validate a single DNA sequence against character and length constraints.

    Args:
        sequence: The DNA nucleotide sequence string to validate.
        allowed_bases: Set or string of valid uppercase nucleotide characters (e.g., 'ACGTN').
        min_length: Minimum valid sequence length (ignored if allow_empty is True and sequence is empty).
        max_length: Optional maximum allowed sequence length.
        allow_empty: Whether an empty string sequence is permissible.
        raise_exception: If True, raises SequenceFormatError on failure instead of returning False.

    Returns:
        True if the sequence is valid, False otherwise.

    Raises:
        SequenceFormatError: If validation fails and raise_exception is True.
    """
    if not isinstance(sequence, str):
        msg = f"Expected sequence of type str, got {type(sequence).__name__}"
        if raise_exception:
            raise SequenceFormatError(msg)
        return False

    seq_len = len(sequence)

    if seq_len == 0:
        if allow_empty:
            return True
        msg = "DNA sequence cannot be empty."
        if raise_exception:
            raise SequenceFormatError(msg)
        return False

    if seq_len < min_length:
        msg = f"Sequence length ({seq_len}) is below minimum required length ({min_length})."
        if raise_exception:
            raise SequenceFormatError(msg)
        return False

    if max_length is not None and seq_len > max_length:
        msg = f"Sequence length ({seq_len}) exceeds maximum allowed length ({max_length})."
        if raise_exception:
            raise SequenceFormatError(msg)
        return False

    allowed_set: Set[str] = {b.upper() for b in allowed_bases}
    upper_seq = sequence.upper()
    invalid_chars = set(upper_seq) - allowed_set

    if invalid_chars:
        msg = f"Sequence contains invalid nucleotide characters: {sorted(invalid_chars)}. Allowed: {sorted(allowed_set)}"
        if raise_exception:
            raise SequenceFormatError(msg)
        return False

    return True


class SequenceValidator:
    """Configurable validator for DNA nucleotide sequences."""

    def __init__(
        self,
        allowed_bases: Iterable[str] = STANDARD_DNA_BASES,
        min_length: int = 1,
        max_length: Optional[int] = None,
        allow_empty: bool = False,
    ) -> None:
        self.allowed_bases = frozenset(b.upper() for b in allowed_bases)
        self.min_length = min_length
        self.max_length = max_length
        self.allow_empty = allow_empty

    def is_valid(self, sequence: str) -> bool:
        """Check if sequence is valid according to configured rules."""
        return validate_dna_sequence(
            sequence=sequence,
            allowed_bases=self.allowed_bases,
            min_length=self.min_length,
            max_length=self.max_length,
            allow_empty=self.allow_empty,
            raise_exception=False,
        )

    def validate(self, sequence: str) -> None:
        """Validate sequence and raise SequenceFormatError if invalid."""
        validate_dna_sequence(
            sequence=sequence,
            allowed_bases=self.allowed_bases,
            min_length=self.min_length,
            max_length=self.max_length,
            allow_empty=self.allow_empty,
            raise_exception=True,
        )


def validate_dataset_schema(
    df: Any,
    required_columns: List[str],
    raise_exception: bool = True,
) -> bool:
    """Validate that a DataFrame contains expected required columns and non-empty rows.

    Args:
        df: A pandas DataFrame (or duck-typed object supporting columns and __len__).
        required_columns: List of column names that must be present.
        raise_exception: If True, raises DataValidationError on failure.

    Returns:
        True if schema is valid, False otherwise.

    Raises:
        DataValidationError: If columns are missing or dataset is empty.
    """
    if df is None:
        msg = "Dataset is None."
        if raise_exception:
            raise DataValidationError(msg)
        return False

    try:
        cols = set(df.columns)
    except AttributeError:
        msg = "Provided dataset does not possess a 'columns' attribute."
        if raise_exception:
            raise DataValidationError(msg)
        return False

    missing = [c for c in required_columns if c not in cols]
    if missing:
        msg = f"Dataset is missing required columns: {missing}. Present columns: {list(cols)}"
        if raise_exception:
            raise DataValidationError(msg)
        return False

    if len(df) == 0:
        msg = "Dataset contains 0 rows."
        if raise_exception:
            raise DataValidationError(msg)
        return False

    return True

