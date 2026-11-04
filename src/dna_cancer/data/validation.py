"""Dataset and DNA sequence validation module."""

from typing import Any, Dict, Iterable, List, Optional, Set

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


def validate_dataset_records(
    df: Any,
    sequence_column: str = "sequence",
    target_column: Optional[str] = "cancer",
    expected_classes: Optional[Iterable[str]] = None,
    validator: Optional[SequenceValidator] = None,
    allow_missing_labels: bool = False,
    raise_exception: bool = True,
) -> Dict[str, Any]:
    """Validate row-level records within a dataset for missing sequences, invalid bases, and invalid labels.

    Args:
        df: DataFrame or duck-typed object supporting .columns and .iterrows().
        sequence_column: Name of the nucleotide sequence column.
        target_column: Optional name of target cancer classification column.
        expected_classes: Optional iterable of allowed target class names.
        validator: Optional SequenceValidator instance.
        allow_missing_labels: Whether None/empty labels are permissible.
        raise_exception: If True, raises validation errors on the first failure encountered.

    Returns:
        Dictionary with record counts and validation summary.

    Raises:
        DataValidationError: If required columns, sequences, or labels are invalid and raise_exception is True.
        SequenceFormatError: If sequence nucleotide composition is invalid and raise_exception is True.
    """
    validate_dataset_schema(df, required_columns=[sequence_column], raise_exception=raise_exception)

    seq_validator = validator or SequenceValidator()
    expected_set = set(expected_classes) if expected_classes else None

    total_records = len(df)
    valid_count = 0
    invalid_seq_indices: List[int] = []
    invalid_label_indices: List[int] = []

    # Iterate rows across pandas or SimpleDataFrame
    iterator = df.iterrows() if hasattr(df, "iterrows") else enumerate(df)

    for idx, row in iterator:
        # 1. Validate Sequence
        raw_seq = row.get(sequence_column) if hasattr(row, "get") else row[sequence_column]
        if raw_seq is None or (isinstance(raw_seq, float) and str(raw_seq) == "nan"):
            if raise_exception:
                raise DataValidationError(f"Row {idx}: missing nucleotide sequence (value is null/NaN).")
            invalid_seq_indices.append(idx)
            continue

        seq_str = str(raw_seq).strip()
        if not seq_str:
            if raise_exception:
                raise DataValidationError(f"Row {idx}: empty nucleotide sequence.")
            invalid_seq_indices.append(idx)
            continue

        if not seq_validator.is_valid(seq_str):
            if raise_exception:
                raise SequenceFormatError(f"Row {idx}: sequence failed validation (length or invalid bases).")
            invalid_seq_indices.append(idx)
            continue

        # 2. Validate Label
        if target_column and target_column in getattr(df, "columns", []):
            raw_label = row.get(target_column) if hasattr(row, "get") else row[target_column]
            if raw_label is None or (isinstance(raw_label, float) and str(raw_label) == "nan") or str(raw_label).strip() == "":
                if not allow_missing_labels:
                    if raise_exception:
                        raise DataValidationError(f"Row {idx}: missing target label in column '{target_column}'.")
                    invalid_label_indices.append(idx)
                    continue
            else:
                label_str = str(raw_label).strip()
                if expected_set and label_str not in expected_set:
                    if raise_exception:
                        raise DataValidationError(
                            f"Row {idx}: invalid label '{label_str}'. Expected one of: {sorted(expected_set)}"
                        )
                    invalid_label_indices.append(idx)
                    continue

        valid_count += 1

    summary = {
        "total_records": total_records,
        "valid_records": valid_count,
        "invalid_sequences": len(invalid_seq_indices),
        "invalid_labels": len(invalid_label_indices),
    }
    return summary
