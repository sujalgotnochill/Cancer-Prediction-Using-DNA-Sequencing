"""Preprocessing orchestration for DNA sequence datasets."""

from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.data.validation import SequenceValidator
from dna_cancer.preprocessing.sequence_cleaner import SequenceCleaner
from dna_cancer.utils.exceptions import PreprocessingError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class SequenceProcessor:
    """Orchestrates DNA sequence cleaning, validation, and dataset filtering."""

    def __init__(
        self,
        cleaner: Optional[SequenceCleaner] = None,
        validator: Optional[SequenceValidator] = None,
        drop_invalid: bool = True,
    ) -> None:
        self.cleaner = cleaner or SequenceCleaner()
        self.validator = validator or SequenceValidator()
        self.drop_invalid = drop_invalid

    def process_single(self, sequence: str) -> Optional[str]:
        """Clean and validate a single sequence.

        Returns cleaned sequence if valid; None if invalid and drop_invalid=True.
        """
        cleaned = self.cleaner.clean(sequence)
        if self.validator.is_valid(cleaned):
            return cleaned

        if not self.drop_invalid:
            raise PreprocessingError(f"Sequence failed validation after cleaning: {sequence[:30]}...")
        return None

    def process_batch(self, sequences: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Process a list of sequences, returning valid cleaned sequences and statistics."""
        cleaned_list: List[str] = []
        stats = {"total": len(sequences), "valid": 0, "dropped": 0}

        for seq in sequences:
            result = self.process_single(seq)
            if result is not None:
                cleaned_list.append(result)
                stats["valid"] += 1
            else:
                stats["dropped"] += 1

        logger.info("Batch preprocessing finished: %s", stats)
        return cleaned_list, stats

    def process_dataframe(
        self,
        df: Any,
        sequence_column: str = "sequence",
        output_column: Optional[str] = None,
    ) -> Any:
        """Process sequences within a pandas DataFrame.

        Filters out invalid rows and updates sequence lengths.
        """
        if pd is None:
            raise PreprocessingError("pandas is required for process_dataframe. Please install pandas.")

        if sequence_column not in df.columns:
            raise PreprocessingError(f"Column '{sequence_column}' not found in DataFrame.")

        target_col = output_column or sequence_column
        df_copy = df.copy()

        # Clean sequences
        cleaned_series = df_copy[sequence_column].astype(str).apply(self.cleaner.clean)

        # Validate
        valid_mask = cleaned_series.apply(self.validator.is_valid)
        dropped_count = int((~valid_mask).sum())

        if dropped_count > 0:
            logger.warning("Dropping %d invalid sequence rows out of %d", dropped_count, len(df_copy))
            df_copy = df_copy[valid_mask].copy()
            cleaned_series = cleaned_series[valid_mask]

        df_copy[target_col] = cleaned_series
        df_copy["len"] = df_copy[target_col].str.len()

        return df_copy
