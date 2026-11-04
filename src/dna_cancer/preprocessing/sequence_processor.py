"""Preprocessing orchestration for DNA sequence datasets."""

from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.data.ingestion import SimpleDataFrame
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

    def process_records(
        self,
        records: List[Dict[str, Any]],
        sequence_key: str = "sequence",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Process a list of record dictionaries in pure Python without requiring pandas."""
        cleaned_records: List[Dict[str, Any]] = []
        stats = {"total": len(records), "valid": 0, "dropped": 0}

        for r in records:
            if sequence_key not in r:
                stats["dropped"] += 1
                if not self.drop_invalid:
                    raise PreprocessingError(f"Record missing sequence key '{sequence_key}': {r}")
                continue

            cleaned = self.process_single(str(r[sequence_key]))
            if cleaned is not None:
                record_copy = dict(r)
                record_copy[sequence_key] = cleaned
                record_copy["len"] = len(cleaned)
                cleaned_records.append(record_copy)
                stats["valid"] += 1
            else:
                stats["dropped"] += 1

        logger.info("Record preprocessing finished: %s", stats)
        return cleaned_records, stats

    def process_dataframe(
        self,
        df: Any,
        sequence_column: str = "sequence",
        output_column: Optional[str] = None,
    ) -> Any:
        """Process sequences within a pandas DataFrame or SimpleDataFrame.

        Filters out invalid rows and updates sequence lengths.
        """
        if sequence_column not in getattr(df, "columns", []):
            raise PreprocessingError(f"Column '{sequence_column}' not found in dataset.")

        target_col = output_column or sequence_column

        if pd is not None and isinstance(df, pd.DataFrame):
            df_copy = df.copy()
            cleaned_series = df_copy[sequence_column].astype(str).apply(self.cleaner.clean)
            valid_mask = cleaned_series.apply(self.validator.is_valid)
            dropped_count = int((~valid_mask).sum())

            if dropped_count > 0:
                logger.warning("Dropping %d invalid sequence rows out of %d", dropped_count, len(df_copy))
                df_copy = df_copy[valid_mask].copy()
                cleaned_series = cleaned_series[valid_mask]

            df_copy[target_col] = cleaned_series
            df_copy["len"] = df_copy[target_col].str.len()
            return df_copy
        else:
            # Fallback for SimpleDataFrame or dict-like tables
            records = df.to_dict(orient="records") if hasattr(df, "to_dict") else [dict(r) for _, r in df.iterrows()]
            cleaned_records, _ = self.process_records(records, sequence_key=sequence_column)
            if target_col != sequence_column:
                for r in cleaned_records:
                    r[target_col] = r[sequence_column]
            return SimpleDataFrame(cleaned_records)
