"""DNA sequence cleaning and normalization module."""

import re
from typing import Optional

from dna_cancer.utils.exceptions import PreprocessingError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

# Matches any character that is not standard IUPAC nucleotide or N
_NON_DNA_REGEX = re.compile(r"[^ACGTNacgtn]")
_WHITESPACE_REGEX = re.compile(r"\s+")


def clean_sequence(
    sequence: str,
    to_upper: bool = True,
    remove_whitespace: bool = True,
    remove_non_nucleotide: bool = True,
    replace_unknown_with: Optional[str] = None,
) -> str:
    """Clean and standardize a raw DNA nucleotide sequence string.

    Args:
        sequence: Raw sequence string (may contain whitespace, newlines, digits, mixed case).
        to_upper: Convert all nucleotides to uppercase.
        remove_whitespace: Strip spaces, tabs, and newlines.
        remove_non_nucleotide: Strip any non-ACGTN characters (e.g., numbers, punctuation).
        replace_unknown_with: If set, replaces non-ACGT characters with this character (e.g., 'N').

    Returns:
        Standardized DNA nucleotide string.

    Raises:
        PreprocessingError: If input is not a string.
    """
    if not isinstance(sequence, str):
        raise PreprocessingError(f"Sequence to clean must be str, got {type(sequence).__name__}")

    cleaned = sequence

    if remove_whitespace:
        cleaned = _WHITESPACE_REGEX.sub("", cleaned)

    if replace_unknown_with is not None:
        cleaned = re.sub(r"[^ACGTacgt]", replace_unknown_with, cleaned)
    elif remove_non_nucleotide:
        cleaned = _NON_DNA_REGEX.sub("", cleaned)

    if to_upper:
        cleaned = cleaned.upper()

    return cleaned


class SequenceCleaner:
    """Configurable cleaner for DNA nucleotide sequences."""

    def __init__(
        self,
        to_upper: bool = True,
        remove_whitespace: bool = True,
        remove_non_nucleotide: bool = True,
        replace_unknown_with: Optional[str] = None,
    ) -> None:
        self.to_upper = to_upper
        self.remove_whitespace = remove_whitespace
        self.remove_non_nucleotide = remove_non_nucleotide
        self.replace_unknown_with = replace_unknown_with

    def clean(self, sequence: str) -> str:
        """Standardize the input DNA sequence."""
        return clean_sequence(
            sequence=sequence,
            to_upper=self.to_upper,
            remove_whitespace=self.remove_whitespace,
            remove_non_nucleotide=self.remove_non_nucleotide,
            replace_unknown_with=self.replace_unknown_with,
        )

