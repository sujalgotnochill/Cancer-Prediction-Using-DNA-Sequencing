"""Configurable k-mer extraction for DNA sequences."""

from typing import List, Union

from dna_cancer.utils.exceptions import FeatureExtractionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


_REVCOMP_TRANS_LOWER = str.maketrans("acgtn", "tgcan")
_REVCOMP_TRANS_UPPER = str.maketrans("ACGTN", "TGCAN")


def extract_kmers(
    sequence: str,
    k: int = 6,
    stride: int = 1,
    to_lower: bool = True,
    as_sentence: bool = False,
    canonical: bool = False,
) -> Union[List[str], str]:
    """Extract overlapping or strided k-mers from a DNA sequence string.

    Args:
        sequence: The DNA nucleotide sequence string.
        k: Length of each substring k-mer (default 6).
        stride: Step size between consecutive k-mers (default 1).
        to_lower: Convert k-mers to lowercase (default True, matching NLP vectorizer norms).
        as_sentence: If True, returns space-separated k-mers as a string sentence.
        canonical: If True, maps each k-mer to the lexicographical minimum of itself
            and its reverse complement (strand-invariant representation).

    Returns:
        List of k-mer strings, or a space-separated sentence string.

    Raises:
        FeatureExtractionError: If k or stride are invalid (< 1) or sequence is not str.
    """
    if not isinstance(sequence, str):
        raise FeatureExtractionError(f"Expected sequence string, got {type(sequence).__name__}")

    if k < 1:
        raise FeatureExtractionError(f"k-mer size k must be >= 1, got {k}")

    if stride < 1:
        raise FeatureExtractionError(f"Stride must be >= 1, got {stride}")

    seq = sequence.lower() if to_lower else sequence
    seq_len = len(seq)

    if seq_len < k:
        return "" if as_sentence else []

    if canonical:
        trans = _REVCOMP_TRANS_LOWER if to_lower else _REVCOMP_TRANS_UPPER
        kmers = []
        for i in range(0, seq_len - k + 1, stride):
            w = seq[i : i + k]
            rc = w.translate(trans)[::-1]
            kmers.append(min(w, rc))
    else:
        kmers = [seq[i : i + k] for i in range(0, seq_len - k + 1, stride)]

    if as_sentence:
        return " ".join(kmers)
    return kmers


class KMerExtractor:
    """Class wrapper for configurable k-mer extraction from DNA sequences."""

    def __init__(
        self,
        k: int = 6,
        stride: int = 1,
        to_lower: bool = True,
        as_sentence: bool = True,
        canonical: bool = False,
    ) -> None:
        self.k = k
        self.stride = stride
        self.to_lower = to_lower
        self.as_sentence = as_sentence
        self.canonical = canonical

    def extract(self, sequence: str) -> Union[List[str], str]:
        """Extract k-mers from a single sequence."""
        return extract_kmers(
            sequence=sequence,
            k=self.k,
            stride=self.stride,
            to_lower=self.to_lower,
            as_sentence=self.as_sentence,
            canonical=self.canonical,
        )

    def extract_batch(self, sequences: List[str]) -> List[Union[List[str], str]]:
        """Extract k-mers from a batch of sequences."""
        return [self.extract(s) for s in sequences]

