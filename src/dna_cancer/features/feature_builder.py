"""Model feature preparation using k-mer vectorization and label encoding."""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    CountVectorizer = None  # type: ignore
    TfidfVectorizer = None  # type: ignore
    LabelEncoder = None  # type: ignore

from dna_cancer.features.kmer_extractor import KMerExtractor
from dna_cancer.utils.exceptions import FeatureExtractionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Prepares numerical features from raw DNA sequences using k-mers and vectorizers."""

    def __init__(
        self,
        k: int = 6,
        stride: int = 1,
        vectorizer_type: str = "count",
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
        return_sparse: bool = False,
        canonical: bool = False,
    ) -> None:
        self.k = k
        self.stride = stride
        self.canonical = canonical
        self.kmer_extractor = KMerExtractor(
            k=k, stride=stride, to_lower=True, as_sentence=True, canonical=canonical
        )
        self.vectorizer_type = vectorizer_type.lower()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.return_sparse = return_sparse

        self.vectorizer = None
        if self.vectorizer_type == "count":
            if CountVectorizer is not None:
                self.vectorizer = CountVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    token_pattern=r"\S+",
                )
        elif self.vectorizer_type == "tfidf":
            if TfidfVectorizer is not None:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    token_pattern=r"\S+",
                )
        elif self.vectorizer_type in ("frequency", "normalized"):
            if TfidfVectorizer is not None:
                self.vectorizer = TfidfVectorizer(
                    use_idf=False,
                    norm="l1",
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    token_pattern=r"\S+",
                )
        else:
            raise FeatureExtractionError(f"Unsupported vectorizer type: {vectorizer_type}")

        self.label_encoder = LabelEncoder() if LabelEncoder is not None else None
        self.is_fitted = False

    def _ensure_vectorizer(self) -> None:
        """Verify that scikit-learn is installed and vectorizer is initialized."""
        if self.vectorizer is None:
            raise FeatureExtractionError(
                "scikit-learn is required for vectorization. Please install scikit-learn."
            )

    def fit(self, sequences: List[str], labels: Optional[List[str]] = None) -> "FeatureBuilder":
        """Fit vectorizer on sequences and optionally label encoder on target classes."""
        self._ensure_vectorizer()
        if not sequences:
            raise FeatureExtractionError("Cannot fit FeatureBuilder on an empty list of sequences.")

        logger.info("Extracting k-mers for feature fitting on %d sequences...", len(sequences))
        sentences = [self.kmer_extractor.extract(s) for s in sequences]

        # Ensure at least one sentence has tokens
        if not any(bool(s) for s in sentences):
            raise FeatureExtractionError(
                f"All input sequences yielded empty k-mer sentences (check if all sequences are shorter than k={self.kmer_extractor.k})."
            )

        self.vectorizer.fit(sentences)

        if labels is not None:
            if self.label_encoder is None:
                raise FeatureExtractionError("scikit-learn is required for LabelEncoder.")
            if len(labels) != len(sequences):
                raise FeatureExtractionError(
                    f"Number of labels ({len(labels)}) does not match sequences ({len(sequences)})."
                )
            self.label_encoder.fit(labels)

        self.is_fitted = True
        return self

    def transform_sequences(
        self,
        sequences: List[str],
        return_sparse: Optional[bool] = None,
    ) -> Any:
        """Transform DNA sequences into vector feature matrix.

        Args:
            sequences: List of raw DNA nucleotide strings.
            return_sparse: If True, returns scipy sparse matrix. If False, returns numpy ndarray.
                If None, defaults to self.return_sparse.
        """
        if not self.is_fitted:
            raise FeatureExtractionError("FeatureBuilder has not been fitted yet.")
        self._ensure_vectorizer()

        if not sequences:
            raise FeatureExtractionError("Cannot transform empty list of sequences.")

        sentences = [self.kmer_extractor.extract(s) for s in sequences]
        matrix = self.vectorizer.transform(sentences)
        
        use_sparse = self.return_sparse if return_sparse is None else return_sparse
        if use_sparse:
            return matrix
        return matrix.toarray()

    @property
    def n_features(self) -> int:
        """Return number of extracted features."""
        if not self.is_fitted or self.vectorizer is None:
            return 0
        if hasattr(self.vectorizer, "vocabulary_"):
            return len(self.vectorizer.vocabulary_)
        return 0

    def get_feature_names(self) -> List[str]:
        """Return names of vocabulary features."""
        if not self.is_fitted or self.vectorizer is None:
            return []
        if hasattr(self.vectorizer, "get_feature_names_out"):
            return list(self.vectorizer.get_feature_names_out())
        return []

    def transform_labels(self, labels: List[str]) -> Any:
        """Transform string cancer labels into integer classes."""
        if self.label_encoder is None or not hasattr(self.label_encoder, "classes_"):
            raise FeatureExtractionError("Label encoder has not been fitted.")
        return self.label_encoder.transform(labels)

    def inverse_transform_labels(self, encoded_labels: Any) -> List[str]:
        """Convert integer class indices back to string cancer labels."""
        if self.label_encoder is None or not hasattr(self.label_encoder, "classes_"):
            raise FeatureExtractionError("Label encoder has not been fitted.")
        return self.label_encoder.inverse_transform(encoded_labels).tolist()

    def save(
        self,
        vectorizer_path: Union[str, Path],
        encoder_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Save vectorizer and label encoder artifacts to disk."""
        if joblib is None:
            raise FeatureExtractionError("joblib is required to persist feature artifacts.")

        v_path = Path(vectorizer_path)
        v_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, v_path)
        logger.info("Saved vectorizer artifact to %s", v_path)

        if encoder_path and self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            e_path = Path(encoder_path)
            e_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.label_encoder, e_path)
            logger.info("Saved label encoder artifact to %s", e_path)

    def load(
        self,
        vectorizer_path: Union[str, Path],
        encoder_path: Optional[Union[str, Path]] = None,
    ) -> "FeatureBuilder":
        """Load vectorizer and label encoder artifacts from disk."""
        if joblib is None:
            raise FeatureExtractionError("joblib is required to load feature artifacts.")

        v_path = Path(vectorizer_path)
        if not v_path.exists():
            raise FeatureExtractionError(f"Vectorizer file not found: {v_path}")

        self.vectorizer = joblib.load(v_path)
        if hasattr(self.vectorizer, "vocabulary_") and self.vectorizer.vocabulary_:
            sample_token = next(iter(self.vectorizer.vocabulary_.keys()))
            if sample_token and len(sample_token) > 0:
                self.k = len(sample_token)
                self.kmer_extractor = KMerExtractor(
                    k=self.k,
                    stride=self.stride,
                    to_lower=True,
                    as_sentence=True,
                    canonical=self.canonical,
                )
        if encoder_path:
            e_path = Path(encoder_path)
            if e_path.exists():
                self.label_encoder = joblib.load(e_path)

        self.is_fitted = True
        logger.info("Loaded feature artifacts from %s", v_path)
        return self
