"""Dataset representation and access module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CancerSequenceSample:
    """Representation of an individual DNA sequence sample."""
    sequence: str
    label: Optional[str] = None
    sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Length of nucleotide sequence."""
        return len(self.sequence)


class DNADataset:
    """In-memory dataset container for DNA sequence classification."""

    def __init__(
        self,
        samples: List[CancerSequenceSample],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CancerSequenceSample:
        return self.samples[idx]

    def get_sequences(self) -> List[str]:
        """Extract all raw DNA sequences."""
        return [s.sequence for s in self.samples]

    def get_labels(self) -> List[Optional[str]]:
        """Extract all target labels."""
        return [s.label for s in self.samples]

    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate frequency of each target class."""
        distribution: Dict[str, int] = {}
        for s in self.samples:
            lbl = s.label or "Unlabeled"
            distribution[lbl] = distribution.get(lbl, 0) + 1
        return distribution

    def to_dataframe(self) -> Any:
        """Export dataset as a pandas DataFrame."""
        if pd is None:
            raise DataValidationError("pandas is required for to_dataframe. Please install pandas.")
        rows = [
            {
                "id": s.sample_id,
                "sequence": s.sequence,
                "cancer": s.label,
                "len": s.length,
            }
            for s in self.samples
        ]
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        sequence_column: str = "sequence",
        target_column: Optional[str] = "cancer",
        id_column: Optional[str] = "id",
    ) -> "DNADataset":
        """Instantiate a DNADataset from a DataFrame or dict-like object."""
        try:
            cols = list(df.columns)
        except AttributeError:
            raise DataValidationError("Provided dataset object must possess a 'columns' attribute.")

        if sequence_column not in cols:
            raise DataValidationError(f"Missing required sequence column: '{sequence_column}'")

        samples: List[CancerSequenceSample] = []
        # Support iterating rows across pandas DataFrame or custom table objects
        if hasattr(df, "iterrows"):
            for idx, row in df.iterrows():
                seq = str(row[sequence_column])
                label = str(row[target_column]) if target_column and target_column in cols and (pd is None or pd.notna(row[target_column])) else None
                s_id = str(row[id_column]) if id_column and id_column in cols and (pd is None or pd.notna(row[id_column])) else str(idx)
                samples.append(CancerSequenceSample(sequence=seq, label=label, sample_id=s_id))
        else:
            # Fallback for dict/list rows
            for idx, item in enumerate(df):
                samples.append(CancerSequenceSample(
                    sequence=str(item.get(sequence_column, "")),
                    label=str(item.get(target_column)) if target_column else None,
                    sample_id=str(item.get(id_column, idx)),
                ))

        return cls(samples=samples)

    def split(
        self,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple["DNADataset", "DNADataset", "DNADataset"]:
        """Split dataset into train, validation, and test subsets."""
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-4:
            raise DataValidationError(f"Split sizes must sum to 1.0, got {total}")

        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise DataValidationError("scikit-learn is required for dataset splitting. Please install scikit-learn.")

        labels = self.get_labels()
        strat = labels if stratify and all(l is not None for l in labels) else None

        temp_ratio = val_size + test_size
        train_indices, temp_indices = train_test_split(
            range(len(self.samples)),
            test_size=temp_ratio,
            random_state=random_state,
            stratify=strat,
        )

        temp_strat = [labels[i] for i in temp_indices] if strat is not None else None
        test_relative_ratio = test_size / temp_ratio

        val_sub_indices, test_sub_indices = train_test_split(
            range(len(temp_indices)),
            test_size=test_relative_ratio,
            random_state=random_state,
            stratify=temp_strat,
        )

        val_indices = [temp_indices[i] for i in val_sub_indices]
        test_indices = [temp_indices[i] for i in test_sub_indices]

        train_ds = DNADataset([self.samples[i] for i in train_indices])
        val_ds = DNADataset([self.samples[i] for i in val_indices])
        test_ds = DNADataset([self.samples[i] for i in test_indices])

        logger.info(
            "Split dataset (%d samples): train=%d, val=%d, test=%d",
            len(self.samples), len(train_ds), len(val_ds), len(test_ds)
        )
        return train_ds, val_ds, test_ds
