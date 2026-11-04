"""Dataset representation and access module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.data.ingestion import SimpleDataFrame
from dna_cancer.data.schema import CancerSequenceSample, DNASample, SampleMetadata
from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


class DNADataset:
    """In-memory dataset container for DNA sequence classification."""

    def __init__(
        self,
        samples: List[DNASample],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DNASample:
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
        """Export dataset as a pandas DataFrame or SimpleDataFrame."""
        rows = [
            {
                "id": s.sample_id,
                "sequence": s.sequence,
                "cancer": s.label,
                "len": s.length,
            }
            for s in self.samples
        ]
        if pd is not None:
            return pd.DataFrame(rows)
        return SimpleDataFrame(rows)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        sequence_column: str = "sequence",
        target_column: Optional[str] = "cancer",
        id_column: Optional[str] = "id",
    ) -> "DNADataset":
        """Instantiate a DNADataset from a DataFrame or SimpleDataFrame."""
        try:
            cols = list(df.columns)
        except AttributeError:
            raise DataValidationError("Provided dataset object must possess a 'columns' attribute.")

        if sequence_column not in cols:
            raise DataValidationError(f"Missing required sequence column: '{sequence_column}'")

        samples: List[CancerSequenceSample] = []
        if hasattr(df, "iterrows"):
            for idx, row in df.iterrows():
                seq = str(row[sequence_column])
                raw_label = row.get(target_column) if hasattr(row, "get") else (row[target_column] if target_column in cols else None)
                label = str(raw_label) if raw_label is not None and str(raw_label) != "nan" else None
                raw_id = row.get(id_column) if hasattr(row, "get") else (row[id_column] if id_column in cols else None)
                s_id = str(raw_id) if raw_id is not None and str(raw_id) != "nan" else str(idx)
                samples.append(CancerSequenceSample(sequence=seq, label=label, sample_id=s_id))
        else:
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
        if len(self.samples) == 0:
            raise DataValidationError("Cannot split an empty dataset.")

        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-4:
            raise DataValidationError(f"Split sizes must sum to 1.0, got {total}")

        if len(self.samples) < 3:
            raise DataValidationError(
                f"Dataset must contain at least 3 samples to perform 3-way split, got {len(self.samples)}"
            )

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
