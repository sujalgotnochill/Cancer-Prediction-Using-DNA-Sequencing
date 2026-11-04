"""Unified dataset schema and sample representation for DNA sequences."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

COMPLEMENT_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def compute_reverse_complement(sequence: str) -> str:
    """Compute the reverse complement of a DNA nucleotide sequence.

    Args:
        sequence: Nucleotide sequence string (e.g., 'ATCGN').

    Returns:
        Reverse complement sequence string (e.g., 'CGATN').
    """
    return sequence.translate(COMPLEMENT_MAP)[::-1]


@dataclass
class SampleMetadata:
    """Standardized biological and experimental metadata for a DNA sample.

    Captures provenance, genomic coordinates, and sequencing technology without
    fabricating unverified biological values.
    """
    dataset_id: Optional[str] = None
    source: Optional[str] = None
    organism: Optional[str] = None
    gene: Optional[str] = None
    locus: Optional[str] = None
    chromosome: Optional[str] = None
    sequence_type: Optional[str] = None
    collection_date: Optional[str] = None
    sequencing_platform: Optional[str] = None
    patient_id: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata fields into a dictionary."""
        d = asdict(self)
        # Merge additional metadata cleanly
        extra = d.pop("additional_metadata", {}) or {}
        d.update(extra)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleMetadata":
        """Instantiate SampleMetadata from a dictionary, separating known from custom fields."""
        known_keys = {
            "dataset_id", "source", "organism", "gene", "locus", "chromosome",
            "sequence_type", "collection_date", "sequencing_platform", "patient_id"
        }
        known_args = {}
        extra_args = {}

        for k, v in data.items():
            if k in known_keys:
                known_args[k] = v
            elif k not in ("sample_id", "sequence", "label", "cancer", "id", "len", "length"):
                extra_args[k] = v

        return cls(**known_args, additional_metadata=extra_args)


class DNASample:
    """Unified internal representation of an individual DNA sequence sample.

    Adheres to the strict schema contract:
    - Required: sample_id, sequence, label (optional only for unannotated samples).
    - Metadata: biological and experimental provenance kept strictly separate from label.
    """

    def __init__(
        self,
        sample_id: str,
        sequence: str,
        label: Optional[str] = None,
        metadata: Optional[Union[SampleMetadata, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        self.sample_id = str(sample_id) if sample_id is not None else ""
        self.sequence = str(sequence) if sequence is not None else ""
        self.label = str(label) if (label is not None and str(label) != "nan") else None

        if isinstance(metadata, SampleMetadata):
            self.metadata = metadata
        elif isinstance(metadata, dict):
            self.metadata = SampleMetadata.from_dict(metadata)
        else:
            self.metadata = SampleMetadata()

        # Fold extra keyword arguments into metadata
        for k, v in kwargs.items():
            if hasattr(self.metadata, k):
                setattr(self.metadata, k, v)
            else:
                self.metadata.additional_metadata[k] = v

    @property
    def length(self) -> int:
        """Length of the nucleotide sequence."""
        return len(self.sequence)

    @property
    def patient_id(self) -> Optional[str]:
        return self.metadata.patient_id

    @patient_id.setter
    def patient_id(self, val: Optional[str]) -> None:
        self.metadata.patient_id = val

    @property
    def dataset_id(self) -> Optional[str]:
        return self.metadata.dataset_id

    @dataset_id.setter
    def dataset_id(self, val: Optional[str]) -> None:
        self.metadata.dataset_id = val

    @property
    def source(self) -> Optional[str]:
        return self.metadata.source

    @property
    def organism(self) -> Optional[str]:
        return self.metadata.organism

    @property
    def gene(self) -> Optional[str]:
        return self.metadata.gene

    def reverse_complement(self) -> str:
        """Compute reverse complement of the sample's nucleotide sequence."""
        return compute_reverse_complement(self.sequence)

    def to_dict(self, flatten: bool = True) -> Dict[str, Any]:
        """Serialize sample to dictionary representation."""
        base = {
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "label": self.label,
            "length": self.length,
        }
        meta_dict = self.metadata.to_dict()
        if flatten:
            base.update(meta_dict)
        else:
            base["metadata"] = meta_dict
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNASample":
        """Factory method to construct DNASample from a dictionary."""
        s_id = data.get("sample_id") or data.get("id") or ""
        seq = data.get("sequence") or data.get("seq") or ""
        lbl = data.get("label") or data.get("cancer")
        meta_input = data.get("metadata")
        if isinstance(meta_input, dict):
            metadata = SampleMetadata.from_dict(meta_input)
        else:
            metadata = SampleMetadata.from_dict(data)

        return cls(
            sample_id=str(s_id),
            sequence=str(seq),
            label=str(lbl) if lbl is not None else None,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"DNASample(sample_id='{self.sample_id}', length={self.length}, "
            f"label={repr(self.label)}, patient_id={repr(self.patient_id)})"
        )


class CancerSequenceSample(DNASample):
    """Backwards-compatible subclass of DNASample for legacy callers."""

    def __init__(
        self,
        sequence: str,
        label: Optional[str] = None,
        sample_id: Optional[str] = None,
        metadata: Optional[Union[SampleMetadata, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        actual_id = sample_id if sample_id is not None else kwargs.pop("id", "")
        super().__init__(
            sample_id=str(actual_id),
            sequence=sequence,
            label=label,
            metadata=metadata,
            **kwargs,
        )

