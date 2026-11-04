"""Data ingestion module supporting CSV, FASTA, JSON, and JSONL formats."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.data.schema import CancerSequenceSample, DNASample, SampleMetadata
from dna_cancer.utils.exceptions import DataIngestionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnMapping:
    """Configurable column name mapping for tabular or JSON datasets.

    Allows ingesting external datasets that use custom column names (e.g., 'DNA_SEQ', 'DIAGNOSIS').
    """
    sequence_col: str = "sequence"
    label_col: Optional[str] = "cancer"
    id_col: Optional[str] = "id"
    patient_id_col: Optional[str] = "patient_id"
    organism_col: Optional[str] = "organism"
    gene_col: Optional[str] = "gene"
    locus_col: Optional[str] = "locus"
    chromosome_col: Optional[str] = "chromosome"
    source_col: Optional[str] = "source"
    platform_col: Optional[str] = "sequencing_platform"
    custom_mappings: Dict[str, str] = field(default_factory=dict)


class FastaLabelMapper:
    """Explicit mapping from FASTA record identifiers to cancer labels.

    FASTA files do not inherently store phenotypic or disease labels. This mapper enforces
    explicit, documented label association without inferring labels from sequence IDs.
    """

    def __init__(
        self,
        mapping: Optional[Dict[str, str]] = None,
        default_label: Optional[str] = None,
    ) -> None:
        self.mapping = mapping or {}
        self.default_label = default_label

    def get_label(self, record_id: str) -> Optional[str]:
        """Look up cancer label for a FASTA record identifier."""
        if record_id in self.mapping:
            return self.mapping[record_id]
        return self.default_label

    @classmethod
    def from_dict(cls, mapping_dict: Dict[str, str], default_label: Optional[str] = None) -> "FastaLabelMapper":
        return cls(mapping=mapping_dict, default_label=default_label)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        id_col: str = "id",
        label_col: str = "label",
        default_label: Optional[str] = None,
    ) -> "FastaLabelMapper":
        """Load ID-to-label mapping from a CSV or JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise DataIngestionError(f"FASTA label mapping file not found: {path}")

        mapping: Dict[str, str] = {}
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    mapping = {str(k): str(v) for k, v in data.items()}
                else:
                    raise DataIngestionError(f"Expected JSON object in {path}, got {type(data).__name__}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if id_col in row and label_col in row:
                        mapping[str(row[id_col])] = str(row[label_col])

        logger.info("Loaded %d explicit FASTA label mappings from %s", len(mapping), path)
        return cls(mapping=mapping, default_label=default_label)


class SimpleDataFrame:
    """Lightweight tabular container providing a duck-typed DataFrame interface.

    Allows basic tabular operations when pandas is not installed in the environment.
    """

    def __init__(self, data: Union[List[Dict[str, Any]], Dict[str, List[Any]]]) -> None:
        if isinstance(data, dict):
            self._cols = list(data.keys())
            row_count = len(next(iter(data.values()))) if data else 0
            self._rows = [
                {col: data[col][i] for col in self._cols}
                for i in range(row_count)
            ]
        elif isinstance(data, list):
            self._rows = data
            self._cols = list(data[0].keys()) if data else []
        else:
            raise DataIngestionError(f"Unsupported data type for SimpleDataFrame: {type(data)}")

    @property
    def columns(self) -> List[str]:
        return list(self._cols)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> List[Any]:
        if key not in self._cols:
            raise KeyError(f"Column '{key}' not found.")
        return [r[key] for r in self._rows]

    def iterrows(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for idx, row in enumerate(self._rows):
            yield idx, row

    def to_dict(self, orient: str = "records") -> Any:
        if orient == "records":
            return [dict(r) for r in self._rows]
        raise NotImplementedError(f"Orient '{orient}' not implemented in SimpleDataFrame.")


def load_csv(
    file_path: Union[str, Path],
    expected_columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Load a CSV file containing DNA sequences and metadata."""
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"CSV file not found: {path}")

    if path.stat().st_size == 0:
        raise DataIngestionError(f"CSV file is empty: {path}")

    if pd is not None:
        try:
            df = pd.read_csv(path, **kwargs)
            logger.info("Loaded CSV with %d rows and %d columns from %s", len(df), len(df.columns), path)
        except Exception as exc:
            raise DataIngestionError(f"Failed to load CSV from {path}: {exc}") from exc
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not reader.fieldnames:
                    raise DataIngestionError(f"CSV file contains no header row: {path}")
                df = SimpleDataFrame(rows)
            logger.info("Loaded CSV with %d rows using standard-library reader from %s", len(df), path)
        except Exception as exc:
            if isinstance(exc, DataIngestionError):
                raise
            raise DataIngestionError(f"Failed reading CSV at {path}: {exc}") from exc

    if len(df) == 0:
        raise DataIngestionError(f"CSV file contains 0 data rows: {path}")

    if expected_columns:
        present_cols = set(df.columns)
        missing = [col for col in expected_columns if col not in present_cols]
        if missing:
            raise DataIngestionError(f"CSV file missing expected columns: {missing}. Present: {list(present_cols)}")

    return df


def parse_fasta_records(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse a FASTA file using standard Python line parsing without external dependencies."""
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"FASTA file not found: {path}")

    if path.stat().st_size == 0:
        raise DataIngestionError(f"FASTA file is empty: {path}")

    records: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    current_desc: str = ""
    current_seq_parts: List[str] = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                if line_str.startswith(">"):
                    if current_id is not None:
                        full_seq = "".join(current_seq_parts)
                        if not full_seq:
                            raise DataIngestionError(
                                f"Malformed FASTA file: record '{current_id}' has no sequence data (line {line_no} in {path})"
                            )
                        records.append({
                            "id": current_id,
                            "description": current_desc,
                            "sequence": full_seq,
                            "len": len(full_seq),
                        })

                    header_content = line_str[1:].strip()
                    if not header_content:
                        raise DataIngestionError(
                            f"Malformed FASTA file: empty record header '>' at line {line_no} in {path}"
                        )
                    parts = header_content.split(maxsplit=1)
                    current_id = parts[0]
                    current_desc = parts[1] if len(parts) > 1 else ""
                    current_seq_parts = []
                else:
                    if current_id is None:
                        raise DataIngestionError(
                            f"Malformed FASTA file: sequence data encountered before any header (line {line_no} in {path})"
                        )
                    current_seq_parts.append(line_str)

            if current_id is not None:
                full_seq = "".join(current_seq_parts)
                if not full_seq:
                    raise DataIngestionError(
                        f"Malformed FASTA file: trailing record '{current_id}' has no sequence data in {path}"
                    )
                records.append({
                    "id": current_id,
                    "description": current_desc,
                    "sequence": full_seq,
                    "len": len(full_seq),
                })
    except Exception as exc:
        if isinstance(exc, DataIngestionError):
            raise
        raise DataIngestionError(f"Failed reading FASTA file at {path}: {exc}") from exc

    if not records:
        raise DataIngestionError(f"No FASTA records found in {path}")

    logger.info("Parsed %d FASTA records from %s", len(records), path)
    return records


def load_fasta_as_dataframe(
    file_path: Union[str, Path],
    cancer_label: Optional[str] = None,
) -> Any:
    """Load a FASTA file into a DataFrame or SimpleDataFrame with optional target label."""
    if cancer_label is not None:
        if not isinstance(cancer_label, str) or not cancer_label.strip():
            raise DataIngestionError(f"cancer_label must be a non-empty string, got {cancer_label}")

    records = parse_fasta_records(file_path)
    if cancer_label is not None:
        for r in records:
            r["cancer"] = cancer_label

    if pd is not None:
        return pd.DataFrame(records)
    return SimpleDataFrame(records)


def load_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from a standard JSON file.

    Accepts either a JSON array of objects `[{...}]` or an object containing a 'samples' or 'records' key.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"JSON file not found: {path}")
    if path.stat().st_size == 0:
        raise DataIngestionError(f"JSON file is empty: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise DataIngestionError(f"Failed to parse JSON file {path}: {exc}") from exc

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        if "records" in data and isinstance(data["records"], list):
            return data["records"]
        return [data]
    else:
        raise DataIngestionError(f"Unexpected JSON structure in {path}: expected array or dict.")


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from a newline-delimited JSON (JSONL) file."""
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"JSONL file not found: {path}")
    if path.stat().st_size == 0:
        raise DataIngestionError(f"JSONL file is empty: {path}")

    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        raise DataIngestionError(f"Line {line_no} in {path} is not a JSON object.")
                except json.JSONDecodeError as err:
                    raise DataIngestionError(f"Invalid JSON at line {line_no} in {path}: {err}") from err
    except Exception as exc:
        if isinstance(exc, DataIngestionError):
            raise
        raise DataIngestionError(f"Error reading JSONL file {path}: {exc}") from exc

    if not records:
        raise DataIngestionError(f"No records found in JSONL file: {path}")

    return records


def load_samples(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    column_mapping: Optional[ColumnMapping] = None,
    label_mapper: Optional[Union[FastaLabelMapper, Dict[str, str]]] = None,
    default_label: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> List[DNASample]:
    """Universal sequence loader ingesting CSV, FASTA, JSON, or JSONL into standardized DNASample objects.

    Args:
        file_path: Path to dataset file.
        file_format: Optional format identifier ('csv', 'fasta', 'json', 'jsonl'). Auto-detected if None.
        column_mapping: Optional ColumnMapping for renaming custom columns.
        label_mapper: Optional FastaLabelMapper or dict mapping record IDs to labels.
        default_label: Fallback label assigned if record does not have a mapped label.
        dataset_id: Optional dataset identifier tagged onto sample metadata.

    Returns:
        List of standardized DNASample instances.
    """
    path = Path(file_path)
    fmt = (file_format or path.suffix.lstrip(".").lower()).strip()
    mapping = column_mapping or ColumnMapping()

    # Normalize fasta format variants
    if fmt in ("fa", "fna", "fasta"):
        fmt = "fasta"
    elif fmt in ("tsv", "csv"):
        fmt = "csv"

    samples: List[DNASample] = []

    if fmt == "fasta":
        records = parse_fasta_records(path)
        mapper = (
            label_mapper
            if isinstance(label_mapper, FastaLabelMapper)
            else (FastaLabelMapper(mapping=label_mapper, default_label=default_label) if label_mapper else FastaLabelMapper(default_label=default_label))
        )

        for rec in records:
            r_id = rec["id"]
            lbl = mapper.get_label(r_id)
            meta = SampleMetadata(
                dataset_id=dataset_id,
                additional_metadata={"description": rec.get("description", "")},
            )
            samples.append(DNASample(sample_id=r_id, sequence=rec["sequence"], label=lbl, metadata=meta))

    elif fmt in ("csv", "json", "jsonl"):
        if fmt == "csv":
            df = load_csv(path)
            rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else [dict(r) for _, r in df.iterrows()]
        elif fmt == "json":
            rows = load_json(path)
        else:
            rows = load_jsonl(path)

        for idx, row in enumerate(rows):
            id_key = mapping.id_col or "id"
            raw_id = row.get(id_key)
            if raw_id is None and id_key == "id":
                raw_id = row.get("sample_id")
            s_id = str(raw_id) if raw_id is not None else f"sample_{idx}"

            seq = str(row.get(mapping.sequence_col, ""))

            raw_label = row.get(mapping.label_col) if mapping.label_col else None
            if raw_label is None:
                if mapping.label_col == "cancer":
                    raw_label = row.get("label")
                elif mapping.label_col == "label":
                    raw_label = row.get("cancer")
            lbl = str(raw_label) if raw_label is not None and str(raw_label) != "nan" else default_label

            meta = SampleMetadata(
                dataset_id=dataset_id or row.get("dataset_id"),
                source=row.get(mapping.source_col) if mapping.source_col else row.get("source"),
                organism=row.get(mapping.organism_col) if mapping.organism_col else row.get("organism"),
                gene=row.get(mapping.gene_col) if mapping.gene_col else row.get("gene"),
                locus=row.get(mapping.locus_col) if mapping.locus_col else row.get("locus"),
                chromosome=row.get(mapping.chromosome_col) if mapping.chromosome_col else row.get("chromosome"),
                sequencing_platform=row.get(mapping.platform_col) if mapping.platform_col else row.get("sequencing_platform"),
                patient_id=row.get(mapping.patient_id_col) if mapping.patient_id_col else row.get("patient_id"),
            )

            # Map any custom columns
            for custom_key, target_field in mapping.custom_mappings.items():
                if custom_key in row:
                    meta.additional_metadata[target_field] = row[custom_key]

            samples.append(DNASample(sample_id=s_id, sequence=seq, label=lbl, metadata=meta))

    else:
        raise DataIngestionError(f"Unsupported dataset format: '{fmt}'. Supported: csv, fasta, json, jsonl")

    logger.info("Ingested %d DNASample objects from %s (format=%s)", len(samples), path, fmt)
    return samples
