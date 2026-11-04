"""Data ingestion module for loading sequences and datasets from various formats."""

import csv
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.utils.exceptions import DataIngestionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


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
    """Load a CSV file containing DNA sequences and metadata.

    Uses pandas if installed; falls back to standard-library csv.DictReader if unavailable.

    Args:
        file_path: Path to the CSV file.
        expected_columns: Optional list of column names to check after loading.
        **kwargs: Additional keyword arguments passed to pandas.read_csv if pandas is available.

    Returns:
        pandas.DataFrame or SimpleDataFrame instance.

    Raises:
        DataIngestionError: If file not found, empty, or missing required columns.
    """
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
        # Standard-library fallback
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
    """Parse a FASTA file using standard Python line parsing without external dependencies.

    Validates header structure and sequence completeness.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        List of dictionaries containing 'id', 'description', 'sequence', and 'len'.

    Raises:
        DataIngestionError: If file is missing, empty, malformed, or missing sequence lines.
    """
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
    """Load a FASTA file into a DataFrame or SimpleDataFrame with optional target label.

    Args:
        file_path: Path to the FASTA file.
        cancer_label: Optional cancer class label to assign to all sequences.

    Returns:
        DataFrame or SimpleDataFrame with columns ['id', 'description', 'sequence', 'len', 'cancer'].

    Raises:
        DataIngestionError: If file parsing fails or cancer_label is invalid.
    """
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
