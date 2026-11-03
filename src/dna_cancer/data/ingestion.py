"""Data ingestion module for loading sequences and datasets from various formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from dna_cancer.utils.exceptions import DataIngestionError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(
    file_path: Union[str, Path],
    expected_columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Load a CSV file containing DNA sequences and metadata into a DataFrame.

    Args:
        file_path: Path to the CSV file.
        expected_columns: Optional list of column names to check after loading.
        **kwargs: Additional keyword arguments passed to pandas.read_csv.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DataIngestionError: If the file does not exist, pandas is missing, or parsing fails.
    """
    if pd is None:
        raise DataIngestionError("pandas is required for load_csv. Please install pandas.")

    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path, **kwargs)
        logger.info("Loaded CSV with %d rows and %d columns from %s", len(df), len(df.columns), path)
    except Exception as exc:
        raise DataIngestionError(f"Failed to load CSV from {path}: {exc}") from exc

    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise DataIngestionError(f"CSV file missing expected columns: {missing}")

    return df


def parse_fasta_records(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse a FASTA file using standard Python line parsing without mandatory external dependencies.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        List of dictionaries containing 'id', 'description', 'sequence', and 'len'.

    Raises:
        DataIngestionError: If the file cannot be read or is invalid FASTA.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"FASTA file not found: {path}")

    records: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    current_desc: str = ""
    current_seq_parts: List[str] = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                if line_str.startswith(">"):
                    if current_id is not None:
                        full_seq = "".join(current_seq_parts)
                        records.append({
                            "id": current_id,
                            "description": current_desc,
                            "sequence": full_seq,
                            "len": len(full_seq),
                        })
                    header = line_str[1:].strip()
                    parts = header.split(maxsplit=1)
                    current_id = parts[0]
                    current_desc = parts[1] if len(parts) > 1 else ""
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line_str)

            if current_id is not None:
                full_seq = "".join(current_seq_parts)
                records.append({
                    "id": current_id,
                    "description": current_desc,
                    "sequence": full_seq,
                    "len": len(full_seq),
                })
    except Exception as exc:
        raise DataIngestionError(f"Failed reading FASTA file at {path}: {exc}") from exc

    logger.info("Parsed %d FASTA records from %s", len(records), path)
    return records


def load_fasta_as_dataframe(
    file_path: Union[str, Path],
    cancer_label: Optional[str] = None,
) -> Any:
    """Load a FASTA file into a DataFrame, optionally appending a cancer type label.

    Args:
        file_path: Path to the FASTA file.
        cancer_label: Optional cancer class label to assign to all sequences.

    Returns:
        DataFrame with columns ['id', 'description', 'sequence', 'len'] (and 'cancer' if label given).
    """
    if pd is None:
        raise DataIngestionError("pandas is required for load_fasta_as_dataframe. Please install pandas.")

    records = parse_fasta_records(file_path)
    df = pd.DataFrame(records)
    if cancer_label is not None:
        df["cancer"] = cancer_label
    return df
