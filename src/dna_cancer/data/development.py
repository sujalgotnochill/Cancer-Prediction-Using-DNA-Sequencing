"""Development cohort data access layer.

Strictly isolates development workflows (training, cross-validation, hyperparameter tuning,
and feature engineering) to the Tier 1 Development Cohort (N=960).
Prevents silent fallback to unpartitioned raw data or sealed holdout files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dna_cancer.data.ingestion import ColumnMapping, load_csv, load_samples, validate_not_sealed_vault_path
from dna_cancer.data.registry import calculate_file_checksum
from dna_cancer.data.schema import DNASample
from dna_cancer.data.splitter import split_gene_family_cv_folds
from dna_cancer.utils.exceptions import DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DEV_COHORT_PATH = "data/processed/development_cohort.csv"
DEFAULT_MANIFEST_PATH = "artifacts/vault/test_sealed_manifest.json"


def load_development_cohort(
    path: Union[str, Path] = DEFAULT_DEV_COHORT_PATH,
    verify_checksum: bool = True,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> List[DNASample]:
    """Load the Tier 1 Development Cohort (N=960) with integrity verification.

    Args:
        path: Path to the development cohort CSV file.
        verify_checksum: Whether to verify the file SHA-256 against the manifest.
        manifest_path: Path to the sealed vault manifest.

    Returns:
        List of standardized DNASample instances.

    Raises:
        PermissionError: If path references the sealed test vault.
        FileNotFoundError: If the development cohort CSV file does not exist.
        DataValidationError: If checksum mismatch or partition error is detected.
    """
    validate_not_sealed_vault_path(path)
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Development cohort file not found at '{p}'. "
            "Silent fallback to historical or sealed holdout data is strictly prohibited."
        )

    if verify_checksum:
        m_p = Path(manifest_path)
        if m_p.exists():
            try:
                with open(m_p, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                expected_sha = manifest.get("development_cohort", {}).get("sha256")
                if expected_sha:
                    actual_sha = calculate_file_checksum(p)
                    if actual_sha != expected_sha:
                        raise DataValidationError(
                            f"Development cohort checksum mismatch: expected {expected_sha}, got {actual_sha}. "
                            "Dataset may have been modified or corrupted."
                        )
            except json.JSONDecodeError as err:
                raise DataValidationError(f"Vault manifest at '{m_p}' is corrupted: {err}") from err

    mapping = ColumnMapping(
        sequence_col="sequence",
        label_col="cancer",
        id_col="Unnamed: 0",
        gene_col="gene",
        locus_col="locus",
    )
    samples = load_samples(p, file_format="csv", column_mapping=mapping)

    logger.info("Loaded %d development samples from %s (verified=%s)", len(samples), p, verify_checksum)
    return samples


def load_development_dataframe(
    path: Union[str, Path] = DEFAULT_DEV_COHORT_PATH,
    verify_checksum: bool = True,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> Any:
    """Load the Tier 1 Development Cohort as a tabular DataFrame.

    Args:
        path: Path to development cohort CSV file.
        verify_checksum: Whether to verify SHA-256 against manifest.
        manifest_path: Path to manifest.

    Returns:
        pandas.DataFrame or SimpleDataFrame.
    """
    validate_not_sealed_vault_path(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Development cohort file not found at '{p}'. "
            "Silent fallback to historical or sealed holdout data is strictly prohibited."
        )

    if verify_checksum:
        m_p = Path(manifest_path)
        if m_p.exists():
            with open(m_p, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            expected_sha = manifest.get("development_cohort", {}).get("sha256")
            if expected_sha:
                actual_sha = calculate_file_checksum(p)
                if actual_sha != expected_sha:
                    raise DataValidationError(
                        f"Development cohort checksum mismatch: expected {expected_sha}, got {actual_sha}."
                    )

    return load_csv(p)


def get_development_cv_folds(
    samples: Optional[List[DNASample]] = None,
    n_splits: int = 3,
    seed: int = 42,
) -> List[Tuple[List[DNASample], List[DNASample]]]:
    """Generate group-balanced cross-validation folds strictly on the development cohort.

    Args:
        samples: Optional pre-loaded development samples; if None, loads automatically.
        n_splits: Number of cross-validation folds (default: 3).
        seed: Random seed for group allocation.

    Returns:
        List of (train_samples, val_samples) fold tuples.
    """
    if samples is None:
        samples = load_development_cohort()
    return split_gene_family_cv_folds(samples, n_splits=n_splits, seed=seed)

