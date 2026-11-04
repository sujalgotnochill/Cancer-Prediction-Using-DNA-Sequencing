"""Persistent dataset registry and version management."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dna_cancer.config.settings import get_project_root, resolve_path
from dna_cancer.data.ingestion import ColumnMapping, FastaLabelMapper, load_samples
from dna_cancer.data.schema import DNASample
from dna_cancer.utils.exceptions import ConfigurationError, DataValidationError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_file_checksum(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for checksum integrity and version tracking.

    Args:
        file_path: Path to target file.
        chunk_size: Bytes to read per iteration (default 64KB).

    Returns:
        Hexadecimal SHA-256 string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot calculate checksum: file not found at {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DatasetMetadata:
    """Metadata container for a registered dataset version."""
    dataset_id: str
    name: str
    version: int
    file_path: str
    format: str
    schema_version: str
    sample_count: int
    checksum_sha256: str
    created_at: str
    updated_at: str
    source: Optional[str] = None
    description: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    label_mapping: Optional[Dict[str, str]] = None
    validation_report_path: Optional[str] = None
    profile_report_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Reconstruct DatasetMetadata from dictionary."""
        return cls(**data)


class DatasetRegistry:
    """Persistent, transparent JSON-backed registry for dataset management and versioning."""

    def __init__(self, registry_file: Optional[Union[str, Path]] = None) -> None:
        if registry_file is None:
            self.registry_file = get_project_root() / "artifacts" / "datasets" / "registry.json"
        else:
            self.registry_file = resolve_path(registry_file)

        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Create parent directory and registry JSON file if not existing."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({"schema_version": "1.0.0", "datasets": {}}, f, indent=2)

    def _read_registry_data(self) -> Dict[str, Any]:
        """Read full registry content."""
        self._ensure_storage()
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Error reading registry file %s: %exc. Initializing empty registry.", self.registry_file)
            return {"schema_version": "1.0.0", "datasets": {}}

    def _write_registry_data(self, data: Dict[str, Any]) -> None:
        """Write registry content atomically."""
        self._ensure_storage()
        temp_file = self.registry_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.registry_file)

    def register_dataset(
        self,
        dataset_id: str,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        file_format: Optional[str] = None,
        source: Optional[str] = None,
        description: Optional[str] = None,
        sample_count: Optional[int] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        label_mapping: Optional[Dict[str, str]] = None,
        validation_report_path: Optional[str] = None,
        profile_report_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        allow_version_increment: bool = True,
    ) -> DatasetMetadata:
        """Register a dataset file, assign versioning, compute checksum, and persist metadata.

        Args:
            dataset_id: Unique string identifier for the dataset.
            file_path: Path to dataset file on disk.
            name: Human-readable dataset name.
            file_format: Optional format string ('csv', 'fasta', 'json', 'jsonl').
            source: Dataset origin or public accession source.
            description: Detailed dataset description.
            sample_count: Total sequence samples in the dataset.
            column_mapping: Custom column mappings dictionary.
            label_mapping: Explicit FASTA label mappings dictionary.
            validation_report_path: Path to quality check report.
            profile_report_path: Path to profiling report.
            tags: List of categorization tags.
            allow_version_increment: Whether to increment version if content changes.

        Returns:
            Registered DatasetMetadata instance.
        """
        path = resolve_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot register dataset: file does not exist at {path}")

        checksum = calculate_file_checksum(path)
        fmt = (file_format or path.suffix.lstrip(".").lower()).strip()
        now = datetime.now(timezone.utc).isoformat()

        # If sample_count not supplied, count samples automatically
        if sample_count is None:
            samples = load_samples(path, file_format=fmt, default_label=None)
            sample_count = len(samples)

        data = self._read_registry_data()
        datasets_map = data.get("datasets", {})

        # Compute relative file path for portability across environments
        try:
            rel_path = str(path.relative_to(get_project_root()))
        except ValueError:
            rel_path = str(path)

        if dataset_id in datasets_map:
            versions: List[Dict[str, Any]] = datasets_map[dataset_id]
            # Check for identical checksum
            for v_entry in versions:
                if v_entry.get("checksum_sha256") == checksum:
                    logger.info("Dataset '%s' with matching checksum already registered as version %d", dataset_id, v_entry["version"])
                    return DatasetMetadata.from_dict(v_entry)

            if not allow_version_increment:
                raise ConfigurationError(
                    f"Dataset ID '{dataset_id}' already registered with different contents. "
                    "Set allow_version_increment=True to register a new version."
                )

            next_version = max(v["version"] for v in versions) + 1
        else:
            versions = []
            next_version = 1

        meta = DatasetMetadata(
            dataset_id=dataset_id,
            name=name or dataset_id,
            version=next_version,
            file_path=rel_path,
            format=fmt,
            schema_version="1.0.0",
            sample_count=sample_count,
            checksum_sha256=checksum,
            created_at=now,
            updated_at=now,
            source=source,
            description=description,
            column_mapping=column_mapping,
            label_mapping=label_mapping,
            validation_report_path=validation_report_path,
            profile_report_path=profile_report_path,
            tags=tags or [],
        )

        versions.append(meta.to_dict())
        datasets_map[dataset_id] = versions
        data["datasets"] = datasets_map
        self._write_registry_data(data)

        logger.info("Registered dataset '%s' (version=%d, samples=%d) in %s", dataset_id, next_version, sample_count, self.registry_file)
        return meta

    def get_dataset(self, dataset_id: str, version: Optional[int] = None) -> DatasetMetadata:
        """Retrieve dataset metadata for a given ID and optional version (defaults to latest)."""
        data = self._read_registry_data()
        datasets_map = data.get("datasets", {})

        if dataset_id not in datasets_map:
            raise DataValidationError(f"Dataset '{dataset_id}' not found in registry at {self.registry_file}")

        versions: List[Dict[str, Any]] = datasets_map[dataset_id]
        if not versions:
            raise DataValidationError(f"No versions registered for dataset '{dataset_id}'.")

        if version is not None:
            for v in versions:
                if v["version"] == version:
                    return DatasetMetadata.from_dict(v)
            raise DataValidationError(f"Version {version} not found for dataset '{dataset_id}'. Available: {[v['version'] for v in versions]}")

        # Return latest version
        latest = max(versions, key=lambda x: x["version"])
        return DatasetMetadata.from_dict(latest)

    def has_dataset(self, dataset_id: str, version: Optional[int] = None) -> bool:
        """Check whether dataset ID and version exist in registry."""
        try:
            self.get_dataset(dataset_id, version=version)
            return True
        except DataValidationError:
            return False

    def list_datasets(self, tag: Optional[str] = None, all_versions: bool = False) -> List[DatasetMetadata]:
        """List datasets registered in the registry."""
        data = self._read_registry_data()
        datasets_map = data.get("datasets", {})
        results: List[DatasetMetadata] = []

        for ds_id, v_list in datasets_map.items():
            if all_versions:
                targets = v_list
            else:
                targets = [max(v_list, key=lambda x: x["version"])] if v_list else []

            for entry in targets:
                meta = DatasetMetadata.from_dict(entry)
                if tag is None or tag in meta.tags:
                    results.append(meta)

        return sorted(results, key=lambda x: (x.dataset_id, x.version))

    def load_dataset_samples(
        self,
        dataset_id: str,
        version: Optional[int] = None,
    ) -> List[DNASample]:
        """Load actual sequence samples for a registered dataset."""
        meta = self.get_dataset(dataset_id, version=version)
        path = resolve_path(meta.file_path)

        col_map = ColumnMapping(**(meta.column_mapping or {})) if meta.column_mapping else None
        lbl_map = FastaLabelMapper(mapping=meta.label_mapping) if meta.label_mapping else None

        return load_samples(
            file_path=path,
            file_format=meta.format,
            column_mapping=col_map,
            label_mapper=lbl_map,
            dataset_id=meta.dataset_id,
        )

