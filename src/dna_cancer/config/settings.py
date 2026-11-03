"""Configuration loading and management module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dna_cancer.utils.exceptions import ConfigurationError


def _parse_scalar(val: str) -> Any:
    """Parse scalar YAML string into appropriate Python type."""
    val = val.strip()
    if not val:
        return None
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.startswith("[") and val.endswith("]"):
        items = [x.strip() for x in val[1:-1].split(",") if x.strip()]
        return [_parse_scalar(x) for x in items]
    val_lower = val.lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    if val_lower in ("null", "none", "~"):
        return None
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def _parse_simple_yaml(file_path: Path) -> Dict[str, Any]:
    """Lightweight recursive/indentation fallback parser for YAML files.

    Handles basic key-value mappings, nested dictionaries, and lists.
    """
    root: Dict[str, Any] = {}
    # stack holds tuples of (indent_level, container_dict_or_list, key_in_parent)
    stack: List[tuple] = [(-1, root, None)]

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))

            # Pop back up the stack to current indent level
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()

            parent_indent, current_container, _ = stack[-1]

            if stripped.startswith("- "):
                # List item
                item_val = _parse_scalar(stripped[2:])
                if isinstance(current_container, list):
                    current_container.append(item_val)
                continue

            if ":" in stripped:
                key, val = [p.strip() for p in stripped.split(":", 1)]
                if not val:
                    # Could be start of a nested dict or list
                    # Peek or default to dict
                    new_container: Any = {}
                    if isinstance(current_container, dict):
                        current_container[key] = new_container
                    stack.append((indent, new_container, key))
                else:
                    parsed_v = _parse_scalar(val)
                    if isinstance(current_container, dict):
                        current_container[key] = parsed_v

    # Convert empty dicts that actually contain lists if any
    return root


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file safely.

    Uses PyYAML if installed; falls back to standard-library parser if unavailable.
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except ImportError:
        return _parse_simple_yaml(file_path)
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML file {file_path}: {exc}") from exc


@dataclass
class ProjectSettings:
    """General project metadata and runtime settings."""
    name: str = "Cancer Prediction Using DNA Sequencing"
    version: str = "0.1.0"
    seed: int = 42
    environment: str = "development"


@dataclass
class PathsSettings:
    """Directory paths for data, artifacts, and logs."""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    interim_data_dir: str = "data/interim"
    processed_data_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    vectorizers_dir: str = "artifacts/vectorizers"
    encoders_dir: str = "artifacts/encoders"
    evaluation_dir: str = "artifacts/evaluation"
    logs_dir: str = "logs"


@dataclass
class DataSettings:
    """Dataset file locations, schema column names, and split parameters."""
    name: str = "dna_cancer_dataset"
    raw_file: str = "data/raw/dna_sequences.csv"
    interim_file: str = "data/interim/cleaned_sequences.csv"
    processed_file: str = "data/processed/features.csv"
    sequence_column: str = "sequence"
    target_column: str = "cancer"
    id_column: str = "id"
    length_column: str = "len"
    expected_classes: List[str] = field(default_factory=lambda: [
        "Cervical Cancer",
        "Colorectal Cancer",
        "Endometrial Cancer",
        "Liver Cancer",
    ])
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    min_sequence_length: int = 20
    max_sequence_length: int = 1000000
    allowed_bases: str = "ACGTN"
    allow_empty: bool = False


@dataclass
class FeatureSettings:
    """K-mer and vectorization settings."""
    k: int = 6
    stride: int = 1
    to_lower: bool = True
    as_sentence: bool = True
    vectorizer_type: str = "count"
    ngram_range: tuple = (4, 4)
    max_features: int = 10000


@dataclass
class ModelSettings:
    """Model architecture and training hyperparameters."""
    name: str = "dense_classifier"
    type: str = "neural_network"
    num_classes: int = 4
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"


@dataclass
class AppConfig:
    """Root configuration holding all settings sections."""
    project: ProjectSettings = field(default_factory=ProjectSettings)
    paths: PathsSettings = field(default_factory=PathsSettings)
    data: DataSettings = field(default_factory=DataSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    model: ModelSettings = field(default_factory=ModelSettings)


def get_project_root() -> Path:
    """Locate the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_dir: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load and merge configuration from YAML files in the config directory.

    Args:
        config_dir: Path to directory containing YAML configs.
                    Defaults to 'configs/' at project root.

    Returns:
        Populated AppConfig instance.
    """
    if config_dir is None:
        root = get_project_root()
        config_path = root / "configs"
    else:
        config_path = Path(config_dir)

    app_config = AppConfig()

    project_file = config_path / "project_config.yaml"
    if project_file.exists():
        p_data = _load_yaml(project_file)
        proj = p_data.get("project") or {}
        paths = p_data.get("paths") or {}
        app_config.project = ProjectSettings(
            name=proj.get("name", app_config.project.name),
            version=proj.get("version", app_config.project.version),
            seed=proj.get("seed", app_config.project.seed),
            environment=proj.get("environment", app_config.project.environment),
        )
        app_config.paths = PathsSettings(
            data_dir=paths.get("data_dir", app_config.paths.data_dir),
            raw_data_dir=paths.get("raw_data_dir", app_config.paths.raw_data_dir),
            interim_data_dir=paths.get("interim_data_dir", app_config.paths.interim_data_dir),
            processed_data_dir=paths.get("processed_data_dir", app_config.paths.processed_data_dir),
            artifacts_dir=paths.get("artifacts_dir", app_config.paths.artifacts_dir),
            models_dir=paths.get("models_dir", app_config.paths.models_dir),
            vectorizers_dir=paths.get("vectorizers_dir", app_config.paths.vectorizers_dir),
            encoders_dir=paths.get("encoders_dir", app_config.paths.encoders_dir),
            evaluation_dir=paths.get("evaluation_dir", app_config.paths.evaluation_dir),
            logs_dir=paths.get("logs_dir", app_config.paths.logs_dir),
        )

    data_file = config_path / "data_config.yaml"
    if data_file.exists():
        d_data = _load_yaml(data_file)
        dataset = d_data.get("dataset") or {}
        validation = d_data.get("validation") or {}
        split = dataset.get("split") or {}
        app_config.data = DataSettings(
            name=dataset.get("name", app_config.data.name),
            raw_file=dataset.get("raw_file", app_config.data.raw_file),
            interim_file=dataset.get("interim_file", app_config.data.interim_file),
            processed_file=dataset.get("processed_file", app_config.data.processed_file),
            sequence_column=dataset.get("sequence_column", app_config.data.sequence_column),
            target_column=dataset.get("target_column", app_config.data.target_column),
            id_column=dataset.get("id_column", app_config.data.id_column),
            length_column=dataset.get("length_column", app_config.data.length_column),
            expected_classes=dataset.get("expected_classes", app_config.data.expected_classes),
            train_size=split.get("train_size", app_config.data.train_size),
            val_size=split.get("val_size", app_config.data.val_size),
            test_size=split.get("test_size", app_config.data.test_size),
            min_sequence_length=validation.get("min_sequence_length", app_config.data.min_sequence_length),
            max_sequence_length=validation.get("max_sequence_length", app_config.data.max_sequence_length),
            allowed_bases=validation.get("allowed_bases", app_config.data.allowed_bases),
            allow_empty=validation.get("allow_empty", app_config.data.allow_empty),
        )

    feature_file = config_path / "feature_config.yaml"
    if feature_file.exists():
        f_data = _load_yaml(feature_file)
        kmers = f_data.get("kmers") or {}
        vec = f_data.get("vectorizer") or {}
        raw_ngram = vec.get("ngram_range", (4, 4))
        ngram = tuple(raw_ngram) if isinstance(raw_ngram, (list, tuple)) else (4, 4)
        app_config.features = FeatureSettings(
            k=kmers.get("k", app_config.features.k),
            stride=kmers.get("stride", app_config.features.stride),
            to_lower=kmers.get("to_lower", app_config.features.to_lower),
            as_sentence=kmers.get("as_sentence", app_config.features.as_sentence),
            vectorizer_type=vec.get("type", app_config.features.vectorizer_type),
            ngram_range=ngram,
            max_features=vec.get("max_features", app_config.features.max_features),
        )

    model_file = config_path / "model_config.yaml"
    if model_file.exists():
        m_data = _load_yaml(model_file)
        m = m_data.get("model") or {}
        tr = m.get("training") or {}
        app_config.model = ModelSettings(
            name=m.get("name", app_config.model.name),
            type=m.get("type", app_config.model.type),
            num_classes=m.get("num_classes", app_config.model.num_classes),
            batch_size=tr.get("batch_size", app_config.model.batch_size),
            epochs=tr.get("epochs", app_config.model.epochs),
            learning_rate=tr.get("learning_rate", app_config.model.learning_rate),
            optimizer=tr.get("optimizer", app_config.model.optimizer),
            loss=tr.get("loss", app_config.model.loss),
        )

    return app_config
