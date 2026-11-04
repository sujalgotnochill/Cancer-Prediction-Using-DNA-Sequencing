"""Configuration loading and management module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dna_cancer.utils.exceptions import ConfigurationError


def get_project_root() -> Path:
    """Locate the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a path relative to project root or specified base directory.

    Args:
        path: Path string or Path object (absolute or relative).
        base_dir: Optional base directory. Defaults to project root.

    Returns:
        Absolute Path object normalized for the current operating system.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(base_dir) if base_dir is not None else get_project_root()
    return (base / p).resolve()


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


def _parse_simple_yaml_text(text: str) -> Dict[str, Any]:
    """Lightweight recursive/indentation fallback parser for YAML text.

    Correctly handles nested dictionaries, lists of scalars, and lists of dicts.
    """
    raw_lines = text.splitlines()
    items: List[Tuple[int, str]] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        items.append((indent, stripped))

    def parse_block(idx: int, current_indent: int) -> Tuple[Any, int]:
        if idx >= len(items):
            return {}, idx

        first_indent, first_stripped = items[idx]
        if first_stripped.startswith("- "):
            res_list: List[Any] = []
            while idx < len(items):
                indent, stripped = items[idx]
                if indent < current_indent:
                    break
                if stripped.startswith("- "):
                    val_str = stripped[2:].strip()
                    if ":" in val_str and not (val_str.startswith('"') or val_str.startswith("'")):
                        k, v = val_str.split(":", 1)
                        dict_item: Dict[str, Any] = {k.strip(): _parse_scalar(v)}
                        idx += 1
                        while idx < len(items) and items[idx][0] > indent and not items[idx][1].startswith("- "):
                            if ":" in items[idx][1]:
                                sub_k, sub_v = items[idx][1].split(":", 1)
                                dict_item[sub_k.strip()] = _parse_scalar(sub_v)
                            idx += 1
                        res_list.append(dict_item)
                    else:
                        res_list.append(_parse_scalar(val_str))
                        idx += 1
                else:
                    break
            return res_list, idx
        else:
            res_dict: Dict[str, Any] = {}
            while idx < len(items):
                indent, stripped = items[idx]
                if indent < current_indent:
                    break
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    idx += 1
                    if not v:
                        if idx < len(items) and items[idx][0] > indent:
                            sub_res, idx = parse_block(idx, items[idx][0])
                            res_dict[k] = sub_res
                        else:
                            res_dict[k] = None
                    else:
                        res_dict[k] = _parse_scalar(v)
                else:
                    idx += 1
            return res_dict, idx

    res, _ = parse_block(0, 0)
    return res if isinstance(res, dict) else {}


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file safely.

    Uses PyYAML if installed; falls back to robust standard-library parser if unavailable.
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except ImportError:
        text = file_path.read_text(encoding="utf-8")
        return _parse_simple_yaml_text(text)
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

    def resolve(self, path_attr: str) -> Path:
        """Resolve a specific directory attribute relative to project root."""
        val = getattr(self, path_attr, None)
        if val is None:
            raise ConfigurationError(f"Unknown path attribute: '{path_attr}'")
        return resolve_path(val)

    def get_data_dir(self) -> Path:
        return resolve_path(self.data_dir)

    def get_raw_data_dir(self) -> Path:
        return resolve_path(self.raw_data_dir)

    def get_interim_data_dir(self) -> Path:
        return resolve_path(self.interim_data_dir)

    def get_processed_data_dir(self) -> Path:
        return resolve_path(self.processed_data_dir)

    def get_artifacts_dir(self) -> Path:
        return resolve_path(self.artifacts_dir)

    def get_models_dir(self) -> Path:
        return resolve_path(self.models_dir)

    def get_vectorizers_dir(self) -> Path:
        return resolve_path(self.vectorizers_dir)

    def get_encoders_dir(self) -> Path:
        return resolve_path(self.encoders_dir)

    def get_evaluation_dir(self) -> Path:
        return resolve_path(self.evaluation_dir)

    def get_logs_dir(self) -> Path:
        return resolve_path(self.logs_dir)


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

    def get_raw_file_path(self) -> Path:
        return resolve_path(self.raw_file)

    def get_interim_file_path(self) -> Path:
        return resolve_path(self.interim_file)

    def get_processed_file_path(self) -> Path:
        return resolve_path(self.processed_file)


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

    def validate(self) -> None:
        """Validate integrity and biological/mathematical constraints of loaded configurations.

        Raises:
            ConfigurationError: If any configuration value is invalid or out of permissible bounds.
        """
        # Split ratio validation
        total_split = self.data.train_size + self.data.val_size + self.data.test_size
        if abs(total_split - 1.0) > 1e-4:
            raise ConfigurationError(
                f"Data split ratios must sum to 1.0; got train={self.data.train_size}, "
                f"val={self.data.val_size}, test={self.data.test_size} (sum={total_split})"
            )
        if self.data.train_size <= 0:
            raise ConfigurationError(f"train_size must be positive, got {self.data.train_size}")
        if self.data.val_size < 0:
            raise ConfigurationError(f"val_size cannot be negative, got {self.data.val_size}")
        if self.data.test_size < 0:
            raise ConfigurationError(f"test_size cannot be negative, got {self.data.test_size}")

        # Sequence constraints validation
        if self.data.min_sequence_length < 1:
            raise ConfigurationError(
                f"min_sequence_length must be >= 1, got {self.data.min_sequence_length}"
            )
        if self.data.max_sequence_length < self.data.min_sequence_length:
            raise ConfigurationError(
                f"max_sequence_length ({self.data.max_sequence_length}) cannot be less than "
                f"min_sequence_length ({self.data.min_sequence_length})"
            )
        if not self.data.allowed_bases:
            raise ConfigurationError("allowed_bases cannot be empty string")

        # Feature constraints validation
        if self.features.k < 1:
            raise ConfigurationError(f"k-mer size k must be >= 1, got {self.features.k}")
        if self.features.stride < 1:
            raise ConfigurationError(f"k-mer stride must be >= 1, got {self.features.stride}")
        if self.features.vectorizer_type.lower() not in ("count", "tfidf"):
            raise ConfigurationError(
                f"Unsupported vectorizer_type: '{self.features.vectorizer_type}'. Must be 'count' or 'tfidf'"
            )
        if (
            not isinstance(self.features.ngram_range, (tuple, list))
            or len(self.features.ngram_range) != 2
            or self.features.ngram_range[0] < 1
            or self.features.ngram_range[1] < self.features.ngram_range[0]
        ):
            raise ConfigurationError(
                f"Invalid ngram_range: {self.features.ngram_range}. Must be (min_n, max_n) with min_n <= max_n"
            )

        # Model constraints validation
        if self.model.num_classes < 2:
            raise ConfigurationError(f"num_classes must be >= 2 for classification, got {self.model.num_classes}")
        if self.model.batch_size < 1:
            raise ConfigurationError(f"batch_size must be >= 1, got {self.model.batch_size}")
        if self.model.epochs < 1:
            raise ConfigurationError(f"epochs must be >= 1, got {self.model.epochs}")
        if self.model.learning_rate <= 0:
            raise ConfigurationError(f"learning_rate must be > 0, got {self.model.learning_rate}")


def load_config(config_dir: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load, merge, and validate configuration from YAML files in the config directory.

    Args:
        config_dir: Path to directory containing YAML configs.
                    Defaults to 'configs/' at project root.

    Returns:
        Populated and validated AppConfig instance.

    Raises:
        ConfigurationError: If config_dir or any required YAML config file is missing,
                            or if loaded values violate validation constraints.
    """
    if config_dir is None:
        config_path = get_project_root() / "configs"
    else:
        config_path = resolve_path(config_dir)

    if not config_path.exists() or not config_path.is_dir():
        raise ConfigurationError(f"Configuration directory not found: {config_path}")

    # Enforce that all required configuration files are present
    required_files = {
        "project": config_path / "project_config.yaml",
        "data": config_path / "data_config.yaml",
        "features": config_path / "feature_config.yaml",
        "model": config_path / "model_config.yaml",
    }

    for name, fpath in required_files.items():
        if not fpath.exists():
            raise ConfigurationError(f"Required configuration file missing: {fpath}")

    app_config = AppConfig()

    # 1. Project & Paths
    p_data = _load_yaml(required_files["project"])
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

    # 2. Data
    d_data = _load_yaml(required_files["data"])
    dataset = d_data.get("dataset") or {}
    validation = d_data.get("validation") or {}
    split = dataset.get("split") or {}
    raw_classes = dataset.get("expected_classes")
    expected_classes = (
        list(raw_classes) if isinstance(raw_classes, (list, tuple)) else app_config.data.expected_classes
    )

    app_config.data = DataSettings(
        name=dataset.get("name", app_config.data.name),
        raw_file=dataset.get("raw_file", app_config.data.raw_file),
        interim_file=dataset.get("interim_file", app_config.data.interim_file),
        processed_file=dataset.get("processed_file", app_config.data.processed_file),
        sequence_column=dataset.get("sequence_column", app_config.data.sequence_column),
        target_column=dataset.get("target_column", app_config.data.target_column),
        id_column=dataset.get("id_column", app_config.data.id_column),
        length_column=dataset.get("length_column", app_config.data.length_column),
        expected_classes=expected_classes,
        train_size=split.get("train_size", app_config.data.train_size),
        val_size=split.get("val_size", app_config.data.val_size),
        test_size=split.get("test_size", app_config.data.test_size),
        min_sequence_length=validation.get("min_sequence_length", app_config.data.min_sequence_length),
        max_sequence_length=validation.get("max_sequence_length", app_config.data.max_sequence_length),
        allowed_bases=validation.get("allowed_bases", app_config.data.allowed_bases),
        allow_empty=validation.get("allow_empty", app_config.data.allow_empty),
    )

    # 3. Features
    f_data = _load_yaml(required_files["features"])
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

    # 4. Model
    m_data = _load_yaml(required_files["model"])
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

    # Validate all loaded configurations
    app_config.validate()

    return app_config
