"""Tests for configuration loading, path resolution, and validation constraints."""

import sys
import tempfile
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.config.settings import (
    AppConfig,
    DataSettings,
    FeatureSettings,
    ModelSettings,
    PathsSettings,
    ProjectSettings,
    get_project_root,
    load_config,
    resolve_path,
)
from dna_cancer.utils.exceptions import ConfigurationError


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading from project configs/ directory."""

    def test_load_default_config(self):
        config = load_config()
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.project.name, "Cancer Prediction Using DNA Sequencing")
        self.assertEqual(config.data.sequence_column, "sequence")
        self.assertEqual(config.data.target_column, "cancer")
        self.assertIsInstance(config.data.expected_classes, list)
        self.assertIn("Colorectal Cancer", config.data.expected_classes)
        self.assertEqual(config.features.k, 6)
        self.assertEqual(config.features.stride, 1)

    def test_path_resolution(self):
        root = get_project_root()
        raw_dir = resolve_path("data/raw")
        self.assertEqual(raw_dir, root / "data" / "raw")

        config = load_config()
        self.assertEqual(config.paths.get_raw_data_dir(), root / "data" / "raw")
        self.assertEqual(config.paths.get_models_dir(), root / "artifacts" / "models")
        self.assertEqual(config.data.get_raw_file_path(), root / "data" / "raw" / "dna_sequences.csv")

    def test_missing_config_directory_raises_error(self):
        non_existent = Path("non_existent_config_dir_xyz_123")
        with self.assertRaises(ConfigurationError) as ctx:
            load_config(non_existent)
        self.assertIn("Configuration directory not found", str(ctx.exception))

    def test_missing_required_config_file_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create only project_config.yaml, omit the rest
            (tmp_path / "project_config.yaml").write_text("project:\n  name: Test\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(tmp_path)
            self.assertIn("Required configuration file missing", str(ctx.exception))


class TestConfigurationValidation(unittest.TestCase):
    """Test validation constraints on AppConfig."""

    def test_valid_default_passes_validation(self):
        config = AppConfig()
        # Should not raise
        config.validate()

    def test_invalid_split_sum_rejected(self):
        config = AppConfig()
        config.data.train_size = 0.80
        config.data.val_size = 0.20
        config.data.test_size = 0.20  # Sum = 1.20
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("must sum to 1.0", str(ctx.exception))

    def test_negative_split_rejected(self):
        config = AppConfig()
        config.data.train_size = 1.10
        config.data.val_size = -0.10
        config.data.test_size = 0.0
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_invalid_sequence_lengths_rejected(self):
        config = AppConfig()
        config.data.min_sequence_length = 100
        config.data.max_sequence_length = 50  # max < min
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("cannot be less than min_sequence_length", str(ctx.exception))

    def test_invalid_kmer_parameters_rejected(self):
        config = AppConfig()
        config.features.k = 0
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("k-mer size k must be >= 1", str(ctx.exception))

        config.features.k = 6
        config.features.stride = 0
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("k-mer stride must be >= 1", str(ctx.exception))

    def test_invalid_vectorizer_type_rejected(self):
        config = AppConfig()
        config.features.vectorizer_type = "invalid_vectorizer"
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("Unsupported vectorizer_type", str(ctx.exception))

    def test_invalid_model_parameters_rejected(self):
        config = AppConfig()
        config.model.num_classes = 1
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("num_classes must be >= 2", str(ctx.exception))

        config.model.num_classes = 4
        config.model.epochs = 0
        with self.assertRaises(ConfigurationError) as ctx:
            config.validate()
        self.assertIn("epochs must be >= 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

