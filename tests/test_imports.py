"""Tests for verifying package and module imports."""

import sys
import unittest
from pathlib import Path

# Add src to sys.path for local discovery
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class TestProjectImports(unittest.TestCase):
    """Test importing all modules and ensuring package contracts exist."""

    def test_import_root_package(self):
        import dna_cancer
        self.assertTrue(hasattr(dna_cancer, "__version__"))

    def test_import_config_package(self):
        from dna_cancer.config import AppConfig, load_config
        config = load_config()
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.project.name, "Cancer Prediction Using DNA Sequencing")

    def test_import_data_package(self):
        from dna_cancer.data import (
            CancerSequenceSample,
            DNADataset,
            SequenceValidator,
            load_csv,
            load_fasta_as_dataframe,
            validate_dataset_schema,
            validate_dna_sequence,
        )
        self.assertTrue(callable(validate_dna_sequence))
        self.assertTrue(callable(validate_dataset_schema))

    def test_import_preprocessing_package(self):
        from dna_cancer.preprocessing import SequenceCleaner, SequenceProcessor, clean_sequence
        self.assertTrue(callable(clean_sequence))
        self.assertTrue(callable(SequenceCleaner))
        self.assertTrue(callable(SequenceProcessor))

    def test_import_features_package(self):
        from dna_cancer.features import FeatureBuilder, KMerExtractor, extract_kmers
        self.assertTrue(callable(extract_kmers))
        self.assertTrue(callable(KMerExtractor))
        self.assertTrue(callable(FeatureBuilder))

    def test_import_models_package(self):
        from dna_cancer.models import (
            BaseDNAClassifier,
            ModelFactory,
            get_model_class,
            list_registered_models,
            register_model,
        )
        self.assertTrue(callable(register_model))
        self.assertTrue(callable(get_model_class))
        self.assertTrue(callable(list_registered_models))

    def test_import_training_package(self):
        from dna_cancer.training import Trainer, TrainerConfig, TrainingPipeline
        self.assertTrue(callable(Trainer))
        self.assertTrue(callable(TrainingPipeline))

    def test_import_evaluation_package(self):
        from dna_cancer.evaluation import Evaluator, compute_classification_metrics
        self.assertTrue(callable(compute_classification_metrics))
        self.assertTrue(callable(Evaluator))

    def test_import_prediction_package(self):
        from dna_cancer.prediction import DNAPredictor
        self.assertTrue(callable(DNAPredictor))

    def test_import_utils_package(self):
        from dna_cancer.utils import DNACancerError, get_logger
        logger = get_logger("test_logger")
        self.assertIsNotNone(logger)
        self.assertTrue(issubclass(DNACancerError, Exception))


if __name__ == "__main__":
    unittest.main()

