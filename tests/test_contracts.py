"""Tests for interface contracts, model registry, factory, dataset, and predictor."""

import sys
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.data.dataset import CancerSequenceSample, DNADataset
from dna_cancer.features.feature_builder import FeatureBuilder
from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.models.model_factory import ModelFactory
from dna_cancer.models.registry import get_model_class, list_registered_models, register_model
from dna_cancer.prediction.predictor import DNAPredictor
from dna_cancer.utils.exceptions import (
    DataValidationError,
    FeatureExtractionError,
    ModelNotFoundError,
    PredictionError,
)


class TestModelContracts(unittest.TestCase):
    """Test model interface, registry, and factory contracts."""

    def test_base_classifier_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BaseDNAClassifier()  # type: ignore

    def test_model_registry_valid_registration(self):
        # Create a compliant dummy classifier solely for testing registry contracts
        class MockClassifier(BaseDNAClassifier):
            def fit(self, X, y, validation_data=None, **kwargs):
                return self

            def predict(self, X):
                return [0] * len(X)

            def predict_proba(self, X):
                return [[1.0, 0.0]] * len(X)

            def save(self, path):
                pass

            @classmethod
            def load(cls, path, **kwargs):
                return cls()

        register_model("mock_test_classifier")(MockClassifier)
        self.assertIn("mock_test_classifier", list_registered_models())
        retrieved_cls = get_model_class("mock_test_classifier")
        self.assertIs(retrieved_cls, MockClassifier)

        # Factory create
        instance = ModelFactory.create("mock_test_classifier")
        self.assertIsInstance(instance, MockClassifier)

    def test_model_registry_rejects_non_subclass(self):
        class NonCompliantClass:
            pass

        with self.assertRaises(TypeError) as ctx:
            register_model("non_compliant")(NonCompliantClass)  # type: ignore
        self.assertIn("must inherit from BaseDNAClassifier", str(ctx.exception))

    def test_model_registry_unknown_model_raises_error(self):
        with self.assertRaises(ModelNotFoundError) as ctx:
            get_model_class("completely_unknown_model_xyz")
        self.assertIn("not found in registry", str(ctx.exception))


class TestPredictionContracts(unittest.TestCase):
    """Test prediction contracts and error enforcement."""

    def test_predictor_without_model_raises_informative_error(self):
        predictor = DNAPredictor(model=None)
        with self.assertRaises(PredictionError) as ctx:
            predictor.predict_sequence("ATCGATCG")
        self.assertIn("Model has not been initialized or loaded in DNAPredictor", str(ctx.exception))

    def test_predictor_with_invalid_sequence_raises_error(self):
        # Predictor without fitted feature builder or with invalid sequence
        class MockPredictModel(BaseDNAClassifier):
            def fit(self, X, y, validation_data=None, **kwargs):
                return self
            def predict(self, X):
                return [0]
            def predict_proba(self, X):
                return [[1.0, 0.0]]
            def save(self, path):
                pass
            @classmethod
            def load(cls, path, **kwargs):
                return cls()

        predictor = DNAPredictor(model=MockPredictModel())
        with self.assertRaises(PredictionError) as ctx:
            predictor.predict_sequence("INVALID_DNA_XYZ!")
        self.assertIn("Sequence failed validation", str(ctx.exception))

    def test_predictor_with_unfitted_features_raises_error(self):
        class MockPredictModel(BaseDNAClassifier):
            def fit(self, X, y, validation_data=None, **kwargs):
                return self
            def predict(self, X):
                return [0]
            def predict_proba(self, X):
                return [[1.0, 0.0]]
            def save(self, path):
                pass
            @classmethod
            def load(cls, path, **kwargs):
                return cls()

        fb = FeatureBuilder()  # Not fitted!
        predictor = DNAPredictor(model=MockPredictModel(), feature_builder=fb)
        with self.assertRaises(FeatureExtractionError) as ctx:
            predictor.predict_sequence("ATCGATCGATCG")
        self.assertIn("FeatureBuilder has not been fitted yet", str(ctx.exception))


class TestDatasetContracts(unittest.TestCase):
    """Test dataset abstraction contracts and error handling."""

    def test_dataset_sample_properties(self):
        sample = CancerSequenceSample(sequence="ATCGATCG", label="Colorectal Cancer", sample_id="s1")
        self.assertEqual(sample.length, 8)
        self.assertEqual(sample.sequence, "ATCGATCG")
        self.assertEqual(sample.label, "Colorectal Cancer")

    def test_dataset_container_methods(self):
        samples = [
            CancerSequenceSample(sequence="ATCGATCG", label="Cervical Cancer", sample_id="s1"),
            CancerSequenceSample(sequence="GCTAGCTA", label="Liver Cancer", sample_id="s2"),
            CancerSequenceSample(sequence="TTTTAAAA", label="Liver Cancer", sample_id="s3"),
        ]
        ds = DNADataset(samples)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].sample_id, "s1")
        self.assertEqual(ds.get_sequences(), ["ATCGATCG", "GCTAGCTA", "TTTTAAAA"])
        self.assertEqual(ds.get_labels(), ["Cervical Cancer", "Liver Cancer", "Liver Cancer"])

        dist = ds.get_class_distribution()
        self.assertEqual(dist["Cervical Cancer"], 1)
        self.assertEqual(dist["Liver Cancer"], 2)

        df = ds.to_dataframe()
        self.assertEqual(len(df), 3)

    def test_dataset_split_empty_raises_error(self):
        ds = DNADataset([])
        with self.assertRaises(DataValidationError) as ctx:
            ds.split(0.7, 0.15, 0.15)
        self.assertIn("Cannot split an empty dataset", str(ctx.exception))

    def test_dataset_split_insufficient_samples_raises_error(self):
        ds = DNADataset([
            CancerSequenceSample(sequence="ATCG", label="A"),
        ])
        with self.assertRaises(DataValidationError) as ctx:
            ds.split(0.7, 0.15, 0.15)
        self.assertIn("at least 3 samples", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

