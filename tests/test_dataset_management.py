"""Unit and integration tests for dataset management, schema, quality, profiling, and registry."""

import json
import tempfile
from pathlib import Path
import unittest

from dna_cancer.data.ingestion import (
    ColumnMapping,
    FastaLabelMapper,
    load_samples,
    load_json,
    load_jsonl,
)
from dna_cancer.data.profiler import DatasetProfiler, _calculate_length_statistics, _calculate_nucleotide_frequencies
from dna_cancer.data.quality import DatasetValidator, DuplicatePolicy, QualityCheckResult
from dna_cancer.data.registry import DatasetRegistry, calculate_file_checksum
from dna_cancer.data.schema import CancerSequenceSample, DNASample, SampleMetadata, compute_reverse_complement
from dna_cancer.utils.exceptions import DataIngestionError, DataValidationError


class TestDNASampleSchema(unittest.TestCase):
    """Test unified DNA sample schema, metadata, and transformations."""

    def test_reverse_complement(self) -> None:
        seq = "ATCGNATCG"
        # A->T, T->A, C->G, G->C, N->N reversed
        # ATCGNATCG -> CGATNCGAT reversed: CGATNCGAT
        rc = compute_reverse_complement(seq)
        self.assertEqual(rc, "CGATNCGAT")
        self.assertEqual(compute_reverse_complement("AAAA"), "TTTT")
        self.assertEqual(compute_reverse_complement("GCGC"), "GCGC")

    def test_sample_creation_and_attributes(self) -> None:
        sample = DNASample(
            sample_id="SAMPLE_001",
            sequence="ATGCATGC",
            label="LUAD",
            patient_id="PAT_42",
            organism="Homo sapiens",
            gene="TP53",
            chromosome="chr17",
        )
        self.assertEqual(sample.sample_id, "SAMPLE_001")
        self.assertEqual(sample.sequence, "ATGCATGC")
        self.assertEqual(sample.label, "LUAD")
        self.assertEqual(sample.length, 8)
        self.assertEqual(sample.patient_id, "PAT_42")
        self.assertEqual(sample.organism, "Homo sapiens")
        self.assertEqual(sample.gene, "TP53")
        self.assertEqual(sample.reverse_complement(), "GCATGCAT")

    def test_sample_serialization_and_deserialization(self) -> None:
        meta = SampleMetadata(
            dataset_id="test_ds",
            patient_id="P1",
            gene="BRCA1",
            additional_metadata={"custom_flag": "valid"},
        )
        sample = DNASample(
            sample_id="S1",
            sequence="ACGTACGT",
            label="BRCA",
            metadata=meta,
        )

        flat_dict = sample.to_dict(flatten=True)
        self.assertEqual(flat_dict["sample_id"], "S1")
        self.assertEqual(flat_dict["label"], "BRCA")
        self.assertEqual(flat_dict["gene"], "BRCA1")
        self.assertEqual(flat_dict["custom_flag"], "valid")

        nested_dict = sample.to_dict(flatten=False)
        self.assertIn("metadata", nested_dict)
        self.assertEqual(nested_dict["metadata"]["gene"], "BRCA1")

        # Roundtrip from dict
        restored = DNASample.from_dict(flat_dict)
        self.assertEqual(restored.sample_id, sample.sample_id)
        self.assertEqual(restored.sequence, sample.sequence)
        self.assertEqual(restored.label, sample.label)
        self.assertEqual(restored.patient_id, "P1")

    def test_cancer_sequence_sample_backwards_compatibility(self) -> None:
        legacy = CancerSequenceSample(
            sequence="AACCGGTT",
            label="COAD",
            id="LEGACY_1",
            patient_id="P_LEGACY",
        )
        self.assertEqual(legacy.sample_id, "LEGACY_1")
        self.assertEqual(legacy.label, "COAD")
        self.assertEqual(legacy.length, 8)
        self.assertEqual(legacy.patient_id, "P_LEGACY")


class TestMultiFormatIngestion(unittest.TestCase):
    """Test ingestion from CSV, FASTA, JSON, and JSONL with mappings."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_csv_ingestion_with_custom_mapping(self) -> None:
        csv_path = self.dir_path / "custom_data.csv"
        csv_content = (
            "PATIENT_BARCODE,RAW_READ,TUMOR_TYPE,GENE_SYMBOL\n"
            "TCGA-01,ATGCATGCATGC,LUAD,EGFR\n"
            "TCGA-02,CGTACGTACGTA,BRCA,BRCA2\n"
        )
        csv_path.write_text(csv_content, encoding="utf-8")

        mapping = ColumnMapping(
            sequence_col="RAW_READ",
            label_col="TUMOR_TYPE",
            id_col="PATIENT_BARCODE",
            patient_id_col="PATIENT_BARCODE",
            gene_col="GENE_SYMBOL",
        )

        samples = load_samples(csv_path, column_mapping=mapping)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].sample_id, "TCGA-01")
        self.assertEqual(samples[0].sequence, "ATGCATGCATGC")
        self.assertEqual(samples[0].label, "LUAD")
        self.assertEqual(samples[0].gene, "EGFR")
        self.assertEqual(samples[1].label, "BRCA")

    def test_fasta_ingestion_with_explicit_label_mapping(self) -> None:
        fasta_path = self.dir_path / "sequences.fasta"
        fasta_content = (
            ">seq1 Description A\nATGCATGC\n"
            ">seq2 Description B\nCGTACGTA\n"
            ">seq3 Description C\nGGCCAATT\n"
        )
        fasta_path.write_text(fasta_content, encoding="utf-8")

        mapper = FastaLabelMapper(
            mapping={"seq1": "LUAD", "seq2": "BRCA"},
            default_label="UNANNOTATED",
        )

        samples = load_samples(fasta_path, label_mapper=mapper)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].sample_id, "seq1")
        self.assertEqual(samples[0].label, "LUAD")
        self.assertEqual(samples[1].sample_id, "seq2")
        self.assertEqual(samples[1].label, "BRCA")
        self.assertEqual(samples[2].sample_id, "seq3")
        self.assertEqual(samples[2].label, "UNANNOTATED")

    def test_fasta_mapper_from_csv_and_json(self) -> None:
        # CSV mapping file
        map_csv = self.dir_path / "mapping.csv"
        map_csv.write_text("id,label\nseqA,LUAD\nseqB,COAD\n", encoding="utf-8")
        csv_mapper = FastaLabelMapper.from_file(map_csv)
        self.assertEqual(csv_mapper.get_label("seqA"), "LUAD")
        self.assertEqual(csv_mapper.get_label("seqB"), "COAD")
        self.assertIsNone(csv_mapper.get_label("seqC"))

        # JSON mapping file
        map_json = self.dir_path / "mapping.json"
        map_json.write_text(json.dumps({"seqA": "KIRC", "seqB": "THCA"}), encoding="utf-8")
        json_mapper = FastaLabelMapper.from_file(map_json)
        self.assertEqual(json_mapper.get_label("seqA"), "KIRC")
        self.assertEqual(json_mapper.get_label("seqB"), "THCA")

    def test_json_and_jsonl_ingestion(self) -> None:
        # JSON array
        json_path = self.dir_path / "data.json"
        records = [
            {"id": "J1", "sequence": "ACGTACGT", "label": "LUAD", "patient_id": "P1"},
            {"id": "J2", "sequence": "CGTACGTA", "label": "COAD", "patient_id": "P2"},
        ]
        json_path.write_text(json.dumps(records), encoding="utf-8")
        samples_json = load_samples(json_path)
        self.assertEqual(len(samples_json), 2)
        self.assertEqual(samples_json[0].sample_id, "J1")
        self.assertEqual(samples_json[1].label, "COAD")

        # JSONL format
        jsonl_path = self.dir_path / "data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        samples_jsonl = load_samples(jsonl_path)
        self.assertEqual(len(samples_jsonl), 2)
        self.assertEqual(samples_jsonl[0].sequence, "ACGTACGT")

    def test_ingestion_error_handling(self) -> None:
        with self.assertRaises(DataIngestionError):
            load_samples(self.dir_path / "non_existent.csv")

        empty_file = self.dir_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with self.assertRaises(DataIngestionError):
            load_jsonl(empty_file)

        bad_format = self.dir_path / "unsupported.xyz"
        bad_format.write_text("dummy", encoding="utf-8")
        with self.assertRaises(DataIngestionError):
            load_samples(bad_format)


class TestDatasetValidatorAndQuality(unittest.TestCase):
    """Test validation checks, duplicate policies, and anomaly detection."""

    def setUp(self) -> None:
        self.validator = DatasetValidator(
            expected_classes=["LUAD", "BRCA", "COAD"],
            min_length=10,
            max_length=1000,
            duplicate_policy=DuplicatePolicy.REPORT_ONLY,
        )

    def test_clean_valid_dataset(self) -> None:
        samples = [
            DNASample("S1", "ACGTACGTACGT", "LUAD"),
            DNASample("S2", "TGCATGCATGCA", "BRCA"),
            DNASample("S3", "CGATCGATCGAT", "COAD"),
        ]
        result = self.validator.validate_dataset(samples)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.valid_samples, 3)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_characters_and_missing_labels(self) -> None:
        samples = [
            DNASample("S1", "ACGT1234ACGT", "LUAD"),  # Numbers in sequence
            DNASample("S2", "ACGTACGTACGT", None),    # Missing label
            DNASample("S3", "ACGT", "LUAD"),          # Too short (< 10)
            DNASample("S4", "ACGTACGTACGT", "UNKNOWN_CANCER"), # Unexpected class
        ]
        result = self.validator.validate_dataset(samples)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.invalid_sequences), 2)  # invalid chars & too short
        self.assertEqual(len(result.missing_labels), 1)
        self.assertEqual(len(result.unexpected_labels), 1)

    def test_duplicate_sequences_and_conflicting_labels(self) -> None:
        # S1 and S2 are identical sequence with conflicting labels (LUAD vs BRCA)
        # S4 is reverse complement of seq1 (non-palindromic)
        seq1 = "AACCGGTTTT"
        rc_seq1 = compute_reverse_complement(seq1)

        samples = [
            DNASample("S1", seq1, "LUAD"),
            DNASample("S2", seq1, "BRCA"),  # Exact duplicate & conflict!
            DNASample("S3", seq1, "LUAD"),  # Duplicate with same label
            DNASample("S4", rc_seq1, "LUAD"), # Reverse complement of seq1
        ]
        result = self.validator.validate_dataset(samples)

        # Duplicate sequences
        self.assertEqual(len(result.duplicate_sequences), 1)
        self.assertEqual(result.duplicate_sequences[0]["count"], 3)

        # Conflicting labels
        self.assertEqual(len(result.conflicting_labels), 1)
        conflict = result.conflicting_labels[0]
        self.assertIn("LUAD", conflict["labels"])
        self.assertIn("BRCA", conflict["labels"])

        # Reverse complement detected
        self.assertEqual(len(result.reverse_complement_matches), 1)

    def test_duplicate_policy_filtering(self) -> None:
        samples = [
            DNASample("S1", "ACGTACGTACGT", "LUAD"),
            DNASample("S2", "ACGTACGTACGT", "LUAD"),
            DNASample("S3", "TGCATGCATGCA", "BRCA"),
        ]

        # Policy: REPORT_ONLY
        val_report = DatasetValidator(duplicate_policy=DuplicatePolicy.REPORT_ONLY)
        kept_report = val_report.apply_duplicate_policy(samples)
        self.assertEqual(len(kept_report), 3)

        # Policy: KEEP_FIRST
        val_keep_first = DatasetValidator(duplicate_policy=DuplicatePolicy.KEEP_FIRST)
        kept_first = val_keep_first.apply_duplicate_policy(samples)
        self.assertEqual(len(kept_first), 2)
        self.assertEqual([s.sample_id for s in kept_first], ["S1", "S3"])

        # Policy: REMOVE_ALL
        val_remove_all = DatasetValidator(duplicate_policy=DuplicatePolicy.REMOVE_ALL)
        kept_none = val_remove_all.apply_duplicate_policy(samples)
        self.assertEqual(len(kept_none), 1)
        self.assertEqual(kept_none[0].sample_id, "S3")

    def test_quality_report_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            samples = [
                DNASample("S1", "ACGTACGTACGT", "LUAD"),
                DNASample("S2", "TGCATGCATGCA", "BRCA"),
            ]
            result = self.validator.validate_dataset(samples)
            json_p, md_p = result.save_reports(tmp_dir, "test_eval")
            self.assertTrue(json_p.exists())
            self.assertTrue(md_p.exists())
            self.assertIn("Dataset Quality & Validation Report", md_p.read_text(encoding="utf-8"))


class TestDatasetProfiler(unittest.TestCase):
    """Test statistical profiling, length distributions, and nucleotide composition."""

    def test_length_statistics_helper(self) -> None:
        lengths = [10, 20, 30, 40, 50]
        stats = _calculate_length_statistics(lengths)
        self.assertEqual(stats["min"], 10)
        self.assertEqual(stats["max"], 50)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)

    def test_nucleotide_frequency_helper(self) -> None:
        seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
        # Total bases = 16, GC = 8 -> 50%
        nuc = _calculate_nucleotide_frequencies(seqs)
        self.assertEqual(nuc["total_bases"], 16)
        self.assertEqual(nuc["gc_content"], 50.0)
        self.assertEqual(nuc["base_counts"]["A"], 4)
        self.assertEqual(nuc["base_counts"]["C"], 4)

    def test_dataset_profiler_orchestration(self) -> None:
        samples = [
            DNASample("S1", "ACGTACGTACGT", "LUAD"),
            DNASample("S2", "ACGTACGTACGT", "LUAD"), # duplicate sequence
            DNASample("S3", "GGCCGGCCGGCC", "BRCA"),
        ]
        profiler = DatasetProfiler()
        profile = profiler.profile(samples, dataset_id="synthetic_profile")

        self.assertEqual(profile.total_samples, 3)
        self.assertEqual(profile.unique_sequences, 2)
        self.assertEqual(profile.length_stats["min"], 12)
        self.assertEqual(profile.length_stats["max"], 12)

        p_dict = profile.to_dict()
        self.assertEqual(p_dict["dataset_id"], "synthetic_profile")
        self.assertEqual(p_dict["total_samples"], 3)
        self.assertIn("class_distribution", p_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_p, md_p = profile.save_reports(tmp_dir)
            self.assertTrue(json_p.exists())
            self.assertTrue(md_p.exists())
            content = md_p.read_text(encoding="utf-8")
            self.assertIn("Dataset Statistical Profile", content)
            self.assertIn("Sequence Length Statistics", content)


class TestDatasetRegistry(unittest.TestCase):
    """Test persistent dataset registry, version tracking, and SHA-256 checksumming."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.registry_file = Path(self.tmp_dir.name) / "registry.json"
        self.registry = DatasetRegistry(registry_file=self.registry_file)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_checksum_calculation(self) -> None:
        test_file = Path(self.tmp_dir.name) / "test.txt"
        test_file.write_text("antigravity dna test content", encoding="utf-8")
        c1 = calculate_file_checksum(test_file)
        c2 = calculate_file_checksum(test_file)
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 64)

    def test_register_and_retrieve_dataset(self) -> None:
        csv_file = Path(self.tmp_dir.name) / "sample_data.csv"
        csv_file.write_text("id,sequence,cancer\nS1,ACGTACGTACGT,LUAD\nS2,TGCATGCATGCA,BRCA\n", encoding="utf-8")

        meta = self.registry.register_dataset(
            dataset_id="test_dataset_1",
            file_path=csv_file,
            name="Test Dataset 1",
            source="Synthetic Provider",
            description="Unit test dataset",
            tags=["synthetic", "test"],
        )

        self.assertEqual(meta.dataset_id, "test_dataset_1")
        self.assertEqual(meta.version, 1)
        self.assertEqual(meta.sample_count, 2)
        self.assertEqual(meta.format, "csv")

        # Retrieve metadata
        retrieved = self.registry.get_dataset("test_dataset_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Dataset 1")

        # List datasets
        ds_list = self.registry.list_datasets()
        self.assertEqual(len(ds_list), 1)
        self.assertEqual(ds_list[0].dataset_id, "test_dataset_1")

        # Load samples through registry
        samples = self.registry.load_dataset_samples("test_dataset_1")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].sample_id, "S1")
        self.assertEqual(samples[0].label, "LUAD")

    def test_version_increment_on_content_change(self) -> None:
        csv_file = Path(self.tmp_dir.name) / "versioned.csv"
        csv_file.write_text("id,sequence,cancer\nS1,ACGTACGTACGT,LUAD\n", encoding="utf-8")

        m1 = self.registry.register_dataset(dataset_id="v_dataset", file_path=csv_file)
        self.assertEqual(m1.version, 1)

        # Modify content and re-register
        csv_file.write_text("id,sequence,cancer\nS1,ACGTACGTACGT,LUAD\nS2,TGCATGCATGCA,BRCA\n", encoding="utf-8")
        m2 = self.registry.register_dataset(dataset_id="v_dataset", file_path=csv_file)
        self.assertEqual(m2.version, 2)
        self.assertEqual(m2.sample_count, 2)


if __name__ == "__main__":
    unittest.main()
