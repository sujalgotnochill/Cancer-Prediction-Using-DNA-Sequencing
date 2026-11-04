"""Integration tests for complete data-processing workflows with synthetic datasets."""

import sys
import tempfile
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dna_cancer.config.settings import load_config
from dna_cancer.data.dataset import DNADataset
from dna_cancer.data.ingestion import load_csv, load_fasta_as_dataframe, parse_fasta_records
from dna_cancer.data.validation import validate_dataset_records, validate_dataset_schema
from dna_cancer.features.kmer_extractor import KMerExtractor
from dna_cancer.preprocessing.sequence_processor import SequenceProcessor


class TestDataProcessingIntegration(unittest.TestCase):
    """Integration tests verifying end-to-end data processing workflow."""

    def setUp(self):
        self.config = load_config()

    def test_end_to_end_csv_workflow(self):
        # 1. Prepare synthetic dataset with varied formatting (whitespace, casing, valid classes)
        csv_data = (
            "id,sequence,cancer\n"
            "sample_1,  atgcgatcgatcgatc  ,Colorectal Cancer\n"
            "sample_2,GCTAGCTAGCTAGCTA,Liver Cancer\n"
            "sample_3,  tccgaatcggatcga\t,Cervical Cancer\n"
            "sample_4,AAAATTTTCCCCGGGG,Endometrial Cancer\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_data)
            tmp_path = tmp.name

        try:
            # 2. Ingestion
            df = load_csv(tmp_path, expected_columns=["sequence", "cancer"])
            self.assertEqual(len(df), 4)

            # 3. Schema & Record Validation
            self.assertTrue(validate_dataset_schema(df, required_columns=["sequence", "cancer"]))
            summary = validate_dataset_records(
                df,
                sequence_column="sequence",
                target_column="cancer",
                expected_classes=self.config.data.expected_classes,
                raise_exception=True,
            )
            self.assertEqual(summary["valid_records"], 4)

            # 4. Preprocessing & Normalization
            processor = SequenceProcessor()
            df_cleaned = processor.process_dataframe(df, sequence_column="sequence")
            self.assertEqual(len(df_cleaned), 4)

            # 5. Dataset container representation
            dataset = DNADataset.from_dataframe(
                df_cleaned,
                sequence_column="sequence",
                target_column="cancer",
            )
            self.assertEqual(len(dataset), 4)
            dist = dataset.get_class_distribution()
            self.assertEqual(len(dist), 4)
            self.assertIn("Colorectal Cancer", dist)
            self.assertIn("Liver Cancer", dist)

            # 6. K-mer feature tokenization
            kmer_extractor = KMerExtractor(
                k=self.config.features.k,
                stride=self.config.features.stride,
                to_lower=self.config.features.to_lower,
                as_sentence=self.config.features.as_sentence,
            )
            sequences = dataset.get_sequences()
            sentences = kmer_extractor.extract_batch(sequences)
            self.assertEqual(len(sentences), 4)

            for sentence in sentences:
                self.assertIsInstance(sentence, str)
                words = sentence.split()
                self.assertGreater(len(words), 0)
                # Verify each word is of length k=6
                for word in words:
                    self.assertEqual(len(word), 6)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_end_to_end_fasta_workflow(self):
        # 1. Prepare synthetic multi-record FASTA
        fasta_data = (
            ">col_1 Colorectal Biomarker Fragment\n"
            "ATCGATCGATCGATCG\n"
            "GCTAGCTAGCTAGCTA\n"
            ">col_2 Colorectal Control Fragment\n"
            "AAAAATTTTTCCCCCG\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tmp:
            tmp.write(fasta_data)
            tmp_path = tmp.name

        try:
            # 2. Parse FASTA records
            records = parse_fasta_records(tmp_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["id"], "col_1")
            self.assertEqual(records[0]["len"], 32)

            # 3. Load as DataFrame with target class label
            df = load_fasta_as_dataframe(tmp_path, cancer_label="Colorectal Cancer")
            self.assertEqual(len(df), 2)
            self.assertIn("cancer", df.columns)

            # 4. Clean and Extract K-mers
            processor = SequenceProcessor()
            df_cleaned = processor.process_dataframe(df, sequence_column="sequence")
            self.assertEqual(len(df_cleaned), 2)

            kmer_extractor = KMerExtractor(k=6, stride=1, as_sentence=True)
            kmer_sentences = kmer_extractor.extract_batch(df_cleaned["sequence"])
            self.assertEqual(len(kmer_sentences), 2)
            # Record 1 length is 32 -> 32 - 6 + 1 = 27 kmers
            self.assertEqual(len(kmer_sentences[0].split()), 27)

        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
