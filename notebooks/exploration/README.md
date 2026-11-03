# Exploratory Notebooks

This directory is designated for interactive data exploration, sequence analysis, and feature visualization notebooks.

- Legacy second-year college experiment notebooks have been preserved in `legacy/old_notebooks/`.
- New notebooks should follow the modular package imports:
  ```python
  from dna_cancer.config import load_config
  from dna_cancer.features import KMerExtractor
  from dna_cancer.data import load_csv
  ```

