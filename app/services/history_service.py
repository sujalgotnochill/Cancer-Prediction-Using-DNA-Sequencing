"""Local SQLite persistence service for sequence classification history.

Stores minimal metadata (timestamp, predicted class, sequence length, GC content, model ID).
Complete DNA sequences and patient information are strictly NEVER stored.
Parameter-safe queries are used for all database operations.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("artifacts/history/prediction_history.db")


class HistoryService:
    """Manages local SQLite prediction history."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Obtain a SQLite database connection with row factory enabled."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize prediction history table schema if not already present."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        predicted_class TEXT NOT NULL,
                        sequence_length INTEGER NOT NULL,
                        gc_content REAL NOT NULL,
                        model_identifier TEXT NOT NULL,
                        record_label TEXT
                    );
                    """
                )
                conn.commit()
            logger.info("Initialized local SQLite history database at %s", self.db_path)
        except Exception as exc:
            logger.error("Failed to initialize SQLite history database: %s", exc)
            raise

    def record_prediction(
        self,
        predicted_class: str,
        sequence_length: int,
        gc_content: float,
        model_identifier: str,
        record_label: Optional[str] = None,
    ) -> int:
        """Record an inference event with parameter-safe SQL.

        Args:
            predicted_class: Historical dataset label predicted by model.
            sequence_length: Nucleotide count of cleaned input.
            gc_content: Percentage GC content (0.0 to 100.0).
            model_identifier: Model name/version string.
            record_label: Optional non-sensitive label/identifier (e.g. FASTA header).

        Returns:
            The inserted row id.
        """
        # Truncate label if excessively long to prevent database bloat
        safe_label = (record_label[:120] if record_label else None)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_history (
                    timestamp, predicted_class, sequence_length, gc_content, model_identifier, record_label
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    timestamp_str,
                    str(predicted_class),
                    int(sequence_length),
                    round(float(gc_content), 2),
                    str(model_identifier),
                    safe_label,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info("Recorded prediction entry id=%d (%s, len=%d)", row_id, predicted_class, sequence_length)
            return row_id

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent prediction history entries in descending order."""
        safe_limit = max(1, min(int(limit), 500))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, predicted_class, sequence_length, gc_content, model_identifier, record_label
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?;
                """,
                (safe_limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count_predictions(self) -> int:
        """Return total count of recorded predictions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prediction_history;")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def clear_history(self) -> int:
        """Clear all prediction records upon user confirmation.

        Returns:
            The number of records deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prediction_history;")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM prediction_history;")
            conn.commit()
            logger.info("Cleared %d prediction history records from database", count)
            return int(count)

