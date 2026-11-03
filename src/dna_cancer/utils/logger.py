"""Centralized logging configuration for DNA Cancer Classification."""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "dna_cancer", level: Optional[str] = None) -> logging.Logger:
    """Get or configure a logger instance with consistent formatting.

    Args:
        name: Name of the logger, typically __name__ or package identifier.
        level: Optional log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL env var or INFO if not specified.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level_str = level or os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger

