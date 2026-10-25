"""Shared file logging configuration for data workflow commands."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import PROJECT_ROOT


def configure_file_logging(log_name: str = "data_workflow.log") -> Path:
    """Configure diagnostic logging under the project logs directory."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    root_logger = logging.getLogger()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path.resolve()
        for handler in root_logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return log_path
