"""Inspect the project CSV and save a human-readable data summary."""

from __future__ import annotations

import logging
from pathlib import Path

from student_performance.config import DEFAULT_DATASET_PATH, PROJECT_ROOT
from student_performance.data.inspection import build_dataset_inspection
from student_performance.data.loader import load_dataset
from student_performance.data.validation import validate_dataset
from student_performance.logging_config import configure_file_logging

LOGGER = logging.getLogger(__name__)
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_inspection.txt"


def inspect_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    report_path: str | Path = REPORT_PATH,
) -> tuple[str, bool]:
    """Inspect a CSV, persist details, and report whether the target is usable."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = load_dataset(dataset_path)
        report = build_dataset_inspection(df, dataset_path)
        report_path.write_text(report, encoding="utf-8")
        validate_dataset(df)
        return report, True
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        LOGGER.exception("Dataset inspection failed")
        if "df" in locals():
            report = build_dataset_inspection(df, dataset_path)
        else:
            report = (
                "Dataset inspection\n"
                f"CSV path: {dataset_path.resolve()}\n"
                f"ERROR: {exc}\n"
            )
        report_path.write_text(report, encoding="utf-8")
        return report, False


def main() -> int:
    """Run inspection with concise terminal output and file-based diagnostics."""
    log_path = configure_file_logging()
    try:
        report, valid = inspect_dataset()
        if not valid:
            print(
                f"ERROR: Dataset inspection could not validate the CSV. "
                f"Review {REPORT_PATH} and {log_path}."
            )
            return 1

        lines = report.splitlines()
        dimensions = next(line for line in lines if line.startswith("Rows:"))
        column_count = next(line for line in lines if line.startswith("Columns:"))
        print(
            f"Inspection completed: {dimensions.split(': ', 1)[1]} rows, "
            f"{column_count.split(': ', 1)[1]} columns. Report: {REPORT_PATH}"
        )
        return 0
    except OSError as exc:
        LOGGER.exception("Could not save inspection report")
        print(f"ERROR: Could not save inspection report: {exc}. Details: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
