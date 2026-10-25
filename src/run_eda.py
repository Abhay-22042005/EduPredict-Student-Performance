"""Run the project EDA workflow and save reports and charts."""

from __future__ import annotations

import logging

from student_performance.analysis.eda import generate_eda
from student_performance.config import DEFAULT_DATASET_PATH, PROJECT_ROOT, TARGET_COLUMN
from student_performance.data.loader import load_dataset
from student_performance.logging_config import configure_file_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Load and analyze the configured dataset, returning a process exit code."""
    log_path = configure_file_logging()
    try:
        df = load_dataset(DEFAULT_DATASET_PATH)
        artifacts = generate_eda(
            df,
            project_root=PROJECT_ROOT,
            dataset_path=DEFAULT_DATASET_PATH,
        )
        print(f"EDA completed for {len(df)} records; target: {TARGET_COLUMN}.")
        print(f"Reports: {', '.join(str(path) for path in artifacts[:3])}")
        print(
            f"Figures: {PROJECT_ROOT / 'reports' / 'figures'} "
            f"({len(artifacts) - 3} PNG files)"
        )
        return 0
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        LOGGER.exception("EDA workflow failed")
        print(f"ERROR: EDA workflow failed: {exc}\nDetails: {log_path}")
        return 1
    except Exception as exc:
        LOGGER.exception("Unexpected EDA failure")
        print(
            f"ERROR: Unexpected EDA failure ({type(exc).__name__}). "
            f"See diagnostic log: {log_path}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
