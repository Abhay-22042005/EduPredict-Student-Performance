"""Train and evaluate project regression models using the real project CSV."""

from __future__ import annotations

import logging

from student_performance.logging_config import configure_file_logging
from student_performance.models.train import train_from_csv

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run training with concise console output and detailed file diagnostics."""
    log_path = configure_file_logging("model_training.log")
    try:
        result = train_from_csv()
    except Exception as exc:
        LOGGER.exception("Model training workflow failed")
        print(
            f"ERROR: Model training failed ({type(exc).__name__}): {exc}. "
            f"See diagnostic log: {log_path}"
        )
        return 1

    test = result["metrics"]["selected_model_test"]
    print(f"Selected model by CV MAE: {result['selected_candidate']}")
    print(
        f"Holdout metrics: MAE={test['mae']:.4f}, RMSE={test['rmse']:.4f}, "
        f"R-squared={test['r2']:.4f}"
    )
    print(f"Comparison: {result['artifacts']['comparison']}")
    print(f"Model artifact: {result['artifacts']['model']}")
    print(f"Summary: {result['artifacts']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
