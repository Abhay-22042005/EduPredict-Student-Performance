"""Environment-based settings for the local prediction API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    BEST_MODEL_PATH,
    DEFAULT_DATASET_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)


def _path_setting(environment_name: str, default: Path) -> Path:
    value = os.getenv(environment_name)
    if value is None or not value.strip():
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _origins_setting() -> tuple[str, ...]:
    raw_origins = os.getenv(
        "STUDENT_PERFORMANCE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = tuple(origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip())
    if "*" in origins:
        raise ValueError(
            "STUDENT_PERFORMANCE_CORS_ORIGINS must not contain wildcard origins."
        )
    return origins


@dataclass(frozen=True)
class ApiSettings:
    """Resolved backend artifact and runtime configuration."""

    model_path: Path
    metadata_path: Path
    dataset_path: Path
    model_comparison_path: Path
    metrics_path: Path
    feature_importance_path: Path
    allowed_origins: tuple[str, ...]
    max_request_bytes: int

    @classmethod
    def from_environment(cls) -> ApiSettings:
        """Read artifact paths, CORS origins, and request-size limit from env."""
        max_request_bytes = int(
            os.getenv("STUDENT_PERFORMANCE_MAX_REQUEST_BYTES", "16384")
        )
        if not 1024 <= max_request_bytes <= 1_048_576:
            raise ValueError(
                "STUDENT_PERFORMANCE_MAX_REQUEST_BYTES must be between 1024 and 1048576."
            )
        return cls(
            model_path=_path_setting(
                "STUDENT_PERFORMANCE_MODEL_PATH", BEST_MODEL_PATH
            ),
            metadata_path=_path_setting(
                "STUDENT_PERFORMANCE_METADATA_PATH", MODEL_METADATA_PATH
            ),
            dataset_path=_path_setting(
                "STUDENT_PERFORMANCE_DATASET_PATH", DEFAULT_DATASET_PATH
            ),
            model_comparison_path=_path_setting(
                "STUDENT_PERFORMANCE_MODEL_COMPARISON_PATH",
                REPORTS_DIR / "model_comparison.csv",
            ),
            metrics_path=_path_setting(
                "STUDENT_PERFORMANCE_METRICS_PATH",
                REPORTS_DIR / "final_metrics.json",
            ),
            feature_importance_path=_path_setting(
                "STUDENT_PERFORMANCE_FEATURE_IMPORTANCE_PATH",
                REPORTS_DIR / "feature_importance.csv",
            ),
            allowed_origins=_origins_setting(),
            max_request_bytes=max_request_bytes,
        )
