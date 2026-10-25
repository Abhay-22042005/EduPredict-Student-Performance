from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import REQUIRED_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class ValidationSummary:
    """Stores a concise validation summary for the dataset."""

    n_rows: int
    n_columns: int
    duplicates: int
    missing_by_column: dict[str, int] = field(default_factory=dict)
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    target_missing_count: int = 0
    target_min: float | None = None
    target_max: float | None = None
    target_mean: float | None = None
    target_dtype: str = "unknown"


def validate_dataset(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> ValidationSummary:
    """Validate the dataset schema, nulls, duplicates, and target quality.

    This is intentionally strict about schema conformity and leaves data cleaning to
    later stages of the project workflow, in line with the requirement to inspect and
    validate before transformation.
    """
    if df.empty:
        raise ValueError("Dataset is empty. The CSV appears to contain no rows.")
    if df.columns.has_duplicates:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset has duplicate column names: {duplicates}")

    required = list(required_columns)
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    target_series = df[target_column]
    if not pd.api.types.is_numeric_dtype(target_series):
        raise TypeError(
            f"Target column '{target_column}' must be numeric but is {target_series.dtype}."
        )

    observed_target = target_series.dropna()
    if observed_target.empty:
        raise ValueError(f"Target column '{target_column}' has no non-missing values.")
    if not np.isfinite(observed_target).all():
        raise ValueError(f"Target column '{target_column}' contains non-finite values.")

    summary = ValidationSummary(
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        duplicates=int(df.duplicated().sum()),
        missing_by_column={column: int(df[column].isna().sum()) for column in df.columns},
        missing_columns=missing_columns,
        unexpected_columns=[column for column in df.columns if column not in required],
        feature_columns=[column for column in df.columns if column != target_column],
        target_missing_count=int(target_series.isna().sum()),
        target_min=float(observed_target.min()),
        target_max=float(observed_target.max()),
        target_mean=float(observed_target.mean()),
        target_dtype=str(target_series.dtype),
    )
    return summary
