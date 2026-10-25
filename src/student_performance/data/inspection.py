"""Tabular dataset inspection and quality-report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ..config import REQUIRED_COLUMNS, TARGET_COLUMN


def is_categorical_column(series: pd.Series) -> bool:
    """Return whether a pandas Series has a categorical or textual dtype."""
    return (
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _category_label_conflicts(series: pd.Series) -> dict[str, list[str]]:
    """Find distinct category spellings that collapse after trim/case normalization."""
    groups: dict[str, set[str]] = {}
    for value in series.dropna().unique():
        raw_value = str(value)
        groups.setdefault(raw_value.strip().casefold(), set()).add(raw_value)
    return {
        normalized: sorted(raw_values)
        for normalized, raw_values in groups.items()
        if len(raw_values) > 1
    }


def build_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build one data-quality row per observed column without altering input data."""
    report_rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        row: dict[str, Any] = {
            "column": column,
            "role": (
                "target"
                if column == TARGET_COLUMN
                else "numeric_feature"
                if is_numeric_dtype(series.dtype)
                else "categorical_feature"
                if is_categorical_column(series)
                else "other_feature"
            ),
            "dtype": str(series.dtype),
            "non_missing_count": int(series.notna().sum()),
            "missing_count": missing_count,
            "missing_percent": round(100 * missing_count / len(df), 4) if len(df) else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
            "min": None,
            "max": None,
            "negative_count": None,
            "non_finite_count": None,
            "iqr_outlier_candidate_count": None,
            "iqr_lower_bound": None,
            "iqr_upper_bound": None,
            "observed_categories": None,
            "normalized_label_conflicts": None,
            "whitespace_only_count": None,
            "target_above_100_count": None,
        }

        if is_numeric_dtype(series.dtype):
            numeric_values = series.dropna().to_numpy(dtype=float)
            finite_values = numeric_values[np.isfinite(numeric_values)]
            row["non_finite_count"] = int((~np.isfinite(numeric_values)).sum())
            row["negative_count"] = int((finite_values < 0).sum())
            if finite_values.size:
                q1, q3 = np.quantile(finite_values, [0.25, 0.75])
                iqr = q3 - q1
                lower_bound = float(q1 - 1.5 * iqr)
                upper_bound = float(q3 + 1.5 * iqr)
                row.update(
                    {
                        "min": float(finite_values.min()),
                        "max": float(finite_values.max()),
                        "iqr_outlier_candidate_count": int(
                            ((finite_values < lower_bound) | (finite_values > upper_bound)).sum()
                        ),
                        "iqr_lower_bound": lower_bound,
                        "iqr_upper_bound": upper_bound,
                    }
                )
            if column == TARGET_COLUMN:
                row["target_above_100_count"] = int((finite_values > 100).sum())

        if is_categorical_column(series):
            categories = series.dropna().unique().tolist()
            row["observed_categories"] = json.dumps(
                [str(value) for value in categories], ensure_ascii=False
            )
            row["normalized_label_conflicts"] = json.dumps(
                _category_label_conflicts(series), ensure_ascii=False, sort_keys=True
            )
            row["whitespace_only_count"] = int(
                series.dropna().map(lambda value: not str(value).strip()).sum()
            )

        report_rows.append(row)
    return pd.DataFrame(report_rows)


def build_dataset_inspection(df: pd.DataFrame, dataset_path: Path) -> str:
    """Format schema, sample, statistics, quality, categories, and target details."""
    numerical_columns = [
        column for column in df.columns if is_numeric_dtype(df[column].dtype)
    ]
    categorical_columns = [
        column for column in df.columns if is_categorical_column(df[column])
    ]
    lines = [
        "Dataset inspection",
        f"CSV path: {dataset_path.resolve()}",
        "",
        "Dimensions",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "Actual column names",
        repr(df.columns.tolist()),
        "",
        "Feature and target types",
        f"Numerical columns: {numerical_columns}",
        f"Categorical columns: {categorical_columns}",
        f"Target column: {TARGET_COLUMN}",
        f"Potential predictors: {[column for column in df.columns if column != TARGET_COLUMN]}",
        f"Expected columns absent: {[column for column in REQUIRED_COLUMNS if column not in df.columns]}",
        f"Observed columns outside the expected schema: "
        f"{[column for column in df.columns if column not in REQUIRED_COLUMNS]}",
        "",
        "Data types",
        df.dtypes.to_string(),
        "",
        "First five rows",
        df.head().to_string(index=False),
        "",
        "Descriptive statistics",
        df.describe(include="all").transpose().to_string(),
        "",
        "Missing values by column",
        df.isna().sum().to_string(),
        "",
        "Duplicate rows",
        str(int(df.duplicated().sum())),
        "",
        "Categorical values and label consistency",
    ]

    for column in categorical_columns:
        values = [str(value) for value in df[column].dropna().unique().tolist()]
        conflicts = _category_label_conflicts(df[column])
        whitespace_only = int(
            df[column].dropna().map(lambda value: not str(value).strip()).sum()
        )
        lines.append(f"{column}: {values!r}")
        if conflicts:
            lines.append(f"  Normalized-label conflicts: {conflicts!r}")
        if whitespace_only:
            lines.append(f"  Whitespace-only values: {whitespace_only}")
        if not conflicts and not whitespace_only:
            lines.append("  No whitespace/case-normalization collisions observed.")

    lines.extend(["", "Numerical ranges and IQR outlier candidates"])
    quality = build_data_quality_report(df)
    for _, row in quality[quality["role"].isin(["numeric_feature", "target"])].iterrows():
        lines.append(
            f"{row['column']}: min={row['min']}, max={row['max']}, "
            f"IQR outlier candidates={row['iqr_outlier_candidate_count']}, "
            f"negative values={row['negative_count']}, "
            f"non-finite values={row['non_finite_count']}"
        )

    lines.extend(
        [
            "",
            "Target distribution",
        ]
    )
    if TARGET_COLUMN not in df.columns:
        lines.append(f"ERROR: Required target column '{TARGET_COLUMN}' is missing.")
    elif not is_numeric_dtype(df[TARGET_COLUMN].dtype):
        lines.append(
            f"ERROR: Target column '{TARGET_COLUMN}' is not numeric "
            f"(dtype={df[TARGET_COLUMN].dtype})."
        )
    else:
        target = df[TARGET_COLUMN].dropna()
        if target.empty:
            lines.append(f"ERROR: Target column '{TARGET_COLUMN}' has no values.")
        elif not np.isfinite(target.to_numpy(dtype=float)).all():
            lines.append(
                f"ERROR: Target column '{TARGET_COLUMN}' contains non-finite values."
            )
        else:
            lines.append(
                f"Target suitability: numeric regression target; "
                f"missing values={int(df[TARGET_COLUMN].isna().sum())}."
            )
            lines.append(target.value_counts().sort_index().to_string())
            above_100 = int((target > 100).sum())
            if above_100:
                lines.append(
                    f"Review flag: {above_100} observed target value(s) exceed 100. "
                    "Confirm the scoring scale and source; no values were changed."
                )
                lines.extend(
                    [
                        "Observed target frequency distribution",
                        target.value_counts().sort_index().to_string(),
                    ]
                )

    lines.extend(
        [
            "",
            "Interpretation notes",
            "- IQR counts identify candidates for review; no records were removed.",
            "- A score above 100 is flagged for source/scale verification, not assumed invalid.",
            "- Observed associations and correlations are descriptive, not causal evidence.",
            "- Category domains are reported as observed; no external domain was assumed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_dataset_inspection(
    df: pd.DataFrame, dataset_path: Path, report_path: Path
) -> Path:
    """Write the inspection report and return its resolved path."""
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_dataset_inspection(df, dataset_path), encoding="utf-8"
    )
    return report_path


def write_data_quality_report(df: pd.DataFrame, report_path: Path) -> Path:
    """Write the per-column quality summary as CSV."""
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    build_data_quality_report(df).to_csv(report_path, index=False)
    return report_path
