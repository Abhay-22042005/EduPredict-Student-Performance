"""Generate descriptive EDA reports and figures from the observed dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import TARGET_COLUMN
from ..data.inspection import (
    build_data_quality_report,
    is_categorical_column,
    write_data_quality_report,
    write_dataset_inspection,
)
from ..data.validation import validate_dataset

LOGGER = logging.getLogger(__name__)
FIGURE_DPI = 150


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved figure: %s", path)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column != TARGET_COLUMN
    ]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        column for column in df.columns if is_categorical_column(df[column])
    ]


def _plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    target = df[TARGET_COLUMN].dropna()
    figure, axis = plt.subplots(figsize=(9, 5))
    bins = min(30, max(8, int(target.nunique())))
    axis.hist(target, bins=bins, color="#3287a8", edgecolor="white", alpha=0.9)
    axis.set(
        title="Distribution of Exam Scores",
        xlabel="Exam score",
        ylabel="Number of records",
    )
    axis.grid(axis="y", alpha=0.2)
    path = output_dir / "exam_score_distribution.png"
    _save_figure(figure, path)
    return path


def _plot_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = _numeric_columns(df)
    if not columns:
        return None

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        axis.hist(df[column].dropna(), bins=25, color="#4967a8", edgecolor="white")
        axis.set(title=column.replace("_", " "), xlabel=column, ylabel="Frequency")
        axis.grid(axis="y", alpha=0.18)
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    path = output_dir / "numerical_feature_distributions.png"
    _save_figure(figure, path)
    return path


def _plot_numeric_boxplots(df: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if df[column].notna().any()
    ]
    if not columns:
        return None

    figure, axis = plt.subplots(figsize=(13, max(5, 0.65 * len(columns))))
    data = [df[column].dropna().to_numpy() for column in columns]
    matplotlib_version = tuple(
        int(component) for component in matplotlib.__version__.split(".")[:2]
    )
    if matplotlib_version >= (3, 10):
        axis.boxplot(
            data,
            orientation="horizontal",
            tick_labels=columns,
            showfliers=True,
        )
    elif matplotlib_version >= (3, 9):
        axis.boxplot(data, vert=False, tick_labels=columns, showfliers=True)
    else:
        axis.boxplot(data, vert=False, labels=columns, showfliers=True)
    axis.set(title="Numerical Feature Ranges and Potential Outliers", xlabel="Value")
    axis.grid(axis="x", alpha=0.2)
    path = output_dir / "numerical_feature_boxplots.png"
    _save_figure(figure, path)
    return path


def _plot_categorical_frequencies(df: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = _categorical_columns(df)
    if not columns:
        return None

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        counts = df[column].value_counts(dropna=False)
        labels = ["Missing" if pd.isna(value) else str(value) for value in counts.index]
        axis.bar(labels, counts.values, color="#7357a6")
        axis.set(
            title=column.replace("_", " "),
            xlabel="Observed category",
            ylabel="Records",
        )
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.18)
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    path = output_dir / "categorical_feature_frequencies.png"
    _save_figure(figure, path)
    return path


def _plot_missing_values(df: pd.DataFrame, output_dir: Path) -> Path:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    figure, axis = plt.subplots(figsize=(10, 5))
    if missing.empty:
        axis.text(0.5, 0.5, "No missing values observed", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.bar(missing.index, missing.values, color="#d58a45")
        axis.set(
            title="Missing Values by Feature",
            xlabel="Feature",
            ylabel="Missing records",
        )
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.2)
    path = output_dir / "missing_values.png"
    _save_figure(figure, path)
    return path


def _plot_numeric_vs_target(df: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = _numeric_columns(df)
    if not columns:
        return None

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        valid = df[[column, TARGET_COLUMN]].dropna()
        axis.scatter(valid[column], valid[TARGET_COLUMN], alpha=0.25, s=12, color="#258b9b")
        axis.set(
            title=f"{column.replace('_', ' ')} and exam score",
            xlabel=column.replace("_", " "),
            ylabel="Exam score",
        )
        axis.grid(alpha=0.18)
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    path = output_dir / "numerical_features_vs_exam_score.png"
    _save_figure(figure, path)
    return path


def _plot_categorical_vs_target(df: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = _categorical_columns(df)
    if not columns:
        return None

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        valid = df[[column, TARGET_COLUMN]].dropna()
        categories = valid[column].drop_duplicates().tolist()
        values = [
            valid.loc[valid[column] == category, TARGET_COLUMN].to_numpy()
            for category in categories
        ]
        if values:
            axis.boxplot(values, showfliers=False)
            axis.set_xticks(range(1, len(categories) + 1))
            axis.set_xticklabels([str(category) for category in categories], rotation=30)
        axis.set(
            title=f"Exam score by {column.replace('_', ' ')}",
            xlabel=column.replace("_", " "),
            ylabel="Exam score",
        )
        axis.grid(axis="y", alpha=0.18)
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    path = output_dir / "categorical_features_vs_exam_score.png"
    _save_figure(figure, path)
    return path


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path | None:
    numeric = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return None

    correlation = numeric.corr(method="pearson")
    figure_size = max(7, min(12, 0.8 * len(correlation.columns)))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    labels = [column.replace("_", " ") for column in correlation.columns]
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Pearson Correlations (Numerical Columns)")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    path = output_dir / "numerical_correlation_heatmap.png"
    _save_figure(figure, path)
    return path


def _build_eda_summary(df: pd.DataFrame, quality: pd.DataFrame, figures: list[Path]) -> str:
    target = df[TARGET_COLUMN].dropna()
    missing = df.isna().sum()
    missing = missing[missing > 0]
    numeric_features = _numeric_columns(df)
    categorical_features = _categorical_columns(df)
    target_above_100 = int((target > 100).sum())
    target_iqr = quality.loc[quality["column"] == TARGET_COLUMN].iloc[0]

    lines = [
        "Exploratory data analysis summary",
        f"Records: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Target: {TARGET_COLUMN} (numeric)",
        f"Numerical predictor columns: {numeric_features}",
        f"Categorical predictor columns: {categorical_features}",
        f"Duplicate records: {int(df.duplicated().sum())}",
        f"Target values: count={len(target)}, min={target.min()}, "
        f"max={target.max()}, mean={target.mean():.4f}, median={target.median():.4f}, "
        f"standard deviation={target.std():.4f}",
        f"Target IQR outlier candidates: {target_iqr['iqr_outlier_candidate_count']} "
        f"(review only; no rows removed)",
    ]
    if missing.empty:
        lines.append("Missing values: none observed.")
    else:
        lines.append("Columns with missing values:")
        lines.extend(
            f"- {column}: {int(count)} ({100 * count / len(df):.2f}%)"
            for column, count in missing.items()
        )

    if target_above_100:
        lines.append(
            f"Review flag: {target_above_100} target value(s) exceed 100. "
            "Confirm the dataset scoring scale; the values are retained unchanged."
        )

    numerical_review = quality[
        quality["role"].isin(["numeric_feature", "target"])
        & (
            quality["negative_count"].fillna(0).gt(0)
            | quality["non_finite_count"].fillna(0).gt(0)
            | quality["iqr_outlier_candidate_count"].fillna(0).gt(0)
        )
    ]
    if not numerical_review.empty:
        lines.append("Numerical values to review (no automatic correction applied):")
        for _, row in numerical_review.iterrows():
            lines.append(
                f"- {row['column']}: negative={int(row['negative_count'])}, "
                f"non-finite={int(row['non_finite_count'])}, "
                f"IQR candidates={int(row['iqr_outlier_candidate_count'])}"
            )

    category_conflicts = quality[
        quality["normalized_label_conflicts"].fillna("{}").ne("{}")
    ]
    whitespace_categories = quality[quality["whitespace_only_count"].fillna(0).gt(0)]
    if category_conflicts.empty and whitespace_categories.empty:
        lines.append(
            "Categorical consistency: no trim/case-normalization collisions or "
            "whitespace-only labels observed."
        )
    else:
        lines.append("Categorical consistency review:")
        for _, row in category_conflicts.iterrows():
            lines.append(
                f"- {row['column']} label collisions: {row['normalized_label_conflicts']}"
            )
        for _, row in whitespace_categories.iterrows():
            lines.append(
                f"- {row['column']} whitespace-only labels: "
                f"{int(row['whitespace_only_count'])}"
            )

    lines.extend(
        [
            "",
            "Interpretation and limitations",
            "- Histograms, box plots, and IQR counts are descriptive; outlier candidates "
            "were not removed or altered.",
            "- Pearson correlation is shown only for numerical columns and does not "
            "establish causation.",
            "- Feature-versus-target plots show observed associations, not causal effects.",
            "- No external category domain or score cap was assumed.",
            "",
            "Generated figures",
        ]
    )
    lines.extend(f"- {path.name}" for path in figures)
    return "\n".join(lines) + "\n"


def generate_eda(
    df: pd.DataFrame,
    *,
    project_root: Path,
    dataset_path: Path,
) -> list[Path]:
    """Validate and save inspection, quality, EDA text, and actual-data figures."""
    project_root = project_root.resolve()
    report_dir = project_root / "reports"
    figure_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    inspection_path = write_dataset_inspection(
        df, dataset_path, report_dir / "dataset_inspection.txt"
    )
    quality_path = write_data_quality_report(
        df, report_dir / "data_quality_report.csv"
    )
    validate_dataset(df)
    quality = build_data_quality_report(df)

    figure_paths: list[Path] = []
    plotting_functions = [
        _plot_target_distribution,
        _plot_numeric_distributions,
        _plot_numeric_boxplots,
        _plot_categorical_frequencies,
        _plot_missing_values,
        _plot_numeric_vs_target,
        _plot_categorical_vs_target,
        _plot_correlation_heatmap,
    ]
    for plot in plotting_functions:
        figure_path = plot(df, figure_dir)
        if figure_path is not None:
            figure_paths.append(figure_path)

    summary_path = report_dir / "eda_summary.txt"
    summary_path.write_text(
        _build_eda_summary(df, quality, figure_paths), encoding="utf-8"
    )
    LOGGER.info(
        "EDA complete: %s, %s, %s, %d figures",
        inspection_path,
        quality_path,
        summary_path,
        len(figure_paths),
    )
    return [inspection_path, quality_path, summary_path, *figure_paths]
