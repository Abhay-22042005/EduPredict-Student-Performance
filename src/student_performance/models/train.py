"""Leakage-safe model comparison, selection, evaluation, and artifact creation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ..config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    DEFAULT_DATASET_PATH,
    FIGURES_DIR,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from ..data.inspection import is_categorical_column
from ..data.loader import load_dataset
from ..data.validation import validate_dataset

LOGGER = logging.getLogger(__name__)

MODEL_SEARCH_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
    "Decision Tree": {
        "model__max_depth": [8, 16, None],
        "model__min_samples_leaf": [2, 8],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_features": [0.7, 1.0],
        "model__min_samples_leaf": [2, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}


def derive_feature_schema(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[list[str], list[str], list[str]]:
    """Derive raw predictors from observed CSV columns and exclude the target."""
    if target_column not in df.columns:
        raise ValueError(f"Required target column '{target_column}' is missing.")
    if not pd.api.types.is_numeric_dtype(df[target_column].dtype):
        raise TypeError(f"Target column '{target_column}' must be numeric.")
    if not np.isfinite(df[target_column].dropna().to_numpy(dtype=float)).all():
        raise ValueError(f"Target column '{target_column}' contains non-finite values.")
    if df[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    feature_names = [name for name in df.columns if name != target_column]
    if not feature_names:
        raise ValueError("No predictor features remain after excluding the target.")
    numeric_features = [
        column
        for column in feature_names
        if pd.api.types.is_numeric_dtype(df[column].dtype)
    ]
    categorical_features = [
        column for column in feature_names if is_categorical_column(df[column])
    ]
    other_features = [
        column
        for column in feature_names
        if column not in numeric_features and column not in categorical_features
    ]
    if other_features:
        raise TypeError(
            "Unsupported predictor data types; explicitly handle these columns "
            f"before training: {other_features}"
        )
    return feature_names, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Create fold-local numeric and categorical preprocessing."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(
    estimator: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = False,
) -> Pipeline:
    """Join all learned preprocessing and the estimator in one sklearn Pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", estimator),
        ]
    )


def split_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return one deterministic holdout split without fitting transformations."""
    if target_column in feature_names:
        raise ValueError("Target column must not be included in predictor features.")
    X = df.loc[:, feature_names].copy()
    y = df[target_column].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def _estimator_specs() -> dict[str, tuple[Any, bool]]:
    """Return baseline estimator definitions and numeric-scaling requirements."""
    return {
        "Dummy": (DummyRegressor(strategy="median"), False),
        "Linear Regression": (LinearRegression(), True),
        "Ridge": (Ridge(), True),
        "Decision Tree": (
            DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=150,
                max_features=0.8,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }


def _pipeline_for(
    estimator: Any,
    scale_numeric: bool,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return build_pipeline(
        estimator,
        numeric_features,
        categorical_features,
        scale_numeric=scale_numeric,
    )


def _model_parameters(pipeline: Pipeline) -> dict[str, Any]:
    params = pipeline.named_steps["model"].get_params(deep=False)
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def _atomic_joblib_dump(model: Pipeline, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}_",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(model, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_artifact_in_fresh_process(
    model_path: Path,
    sample_features: pd.DataFrame,
    expected_prediction: float,
) -> None:
    """Verify a fresh interpreter can deserialize and use the saved pipeline."""
    code = (
        "import json, sys; from pathlib import Path; import joblib, pandas as pd; "
        "sys.path.insert(0, str(Path('src').resolve())); "
        "model=joblib.load(sys.argv[1]); "
        "sample=pd.DataFrame([json.loads(sys.argv[2])]); "
        "prediction=float(model.predict(sample)[0]); "
        "print(json.dumps({'prediction': prediction}))"
    )
    sample_json = sample_features.to_json(orient="records")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(model_path.resolve()),
            json.dumps(json.loads(sample_json)[0]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    prediction = float(json.loads(result.stdout)["prediction"])
    if not np.isclose(prediction, expected_prediction, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Fresh-process model prediction differs from the in-memory prediction: "
            f"{prediction} != {expected_prediction}."
        )


def _save_evaluation_figures(
    y_true: pd.Series, predictions: np.ndarray, figure_dir: Path
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    minimum = float(min(y_true.min(), predictions.min()))
    maximum = float(max(y_true.max(), predictions.max()))
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(y_true, predictions, alpha=0.45, s=18, color="#247e91")
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#b14d55")
    axis.set(
        title="Holdout: Actual vs Predicted Exam Score",
        xlabel="Actual exam score",
        ylabel="Predicted exam score",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    actual_path = figure_dir / "holdout_actual_vs_predicted.png"
    figure.savefig(actual_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    paths.append(actual_path)

    residuals = predictions - y_true.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(predictions, residuals, alpha=0.45, s=18, color="#7357a6")
    axis.axhline(0, linestyle="--", color="#b14d55")
    axis.set(
        title="Holdout Residuals vs Predicted Score",
        xlabel="Predicted exam score",
        ylabel="Residual (predicted - actual)",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    residual_path = figure_dir / "holdout_residuals.png"
    figure.savefig(residual_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    paths.append(residual_path)
    return paths


def _save_importance_figure(importance: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    display = importance.sort_values("importance_mean").tail(15)
    figure, axis = plt.subplots(figsize=(9, max(5, 0.38 * len(display))))
    axis.barh(
        display["feature"],
        display["importance_mean"],
        xerr=display["importance_std"],
        color="#318b8b",
        alpha=0.9,
        capsize=2,
    )
    axis.axvline(0, linewidth=0.8, color="black")
    axis.set(
        title="Permutation Importance on Holdout Data",
        xlabel="Increase in MAE when feature is permuted (higher = more useful)",
        ylabel="Raw predictor feature",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path = figure_dir / "permutation_feature_importance.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _package_version(name: str) -> str:
    return package_metadata.version(name)


def train_model(
    df: pd.DataFrame,
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    reports_dir: str | Path = REPORTS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    permutation_repeats: int = 10,
) -> dict[str, Any]:
    """Compare candidates, tune by train-only CV MAE, evaluate once, save artifacts."""
    validate_dataset(df)
    dataset_path = Path(dataset_path).resolve()
    reports_dir = Path(reports_dir).resolve()
    figures_dir = Path(figures_dir).resolve()
    model_path = Path(model_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names, numeric_features, categorical_features = derive_feature_schema(df)
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        feature_names,
        random_state=random_state,
    )
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X_train, y_train))
    specs = _estimator_specs()
    scale_by_name = {
        "Dummy": False,
        "Linear Regression": True,
        "Ridge": True,
        "Decision Tree": False,
        "Random Forest": False,
        "Gradient Boosting": False,
    }

    comparison_rows: list[dict[str, Any]] = []
    fitted_templates: dict[str, tuple[Pipeline, float, float]] = {}
    for name, (estimator, _) in specs.items():
        pipeline = _pipeline_for(
            estimator,
            scale_by_name[name],
            numeric_features,
            categorical_features,
        )
        scores = -cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=cv_splits,
            n_jobs=1,
            error_score="raise",
        )
        mean_mae = float(scores.mean())
        std_mae = float(scores.std(ddof=1))
        params = _model_parameters(pipeline)
        comparison_rows.append(
            {
                "candidate": name,
                "tuned": False,
                "mean_cv_mae": mean_mae,
                "std_cv_mae": std_mae,
                "parameters_json": json.dumps(params, sort_keys=True),
            }
        )
        fitted_templates[name] = (pipeline, mean_mae, std_mae)
        LOGGER.info(
            "%s CV MAE: %.5f +/- %.5f", name, mean_mae, std_mae
        )

    tunable_names = list(MODEL_SEARCH_GRIDS)
    promising = sorted(
        tunable_names,
        key=lambda name: fitted_templates[name][1],
    )[:2]
    tuned_candidates: dict[str, tuple[Pipeline, float, float, dict[str, Any]]] = {}
    for name in promising:
        base_pipeline = fitted_templates[name][0]
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=MODEL_SEARCH_GRIDS[name],
            scoring="neg_mean_absolute_error",
            cv=cv_splits,
            refit=True,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_index = int(search.best_index_)
        mean_mae = float(-search.cv_results_["mean_test_score"][best_index])
        std_mae = float(search.cv_results_["std_test_score"][best_index])
        best_params = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }
        tuned_name = f"{name} (tuned)"
        comparison_rows.append(
            {
                "candidate": tuned_name,
                "tuned": True,
                "mean_cv_mae": mean_mae,
                "std_cv_mae": std_mae,
                "parameters_json": json.dumps(best_params, sort_keys=True),
            }
        )
        tuned_candidates[tuned_name] = (
            search.best_estimator_,
            mean_mae,
            std_mae,
            best_params,
        )
        LOGGER.info(
            "%s tuned CV MAE: %.5f +/- %.5f with %s",
            name,
            mean_mae,
            std_mae,
            best_params,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["mean_cv_mae", "candidate"], ignore_index=True
    )
    comparison["selection_rank"] = np.arange(1, len(comparison) + 1)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    selected_name = str(comparison.iloc[0]["candidate"])
    if selected_name in tuned_candidates:
        selected_pipeline, selected_cv_mae, selected_cv_std, selected_params = (
            tuned_candidates[selected_name]
        )
        estimator_name = selected_name.removesuffix(" (tuned)")
        selected_tuned = True
    else:
        selected_pipeline, selected_cv_mae, selected_cv_std = fitted_templates[
            selected_name
        ]
        selected_params = _model_parameters(selected_pipeline)
        estimator_name = selected_name
        selected_tuned = False
    selected_pipeline.fit(X_train, y_train)

    baseline_pipeline, _, _ = fitted_templates["Dummy"]
    baseline_pipeline.fit(X_train, y_train)
    predictions = np.asarray(selected_pipeline.predict(X_test), dtype=float)
    baseline_predictions = np.asarray(baseline_pipeline.predict(X_test), dtype=float)
    residuals = predictions - y_test.to_numpy(dtype=float)
    metrics: dict[str, Any] = {
        "selected_model": selected_name,
        "selection_metric": "mean cross-validation MAE (lower is better)",
        "selected_cv_mae_mean": float(selected_cv_mae),
        "selected_cv_mae_std": float(selected_cv_std),
        "test_rows": int(len(y_test)),
        "selected_model_test": {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mean_squared_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
            "residual_mean_predicted_minus_actual": float(residuals.mean()),
            "residual_median_predicted_minus_actual": float(np.median(residuals)),
            "residual_std_predicted_minus_actual": float(residuals.std(ddof=1)),
            "residual_mean_absolute": float(np.abs(residuals).mean()),
        },
        "dummy_baseline_test": {
            "mae": float(mean_absolute_error(y_test, baseline_predictions)),
            "mse": float(mean_squared_error(y_test, baseline_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
            "r2": float(r2_score(y_test, baseline_predictions)),
        },
    }

    figure_paths = _save_evaluation_figures(y_test, predictions, figures_dir)
    importance_result = permutation_importance(
        selected_pipeline,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=permutation_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
            "permutation_repeats": permutation_repeats,
            "interpretation": "MAE increase when raw feature values are permuted",
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    figure_paths.append(_save_importance_figure(importance, figures_dir))

    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metadata = {
        "target_name": TARGET_COLUMN,
        "raw_predictor_names": feature_names,
        "feature_types": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "excluded_columns": {
            TARGET_COLUMN: "Prediction target; excluded from all predictor inputs.",
            "identifier_or_duplicate_columns": [],
            "potential_leakage_columns": [],
        },
        "feature_audit": (
            "No identifier-like or direct target-copy columns were present in the "
            "observed schema. Previous_Scores was retained as prior-history data; "
            "all other non-target columns were retained as plausible predictors. "
            "The dataset source documentation and score scale still require "
            "provenance confirmation."
        ),
        "selected_estimator": estimator_name,
        "selected_candidate": selected_name,
        "selected_hyperparameters": _model_parameters(selected_pipeline),
        "selected_search_parameters": selected_params if selected_tuned else {},
        "selected_candidate_tuned": selected_tuned,
        "split": {
            "test_size": TEST_SIZE,
            "train_size": float(1 - TEST_SIZE),
            "random_state": random_state,
            "shuffle": True,
        },
        "cross_validation": {
            "method": "KFold",
            "folds": cv_folds,
            "shuffle": True,
            "random_state": random_state,
            "selection_metric": "negative_mean_absolute_error",
            "reported_cv_mae_mean": float(selected_cv_mae),
            "reported_cv_mae_std": float(selected_cv_std),
        },
        "dataset": {
            "path_at_training": str(dataset_path),
            "sha256": dataset_sha256,
            "rows": int(len(df)),
            "target_min": float(df[TARGET_COLUMN].min()),
            "target_max": float(df[TARGET_COLUMN].max()),
            "score_above_100_count": int((df[TARGET_COLUMN] > 100).sum()),
            "source_records_modified": False,
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "holdout_metrics": metrics,
    }

    _atomic_joblib_dump(selected_pipeline, model_path)
    _atomic_write_json(metadata_path, metadata)
    expected_prediction = float(selected_pipeline.predict(X_test.iloc[[0]])[0])
    _verify_artifact_in_fresh_process(
        model_path,
        X_test.iloc[[0]],
        expected_prediction,
    )

    _atomic_write_json(reports_dir / "final_metrics.json", metrics)
    summary_lines = [
        "Model training and evaluation summary",
        f"Dataset rows: {len(df)}; predictor columns: {len(feature_names)}.",
        f"Train/test sizes: {len(X_train)}/{len(X_test)} "
        f"(test fraction={TEST_SIZE:.2f}, random_state={random_state}).",
        f"CV selection: {cv_folds}-fold shuffled KFold, train data only, "
        "negative MAE; lowest mean CV MAE selected.",
        f"Selected candidate: {selected_name}.",
        f"Selected mean CV MAE: {selected_cv_mae:.5f} "
        f"(fold standard deviation {selected_cv_std:.5f}).",
        f"Holdout MAE: {metrics['selected_model_test']['mae']:.5f}; "
        f"RMSE: {metrics['selected_model_test']['rmse']:.5f}; "
        f"R-squared: {metrics['selected_model_test']['r2']:.5f}.",
        f"Dummy holdout MAE: {metrics['dummy_baseline_test']['mae']:.5f}; "
        f"RMSE: {metrics['dummy_baseline_test']['rmse']:.5f}; "
        f"R-squared: {metrics['dummy_baseline_test']['r2']:.5f}.",
        f"Dataset SHA-256: {dataset_sha256}.",
        "Preprocessing and estimator were composed in a Pipeline and refit only on "
        "training rows. CV folds were reused for all candidate comparisons and tuning.",
        "The holdout data were used only for final metrics and post-selection "
        "permutation importance; they were not used to select the model.",
        "Scores above 100 and IQR outlier candidates were retained unchanged; verify "
        "the source score scale before interpreting them.",
        "Permutation importance measures predictive association, not causation; "
        "correlated features may share or obscure importance.",
        "Residuals are reported as predicted minus actual. R-squared is a regression "
        "metric and is not classification accuracy.",
        "A fresh Python process reloaded the saved pipeline and matched an in-memory "
        "prediction.",
        "Inference contract: provide a pandas DataFrame with exactly the raw "
        "predictor columns in manifest order and without Exam_Score. See "
        "docs/inference_contract.md.",
        "Intended use is educational analysis and decision support, not definitive "
        "judgment of individual students.",
    ]
    (reports_dir / "model_training_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Selected %s using CV MAE %.5f; holdout MAE %.5f RMSE %.5f R2 %.5f",
        selected_name,
        selected_cv_mae,
        metrics["selected_model_test"]["mae"],
        metrics["selected_model_test"]["rmse"],
        metrics["selected_model_test"]["r2"],
    )
    return {
        "selected_candidate": selected_name,
        "cv_mae_mean": selected_cv_mae,
        "cv_mae_std": selected_cv_std,
        "metrics": metrics,
        "dataset_sha256": dataset_sha256,
        "artifacts": {
            "comparison": reports_dir / "model_comparison.csv",
            "metrics": reports_dir / "final_metrics.json",
            "summary": reports_dir / "model_training_summary.txt",
            "importance": reports_dir / "feature_importance.csv",
            "model": model_path,
            "metadata": metadata_path,
            "figures": figure_paths,
        },
    }


def train_from_csv() -> dict[str, Any]:
    """Load the actual project CSV and train the reproducible candidate suite."""
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    validate_dataset(dataset)
    return train_model(dataset, dataset_path=DEFAULT_DATASET_PATH)
