"""Cached service operations for model inference and real-data analytics."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline

from ..config import TARGET_COLUMN
from ..data.inspection import is_categorical_column
from ..data.loader import load_dataset
from ..data.validation import validate_dataset
from ..models.predict import (
    ModelArtifactError,
    load_model,
    load_model_metadata,
    validate_prediction_features,
)
from .schemas import PredictionRequest
from .settings import ApiSettings

LOGGER = logging.getLogger(__name__)


class ApiServiceError(RuntimeError):
    """An expected API service error with a safe client-facing status."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class BackendService:
    """Load immutable model artifacts once and cache the active CSV for analytics."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.model: Pipeline | None = None
        self.metadata: dict[str, Any] | None = None
        self.model_ready = False
        self._model_failure: str | None = None
        self._dataset: pd.DataFrame | None = None

    def initialize_model(self) -> None:
        """Load and cross-check the saved pipeline and manifest for readiness."""
        try:
            metadata = load_model_metadata(self.settings.metadata_path)
            model = load_model(self.settings.model_path)
            expected_features = metadata["raw_predictor_names"]
            feature_types = metadata["feature_types"]
            if (
                not isinstance(expected_features, list)
                or not expected_features
                or len(expected_features) != len(set(expected_features))
                or TARGET_COLUMN in expected_features
            ):
                raise ModelArtifactError(
                    "Metadata manifest has an invalid raw prediction feature list."
                )
            if not isinstance(feature_types, dict) or not {
                "numeric",
                "categorical",
            }.issubset(feature_types):
                raise ModelArtifactError(
                    "Metadata manifest lacks numerical/categorical feature types."
                )
            numeric_features = set(feature_types["numeric"])
            categorical_features = set(feature_types["categorical"])
            if (
                numeric_features & categorical_features
                or numeric_features | categorical_features != set(expected_features)
            ):
                raise ModelArtifactError(
                    "Metadata feature types do not match the raw feature schema."
                )
            if expected_features != list(PredictionRequest.model_fields):
                raise ModelArtifactError(
                    "Trained feature order does not match the typed API request schema."
                )
            actual_features = list(getattr(model, "feature_names_in_", []))
            if actual_features and actual_features != expected_features:
                raise ModelArtifactError(
                    "Pipeline input schema does not match its metadata manifest."
                )
            if metadata.get("target_name") != TARGET_COLUMN:
                raise ModelArtifactError("Metadata target does not match the API target.")
            self.model = model
            self.metadata = metadata
            self.model_ready = True
            self._model_failure = None
            LOGGER.info("Model and metadata loaded; prediction service is ready.")
        except Exception:
            self.model = None
            self.metadata = None
            self.model_ready = False
            self._model_failure = "Trained model artifacts are unavailable or incompatible."
            LOGGER.exception("Model readiness check failed")

    def require_model(self) -> tuple[Pipeline, dict[str, Any]]:
        """Return the loaded model contract or a safe service-unavailable error."""
        if not self.model_ready or self.model is None or self.metadata is None:
            raise ApiServiceError(
                503,
                "model_unavailable",
                self._model_failure or "The prediction model is not ready.",
            )
        return self.model, self.metadata

    def load_dataset(self) -> pd.DataFrame:
        """Load and validate the actual configured dataset once for analytics."""
        if self._dataset is not None:
            return self._dataset
        try:
            df = load_dataset(self.settings.dataset_path)
            validate_dataset(df)
        except (FileNotFoundError, OSError, TypeError, ValueError):
            LOGGER.exception("Dataset unavailable for analytics")
            raise ApiServiceError(
                503,
                "dataset_unavailable",
                "The configured dataset is unavailable or has an invalid schema.",
            ) from None
        self._dataset = df
        return df

    def health(self) -> dict[str, Any]:
        """Report service liveness separately from model readiness."""
        return {
            "status": "ok" if self.model_ready else "degraded",
            "service_available": True,
            "model_ready": self.model_ready,
        }

    def model_info(self) -> dict[str, Any]:
        """Return the actual trained schema and manifest metadata."""
        _, metadata = self.require_model()
        numerical = set(metadata["feature_types"]["numeric"])
        categorical = set(metadata["feature_types"]["categorical"])
        feature_schema = [
            {
                "name": name,
                "data_type": (
                    "number"
                    if name in numerical
                    else "string"
                    if name in categorical
                    else "unknown"
                ),
                "nullable": True,
            }
            for name in metadata["raw_predictor_names"]
        ]
        return {
            "target_name": metadata["target_name"],
            "selected_estimator": metadata["selected_estimator"],
            "selected_candidate": metadata["selected_candidate"],
            "trained_feature_schema": feature_schema,
            "selected_hyperparameters": metadata["selected_hyperparameters"],
            "training_timestamp_utc": metadata["training_timestamp_utc"],
            "dataset_sha256": metadata["dataset"]["sha256"],
            "cross_validation": metadata["cross_validation"],
            "split": metadata["split"],
            "library_versions": metadata["library_versions"],
            "evaluation_results": metadata["holdout_metrics"],
            "limitations": [
                "Exam_Score values above 100 occur in the observed dataset; confirm "
                "the source score scale before interpreting them.",
                "Predictions are estimates and are not guaranteed academic outcomes.",
                "The model is intended for educational decision support, not definitive "
                "judgments about individual students.",
            ],
        }

    def analytics_summary(self) -> dict[str, Any]:
        """Compute record, predictor, target, missingness, and duplicate summaries."""
        df = self.load_dataset()
        target = df[TARGET_COLUMN].dropna()
        if target.empty or not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ApiServiceError(
                503,
                "dataset_target_unavailable",
                "The configured dataset has no usable numeric target values.",
            )
        features = [column for column in df.columns if column != TARGET_COLUMN]
        numeric_features = [
            column
            for column in features
            if is_numeric_dtype(df[column].dtype)
        ]
        categorical_features = [
            column for column in features if is_categorical_column(df[column])
        ]
        return {
            "record_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "feature_count": int(len(features)),
            "numerical_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "duplicate_record_count": int(df.duplicated().sum()),
            "missing_value_counts": {
                column: int(count)
                for column, count in df.isna().sum().items()
                if count > 0
            },
            "target_name": TARGET_COLUMN,
            "target": {
                "count": int(target.count()),
                "missing_count": int(df[TARGET_COLUMN].isna().sum()),
                "mean": float(target.mean()),
                "median": float(target.median()),
                "standard_deviation": float(target.std()),
                "minimum": float(target.min()),
                "maximum": float(target.max()),
            },
        }

    def distribution(self, feature: str, *, max_categories: int = 25) -> dict[str, Any]:
        """Return bounded numeric histogram or categorical frequency data."""
        df = self.load_dataset()
        if feature not in df.columns:
            raise ApiServiceError(
                422,
                "unsupported_feature",
                "The requested feature is not present in the dataset.",
            )
        series = df[feature]
        missing_count = int(series.isna().sum())
        if is_numeric_dtype(series.dtype):
            values = series.dropna().to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ApiServiceError(
                    503,
                    "distribution_unavailable",
                    "No finite values are available for the requested feature.",
                )
            counts, edges = np.histogram(values, bins=min(20, max(1, len(np.unique(values)))))
            bins = [
                {
                    "start": float(edges[index]),
                    "end": float(edges[index + 1]),
                    "count": int(counts[index]),
                }
                for index in range(len(counts))
            ]
            return {
                "feature": feature,
                "data_type": str(series.dtype),
                "record_count": int(len(df)),
                "missing_count": missing_count,
                "distribution_type": "histogram",
                "bins": bins,
                "categories": [],
                "truncated": False,
            }

        counts = series.fillna("__MISSING__").astype(str).value_counts()
        top_counts = counts.head(max_categories)
        return {
            "feature": feature,
            "data_type": str(series.dtype),
            "record_count": int(len(df)),
            "missing_count": missing_count,
            "distribution_type": "category_frequency",
            "bins": [],
            "categories": [
                {"category": str(category), "count": int(count)}
                for category, count in top_counts.items()
            ],
            "truncated": len(counts) > max_categories,
        }

    def model_metrics(self) -> dict[str, Any]:
        """Read actual CV and holdout records from training output files."""
        try:
            final_metrics = json.loads(
                self.settings.metrics_path.read_text(encoding="utf-8")
            )
            comparison = pd.read_csv(self.settings.model_comparison_path)
            required_columns = {
                "selection_rank",
                "candidate",
                "tuned",
                "mean_cv_mae",
                "std_cv_mae",
                "parameters_json",
            }
            if not required_columns.issubset(comparison.columns):
                raise ValueError("Model comparison report is missing required columns.")
            results = []
            for row in comparison.to_dict(orient="records"):
                parameters = json.loads(row["parameters_json"])
                results.append(
                    {
                        "selection_rank": int(row["selection_rank"]),
                        "candidate": str(row["candidate"]),
                        "tuned": bool(row["tuned"]),
                        "mean_cv_mae": float(row["mean_cv_mae"]),
                        "std_cv_mae": float(row["std_cv_mae"]),
                        "parameters": parameters,
                    }
                )
            if not results:
                raise ValueError("Model comparison report contains no candidate rows.")
            if "selected_model_test" not in final_metrics:
                raise ValueError("Final metrics report lacks holdout test results.")
            return {
                "selection_method": (
                    "Cross-validation MAE on training data; lower is better. "
                    "Holdout test results are final evaluation only."
                ),
                "cross_validation_results": results,
                "holdout_test_results": final_metrics,
            }
        except (OSError, UnicodeError, json.JSONDecodeError, pd.errors.ParserError, ValueError, TypeError, KeyError):
            LOGGER.exception("Model reports unavailable or invalid")
            raise ApiServiceError(
                503,
                "model_reports_unavailable",
                "Model evaluation reports are unavailable or invalid.",
            ) from None

    def feature_importance(self) -> dict[str, Any]:
        """Return saved holdout permutation importance with non-causal interpretation."""
        try:
            importance = pd.read_csv(self.settings.feature_importance_path)
            required = {
                "feature",
                "importance_mean",
                "importance_std",
                "interpretation",
            }
            if not required.issubset(importance.columns):
                raise ValueError("Feature-importance report is missing required columns.")
            if importance.empty:
                raise ValueError("Feature-importance report contains no rows.")
            items = []
            for row in importance.to_dict(orient="records"):
                mean = float(row["importance_mean"])
                std = float(row["importance_std"])
                if not np.isfinite([mean, std]).all():
                    raise ValueError("Feature-importance report contains non-finite values.")
                items.append(
                    {
                        "feature": str(row["feature"]),
                        "importance_mean": mean,
                        "importance_std": std,
                        "importance_direction": (
                            "increases_error_when_permuted"
                            if mean > 0
                            else "decreases_error_when_permuted"
                            if mean < 0
                            else "no_mean_change_when_permuted"
                        ),
                        "interpretation": str(row["interpretation"]),
                    }
                )
            return {
                "method": "Permutation importance scored by increase in MAE.",
                "evaluation_data": "Untouched holdout set, after model selection.",
                "causal_claim": False,
                "items": items,
            }
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError, TypeError, KeyError):
            LOGGER.exception("Feature-importance report unavailable or invalid")
            raise ApiServiceError(
                503,
                "feature_importance_unavailable",
                "Feature-importance results are unavailable or invalid.",
            ) from None

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        """Predict once from ordered raw schema using the already-loaded pipeline."""
        model, metadata = self.require_model()
        feature_names: list[str] = metadata["raw_predictor_names"]
        values = request.model_dump()
        frame = pd.DataFrame(
            [{feature: values[feature] for feature in feature_names}],
            columns=feature_names,
        )
        try:
            validate_prediction_features(frame, feature_names)
            prediction = float(model.predict(frame)[0])
        except (TypeError, ValueError) as exc:
            LOGGER.warning(
                "Prediction input rejected by model preprocessing (%s).",
                type(exc).__name__,
            )
            raise ApiServiceError(
                422,
                "invalid_prediction_input",
                "The supplied feature values could not be processed by the trained model.",
            ) from None
        if not np.isfinite(prediction):
            LOGGER.error("Model returned a non-finite prediction.")
            raise ApiServiceError(
                503,
                "prediction_unavailable",
                "The prediction model returned an invalid result.",
            )
        return {
            "target_name": metadata["target_name"],
            "predicted_score": prediction,
            "is_estimate": True,
            "disclaimer": (
                "This is an estimate for educational decision support, not a "
                "guaranteed academic outcome or definitive judgment."
            ),
        }
