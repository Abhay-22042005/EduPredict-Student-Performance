"""Safe raw-feature inference using a saved, complete sklearn pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import BEST_MODEL_PATH, MODEL_METADATA_PATH


class ModelArtifactError(RuntimeError):
    """Raised when the model artifact cannot be safely loaded or used."""


def load_model(path: str | Path = BEST_MODEL_PATH) -> Pipeline:
    """Load a trusted local sklearn Pipeline artifact with an actionable error."""
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {artifact_path}")
    try:
        pipeline: Any = joblib.load(artifact_path)
    except (
        OSError,
        EOFError,
        ValueError,
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        TypeError,
        KeyError,
        pickle.UnpicklingError,
        BadZipFile,
    ) as exc:
        raise ModelArtifactError(
            f"Could not load model artifact at {artifact_path}: {exc}"
        ) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelArtifactError(
            f"Model artifact at {artifact_path} is not a scikit-learn Pipeline."
        )
    if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ModelArtifactError(
            f"Model artifact at {artifact_path} lacks required preprocessing/model steps."
        )
    return pipeline


def load_model_metadata(path: str | Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    """Load and validate the manifest that defines the inference feature contract."""
    metadata_path = Path(path).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata manifest not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(
            f"Could not read model metadata at {metadata_path}: {exc}"
        ) from exc

    features = metadata.get("raw_predictor_names")
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) for name in features
    ):
        raise ModelArtifactError(
            f"Model metadata at {metadata_path} has no valid raw predictor schema."
        )
    if metadata.get("target_name") in features:
        raise ModelArtifactError(
            f"Model metadata at {metadata_path} incorrectly includes the target "
            "among inference features."
        )
    return metadata


def validate_prediction_features(
    features: pd.DataFrame, expected_features: list[str]
) -> None:
    """Require raw input columns in the exact order recorded at training time."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")
    if features.columns.has_duplicates:
        raise ValueError("Prediction input contains duplicate feature names.")
    actual = features.columns.tolist()
    if actual != expected_features:
        missing = [name for name in expected_features if name not in actual]
        unexpected = [name for name in actual if name not in expected_features]
        raise ValueError(
            "Prediction feature schema/order mismatch. "
            f"Missing={missing}; unexpected={unexpected}; "
            f"expected_order={expected_features}; actual_order={actual}."
        )
    if features.empty:
        raise ValueError("Prediction input contains no records.")


def predict_from_dataframe(
    features: pd.DataFrame,
    *,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
) -> np.ndarray:
    """Predict from raw feature columns using the serialized full pipeline.

    The dataframe must have exactly the trained feature names in their recorded
    order and must not contain the target. Categorical values unseen during training
    are accepted by the saved OneHotEncoder configured with handle_unknown='ignore'.
    """
    metadata = load_model_metadata(metadata_path)
    expected_features: list[str] = metadata["raw_predictor_names"]
    validate_prediction_features(features, expected_features)
    model = load_model(model_path)
    try:
        predictions = np.asarray(model.predict(features), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prediction input could not be processed: {exc}") from exc
    if predictions.ndim != 1 or len(predictions) != len(features):
        raise ModelArtifactError(
            "Loaded model returned a prediction array with an invalid shape."
        )
    if not np.isfinite(predictions).all():
        raise ModelArtifactError("Loaded model returned a non-finite prediction.")
    return predictions
