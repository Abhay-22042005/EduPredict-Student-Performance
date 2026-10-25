import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from student_performance.config import DEFAULT_DATASET_PATH, TARGET_COLUMN
from student_performance.data.loader import load_dataset
from student_performance.models.predict import (
    ModelArtifactError,
    load_model,
    predict_from_dataframe,
    validate_prediction_features,
)
from student_performance.models.train import (
    _atomic_write_json,
    build_pipeline,
    derive_feature_schema,
    split_dataset,
)


@pytest.fixture(scope="module")
def project_df() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATASET_PATH)


def _fit_real_data_pipeline(
    df: pd.DataFrame,
) -> tuple[object, list[str], pd.DataFrame]:
    feature_names, numeric, categorical = derive_feature_schema(df)
    assert TARGET_COLUMN not in feature_names
    X_train, X_test, y_train, _ = split_dataset(df, feature_names)
    pipeline = build_pipeline(
        Ridge(),
        numeric,
        categorical,
        scale_numeric=True,
    )
    sample_count = min(240, len(X_train))
    pipeline.fit(X_train.iloc[:sample_count], y_train.iloc[:sample_count])
    return pipeline, feature_names, X_test.iloc[:3].copy()


def test_feature_schema_excludes_target_and_uses_actual_columns(
    project_df: pd.DataFrame,
) -> None:
    feature_names, numeric, categorical = derive_feature_schema(project_df)
    assert feature_names == [column for column in project_df.columns if column != TARGET_COLUMN]
    assert TARGET_COLUMN not in feature_names
    assert len(feature_names) == 19
    assert len(numeric) == 6
    assert len(categorical) == 13


def test_pipeline_handles_missing_values_and_unseen_categories(
    project_df: pd.DataFrame,
) -> None:
    pipeline, feature_names, prediction_rows = _fit_real_data_pipeline(project_df)
    assert prediction_rows.columns.tolist() == feature_names

    prediction_rows.iloc[0, prediction_rows.columns.get_loc("Teacher_Quality")] = (
        "Unseen_Category"
    )
    prediction_rows.iloc[0, prediction_rows.columns.get_loc("Attendance")] = np.nan
    predictions = pipeline.predict(prediction_rows)

    assert predictions.shape == (len(prediction_rows),)
    assert np.isfinite(predictions).all()


def test_split_is_reproducible_and_isolates_features(
    project_df: pd.DataFrame,
) -> None:
    feature_names, _, _ = derive_feature_schema(project_df)
    first = split_dataset(project_df, feature_names)
    second = split_dataset(project_df, feature_names)

    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[1].index.tolist() == second[1].index.tolist()
    assert len(first[1]) == 1322
    assert TARGET_COLUMN not in first[0].columns
    assert TARGET_COLUMN not in first[1].columns


def test_actual_dataset_training_and_serialized_prediction(
    project_df: pd.DataFrame, tmp_path: Path
) -> None:
    pipeline, feature_names, prediction_rows = _fit_real_data_pipeline(project_df)
    expected = pipeline.predict(prediction_rows)
    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "metadata.json"

    import joblib

    joblib.dump(pipeline, model_path)
    manifest_path.write_text(
        json.dumps(
            {
                "target_name": TARGET_COLUMN,
                "raw_predictor_names": feature_names,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_model(model_path)
    actual = predict_from_dataframe(
        prediction_rows,
        model_path=model_path,
        metadata_path=manifest_path,
    )

    assert np.allclose(loaded.predict(prediction_rows), expected)
    assert np.allclose(actual, expected)


def test_prediction_requires_exact_feature_order(project_df: pd.DataFrame) -> None:
    feature_names, _, _ = derive_feature_schema(project_df)
    reversed_features = list(reversed(feature_names))
    with pytest.raises(ValueError, match="schema/order mismatch"):
        validate_prediction_features(
            pd.DataFrame(columns=reversed_features),
            feature_names,
        )


def test_missing_and_corrupt_model_artifacts_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="artifact not found"):
        load_model(tmp_path / "missing.joblib")

    corrupt_artifact = tmp_path / "corrupt.joblib"
    corrupt_artifact.write_bytes(b"not a joblib artifact")
    with pytest.raises(ModelArtifactError, match="Could not load model artifact"):
        load_model(corrupt_artifact)


def test_metadata_rejects_target_in_predictor_schema(tmp_path: Path) -> None:
    artifact_path = tmp_path / "metadata.json"
    artifact_path.write_text(
        json.dumps(
            {"target_name": TARGET_COLUMN, "raw_predictor_names": [TARGET_COLUMN]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactError, match="incorrectly includes the target"):
        from student_performance.models.predict import load_model_metadata

        load_model_metadata(artifact_path)


def test_metric_json_serialization(tmp_path: Path) -> None:
    output = tmp_path / "final_metrics.json"
    metrics = {
        "selected_model_test": {"mae": 3.25, "mse": 20.5, "rmse": 4.52769257, "r2": 0.1}
    }
    _atomic_write_json(output, metrics)
    assert json.loads(output.read_text(encoding="utf-8")) == metrics
    assert not output.with_suffix(".json.tmp").exists()
