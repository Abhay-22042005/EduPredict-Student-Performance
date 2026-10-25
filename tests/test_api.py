from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from student_performance.api.main import create_app
from student_performance.api.settings import ApiSettings
from student_performance.config import (
    BEST_MODEL_PATH,
    DEFAULT_DATASET_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from student_performance.data.loader import load_dataset
from student_performance.models.predict import predict_from_dataframe


@pytest.fixture(scope="module")
def api_settings() -> ApiSettings:
    return ApiSettings.from_environment()


@pytest.fixture(scope="module")
def client(api_settings: ApiSettings) -> Iterator[TestClient]:
    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def real_prediction_record() -> dict[str, object]:
    df = load_dataset(DEFAULT_DATASET_PATH)
    return df.drop(columns=["Exam_Score"]).iloc[0].to_dict()


def test_health_separates_service_from_model_readiness(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service_available": True,
        "model_ready": True,
    }


def test_model_info_matches_real_training_metadata(client: TestClient) -> None:
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["target_name"] == "Exam_Score"
    assert body["selected_candidate"] == "Ridge (tuned)"
    assert len(body["trained_feature_schema"]) == 19
    assert "Exam_Score" not in [item["name"] for item in body["trained_feature_schema"]]
    assert body["evaluation_results"]["selected_model_test"]["mae"] == pytest.approx(
        0.4497375972213063
    )


def test_analytics_summary_reports_actual_data(client: TestClient) -> None:
    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["record_count"] == 6607
    assert body["feature_count"] == 19
    assert body["numerical_feature_count"] == 6
    assert body["categorical_feature_count"] == 13
    assert body["duplicate_record_count"] == 0
    assert body["missing_value_counts"] == {
        "Teacher_Quality": 78,
        "Parental_Education_Level": 90,
        "Distance_from_Home": 67,
    }
    assert body["target"]["minimum"] == 55
    assert body["target"]["maximum"] == 101


def test_distributions_are_chart_ready_and_feature_allowlisted(
    client: TestClient,
) -> None:
    numeric = client.get(
        "/api/v1/analytics/distributions", params={"feature": "Exam_Score"}
    )
    assert numeric.status_code == 200
    assert numeric.json()["distribution_type"] == "histogram"
    assert sum(item["count"] for item in numeric.json()["bins"]) == 6607

    categorical = client.get(
        "/api/v1/analytics/distributions",
        params={"feature": "School_Type"},
    )
    assert categorical.status_code == 200
    assert categorical.json()["distribution_type"] == "category_frequency"
    assert sum(item["count"] for item in categorical.json()["categories"]) == 6607

    unsupported = client.get(
        "/api/v1/analytics/distributions",
        params={"feature": "Made_Up_Column"},
    )
    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "unsupported_feature"


def test_metrics_and_feature_importance_are_saved_results(client: TestClient) -> None:
    metrics = client.get("/api/v1/models/metrics")
    assert metrics.status_code == 200
    body = metrics.json()
    assert len(body["cross_validation_results"]) == 8
    assert body["cross_validation_results"][0]["candidate"] == "Ridge (tuned)"
    assert body["holdout_test_results"]["selected_model_test"]["rmse"] == pytest.approx(
        1.8030334489497746
    )
    assert "Cross-validation" in body["selection_method"]
    assert "Holdout" in body["selection_method"]

    importance = client.get("/api/v1/models/feature-importance")
    assert importance.status_code == 200
    importance_body = importance.json()
    assert importance_body["causal_claim"] is False
    assert importance_body["evaluation_data"].startswith("Untouched holdout")
    assert len(importance_body["items"]) == 19
    assert all("importance_direction" in item for item in importance_body["items"])


def test_prediction_uses_saved_model_with_real_feature_record(
    client: TestClient,
    real_prediction_record: dict[str, object],
) -> None:
    response = client.post("/api/v1/predict", json=real_prediction_record)
    assert response.status_code == 200
    body = response.json()
    expected = float(
        predict_from_dataframe(
            pd.DataFrame([real_prediction_record]),
            model_path=BEST_MODEL_PATH,
            metadata_path=MODEL_METADATA_PATH,
        )[0]
    )
    assert body["target_name"] == "Exam_Score"
    assert body["predicted_score"] == pytest.approx(expected)
    assert body["is_estimate"] is True
    assert "guaranteed" in body["disclaimer"]


def test_prediction_accepts_missing_values_and_unseen_categories(
    client: TestClient,
    real_prediction_record: dict[str, object],
) -> None:
    body = dict(real_prediction_record)
    body["Teacher_Quality"] = "Previously_Unseen"
    body["Attendance"] = None
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 200
    assert isinstance(response.json()["predicted_score"], float)


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        ("missing", "Hours_Studied"),
        ("extra_target", "Exam_Score"),
        ("invalid_number", "Hours_Studied"),
        ("boolean_number", "Attendance"),
        ("infinite_number", "Physical_Activity"),
        ("invalid_category_type", "School_Type"),
    ],
)
def test_prediction_validation_errors_are_consistent(
    client: TestClient,
    real_prediction_record: dict[str, object],
    mutation: str,
    expected_field: str,
) -> None:
    body = dict(real_prediction_record)
    if mutation == "missing":
        body.pop(expected_field)
    elif mutation == "extra_target":
        body[expected_field] = 70
    elif mutation == "invalid_number":
        body[expected_field] = "many"
    elif mutation == "boolean_number":
        body[expected_field] = True
    elif mutation == "invalid_category_type":
        body[expected_field] = 12
    else:
        body[expected_field] = float("inf")

    if mutation == "infinite_number":
        response = client.post(
            "/api/v1/predict",
            content=json.dumps(body, allow_nan=True),
            headers={"content-type": "application/json"},
        )
    else:
        response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert isinstance(error["message"], str)
    assert expected_field in [detail["field"] for detail in error["details"]]
    assert "input" not in json.dumps(error).lower()


def test_model_unavailable_is_reported_without_stopping_service(
    api_settings: ApiSettings, tmp_path: Path
) -> None:
    missing_settings = replace(
        api_settings,
        model_path=tmp_path / "missing.joblib",
        metadata_path=tmp_path / "missing.json",
    )
    with TestClient(create_app(missing_settings)) as missing_client:
        health = missing_client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["service_available"] is True
        assert health.json()["model_ready"] is False
        model_info = missing_client.get("/api/v1/model-info")
        assert model_info.status_code == 503
        assert model_info.json()["error"]["code"] == "model_unavailable"
        assert str(tmp_path) not in model_info.text


def test_corrupt_model_artifact_marks_readiness_degraded(
    api_settings: ApiSettings, tmp_path: Path
) -> None:
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a model")
    settings = replace(api_settings, model_path=corrupt)
    with TestClient(create_app(settings)) as corrupt_client:
        response = corrupt_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["model_ready"] is False


def test_missing_metadata_marks_readiness_degraded(
    api_settings: ApiSettings, tmp_path: Path
) -> None:
    settings = replace(api_settings, metadata_path=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as missing_client:
        response = missing_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["model_ready"] is False


def test_missing_dataset_is_a_safe_503(
    api_settings: ApiSettings, tmp_path: Path
) -> None:
    settings = replace(api_settings, dataset_path=tmp_path / "missing.csv")
    with TestClient(create_app(settings)) as missing_client:
        response = missing_client.get("/api/v1/analytics/summary")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "dataset_unavailable"
        assert str(tmp_path) not in response.text


def test_missing_and_malformed_reports_are_safe_503(
    api_settings: ApiSettings, tmp_path: Path
) -> None:
    settings = replace(
        api_settings,
        metrics_path=tmp_path / "missing_metrics.json",
        feature_importance_path=tmp_path / "bad_importance.csv",
    )
    settings.feature_importance_path.write_text("bad header\nx\n", encoding="utf-8")
    with TestClient(create_app(settings)) as report_client:
        metrics = report_client.get("/api/v1/models/metrics")
        assert metrics.status_code == 503
        assert metrics.json()["error"]["code"] == "model_reports_unavailable"
        importance = report_client.get("/api/v1/models/feature-importance")
        assert importance.status_code == 503
        assert importance.json()["error"]["code"] == "feature_importance_unavailable"
        assert str(tmp_path) not in metrics.text + importance.text


def test_invalid_route_and_oversized_request_use_common_error_shape(
    client: TestClient,
) -> None:
    not_found = client.get("/api/v1/not-a-route")
    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "not_found"

    oversized = client.post(
        "/api/v1/predict",
        content=b" " * 20_000,
        headers={"content-type": "application/json"},
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "request_too_large"


def test_settings_use_project_relative_paths_and_reject_wildcard_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUDENT_PERFORMANCE_MODEL_PATH", "models/best_model.joblib")
    settings = ApiSettings.from_environment()
    assert settings.model_path == BEST_MODEL_PATH.resolve()
    monkeypatch.setenv("STUDENT_PERFORMANCE_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="wildcard"):
        ApiSettings.from_environment()


def test_api_settings_reference_real_dataset_by_default(
    api_settings: ApiSettings,
) -> None:
    assert api_settings.dataset_path == DEFAULT_DATASET_PATH.resolve()
    assert api_settings.model_comparison_path == (
        REPORTS_DIR / "model_comparison.csv"
    ).resolve()
    assert api_settings.dataset_path.is_file()
    assert PROJECT_ROOT in api_settings.dataset_path.parents
