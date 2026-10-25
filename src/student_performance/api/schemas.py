"""Typed Pydantic request and response contracts for API v1."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

NumericInput = Annotated[int | float | None, Field(description="Finite numeric value or null.")]


class ApiModel(BaseModel):
    """Base response model configured for explicit, predictable serialization."""

    model_config = ConfigDict(extra="forbid")


class ErrorDetail(ApiModel):
    code: str
    message: str


class ErrorResponse(ApiModel):
    error: ErrorDetail


class HealthResponse(ApiModel):
    status: str
    service_available: bool
    model_ready: bool


class FeatureSchemaItem(ApiModel):
    name: str
    data_type: str
    nullable: bool


class ModelInfoResponse(ApiModel):
    target_name: str
    selected_estimator: str
    selected_candidate: str
    trained_feature_schema: list[FeatureSchemaItem]
    selected_hyperparameters: dict[str, Any]
    training_timestamp_utc: str
    dataset_sha256: str
    cross_validation: dict[str, Any]
    split: dict[str, Any]
    library_versions: dict[str, str]
    evaluation_results: dict[str, Any]
    limitations: list[str]


class TargetSummary(ApiModel):
    count: int
    missing_count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float


class AnalyticsSummaryResponse(ApiModel):
    record_count: int
    column_count: int
    feature_count: int
    numerical_feature_count: int
    categorical_feature_count: int
    duplicate_record_count: int
    missing_value_counts: dict[str, int]
    target_name: str
    target: TargetSummary


class HistogramBin(ApiModel):
    start: float
    end: float
    count: int


class CategoryCount(ApiModel):
    category: str
    count: int


class DistributionResponse(ApiModel):
    feature: str
    data_type: str
    record_count: int
    missing_count: int
    distribution_type: str
    bins: list[HistogramBin] = Field(default_factory=list)
    categories: list[CategoryCount] = Field(default_factory=list)
    truncated: bool = False


class CrossValidationResult(ApiModel):
    selection_rank: int
    candidate: str
    tuned: bool
    mean_cv_mae: float
    std_cv_mae: float
    parameters: dict[str, Any]


class ModelMetricsResponse(ApiModel):
    selection_method: str
    cross_validation_results: list[CrossValidationResult]
    holdout_test_results: dict[str, Any]


class FeatureImportanceItem(ApiModel):
    feature: str
    importance_mean: float
    importance_std: float
    importance_direction: str
    interpretation: str


class FeatureImportanceResponse(ApiModel):
    method: str
    evaluation_data: str
    causal_claim: bool
    items: list[FeatureImportanceItem]


class PredictionRequest(ApiModel):
    """One student feature record; every key is required, values may be null."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "example": {
                "Hours_Studied": 23,
                "Attendance": 84,
                "Parental_Involvement": "Low",
                "Access_to_Resources": "High",
                "Extracurricular_Activities": "No",
                "Sleep_Hours": 7,
                "Previous_Scores": 73,
                "Motivation_Level": "Low",
                "Internet_Access": "Yes",
                "Tutoring_Sessions": 0,
                "Family_Income": "Low",
                "Teacher_Quality": "Medium",
                "School_Type": "Public",
                "Peer_Influence": "Positive",
                "Physical_Activity": 3,
                "Learning_Disabilities": "No",
                "Parental_Education_Level": "High School",
                "Distance_from_Home": "Near",
                "Gender": "Male",
            }
        },
    )

    Hours_Studied: NumericInput
    Attendance: NumericInput
    Parental_Involvement: StrictStr | None
    Access_to_Resources: StrictStr | None
    Extracurricular_Activities: StrictStr | None
    Sleep_Hours: NumericInput
    Previous_Scores: NumericInput
    Motivation_Level: StrictStr | None
    Internet_Access: StrictStr | None
    Tutoring_Sessions: NumericInput
    Family_Income: StrictStr | None
    Teacher_Quality: StrictStr | None
    School_Type: StrictStr | None
    Peer_Influence: StrictStr | None
    Physical_Activity: NumericInput
    Learning_Disabilities: StrictStr | None
    Parental_Education_Level: StrictStr | None
    Distance_from_Home: StrictStr | None
    Gender: StrictStr | None

    @field_validator(
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
    )
    @classmethod
    def validate_finite_numbers(cls, value: int | float | None) -> int | float | None:
        if value is None:
            return value
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise ValueError("Numeric inputs must be finite.")
        return value


class PredictionResponse(ApiModel):
    target_name: str
    predicted_score: float
    is_estimate: bool
    disclaimer: str
