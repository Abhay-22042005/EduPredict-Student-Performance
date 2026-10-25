from pathlib import Path

import pandas as pd
import pytest

from student_performance.config import DEFAULT_DATASET_PATH, REQUIRED_COLUMNS, TARGET_COLUMN
from student_performance.data.loader import load_dataset
from student_performance.data.validation import validate_dataset


@pytest.fixture
def dataset_df() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATASET_PATH)


def test_load_dataset_valid(dataset_df: pd.DataFrame) -> None:
    assert not dataset_df.empty
    assert list(dataset_df.columns) == REQUIRED_COLUMNS
    report = validate_dataset(dataset_df)
    assert report.n_rows == len(dataset_df)
    assert report.n_columns == len(dataset_df.columns)
    assert report.duplicates == 0
    assert report.missing_columns == []
    assert report.target_min == pytest.approx(55.0)
    assert report.target_max == pytest.approx(101.0)
    assert report.target_mean == pytest.approx(67.23565914938702)
    assert report.target_missing_count == 0
    assert report.unexpected_columns == []
    assert len(report.feature_columns) == 19
    assert TARGET_COLUMN in dataset_df.columns


def test_missing_dataset_file_raises() -> None:
    missing_path = Path("Dataset") / "missing_dataset.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)


def test_relative_dataset_path_is_project_root_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    df = load_dataset(Path("Dataset") / "StudentPerformanceFactors.csv")
    assert len(df) == 6607


def test_missing_required_columns_raises() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df, required_columns=["A", "B", "C", TARGET_COLUMN])


def test_non_numeric_target_raises() -> None:
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        TARGET_COLUMN: ["low", "medium", "high"],
    })
    with pytest.raises(TypeError, match="must be numeric"):
        validate_dataset(df, required_columns=["feature", TARGET_COLUMN])


def test_target_missing_values_are_reported() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3], TARGET_COLUMN: [70.0, None, 80.0]})
    report = validate_dataset(df, required_columns=["feature", TARGET_COLUMN])
    assert report.target_missing_count == 1
    assert report.target_min == 70.0
    assert report.target_max == 80.0


def test_non_finite_target_raises() -> None:
    df = pd.DataFrame({"feature": [1, 2], TARGET_COLUMN: [70.0, float("inf")]})
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset(df, required_columns=["feature", TARGET_COLUMN])
