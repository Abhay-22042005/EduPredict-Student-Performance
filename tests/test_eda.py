import pandas as pd
import pytest

from student_performance.analysis.eda import generate_eda
from student_performance.config import DEFAULT_DATASET_PATH
from student_performance.data.inspection import build_data_quality_report
from student_performance.data.loader import load_dataset


@pytest.fixture
def dataset_df() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATASET_PATH)


def test_quality_report_documents_missing_values_and_score_review(
    dataset_df: pd.DataFrame,
) -> None:
    quality = build_data_quality_report(dataset_df).set_index("column")
    assert quality.loc["Teacher_Quality", "missing_count"] == 78
    assert quality.loc["Parental_Education_Level", "missing_count"] == 90
    assert quality.loc["Distance_from_Home", "missing_count"] == 67
    assert quality.loc["Exam_Score", "target_above_100_count"] == 1


def test_generate_eda_creates_reports_and_figures(
    dataset_df: pd.DataFrame, tmp_path
) -> None:
    artifacts = generate_eda(
        dataset_df,
        project_root=tmp_path,
        dataset_path=DEFAULT_DATASET_PATH,
    )

    report_dir = tmp_path / "reports"
    assert (report_dir / "dataset_inspection.txt").is_file()
    assert (report_dir / "data_quality_report.csv").is_file()
    assert (report_dir / "eda_summary.txt").is_file()
    assert len(artifacts) == 11
    assert len(list((report_dir / "figures").glob("*.png"))) == 8
    assert all(path.stat().st_size > 0 for path in artifacts)

    summary = (report_dir / "eda_summary.txt").read_text(encoding="utf-8")
    assert "correlation" in summary.lower()
    assert "causation" in summary.lower()
    assert "exceed 100" in summary


def test_generate_eda_stops_when_target_is_missing(tmp_path) -> None:
    df = pd.DataFrame({"Hours_Studied": [1, 2]})
    with pytest.raises(ValueError, match="missing required columns"):
        generate_eda(
            df,
            project_root=tmp_path,
            dataset_path=DEFAULT_DATASET_PATH,
        )

    report_dir = tmp_path / "reports"
    assert (report_dir / "dataset_inspection.txt").is_file()
    assert (report_dir / "data_quality_report.csv").is_file()
    inspection = (report_dir / "dataset_inspection.txt").read_text(encoding="utf-8")
    assert "Required target column 'Exam_Score' is missing" in inspection
    assert not (report_dir / "eda_summary.txt").exists()
