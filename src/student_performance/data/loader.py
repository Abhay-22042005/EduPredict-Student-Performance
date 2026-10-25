from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATASET_PATH, PROJECT_ROOT


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load a project CSV without modifying it.

    Relative paths are resolved from the project root, not the shell's working
    directory.
    """
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset_path = dataset_path.resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Could not parse dataset CSV at {dataset_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset contains no data rows: {dataset_path}")
    if df.columns.has_duplicates:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset has duplicate column names: {duplicates}")
    return df
