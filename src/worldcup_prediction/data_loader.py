from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_csv(path: str | Path, date_columns: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    frame = pd.read_csv(path, **kwargs)
    for column in date_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data
