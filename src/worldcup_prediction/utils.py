from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha1
from typing import Any

import pandas as pd


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], frame_name: str = "dataframe") -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def load_team_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping = pd.read_csv(path)
    ensure_columns(mapping, ["raw_team_name", "standard_team_name"], "team mapping")
    return dict(zip(mapping["raw_team_name"], mapping["standard_team_name"], strict=False))


def standardize_team_name(name: Any, mapping: Mapping[str, str] | None = None) -> str:
    if pd.isna(name):
        raise ValueError("Team name cannot be missing")
    normalized = str(name).strip()
    normalized = " ".join(normalized.split())
    return mapping.get(normalized, normalized) if mapping else normalized


def normalize_tournament_name(name: Any) -> str:
    if pd.isna(name):
        return "Unknown"
    normalized = str(name).strip()
    normalized = " ".join(normalized.split())
    aliases = {
        "FIFA World Cup qualification": "FIFA World Cup qualification",
        "FIFA World Cup Qualifier": "FIFA World Cup qualification",
        "World Cup": "FIFA World Cup",
        "FIFA World Cup": "FIFA World Cup",
        "Friendly": "Friendly",
    }
    return aliases.get(normalized, normalized)


def deterministic_match_id(*parts: Any) -> str:
    raw = "|".join("" if pd.isna(part) else str(part) for part in parts)
    return sha1(raw.encode("utf-8")).hexdigest()[:16]
