from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from worldcup_prediction.utils import (
    deterministic_match_id,
    ensure_columns,
    normalize_tournament_name,
    standardize_team_name,
)

RAW_MATCH_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]

RANKING_COLUMNS = ["rank_date", "team", "rank", "points"]


def clean_matches(matches: pd.DataFrame, team_mapping: Mapping[str, str] | None = None) -> pd.DataFrame:
    ensure_columns(matches, RAW_MATCH_COLUMNS, "matches")
    cleaned = matches.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["team_a"] = cleaned["home_team"].map(lambda value: standardize_team_name(value, team_mapping))
    cleaned["team_b"] = cleaned["away_team"].map(lambda value: standardize_team_name(value, team_mapping))
    cleaned["team_a_score"] = pd.to_numeric(cleaned["home_score"], errors="coerce").astype("Int64")
    cleaned["team_b_score"] = pd.to_numeric(cleaned["away_score"], errors="coerce").astype("Int64")
    cleaned["tournament"] = cleaned["tournament"].map(normalize_tournament_name)
    cleaned["neutral"] = cleaned["neutral"].fillna(False).astype(bool)

    invalid_scores = cleaned["team_a_score"].isna() | cleaned["team_b_score"].isna()
    if invalid_scores.any():
        raise ValueError(f"Found {int(invalid_scores.sum())} matches with invalid scores")
    if cleaned["date"].isna().any():
        raise ValueError("Found matches with invalid dates")

    if "city" not in cleaned.columns:
        cleaned["city"] = ""
    if "country" not in cleaned.columns:
        cleaned["country"] = ""
    if "stage" not in cleaned.columns:
        cleaned["stage"] = ""
    if "group" not in cleaned.columns:
        cleaned["group"] = ""

    cleaned = cleaned.drop_duplicates(
        subset=["date", "team_a", "team_b", "team_a_score", "team_b_score", "tournament"]
    )
    cleaned = cleaned.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)
    cleaned["match_id"] = cleaned.apply(
        lambda row: deterministic_match_id(
            row["date"].date(),
            row["team_a"],
            row["team_b"],
            row["team_a_score"],
            row["team_b_score"],
            row["tournament"],
        ),
        axis=1,
    )
    return cleaned


def clean_rankings(rankings: pd.DataFrame, team_mapping: Mapping[str, str] | None = None) -> pd.DataFrame:
    ensure_columns(rankings, RANKING_COLUMNS, "rankings")
    cleaned = rankings.copy()
    cleaned["rank_date"] = pd.to_datetime(cleaned["rank_date"], errors="coerce")
    cleaned["team"] = cleaned["team"].map(lambda value: standardize_team_name(value, team_mapping))
    cleaned["rank"] = pd.to_numeric(cleaned["rank"], errors="coerce")
    cleaned["points"] = pd.to_numeric(cleaned["points"], errors="coerce")
    if cleaned[["rank_date", "team", "rank", "points"]].isna().any().any():
        raise ValueError("Rankings contain invalid dates, teams, ranks, or points")
    cleaned = cleaned.drop_duplicates(subset=["rank_date", "team"]).sort_values(["team", "rank_date"])
    return cleaned.reset_index(drop=True)
