from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np
import pandas as pd

from worldcup_prediction.utils import ensure_columns

FEATURE_MATCH_COLUMNS = ["date", "team_a", "team_b", "team_a_score", "team_b_score", "tournament", "neutral"]


def add_match_targets(matches: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(matches, ["team_a_score", "team_b_score"], "matches")
    frame = matches.copy()
    frame["target"] = np.select(
        [
            frame["team_a_score"] < frame["team_b_score"],
            frame["team_a_score"] == frame["team_b_score"],
            frame["team_a_score"] > frame["team_b_score"],
        ],
        [0, 1, 2],
    ).astype(int)
    return frame


def add_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(matches, ["tournament", "neutral"], "matches")
    frame = matches.copy()
    tournament = frame["tournament"].str.lower()
    stage = frame.get("stage", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    frame["is_friendly"] = tournament.str.contains("friendly", regex=False).astype(int)
    frame["is_qualifier"] = tournament.str.contains("qualification", regex=False).astype(int)
    frame["is_world_cup"] = (frame["tournament"] == "FIFA World Cup").astype(int)
    frame["is_world_cup_group"] = ((frame["is_world_cup"] == 1) & stage.str.contains("group", regex=False)).astype(int)
    frame["is_world_cup_knockout"] = (
        (frame["is_world_cup"] == 1)
        & stage.str.contains("round|quarter|semi|final|knockout", regex=True)
    ).astype(int)
    frame["is_neutral"] = frame["neutral"].astype(bool).astype(int)
    frame["team_a_home_advantage"] = (~frame["neutral"].astype(bool)).astype(int)
    return frame


def add_rest_features(matches: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(matches, ["date", "team_a", "team_b"], "matches")
    frame = matches.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True).copy()
    last_match_date: dict[str, pd.Timestamp] = {}
    team_a_rest: list[float] = []
    team_b_rest: list[float] = []

    for row in frame.itertuples(index=False):
        match_date = pd.Timestamp(row.date)
        rest_a = (match_date - last_match_date[row.team_a]).days if row.team_a in last_match_date else np.nan
        rest_b = (match_date - last_match_date[row.team_b]).days if row.team_b in last_match_date else np.nan
        team_a_rest.append(rest_a)
        team_b_rest.append(rest_b)
        last_match_date[row.team_a] = match_date
        last_match_date[row.team_b] = match_date

    frame["team_a_days_since_last_match"] = team_a_rest
    frame["team_b_days_since_last_match"] = team_b_rest
    frame["rest_days_diff"] = frame["team_a_days_since_last_match"] - frame["team_b_days_since_last_match"]
    return frame


def _history_summary(history: deque[dict[str, float]], window: int) -> dict[str, float]:
    recent = list(history)[-window:]
    if not recent:
        return {
            "points_per_game": np.nan,
            "win_rate": np.nan,
            "goals_for_avg": np.nan,
            "goals_against_avg": np.nan,
            "goal_difference_avg": np.nan,
        }
    return {
        "points_per_game": float(np.mean([item["points"] for item in recent])),
        "win_rate": float(np.mean([item["win"] for item in recent])),
        "goals_for_avg": float(np.mean([item["goals_for"] for item in recent])),
        "goals_against_avg": float(np.mean([item["goals_against"] for item in recent])),
        "goal_difference_avg": float(np.mean([item["goal_difference"] for item in recent])),
    }


def add_rolling_form_features(matches: pd.DataFrame, windows: Iterable[int] = (5, 10, 20)) -> pd.DataFrame:
    ensure_columns(matches, ["date", "team_a", "team_b", "team_a_score", "team_b_score"], "matches")
    frame = matches.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True).copy()
    windows = tuple(windows)
    history: defaultdict[str, deque[dict[str, float]]] = defaultdict(deque)
    feature_rows: list[dict[str, float]] = []

    for row in frame.itertuples(index=False):
        features: dict[str, float] = {}
        for window in windows:
            summary_a = _history_summary(history[row.team_a], window)
            summary_b = _history_summary(history[row.team_b], window)
            for metric, value in summary_a.items():
                features[f"team_a_{metric}_last_{window}"] = value
            for metric, value in summary_b.items():
                features[f"team_b_{metric}_last_{window}"] = value
            features[f"form_points_diff_{window}"] = (
                features[f"team_a_points_per_game_last_{window}"]
                - features[f"team_b_points_per_game_last_{window}"]
            )
            features[f"goal_diff_form_{window}"] = (
                features[f"team_a_goal_difference_avg_last_{window}"]
                - features[f"team_b_goal_difference_avg_last_{window}"]
            )
        feature_rows.append(features)

        if row.team_a_score > row.team_b_score:
            points_a, points_b = 3.0, 0.0
            win_a, win_b = 1.0, 0.0
        elif row.team_a_score == row.team_b_score:
            points_a, points_b = 1.0, 1.0
            win_a, win_b = 0.0, 0.0
        else:
            points_a, points_b = 0.0, 3.0
            win_a, win_b = 0.0, 1.0

        history[row.team_a].append(
            {
                "points": points_a,
                "win": win_a,
                "goals_for": float(row.team_a_score),
                "goals_against": float(row.team_b_score),
                "goal_difference": float(row.team_a_score - row.team_b_score),
            }
        )
        history[row.team_b].append(
            {
                "points": points_b,
                "win": win_b,
                "goals_for": float(row.team_b_score),
                "goals_against": float(row.team_a_score),
                "goal_difference": float(row.team_b_score - row.team_a_score),
            }
        )

    return pd.concat([frame, pd.DataFrame(feature_rows)], axis=1)


def merge_latest_rankings(matches: pd.DataFrame, rankings: pd.DataFrame, strict_before: bool = True) -> pd.DataFrame:
    ensure_columns(matches, ["date", "team_a", "team_b"], "matches")
    ensure_columns(rankings, ["rank_date", "team", "rank", "points"], "rankings")

    frame = matches.copy()
    rankings = rankings.sort_values(["team", "rank_date"]).copy()
    rankings_by_team = {
        team: team_rankings.reset_index(drop=True)
        for team, team_rankings in rankings.groupby("team", sort=False)
    }

    def lookup(team: str, date: pd.Timestamp, column: str) -> float:
        team_rankings = rankings_by_team.get(team)
        if team_rankings is None or team_rankings.empty:
            return np.nan
        dates = team_rankings["rank_date"].to_numpy(dtype="datetime64[ns]")
        side = "left" if strict_before else "right"
        index = np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side=side) - 1
        if index < 0:
            return np.nan
        return float(team_rankings.iloc[index][column])

    for side_name, team_column in [("team_a", "team_a"), ("team_b", "team_b")]:
        frame[f"{side_name}_fifa_rank"] = [
            lookup(team, date, "rank") for team, date in zip(frame[team_column], frame["date"], strict=False)
        ]
        frame[f"{side_name}_fifa_points"] = [
            lookup(team, date, "points") for team, date in zip(frame[team_column], frame["date"], strict=False)
        ]

    frame["fifa_rank_diff"] = frame["team_b_fifa_rank"] - frame["team_a_fifa_rank"]
    frame["fifa_points_diff"] = frame["team_a_fifa_points"] - frame["team_b_fifa_points"]
    return frame


def build_feature_table(matches: pd.DataFrame, rankings: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_columns(matches, FEATURE_MATCH_COLUMNS, "matches")
    frame = add_match_targets(matches)
    frame = add_context_features(frame)
    frame = add_rest_features(frame)
    frame = add_rolling_form_features(frame)
    if rankings is not None:
        frame = merge_latest_rankings(frame, rankings)
    return frame
