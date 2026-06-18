from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping

import pandas as pd

from worldcup_prediction.utils import ensure_columns

ELO_COLUMNS = ["team_a", "team_b", "team_a_score", "team_b_score", "tournament"]


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def match_result_score(score_a: int | float, score_b: int | float) -> float:
    if score_a > score_b:
        return 1.0
    if score_a == score_b:
        return 0.5
    return 0.0


def default_k_factor(tournament: str, stage: str | None = None) -> float:
    tournament_lower = str(tournament).lower()
    stage_lower = str(stage or "").lower()
    stage_words = set(re.findall(r"[a-z0-9]+", stage_lower))

    if "qualification" in tournament_lower or "qualifier" in tournament_lower:
        return 30.0
    if (
        "world cup" in tournament_lower
        and "final" in stage_words
        and not stage_words.intersection({"semi", "quarter", "round", "knockout"})
    ):
        return 70.0
    if "world cup" in tournament_lower and any(token in stage_lower for token in ["knockout", "round", "quarter", "semi"]):
        return 60.0
    if "world cup" in tournament_lower:
        return 50.0
    if any(token in tournament_lower for token in ["euro", "copa", "afcon", "asian cup", "gold cup"]):
        return 40.0
    if "friendly" in tournament_lower:
        return 20.0
    return 30.0


def add_elo_features(
    matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factors: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    ensure_columns(matches, ELO_COLUMNS, "matches")
    ordered = matches.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True).copy()
    ratings: defaultdict[str, float] = defaultdict(lambda: initial_rating)

    team_a_elos: list[float] = []
    team_b_elos: list[float] = []
    expected_scores: list[float] = []

    for row in ordered.itertuples(index=False):
        rating_a = ratings[row.team_a]
        rating_b = ratings[row.team_b]
        team_a_elos.append(rating_a)
        team_b_elos.append(rating_b)

        expected_a = expected_score(rating_a, rating_b)
        expected_scores.append(expected_a)
        actual_a = match_result_score(row.team_a_score, row.team_b_score)
        if k_factors and row.tournament in k_factors:
            k = float(k_factors[row.tournament])
        else:
            k = default_k_factor(row.tournament, getattr(row, "stage", None))

        delta = k * (actual_a - expected_a)
        ratings[row.team_a] = rating_a + delta
        ratings[row.team_b] = rating_b - delta

    ordered["team_a_elo"] = team_a_elos
    ordered["team_b_elo"] = team_b_elos
    ordered["elo_diff"] = ordered["team_a_elo"] - ordered["team_b_elo"]
    ordered["elo_avg"] = (ordered["team_a_elo"] + ordered["team_b_elo"]) / 2.0
    ordered["elo_abs_diff"] = ordered["elo_diff"].abs()
    ordered["elo_expected_a"] = expected_scores
    return ordered
