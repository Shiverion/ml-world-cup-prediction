from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from worldcup_prediction.utils import ensure_columns, standardize_team_name


HISTORICAL_WORLD_CUP_YEARS = (2002, 2006, 2010, 2014, 2018, 2022)


def knockout_round_key(round_name: Any) -> str | None:
    """Normalize archive round labels and deliberately omit the third-place match."""
    text = str(round_name or "").strip().lower()
    if "round of 16" in text:
        return "round_of_16"
    if "quarter" in text:
        return "quarterfinals"
    if "semi" in text:
        return "semifinals"
    if text == "final":
        return "final"
    return None


def _score_pair(score: Mapping[str, Any], key: str) -> tuple[Any, Any]:
    value = score.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    return value[0], value[1]


def _winner_from_scores(team_a: str, team_b: str, score_a: Any, score_b: Any) -> str | None:
    if score_a is None or score_b is None:
        return None
    if score_a > score_b:
        return team_a
    if score_b > score_a:
        return team_b
    return None


def world_cup_knockout_results_from_payload(
    payload: Mapping[str, Any],
    year: int,
    source_url: str = "",
) -> pd.DataFrame:
    """Convert an openfootball World Cup archive payload into decisive knockout ties."""
    rows: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        round_name = str(match.get("round", ""))
        round_key = knockout_round_key(round_name)
        if round_key is None:
            continue

        team_a = match.get("team1")
        team_b = match.get("team2")
        score = match.get("score") or {}
        if not isinstance(team_a, str) or not isinstance(team_b, str) or not isinstance(score, Mapping):
            raise ValueError(f"World Cup {year} archive has an incomplete knockout match: {match}")

        score_a_ft, score_b_ft = _score_pair(score, "ft")
        score_a_et, score_b_et = _score_pair(score, "et")
        score_a_penalties, score_b_penalties = _score_pair(score, "p")
        has_extra_time = score_a_et is not None and score_b_et is not None
        final_score_a, final_score_b = (
            (score_a_et, score_b_et) if has_extra_time else (score_a_ft, score_b_ft)
        )
        penalty_winner = _winner_from_scores(team_a, team_b, score_a_penalties, score_b_penalties)
        score_winner = _winner_from_scores(team_a, team_b, final_score_a, final_score_b)
        winner = penalty_winner or score_winner
        if winner is None:
            raise ValueError(
                f"World Cup {year} knockout match has no decisive winner: {team_a} vs {team_b} ({round_name})"
            )

        rows.append(
            {
                "year": int(year),
                "date": match.get("date"),
                "round": round_name,
                "round_key": round_key,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_score": final_score_a,
                "team_b_score": final_score_b,
                "team_a_score_ft": score_a_ft,
                "team_b_score_ft": score_b_ft,
                "team_a_score_et": score_a_et,
                "team_b_score_et": score_b_et,
                "team_a_penalties": score_a_penalties,
                "team_b_penalties": score_b_penalties,
                "winner": winner,
                "winner_method": "penalties"
                if penalty_winner is not None
                else "extra_time"
                if has_extra_time
                else "full_time",
                "source_url": source_url,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["date", "round_key", "team_a", "team_b"]).reset_index(drop=True)


def load_historical_knockout_results(
    path: str,
    team_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "year",
        "date",
        "round",
        "round_key",
        "team_a",
        "team_b",
        "team_a_score",
        "team_b_score",
        "winner",
        "winner_method",
    }
    ensure_columns(frame, sorted(required), "historical knockout results")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in ["team_a", "team_b", "winner"]:
        frame[column] = frame[column].map(lambda value: standardize_team_name(value, team_mapping))
    for column in [
        "team_a_score",
        "team_b_score",
        "team_a_score_ft",
        "team_b_score_ft",
        "team_a_score_et",
        "team_b_score_et",
        "team_a_penalties",
        "team_b_penalties",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    expected_rounds = {"round_of_16", "quarterfinals", "semifinals", "final"}
    unexpected_rounds = sorted(set(frame["round_key"]) - expected_rounds)
    if unexpected_rounds:
        raise ValueError(f"Historical knockout results contain unsupported round keys: {unexpected_rounds}")
    invalid_winners = frame[~frame.apply(lambda row: row["winner"] in {row["team_a"], row["team_b"]}, axis=1)]
    if not invalid_winners.empty:
        raise ValueError("Historical knockout results contain a winner outside the participating teams")

    return frame.sort_values(["year", "date", "round_key", "team_a", "team_b"]).reset_index(drop=True)
