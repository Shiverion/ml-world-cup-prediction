from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worldcup_prediction.backtest import DEFAULT_WORLDCUP_WINDOWS, WorldCupWindow, rolling_world_cup_backtest
from worldcup_prediction.calibration import (
    calibration_by_group,
    calibration_table_by_probability_bin,
    probability_sharpness_report,
    top_label_calibration_summary,
)
from worldcup_prediction.cleaning import clean_matches, clean_rankings
from worldcup_prediction.config import CONFIG_DIR, PROJECT_ROOT, RANDOM_SEED
from worldcup_prediction.data_loader import read_csv, read_yaml, write_csv
from worldcup_prediction.elo import add_elo_features, default_k_factor, expected_score, match_result_score
from worldcup_prediction.features import build_feature_table
from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS, make_model, predict_probabilities, train_model
from worldcup_prediction.research import (
    deterministic_interval_seeds,
    match_probability_frame,
    registry_path_reference,
    rolling_model_prediction_records,
    run_ablation_study,
    run_baseline_comparison,
    run_nested_model_selection_backtest,
    simulation_probability_intervals,
    summarize_backtest_like_results,
    write_forecast_registry,
    write_forecast_registry_frames,
)
from worldcup_prediction.simulator import (
    GroupRecord,
    MatchProbabilityFn,
    SimulatedMatch,
    build_round_of_32_bracket,
    knockout_model_vs_monte_carlo_frame,
    normalize_match_probabilities,
    poisson_outcome_probabilities,
    rank_group,
    simulate_tournament_detailed,
)
from worldcup_prediction.utils import ensure_columns, load_team_mapping, standardize_team_name


KNOCKOUT_ROUND_SEQUENCE = ["round_of_32", "round_of_16", "quarterfinals", "semifinals", "third_place", "final"]


def resolve_project_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def available_feature_columns(features: pd.DataFrame, requested_columns: Sequence[str]) -> list[str]:
    return [column for column in requested_columns if column in features.columns]


def parse_world_cup_windows(config: Mapping[str, Any]) -> list[WorldCupWindow]:
    windows = config.get("world_cups")
    if not windows:
        return list(DEFAULT_WORLDCUP_WINDOWS)
    return [
        WorldCupWindow(
            year=int(window["year"]),
            start=str(window["start"]),
            end=str(window["end"]),
        )
        for window in windows
    ]


def load_teams_by_group(tournament_config: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, list[str]]:
    groups = tournament_config.get("groups") or {}
    if groups:
        return {str(group): [str(team) for team in teams] for group, teams in groups.items()}

    groups_path = tournament_config.get("groups_path")
    if not groups_path:
        return {}

    path = resolve_project_path(groups_path, root)
    if not path.exists():
        raise FileNotFoundError(f"Configured groups_path does not exist: {path}")

    frame = read_csv(path)
    ensure_columns(frame, ["group", "team"], "tournament groups")
    teams_by_group: dict[str, list[str]] = {}
    for group, group_frame in frame.groupby("group", sort=False):
        teams_by_group[str(group)] = [str(team) for team in group_frame["team"]]
    return teams_by_group


def resolve_knockout_bracket_config(tournament_config: Mapping[str, Any], root: Path = PROJECT_ROOT) -> Mapping[str, Any] | None:
    bracket_config = tournament_config.get("knockout_bracket")
    if not isinstance(bracket_config, Mapping):
        return bracket_config
    resolved = dict(bracket_config)
    mapping_path = resolved.get("third_place_mapping_path")
    if mapping_path:
        resolved["third_place_mapping_path"] = str(resolve_project_path(str(mapping_path), root))
    return resolved


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _is_knockout_round(round_value: Any) -> bool:
    round_text = _text(round_value).lower()
    return any(token in round_text for token in ["round of", "quarter", "semi", "final", "third place"])


def _is_group_fixture_row(row: pd.Series) -> bool:
    group = _text(row.get("group", "")).replace("Group ", "").strip()
    return bool(group) and not _is_knockout_round(row.get("round", ""))


def _is_unresolved_slot_name(value: Any) -> bool:
    text = _text(value).upper().replace(" ", "")
    if not text:
        return True
    if "/" in text:
        return True
    if text.startswith(("W", "L")) and text[1:].isdigit():
        return True
    return len(text) == 2 and text[0].isdigit() and text[1].isalpha()


def _fixture_stage_name(round_value: Any, group_value: Any) -> str:
    if _is_knockout_round(round_value):
        return _text(round_value)
    if _text(group_value):
        return "Group"
    return _text(round_value)


def _group_stage_completed_dates(fixtures: pd.DataFrame) -> pd.Series:
    completed = fixtures[fixtures["status"].eq("completed")].copy()
    if completed.empty:
        return pd.Series(dtype="datetime64[ns]")
    group_rows = completed.apply(_is_group_fixture_row, axis=1)
    return pd.to_datetime(completed.loc[group_rows, "date"], errors="coerce")


def _completed_fixture_dates(fixtures: pd.DataFrame) -> pd.Series:
    completed = fixtures[fixtures["status"].eq("completed")].copy()
    if completed.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(completed["date"], errors="coerce")


def _pre_knockout_cutoff_from_fixture_frame(fixtures: pd.DataFrame) -> pd.Timestamp | None:
    group_dates = _group_stage_completed_dates(fixtures)
    if group_dates.notna().any():
        return pd.Timestamp(group_dates.max()) + pd.Timedelta(days=1)
    return None


def expected_group_match_count(teams_by_group: Mapping[str, Sequence[str]]) -> int:
    return sum(len(teams) * (len(teams) - 1) // 2 for teams in teams_by_group.values())


def completed_group_matches_from_fixture_frame(
    fixtures: pd.DataFrame,
    team_mapping: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    ensure_columns(fixtures, ["group", "team_a", "team_b", "team_a_score", "team_b_score", "status"], "fixtures")
    completed = fixtures[fixtures["status"].eq("completed")].copy()
    if "round" in completed.columns:
        completed = completed[completed.apply(_is_group_fixture_row, axis=1)]
    completed = completed.dropna(subset=["group", "team_a", "team_b", "team_a_score", "team_b_score"])
    rows: list[dict[str, Any]] = []
    for row in completed.itertuples(index=False):
        rows.append(
            {
                "group": str(row.group).replace("Group ", "").strip(),
                "team_a": standardize_team_name(row.team_a, team_mapping),
                "team_b": standardize_team_name(row.team_b, team_mapping),
                "team_a_score": int(row.team_a_score),
                "team_b_score": int(row.team_b_score),
            }
        )
    return rows


def group_table_from_completed_matches(
    teams_by_group: Mapping[str, Sequence[str]],
    completed_matches: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if not teams_by_group:
        return pd.DataFrame()
    table = {
        str(group): {str(team): GroupRecord(team=str(team), group=str(group)) for team in teams}
        for group, teams in teams_by_group.items()
    }
    played: list[SimulatedMatch] = []
    for match in completed_matches:
        group = str(match["group"])
        team_a = str(match["team_a"])
        team_b = str(match["team_b"])
        if group not in table or team_a not in table[group] or team_b not in table[group]:
            continue
        score_a = int(match["team_a_score"])
        score_b = int(match["team_b_score"])
        record_a = table[group][team_a]
        record_b = table[group][team_b]
        record_a.goals_for += score_a
        record_a.goals_against += score_b
        record_b.goals_for += score_b
        record_b.goals_against += score_a
        if score_a > score_b:
            record_a.points += 3
            record_a.wins += 1
        elif score_b > score_a:
            record_b.points += 3
            record_b.wins += 1
        else:
            record_a.points += 1
            record_b.points += 1
        played.append(SimulatedMatch(group, team_a, team_b, score_a, score_b))

    rows: list[dict[str, Any]] = []
    for group, records_by_team in table.items():
        group_played = [match for match in played if match.group == group]
        ranked = rank_group(list(records_by_team.values()), group_played, np.random.default_rng(0))
        for position, record in enumerate(ranked, start=1):
            rows.append(
                {
                    "group": group,
                    "position": position,
                    "team": record.team,
                    "points": record.points,
                    "goals_for": record.goals_for,
                    "goals_against": record.goals_against,
                    "goal_difference": record.goal_difference,
                    "wins": record.wins,
                }
            )
    return pd.DataFrame(rows)


def completed_fixture_matches_for_training(
    fixtures: pd.DataFrame,
    team_mapping: Mapping[str, str] | None = None,
    include_knockout: bool = True,
) -> pd.DataFrame:
    ensure_columns(fixtures, ["date", "team_a", "team_b", "team_a_score", "team_b_score", "status"], "fixtures")
    completed = fixtures[fixtures["status"].eq("completed")].copy()
    if not include_knockout and "round" in completed.columns:
        completed = completed[completed.apply(_is_group_fixture_row, axis=1)]
    elif include_knockout and "round" in completed.columns:
        unresolved_knockout = completed.apply(
            lambda row: _is_knockout_round(row.get("round", ""))
            and (
                _is_unresolved_slot_name(row.get("team_a", ""))
                or _is_unresolved_slot_name(row.get("team_b", ""))
            ),
            axis=1,
        )
        completed = completed[~unresolved_knockout]
    completed = completed.dropna(subset=["date", "team_a", "team_b", "team_a_score", "team_b_score"])
    if completed.empty:
        return pd.DataFrame()
    stages = [
        _fixture_stage_name(row.get("round", ""), row.get("group", ""))
        for _, row in completed.iterrows()
    ]
    groups = [
        _text(row.get("group", "")).replace("Group ", "").strip() if not _is_knockout_round(row.get("round", "")) else ""
        for _, row in completed.iterrows()
    ]
    raw = pd.DataFrame(
        {
            "date": completed["date"],
            "home_team": completed["team_a"],
            "away_team": completed["team_b"],
            "home_score": completed["team_a_score"].astype(int),
            "away_score": completed["team_b_score"].astype(int),
            "tournament": "FIFA World Cup",
            "city": completed.get("ground", ""),
            "country": "",
            "neutral": True,
            "stage": stages,
            "group": groups,
        }
    )
    return clean_matches(raw, team_mapping)


def merge_live_training_matches(matches_clean: pd.DataFrame, live_training_matches: pd.DataFrame) -> pd.DataFrame:
    if live_training_matches.empty:
        return matches_clean
    return (
        pd.concat([matches_clean, live_training_matches], ignore_index=True)
        .drop_duplicates(
            subset=["date", "team_a", "team_b", "team_a_score", "team_b_score", "tournament"],
            keep="last",
        )
        .sort_values(["date", "team_a", "team_b"])
        .reset_index(drop=True)
    )


def _round_key_from_fixture_round(round_value: Any) -> str | None:
    round_text = _text(round_value).lower()
    if "round of 32" in round_text:
        return "round_of_32"
    if "round of 16" in round_text:
        return "round_of_16"
    if "quarter" in round_text:
        return "quarterfinals"
    if "semi" in round_text:
        return "semifinals"
    if "third" in round_text and "place" in round_text:
        return "third_place"
    if round_text == "final":
        return "final"
    return None


def knockout_round_index(round_key: str | None) -> int | None:
    if round_key not in KNOCKOUT_ROUND_SEQUENCE:
        return None
    return KNOCKOUT_ROUND_SEQUENCE.index(str(round_key))


def fixture_frame_for_reconstructed_round(
    fixtures: pd.DataFrame,
    forecast_round_key: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only information that would be known before a knockout round starts."""
    if forecast_round_key not in KNOCKOUT_ROUND_SEQUENCE:
        raise ValueError(f"Unknown knockout round: {forecast_round_key}")
    ensure_columns(fixtures, ["date", "round", "team_a", "team_b", "team_a_score", "team_b_score", "status"], "fixtures")
    forecast_index = KNOCKOUT_ROUND_SEQUENCE.index(forecast_round_key)
    frame = fixtures.copy()
    frame["_round_key"] = frame["round"].map(_round_key_from_fixture_round)
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce")

    def is_allowed_row(row: pd.Series) -> bool:
        round_key = row["_round_key"]
        if round_key is None:
            return True
        round_index = knockout_round_index(str(round_key))
        return round_index is not None and round_index <= forecast_index

    frame = frame[frame.apply(is_allowed_row, axis=1)].copy()
    round_indices = pd.to_numeric(frame["_round_key"].map(knockout_round_index), errors="coerce").fillna(-1)
    known_future = (round_indices >= forecast_index) | (frame["_date"] >= cutoff)
    frame.loc[known_future, "status"] = "scheduled"
    hidden_columns = [
        "team_a_score",
        "team_b_score",
        "team_a_score_ft",
        "team_b_score_ft",
        "team_a_score_et",
        "team_b_score_et",
        "team_a_penalties",
        "team_b_penalties",
        "winner",
        "winner_method",
    ]
    frame.loc[known_future, [column for column in hidden_columns if column in frame.columns]] = pd.NA
    return frame.drop(columns=["_round_key", "_date"])


def _slot_label(slot: Mapping[str, Any]) -> str:
    if "group" in slot and "position" in slot:
        return f"{int(slot['position'])}{_text(slot['group']).upper()}"
    if "third_place_from" in slot:
        groups = "/".join(_text(group).upper() for group in slot["third_place_from"])
        return f"3{groups}"
    return ""


def _normalized_slot_text(value: Any) -> str:
    return _text(value).upper().replace(" ", "")


def _configured_knockout_match_ids(bracket_config: Mapping[str, Any] | None) -> set[int]:
    if not bracket_config:
        return set()
    ids: set[int] = set()
    for round_key in ["round_of_32", "round_of_16", "quarterfinals", "semifinals", "third_place", "final"]:
        for match in bracket_config.get(round_key, []):
            ids.add(int(match["match"]))
    return ids


def _configured_knockout_lookup(bracket_config: Mapping[str, Any] | None) -> dict[tuple[str, tuple[Any, ...]], int]:
    if not bracket_config:
        return {}
    lookup: dict[tuple[str, tuple[Any, ...]], int] = {}
    for match in bracket_config.get("round_of_32", []):
        teams = list(match.get("teams", []))
        if len(teams) == 2:
            labels = tuple(sorted(_slot_label(slot) for slot in teams))
            lookup[("slots", labels)] = int(match["match"])
    for round_key in ["round_of_16", "quarterfinals", "semifinals", "final"]:
        for match in bracket_config.get(round_key, []):
            sources = tuple(sorted(int(match_id) for match_id in match.get("winners_of", [])))
            if len(sources) == 2:
                lookup[("winners_of", sources)] = int(match["match"])
    for match in bracket_config.get("third_place", []):
        sources = tuple(sorted(int(match_id) for match_id in match.get("losers_of", [])))
        if len(sources) == 2:
            lookup[("losers_of", sources)] = int(match["match"])
    return lookup


def _round_match_configs(bracket_config: Mapping[str, Any] | None, round_key: str) -> list[Mapping[str, Any]]:
    if not bracket_config:
        return []
    return [match for match in bracket_config.get(round_key, [])]


def _knockout_candidate_pools(
    bracket_config: Mapping[str, Any] | None,
    group_table: pd.DataFrame | None,
    third_place_count: int = 8,
) -> tuple[
    dict[int, set[str]],
    dict[int, tuple[int, int]],
    dict[int, tuple[int, int]],
    dict[str, set[int]],
]:
    if not bracket_config:
        return {}, {}, {}, {}
    pools: dict[int, set[str]] = {}
    winner_sources: dict[int, tuple[int, int]] = {}
    loser_sources: dict[int, tuple[int, int]] = {}
    round_ids: dict[str, set[int]] = {}

    if group_table is not None and not group_table.empty:
        for match in build_round_of_32_bracket(group_table, bracket_config, third_place_count=third_place_count):
            match_id = int(match["match"])
            pools[match_id] = {str(match["team_a"]), str(match["team_b"])}
            round_ids.setdefault("round_of_32", set()).add(match_id)

    for round_key in ["round_of_16", "quarterfinals", "semifinals", "third_place", "final"]:
        for match in _round_match_configs(bracket_config, round_key):
            match_id = int(match["match"])
            source_field = "losers_of" if round_key == "third_place" else "winners_of"
            source_ids = tuple(int(source_id) for source_id in match.get(source_field, []))
            if len(source_ids) != 2:
                continue
            sources = loser_sources if round_key == "third_place" else winner_sources
            sources[match_id] = (source_ids[0], source_ids[1])
            round_ids.setdefault(round_key, set()).add(match_id)
            source_pool = set()
            for source_id in source_ids:
                source_pool.update(pools.get(source_id, set()))
            if source_pool:
                pools[match_id] = source_pool
    return pools, winner_sources, loser_sources, round_ids


def _match_id_from_candidate_pools(
    round_key: str | None,
    team_a: str,
    team_b: str,
    candidate_pools: Mapping[int, set[str]],
    source_matches: Mapping[int, tuple[int, int]],
    loser_source_matches: Mapping[int, tuple[int, int]],
    round_ids: Mapping[str, set[int]],
) -> int | None:
    if round_key is None:
        return None
    pair = {team_a, team_b}
    candidates: list[int] = []
    for match_id in round_ids.get(round_key, set()):
        if round_key == "round_of_32":
            if candidate_pools.get(match_id) == pair:
                candidates.append(match_id)
            continue
        sources = loser_source_matches.get(match_id) if round_key == "third_place" else source_matches.get(match_id)
        if not sources:
            continue
        left_pool = candidate_pools.get(sources[0], set())
        right_pool = candidate_pools.get(sources[1], set())
        direct = team_a in left_pool and team_b in right_pool
        reverse = team_a in right_pool and team_b in left_pool
        if direct or reverse:
            candidates.append(match_id)
    return candidates[0] if len(candidates) == 1 else None


def _fixture_knockout_match_id(
    row: pd.Series,
    row_index: Any,
    bracket_config: Mapping[str, Any] | None,
    team_a: str | None = None,
    team_b: str | None = None,
    candidate_pools: Mapping[int, set[str]] | None = None,
    source_matches: Mapping[int, tuple[int, int]] | None = None,
    loser_source_matches: Mapping[int, tuple[int, int]] | None = None,
    round_ids: Mapping[str, set[int]] | None = None,
) -> int | None:
    configured_ids = _configured_knockout_match_ids(bracket_config)
    if "match" in row and _text(row["match"]):
        try:
            candidate = int(float(row["match"]))
            if not configured_ids or candidate in configured_ids:
                return candidate
        except ValueError:
            pass

    round_key = _round_key_from_fixture_round(row.get("round", ""))
    if team_a is not None and team_b is not None and candidate_pools:
        candidate = _match_id_from_candidate_pools(
            round_key,
            team_a,
            team_b,
            candidate_pools,
            source_matches or {},
            loser_source_matches or {},
            round_ids or {},
        )
        if candidate is not None:
            return candidate

    lookup = _configured_knockout_lookup(bracket_config)
    slot_a = _normalized_slot_text(row.get("team_a", ""))
    slot_b = _normalized_slot_text(row.get("team_b", ""))
    winners = []
    for value in [slot_a, slot_b]:
        if value.startswith("W") and value[1:].isdigit():
            winners.append(int(value[1:]))
    if len(winners) == 2:
        match_id = lookup.get(("winners_of", tuple(sorted(winners))))
        if match_id is not None:
            return match_id
    losers = []
    for value in [slot_a, slot_b]:
        if value.startswith("L") and value[1:].isdigit():
            losers.append(int(value[1:]))
    if len(losers) == 2:
        match_id = lookup.get(("losers_of", tuple(sorted(losers))))
        if match_id is not None:
            return match_id
    slot_match_id = lookup.get(("slots", tuple(sorted([slot_a, slot_b]))))
    if slot_match_id is not None:
        return slot_match_id

    try:
        candidate = int(row_index) + 1
        if candidate in configured_ids:
            return candidate
    except (TypeError, ValueError):
        pass
    return None


def _truthy_penalty_marker(value: Any) -> bool:
    text = _text(value).lower()
    if not text:
        return False
    return text in {"1", "true", "yes", "y", "penalty", "penalties", "pens", "pso", "shootout"} or (
        ("pen" in text or "shootout" in text) and "no pen" not in text
    )


def _fixture_decided_by_penalties(row: pd.Series) -> bool:
    explicit_columns = [
        "decided_by_penalties",
        "penalty_shootout",
        "shootout",
        "penalties",
        "went_to_penalties",
    ]
    if any(column in row.index and _truthy_penalty_marker(row.get(column)) for column in explicit_columns):
        return True

    penalty_score_pairs = [
        ("team_a_penalties", "team_b_penalties"),
        ("team_a_penalty_score", "team_b_penalty_score"),
        ("penalty_score_a", "penalty_score_b"),
        ("home_penalties", "away_penalties"),
        ("home_penalty_score", "away_penalty_score"),
    ]
    for left, right in penalty_score_pairs:
        if left in row.index and right in row.index and not pd.isna(row.get(left)) and not pd.isna(row.get(right)):
            return True

    text_columns = ["notes", "note", "remarks", "result_notes", "method", "winner_method"]
    return any(column in row.index and _truthy_penalty_marker(row.get(column)) for column in text_columns)


def completed_knockout_matches_from_fixture_frame(
    fixtures: pd.DataFrame,
    bracket_config: Mapping[str, Any] | None,
    team_mapping: Mapping[str, str] | None = None,
    group_table: pd.DataFrame | None = None,
    third_place_count: int = 8,
) -> list[dict[str, Any]]:
    ensure_columns(fixtures, ["round", "team_a", "team_b", "team_a_score", "team_b_score", "status"], "fixtures")
    knockout = fixtures[fixtures["round"].map(lambda value: _round_key_from_fixture_round(value) is not None)].copy()
    knockout = knockout.dropna(subset=["team_a", "team_b"])
    candidate_pools, source_matches, loser_source_matches, round_ids = _knockout_candidate_pools(
        bracket_config,
        group_table,
        third_place_count=third_place_count,
    )

    fixture_rows: list[dict[str, Any]] = []
    participants_by_match: dict[int, set[str]] = {}
    winners_by_match: dict[int, str] = {}
    winner_sources_by_match: dict[int, str] = {}
    for index, row in knockout.iterrows():
        if _is_unresolved_slot_name(row.get("team_a", "")) or _is_unresolved_slot_name(row.get("team_b", "")):
            continue
        team_a = standardize_team_name(row["team_a"], team_mapping)
        team_b = standardize_team_name(row["team_b"], team_mapping)
        match_id = _fixture_knockout_match_id(
            row,
            index,
            bracket_config,
            team_a=team_a,
            team_b=team_b,
            candidate_pools=candidate_pools,
            source_matches=source_matches,
            loser_source_matches=loser_source_matches,
            round_ids=round_ids,
        )
        if match_id is None:
            continue
        participants_by_match[match_id] = {team_a, team_b}
        score_a = int(float(row["team_a_score"])) if not pd.isna(row.get("team_a_score")) else None
        score_b = int(float(row["team_b_score"])) if not pd.isna(row.get("team_b_score")) else None
        penalties_a = int(float(row["team_a_penalties"])) if "team_a_penalties" in row and not pd.isna(row.get("team_a_penalties")) else None
        penalties_b = int(float(row["team_b_penalties"])) if "team_b_penalties" in row and not pd.isna(row.get("team_b_penalties")) else None
        decided_by_penalties = _fixture_decided_by_penalties(row)
        winner_value = row.get("winner") if "winner" in row else None
        winner = None
        winner_source = None
        if decided_by_penalties and penalties_a is not None and penalties_b is not None:
            if penalties_a > penalties_b:
                winner = team_a
                winner_source = "penalties"
            elif penalties_b > penalties_a:
                winner = team_b
                winner_source = "penalties"
        if winner is None and score_a is not None and score_b is not None:
            if score_a > score_b:
                winner = team_a
                winner_source = "score"
            elif score_b > score_a:
                winner = team_b
                winner_source = "score"
        if winner is None and _text(winner_value):
            winner = standardize_team_name(winner_value, team_mapping)
            winner_source = "fixture"
        if winner is not None:
            winners_by_match[match_id] = winner
            winner_sources_by_match[match_id] = winner_source or "unknown"
        fixture_rows.append(
            {
                "round": _round_key_from_fixture_round(row["round"]),
                "match": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_score": score_a,
                "team_b_score": score_b,
                "team_a_penalties": penalties_a,
                "team_b_penalties": penalties_b,
                "winner": winner,
                "completed": bool(row.get("status") == "completed" and score_a is not None and score_b is not None),
                "decided_by_penalties": decided_by_penalties,
                "winner_method": _text(row.get("winner_method", "")),
            }
        )

    changed = True
    while changed:
        changed = False
        for target_id, source_ids in source_matches.items():
            target_participants = participants_by_match.get(target_id)
            if not target_participants:
                continue
            for source_id in source_ids:
                if source_id in winners_by_match:
                    continue
                source_participants = participants_by_match.get(source_id) or candidate_pools.get(source_id, set())
                inferred = source_participants & target_participants
                if len(inferred) == 1:
                    winners_by_match[source_id] = next(iter(inferred))
                    winner_sources_by_match[source_id] = "next_round"
                    changed = True

    rows: list[dict[str, Any]] = []
    for row in fixture_rows:
        if not row["completed"]:
            continue
        winner = winners_by_match.get(int(row["match"]))
        if winner is None:
            continue
        rows.append(
            {
                "round": row["round"],
                "match": row["match"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "team_a_score": row["team_a_score"],
                "team_b_score": row["team_b_score"],
                "team_a_penalties": row["team_a_penalties"],
                "team_b_penalties": row["team_b_penalties"],
                "winner": winner,
                "winner_source": winner_sources_by_match.get(int(row["match"]), "unknown"),
                "decided_by_penalties": bool(row["decided_by_penalties"]),
                "winner_method": row["winner_method"],
                "decided_after_extra_time": row["winner_method"] == "extra_time",
            }
        )
    return rows


def final_elo_ratings(
    matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    cutoff: pd.Timestamp | None = None,
) -> dict[str, float]:
    required = ["date", "team_a", "team_b", "team_a_score", "team_b_score", "tournament"]
    ensure_columns(matches, required, "matches")
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if cutoff is not None:
        frame = frame[frame["date"] < cutoff]
    ordered = frame.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)
    ratings: defaultdict[str, float] = defaultdict(lambda: initial_rating)

    for row in ordered.itertuples(index=False):
        rating_a = ratings[row.team_a]
        rating_b = ratings[row.team_b]
        expected_a = expected_score(rating_a, rating_b)
        actual_a = match_result_score(row.team_a_score, row.team_b_score)
        k = default_k_factor(row.tournament, getattr(row, "stage", None))
        delta = k * (actual_a - expected_a)
        ratings[row.team_a] = rating_a + delta
        ratings[row.team_b] = rating_b - delta

    return dict(ratings)


def make_elo_probability_predictor(
    ratings: Mapping[str, float],
    initial_rating: float = 1500.0,
    draw_probability: float = 0.24,
) -> MatchProbabilityFn:
    if not 0.0 <= draw_probability < 1.0:
        raise ValueError("draw_probability must be in the range [0, 1)")

    def predict(team_a: str, team_b: str, context: Mapping[str, Any] | None = None) -> dict[str, float]:
        del context
        rating_a = float(ratings.get(team_a, initial_rating))
        rating_b = float(ratings.get(team_b, initial_rating))
        expected_a = expected_score(rating_a, rating_b)
        decisive_mass = 1.0 - draw_probability
        return {
            "team_a_win": decisive_mass * expected_a,
            "draw": draw_probability,
            "team_b_win": decisive_mass * (1.0 - expected_a),
        }

    return predict


def make_elo_poisson_predictor(
    ratings: Mapping[str, float],
    initial_rating: float = 1500.0,
    average_total_goals: float = 2.55,
) -> MatchProbabilityFn:
    if average_total_goals <= 0:
        raise ValueError("average_total_goals must be positive")

    def predict(team_a: str, team_b: str, context: Mapping[str, Any] | None = None) -> dict[str, float]:
        del context
        rating_a = float(ratings.get(team_a, initial_rating))
        rating_b = float(ratings.get(team_b, initial_rating))
        strength_ratio = 10.0 ** ((rating_a - rating_b) / 400.0)
        lambda_b = average_total_goals / (1.0 + strength_ratio)
        lambda_a = average_total_goals - lambda_b
        probabilities = poisson_outcome_probabilities(lambda_a, lambda_b)
        return {
            **probabilities,
            "team_a_goals_lambda": lambda_a,
            "team_b_goals_lambda": lambda_b,
        }

    return predict


def latest_ranking_snapshot(
    rankings: pd.DataFrame | None,
    cutoff: pd.Timestamp,
    inclusive: bool = False,
) -> dict[str, dict[str, float]]:
    if rankings is None or rankings.empty:
        return {}
    ensure_columns(rankings, ["rank_date", "team", "rank", "points"], "rankings")
    frame = rankings.copy()
    frame["rank_date"] = pd.to_datetime(frame["rank_date"], errors="coerce")
    frame = frame.dropna(subset=["rank_date", "team"])
    frame = frame[frame["rank_date"] <= cutoff] if inclusive else frame[frame["rank_date"] < cutoff]
    if frame.empty:
        return {}
    latest = frame.sort_values(["team", "rank_date"]).groupby("team", as_index=False).tail(1)
    return {
        str(row.team): {
            "rank": float(row.rank),
            "points": float(row.points),
        }
        for row in latest.itertuples(index=False)
    }


def recent_form_snapshot(
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    windows: Sequence[int] = (5, 10, 20),
) -> dict[str, dict[str, float]]:
    ensure_columns(matches, ["date", "team_a", "team_b", "team_a_score", "team_b_score"], "matches")
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"] < cutoff].sort_values(["date", "team_a", "team_b"])
    history: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    for row in frame.itertuples(index=False):
        if row.team_a_score > row.team_b_score:
            points_a, points_b = 3.0, 0.0
            win_a, win_b = 1.0, 0.0
        elif row.team_a_score == row.team_b_score:
            points_a, points_b = 1.0, 1.0
            win_a, win_b = 0.0, 0.0
        else:
            points_a, points_b = 0.0, 3.0
            win_a, win_b = 0.0, 1.0

        history[str(row.team_a)].append(
            {
                "points": points_a,
                "win": win_a,
                "goal_difference": float(row.team_a_score - row.team_b_score),
            }
        )
        history[str(row.team_b)].append(
            {
                "points": points_b,
                "win": win_b,
                "goal_difference": float(row.team_b_score - row.team_a_score),
            }
        )

    snapshot: dict[str, dict[str, float]] = {}
    for team, team_history in history.items():
        summary: dict[str, float] = {}
        for window in windows:
            recent = team_history[-window:]
            if not recent:
                summary[f"points_per_game_last_{window}"] = np.nan
                summary[f"goal_difference_avg_last_{window}"] = np.nan
                continue
            summary[f"points_per_game_last_{window}"] = float(np.mean([item["points"] for item in recent]))
            summary[f"goal_difference_avg_last_{window}"] = float(
                np.mean([item["goal_difference"] for item in recent])
            )
        snapshot[team] = summary
    return snapshot


def _numeric_difference(left: float | None, right: float | None) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left) - float(right)


def _fixture_context_features(context: Mapping[str, Any] | None) -> dict[str, float]:
    context = context or {}
    stage = str(context.get("stage", "")).lower()
    is_group_match = int((not stage and "group" in context) or "group" in stage)
    is_knockout = int(any(token in stage for token in ["round", "quarter", "semi", "final", "knockout"]))
    return {
        "is_neutral": 1.0,
        "team_a_home_advantage": 0.0,
        "is_friendly": 0.0,
        "is_qualifier": 0.0,
        "is_world_cup": 1.0,
        "is_world_cup_group": float(is_group_match),
        "is_world_cup_knockout": float(is_knockout),
        "rest_days_diff": 0.0,
    }


def make_ml_outcome_predictor(
    model: object,
    ratings: Mapping[str, float],
    ranking_snapshot: Mapping[str, Mapping[str, float]],
    form_snapshot: Mapping[str, Mapping[str, float]],
    feature_columns: Sequence[str],
    initial_rating: float = 1500.0,
) -> MatchProbabilityFn:
    columns = list(feature_columns)
    cache: dict[tuple[str, str, str], dict[str, float]] = {}

    def form_value(team: str, key: str) -> float:
        return float(form_snapshot.get(team, {}).get(key, np.nan))

    def ranking_value(team: str, key: str) -> float:
        return float(ranking_snapshot.get(team, {}).get(key, np.nan))

    def predict(team_a: str, team_b: str, context: Mapping[str, Any] | None = None) -> dict[str, float]:
        context_features = _fixture_context_features(context)
        context_key = "group" if context_features["is_world_cup_group"] else str((context or {}).get("stage", "neutral"))
        cache_key = (team_a, team_b, context_key)
        if cache_key in cache:
            return dict(cache[cache_key])

        rating_a = float(ratings.get(team_a, initial_rating))
        rating_b = float(ratings.get(team_b, initial_rating))
        rank_a = ranking_value(team_a, "rank")
        rank_b = ranking_value(team_b, "rank")
        points_a = ranking_value(team_a, "points")
        points_b = ranking_value(team_b, "points")

        values: dict[str, float] = {
            "elo_diff": rating_a - rating_b,
            "elo_abs_diff": abs(rating_a - rating_b),
            "elo_expected_a": expected_score(rating_a, rating_b),
            "fifa_rank_diff": _numeric_difference(rank_b, rank_a),
            "fifa_points_diff": _numeric_difference(points_a, points_b),
            "form_points_diff_5": _numeric_difference(
                form_value(team_a, "points_per_game_last_5"),
                form_value(team_b, "points_per_game_last_5"),
            ),
            "form_points_diff_10": _numeric_difference(
                form_value(team_a, "points_per_game_last_10"),
                form_value(team_b, "points_per_game_last_10"),
            ),
            "form_points_diff_20": _numeric_difference(
                form_value(team_a, "points_per_game_last_20"),
                form_value(team_b, "points_per_game_last_20"),
            ),
            "goal_diff_form_5": _numeric_difference(
                form_value(team_a, "goal_difference_avg_last_5"),
                form_value(team_b, "goal_difference_avg_last_5"),
            ),
            "goal_diff_form_10": _numeric_difference(
                form_value(team_a, "goal_difference_avg_last_10"),
                form_value(team_b, "goal_difference_avg_last_10"),
            ),
            "goal_diff_form_20": _numeric_difference(
                form_value(team_a, "goal_difference_avg_last_20"),
                form_value(team_b, "goal_difference_avg_last_20"),
            ),
            **context_features,
        }
        fixture_features = pd.DataFrame([{column: values.get(column, np.nan) for column in columns}])
        probabilities = predict_probabilities(model, fixture_features, columns).iloc[0]
        result = {
            "team_a_loss": float(probabilities["team_a_loss"]),
            "draw": float(probabilities["draw"]),
            "team_a_win": float(probabilities["team_a_win"]),
        }
        cache[cache_key] = result
        return dict(result)

    return predict


def live_model_update_weight(
    completed_knockout_matches: Sequence[Mapping[str, Any]] | int,
    tournament_config: Mapping[str, Any],
) -> float:
    update_config = tournament_config.get("live_model_update") or {}
    if not bool(update_config.get("enabled", True)):
        return 1.0

    completed_count = (
        int(completed_knockout_matches)
        if isinstance(completed_knockout_matches, int)
        else len(completed_knockout_matches)
    )
    if completed_count <= 0:
        return 0.0

    prior_strength = float(
        update_config.get(
            "prior_strength",
            update_config.get("knockout_prior_matches", 80.0),
        )
    )
    max_live_weight = float(update_config.get("max_live_weight", 0.35))
    if prior_strength <= 0:
        raw_weight = 1.0
    else:
        raw_weight = completed_count / (completed_count + prior_strength)
    return max(0.0, min(max_live_weight, raw_weight))


def blend_match_probability_predictors(
    anchor_predictor: MatchProbabilityFn,
    live_predictor: MatchProbabilityFn,
    live_weight: float,
) -> MatchProbabilityFn:
    live_weight = max(0.0, min(1.0, float(live_weight)))
    anchor_weight = 1.0 - live_weight

    def predict(team_a: str, team_b: str, context: Mapping[str, Any] | None = None) -> dict[str, float]:
        anchor_probabilities = normalize_match_probabilities(anchor_predictor(team_a, team_b, context))
        live_probabilities = normalize_match_probabilities(live_predictor(team_a, team_b, context))
        team_a_win = (
            anchor_weight * anchor_probabilities["team_a_win"]
            + live_weight * live_probabilities["team_a_win"]
        )
        draw = anchor_weight * anchor_probabilities["draw"] + live_weight * live_probabilities["draw"]
        team_a_loss = (
            anchor_weight * anchor_probabilities["team_b_win"]
            + live_weight * live_probabilities["team_b_win"]
        )
        return {
            "team_a_loss": team_a_loss,
            "draw": draw,
            "team_a_win": team_a_win,
        }

    return predict


def _live_update_metadata(
    live_weight: float,
    completed_knockout_matches: Sequence[Mapping[str, Any]],
    tournament_config: Mapping[str, Any],
    anchor_cutoff: pd.Timestamp | None = None,
) -> dict[str, Any]:
    update_config = tournament_config.get("live_model_update") or {}
    metadata = {
        "anchored_live_update": True,
        "anchor_model_weight": round(1.0 - live_weight, 4),
        "live_model_weight": round(live_weight, 4),
        "live_knockout_training_matches": len(completed_knockout_matches),
        "live_update_prior_strength": update_config.get(
            "prior_strength",
            update_config.get("knockout_prior_matches", 80.0),
        ),
        "live_update_max_live_weight": update_config.get("max_live_weight", 0.35),
    }
    if anchor_cutoff is not None:
        metadata["anchor_snapshot_cutoff"] = pd.Timestamp(anchor_cutoff).isoformat()
    return metadata


def _train_primary_model_for_cutoff(
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    model_config: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    primary_model_name: str,
    feature_columns: Sequence[str],
    empty_message: str,
) -> object:
    train_frame = features[pd.to_datetime(features["date"], errors="coerce") < cutoff].copy()
    if train_frame.empty:
        raise ValueError(empty_message)
    return train_model(
        _model_from_spec(
            model_specs[primary_model_name],
            primary_model_name,
            int(model_config.get("random_seed", RANDOM_SEED)),
        ),
        train_frame,
        feature_columns,
        target_column=str(model_config.get("target_column", "target")),
    )


def _make_anchored_live_predictor(
    live_model: object,
    anchor_matches_clean: pd.DataFrame,
    rankings_clean: pd.DataFrame | None,
    cutoff: pd.Timestamp,
    model_config: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    primary_model_name: str,
    feature_columns: Sequence[str],
    ratings: Mapping[str, float],
    live_matches_clean: pd.DataFrame,
    completed_knockout_matches: Sequence[Mapping[str, Any]],
    tournament_config: Mapping[str, Any],
    anchor_cutoff: pd.Timestamp | None = None,
) -> tuple[MatchProbabilityFn, dict[str, Any] | None]:
    live_predictor = make_ml_outcome_predictor(
        live_model,
        ratings,
        latest_ranking_snapshot(
            rankings_clean,
            cutoff,
            inclusive=bool(tournament_config.get("ranking_cutoff_inclusive", False)),
        ),
        recent_form_snapshot(live_matches_clean, cutoff),
        feature_columns,
    )
    live_weight = live_model_update_weight(completed_knockout_matches, tournament_config)
    update_config = tournament_config.get("live_model_update") or {}
    if live_weight >= 1.0 or not bool(update_config.get("enabled", True)) or not completed_knockout_matches:
        return live_predictor, None

    frozen_anchor_cutoff = pd.Timestamp(anchor_cutoff) if anchor_cutoff is not None else cutoff
    anchor_features = build_feature_table(add_elo_features(anchor_matches_clean), rankings_clean)
    missing_columns = sorted(set(feature_columns) - set(anchor_features.columns))
    if missing_columns:
        raise ValueError(f"Anchor training data is missing columns: {missing_columns}")
    anchor_model = _train_primary_model_for_cutoff(
        anchor_features,
        frozen_anchor_cutoff,
        model_config,
        model_specs,
        primary_model_name,
        feature_columns,
        f"No anchor training rows before anchor cutoff: {frozen_anchor_cutoff.date()}",
    )
    anchor_ratings = final_elo_ratings(anchor_matches_clean, cutoff=frozen_anchor_cutoff)
    anchor_predictor = make_ml_outcome_predictor(
        anchor_model,
        anchor_ratings,
        latest_ranking_snapshot(
            rankings_clean,
            frozen_anchor_cutoff,
            inclusive=bool(tournament_config.get("ranking_cutoff_inclusive", False)),
        ),
        recent_form_snapshot(anchor_matches_clean, frozen_anchor_cutoff),
        feature_columns,
    )
    return (
        blend_match_probability_predictors(anchor_predictor, live_predictor, live_weight),
        _live_update_metadata(
            live_weight,
            completed_knockout_matches,
            tournament_config,
            anchor_cutoff=frozen_anchor_cutoff,
        ),
    )


def _load_optional_rankings(path: Path, team_mapping: Mapping[str, str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return clean_rankings(read_csv(path), team_mapping)


def _run_backtests(
    features: pd.DataFrame,
    model_config: Mapping[str, Any],
    backtest_config: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    configured_candidates = model_config.get("backtest_model_candidates")
    if configured_candidates:
        model_names = [str(name) for name in configured_candidates]
        missing = sorted(set(model_names) - set(model_specs))
        if missing:
            raise ValueError(f"backtest_model_candidates are not configured under models: {missing}")
    else:
        model_names = list(model_specs)
    windows = parse_world_cup_windows(backtest_config)
    rows: list[pd.DataFrame] = []

    for model_name in model_names:
        spec = model_specs[model_name]
        random_seed = int(model_config.get("random_seed", RANDOM_SEED))
        result = rolling_world_cup_backtest(
            features,
            model_factory=lambda spec=spec, model_name=model_name, random_seed=random_seed: _model_from_spec(
                spec,
                model_name,
                random_seed,
            ),
            feature_columns=feature_columns,
            windows=windows,
            target_column=str(model_config.get("target_column", "target")),
        )
        if not result.empty:
            result.insert(0, "model", model_name)
        rows.append(result)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _model_factories_from_config(model_config: Mapping[str, Any]) -> dict[str, Callable[[], object]]:
    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    random_seed = int(model_config.get("random_seed", RANDOM_SEED))
    factories: dict[str, Callable[[], object]] = {}
    for model_name, spec in model_specs.items():
        factories[str(model_name)] = (
            lambda spec=spec, model_name=str(model_name), random_seed=random_seed: _model_from_spec(
                spec,
                model_name,
                random_seed,
            )
        )
    return factories


def _run_research_evaluation_outputs(
    features: pd.DataFrame,
    model_config: Mapping[str, Any],
    backtest_config: Mapping[str, Any],
    feature_columns: Sequence[str],
    average_total_goals: float,
    draw_probability: float,
    root: Path = PROJECT_ROOT,
) -> dict[str, Path | None]:
    evaluation_dir = root / "outputs" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    windows = parse_world_cup_windows(backtest_config)
    target_column = str(model_config.get("target_column", "target"))
    primary_metric = str(model_config.get("primary_metric", "log_loss"))
    primary_model_name = str(model_config.get("primary_model") or next(iter((model_config.get("models") or {"logistic": {}}).keys())))
    model_factories = _model_factories_from_config(model_config)
    if primary_model_name not in model_factories:
        raise ValueError(f"primary_model is not configured under models: {primary_model_name}")
    primary_factory = model_factories[primary_model_name]
    research_config = model_config.get("research_evaluation") or {}
    nested_candidate_names = [
        str(name)
        for name in research_config.get("nested_model_candidates", [primary_model_name])
        if str(name) in model_factories
    ]
    nested_model_factories = {
        name: model_factories[name]
        for name in nested_candidate_names
    }

    outputs: dict[str, Path | None] = {}

    baseline = run_baseline_comparison(
        features,
        primary_factory,
        feature_columns,
        windows,
        average_total_goals=average_total_goals,
        draw_probability=draw_probability,
        target_column=target_column,
    )
    baseline_path = evaluation_dir / "baseline_comparison.csv"
    baseline_summary_path = evaluation_dir / "baseline_comparison_summary.csv"
    write_csv(baseline, baseline_path)
    write_csv(summarize_backtest_like_results(baseline, "model", primary_metric), baseline_summary_path)
    outputs["baseline_comparison"] = baseline_path
    outputs["baseline_comparison_summary"] = baseline_summary_path

    ablation = run_ablation_study(
        features,
        primary_factory,
        feature_columns,
        windows,
        target_column=target_column,
    )
    ablation_path = evaluation_dir / "ablation_results.csv"
    ablation_summary_path = evaluation_dir / "ablation_summary.csv"
    write_csv(ablation, ablation_path)
    write_csv(summarize_backtest_like_results(ablation, "feature_set", primary_metric), ablation_summary_path)
    outputs["ablation_results"] = ablation_path
    outputs["ablation_summary"] = ablation_summary_path

    nested = (
        run_nested_model_selection_backtest(
            features,
            nested_model_factories,
            feature_columns,
            windows,
            primary_metric=primary_metric,
            target_column=target_column,
        )
        if nested_model_factories
        else pd.DataFrame()
    )
    nested_path = evaluation_dir / "nested_backtest_results.csv"
    write_csv(nested, nested_path)
    outputs["nested_backtest_results"] = nested_path

    predictions = rolling_model_prediction_records(
        features,
        primary_factory,
        feature_columns,
        windows,
        primary_model_name,
        target_column=target_column,
    )
    predictions_path = evaluation_dir / "rolling_prediction_records.csv"
    write_csv(predictions, predictions_path)
    outputs["rolling_prediction_records"] = predictions_path

    if predictions.empty:
        outputs["calibration_table"] = None
        outputs["calibration_summary"] = None
        outputs["calibration_by_world_cup"] = None
        outputs["probability_sharpness_report"] = None
        return outputs

    calibration_table_path = evaluation_dir / "calibration_table_by_probability_bin.csv"
    calibration_summary_path = evaluation_dir / "calibration_summary.csv"
    calibration_by_world_cup_path = evaluation_dir / "calibration_by_world_cup.csv"
    sharpness_path = evaluation_dir / "probability_sharpness_report.csv"

    probabilities = predictions[["team_a_loss", "draw", "team_a_win"]]
    write_csv(
        calibration_table_by_probability_bin(predictions[target_column], probabilities),
        calibration_table_path,
    )
    write_csv(
        pd.DataFrame([top_label_calibration_summary(predictions[target_column], probabilities)]),
        calibration_summary_path,
    )
    write_csv(
        calibration_by_group(predictions, "year", target_column=target_column),
        calibration_by_world_cup_path,
    )
    write_csv(probability_sharpness_report(probabilities), sharpness_path)
    outputs["calibration_table"] = calibration_table_path
    outputs["calibration_summary"] = calibration_summary_path
    outputs["calibration_by_world_cup"] = calibration_by_world_cup_path
    outputs["probability_sharpness_report"] = sharpness_path
    return outputs


def summarize_backtests(backtests: pd.DataFrame, primary_metric: str = "log_loss") -> pd.DataFrame:
    if backtests.empty:
        return pd.DataFrame()
    required = {"model", "accuracy", "top1_accuracy", "log_loss", "brier_score"}
    missing = sorted(required - set(backtests.columns))
    if missing:
        raise ValueError(f"Backtest data is missing columns: {missing}")
    aggregations: dict[str, tuple[str, str]] = {
        "windows": ("year", "count"),
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "log_loss_mean": ("log_loss", "mean"),
        "log_loss_std": ("log_loss", "std"),
        "brier_score_mean": ("brier_score", "mean"),
        "brier_score_std": ("brier_score", "std"),
        "top1_accuracy_mean": ("top1_accuracy", "mean"),
    }
    if "ranked_probability_score" in backtests.columns:
        aggregations["ranked_probability_score_mean"] = ("ranked_probability_score", "mean")
        aggregations["ranked_probability_score_std"] = ("ranked_probability_score", "std")
    summary = backtests.groupby("model", as_index=False).agg(**aggregations).fillna(0.0)
    ascending = primary_metric in {"log_loss", "brier_score"}
    sort_column = f"{primary_metric}_mean" if f"{primary_metric}_mean" in summary.columns else primary_metric
    return summary.sort_values(sort_column, ascending=ascending).reset_index(drop=True)


def _model_from_spec(spec: Mapping[str, Any], model_name: str, random_seed: int):
    kind = str(spec.get("kind", model_name))
    params = {key: value for key, value in spec.items() if key != "kind"}
    return make_model(kind, random_state=random_seed, **params)


def apply_simulation_profile(
    tournament_config: Mapping[str, Any],
    simulation_profile: str | None = None,
) -> dict[str, Any]:
    config = dict(tournament_config)
    selected_profile = simulation_profile or config.get("simulation_profile")
    if not selected_profile:
        return config

    profiles = config.get("simulation_profiles") or {}
    if selected_profile not in profiles:
        available = ", ".join(sorted(str(profile) for profile in profiles)) or "none"
        raise ValueError(f"Unknown simulation profile '{selected_profile}'. Available profiles: {available}")

    profile_config = profiles[selected_profile] or {}
    for key, value in profile_config.items():
        if isinstance(value, Mapping) and isinstance(config.get(key), Mapping):
            config[key] = {**dict(config[key]), **dict(value)}
        else:
            config[key] = value
    config["simulation_profile"] = selected_profile
    return config


def run_reconstructed_live_snapshot(
    forecast_round_key: str,
    cutoff: str | pd.Timestamp,
    data_config_path: str | Path = CONFIG_DIR / "data_config.yaml",
    model_config_path: str | Path = CONFIG_DIR / "model_config.yaml",
    tournament_config_path: str | Path = CONFIG_DIR / "tournament_2026.yaml",
    root: Path = PROJECT_ROOT,
    simulation_profile: str | None = None,
) -> Path:
    cutoff_ts = pd.Timestamp(cutoff)
    data_config = read_yaml(resolve_project_path(data_config_path, root))
    model_config = read_yaml(resolve_project_path(model_config_path, root))
    tournament_config = apply_simulation_profile(
        read_yaml(resolve_project_path(tournament_config_path, root)),
        simulation_profile,
    )

    raw_matches_path = resolve_project_path(data_config["raw_matches_path"], root)
    raw_rankings_path = resolve_project_path(data_config["raw_rankings_path"], root)
    live_fixtures_path = resolve_project_path(
        tournament_config.get("live_results_path", "data/raw/world_cup_2026_matches.csv"),
        root,
    )
    if not raw_matches_path.exists():
        raise FileNotFoundError(f"Raw match data not found: {raw_matches_path}")
    if not live_fixtures_path.exists():
        raise FileNotFoundError(f"Live fixture/results file not found: {live_fixtures_path}")

    team_mapping_path = data_config.get("team_mapping_path")
    team_mapping = load_team_mapping(str(resolve_project_path(team_mapping_path, root))) if team_mapping_path else {}
    teams_by_group = load_teams_by_group(tournament_config, root)
    knockout_bracket = resolve_knockout_bracket_config(tournament_config, root)
    live_fixtures = fixture_frame_for_reconstructed_round(
        read_csv(live_fixtures_path),
        forecast_round_key,
        cutoff_ts,
    )

    matches_clean = clean_matches(read_csv(raw_matches_path), team_mapping)
    anchor_matches_clean = matches_clean.copy()
    completed_group_matches = completed_group_matches_from_fixture_frame(live_fixtures, team_mapping)
    expected_matches = expected_group_match_count(teams_by_group)
    if len(completed_group_matches) < expected_matches:
        raise ValueError(
            "Reconstructed live forecast requires the completed group stage. "
            f"Found {len(completed_group_matches)} completed group matches, expected {expected_matches}."
        )
    anchor_cutoff = _pre_knockout_cutoff_from_fixture_frame(live_fixtures) or cutoff_ts
    completed_group_table = group_table_from_completed_matches(teams_by_group, completed_group_matches)
    completed_knockout_matches = completed_knockout_matches_from_fixture_frame(
        live_fixtures,
        knockout_bracket,
        team_mapping,
        group_table=completed_group_table,
        third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
    )
    live_training_matches = completed_fixture_matches_for_training(
        live_fixtures,
        team_mapping,
        include_knockout=True,
    )
    anchor_live_training_matches = completed_fixture_matches_for_training(
        live_fixtures,
        team_mapping,
        include_knockout=False,
    )
    if not anchor_live_training_matches.empty:
        anchor_matches_clean = merge_live_training_matches(anchor_matches_clean, anchor_live_training_matches)
    if not live_training_matches.empty:
        matches_clean = merge_live_training_matches(matches_clean, live_training_matches)

    rankings_clean = _load_optional_rankings(raw_rankings_path, team_mapping)
    matches_with_elo = add_elo_features(matches_clean)
    features = build_feature_table(matches_with_elo, rankings_clean)
    requested_columns = model_config.get("baseline_feature_columns") or DEFAULT_FEATURE_COLUMNS
    feature_columns = available_feature_columns(features, requested_columns)
    if not feature_columns:
        raise ValueError("No configured feature columns are available in the generated feature table")

    train_frame = features[pd.to_datetime(features["date"], errors="coerce") < cutoff_ts].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows before reconstructed cutoff: {cutoff_ts.date()}")
    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    primary_model_name = str(model_config.get("primary_model") or next(iter(model_specs.keys())))
    if primary_model_name not in model_specs:
        raise ValueError(f"primary_model is not configured under models: {primary_model_name}")
    primary_model = train_model(
        _model_from_spec(
            model_specs[primary_model_name],
            primary_model_name,
            int(model_config.get("random_seed", RANDOM_SEED)),
        ),
        train_frame,
        feature_columns,
        target_column=str(model_config.get("target_column", "target")),
    )

    ratings = final_elo_ratings(matches_clean, cutoff=cutoff_ts)
    rating_frame = matches_clean[pd.to_datetime(matches_clean["date"], errors="coerce") < cutoff_ts]
    average_total_goals = float(
        tournament_config.get(
            "average_total_goals",
            (rating_frame["team_a_score"] + rating_frame["team_b_score"]).mean(),
        )
    )
    simulation_predictor = str(tournament_config.get("simulation_predictor", "elo_poisson"))
    forecast_metadata: dict[str, Any] | None = None
    if simulation_predictor == "elo_poisson":
        predictor = make_elo_poisson_predictor(ratings, average_total_goals=average_total_goals)
    elif simulation_predictor == "elo_baseline":
        predictor = make_elo_probability_predictor(
            ratings,
            draw_probability=float(tournament_config.get("draw_probability", 0.24)),
        )
    elif simulation_predictor == "ml_outcome":
        predictor, forecast_metadata = _make_anchored_live_predictor(
            primary_model,
            anchor_matches_clean,
            rankings_clean,
            cutoff_ts,
            model_config,
            model_specs,
            primary_model_name,
            feature_columns,
            ratings,
            matches_clean,
            completed_knockout_matches,
            tournament_config,
            anchor_cutoff=anchor_cutoff,
        )
    else:
        raise ValueError(f"Unsupported simulation_predictor: {simulation_predictor}")

    simulation_count = int(tournament_config.get("simulation_count", 10_000))
    simulation_outputs = simulate_tournament_detailed(
        teams_by_group,
        predictor,
        n_simulations=simulation_count,
        seed=int(model_config.get("random_seed", RANDOM_SEED)),
        third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
        knockout_bracket=knockout_bracket,
        completed_group_matches=completed_group_matches,
        completed_knockout_matches=completed_knockout_matches,
    )
    group_match_probabilities = match_probability_frame(teams_by_group, predictor)
    knockout_comparison = knockout_model_vs_monte_carlo_frame(simulation_outputs["knockout_bracket"])
    return write_forecast_registry_frames(
        root,
        "reconstructed_live",
        cutoff_ts,
        primary_model_name,
        simulation_predictor,
        simulation_count,
        feature_columns,
        {
            "simulation": simulation_outputs["team_probabilities"],
            "group_positions": simulation_outputs["group_positions"],
            "knockout_bracket": simulation_outputs["knockout_bracket"],
            "knockout_comparison": knockout_comparison,
        },
        match_probabilities=group_match_probabilities,
        metadata={
            "reconstructed": True,
            "forecast_round": forecast_round_key,
            "simulation_profile": tournament_config.get("simulation_profile", ""),
            "source_live_results": registry_path_reference(live_fixtures_path, root),
            **(forecast_metadata or {}),
        },
    )


def run_analysis(
    data_config_path: str | Path = CONFIG_DIR / "data_config.yaml",
    model_config_path: str | Path = CONFIG_DIR / "model_config.yaml",
    backtest_config_path: str | Path = CONFIG_DIR / "backtest_config.yaml",
    tournament_config_path: str | Path = CONFIG_DIR / "tournament_2026.yaml",
    root: Path = PROJECT_ROOT,
    live: bool = False,
    pre_knockout: bool = False,
    simulation_profile: str | None = None,
) -> dict[str, Path | None]:
    if live and pre_knockout:
        raise ValueError("Use either live=True or pre_knockout=True, not both")

    data_config = read_yaml(resolve_project_path(data_config_path, root))
    model_config = read_yaml(resolve_project_path(model_config_path, root))
    backtest_config = read_yaml(resolve_project_path(backtest_config_path, root))
    tournament_config = apply_simulation_profile(
        read_yaml(resolve_project_path(tournament_config_path, root)),
        simulation_profile,
    )

    raw_matches_path = resolve_project_path(data_config["raw_matches_path"], root)
    raw_rankings_path = resolve_project_path(data_config["raw_rankings_path"], root)
    if not raw_matches_path.exists():
        raise FileNotFoundError(f"Raw match data not found: {raw_matches_path}")

    team_mapping_path = data_config.get("team_mapping_path")
    team_mapping = load_team_mapping(str(resolve_project_path(team_mapping_path, root))) if team_mapping_path else {}
    mode = "live" if live else "pre_knockout" if pre_knockout else str(tournament_config.get("mode", "pre_tournament"))
    matches_clean = clean_matches(read_csv(raw_matches_path), team_mapping)
    anchor_matches_clean = matches_clean.copy()
    teams_by_group = load_teams_by_group(tournament_config, root)
    knockout_bracket = resolve_knockout_bracket_config(tournament_config, root)
    completed_group_matches: list[dict[str, Any]] = []
    completed_knockout_matches: list[dict[str, Any]] = []
    completed_group_table = pd.DataFrame()
    live_fixtures_path = resolve_project_path(
        tournament_config.get("live_results_path", "data/raw/world_cup_2026_matches.csv"),
        root,
    )
    live_fixtures: pd.DataFrame | None = None
    anchor_cutoff: pd.Timestamp | None = None
    if mode in {"live", "pre_knockout"}:
        if not live_fixtures_path.exists():
            raise FileNotFoundError(f"Live fixture/results file not found: {live_fixtures_path}")
        live_fixtures = read_csv(live_fixtures_path)
        anchor_cutoff = _pre_knockout_cutoff_from_fixture_frame(live_fixtures)
        completed_group_matches = completed_group_matches_from_fixture_frame(live_fixtures, team_mapping)
        if len(completed_group_matches) >= expected_group_match_count(teams_by_group):
            completed_group_table = group_table_from_completed_matches(teams_by_group, completed_group_matches)
        if mode == "pre_knockout":
            expected_matches = expected_group_match_count(teams_by_group)
            if len(completed_group_matches) < expected_matches:
                raise ValueError(
                    "Pre-knockout forecast requires the completed group stage. "
                    f"Found {len(completed_group_matches)} completed group matches, expected {expected_matches}."
                )
        if mode == "live":
            completed_knockout_matches = completed_knockout_matches_from_fixture_frame(
                live_fixtures,
                knockout_bracket,
                team_mapping,
                group_table=completed_group_table,
                third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
            )
            anchor_live_training_matches = completed_fixture_matches_for_training(
                live_fixtures,
                team_mapping,
                include_knockout=False,
            )
            if not anchor_live_training_matches.empty:
                anchor_matches_clean = merge_live_training_matches(anchor_matches_clean, anchor_live_training_matches)
        live_training_matches = completed_fixture_matches_for_training(
            live_fixtures,
            team_mapping,
            include_knockout=mode == "live",
        )
        if not live_training_matches.empty:
            matches_clean = merge_live_training_matches(matches_clean, live_training_matches)
    rankings_clean = _load_optional_rankings(raw_rankings_path, team_mapping)

    processed_matches_path = resolve_project_path(data_config["processed_matches_path"], root)
    processed_rankings_path = resolve_project_path(data_config["processed_rankings_path"], root)
    features_path = resolve_project_path(data_config["features_path"], root)

    write_csv(matches_clean, processed_matches_path)
    if rankings_clean is not None:
        write_csv(rankings_clean, processed_rankings_path)

    matches_with_elo = add_elo_features(matches_clean)
    features = build_feature_table(matches_with_elo, rankings_clean)
    requested_columns = model_config.get("baseline_feature_columns") or DEFAULT_FEATURE_COLUMNS
    feature_columns = available_feature_columns(features, requested_columns)
    if not feature_columns:
        raise ValueError("No configured feature columns are available in the generated feature table")
    write_csv(features, features_path)

    backtest_output_path = root / "outputs" / "backtest_results" / "model_backtest.csv"
    backtest_summary_output_path = root / "outputs" / "backtest_results" / "model_backtest_summary.csv"
    backtest = _run_backtests(features, model_config, backtest_config, feature_columns)
    backtest_summary = summarize_backtests(backtest, str(model_config.get("primary_metric", "log_loss")))
    write_csv(backtest, backtest_output_path)
    write_csv(backtest_summary, backtest_summary_output_path)
    historical_average_total_goals = float((features["team_a_score"] + features["team_b_score"]).mean())
    evaluation_outputs = _run_research_evaluation_outputs(
        features,
        model_config,
        backtest_config,
        feature_columns,
        average_total_goals=historical_average_total_goals,
        draw_probability=float(tournament_config.get("draw_probability", 0.24)),
        root=root,
    )

    cutoff = pd.Timestamp(tournament_config.get("data_cutoff", data_config.get("pre_tournament_cutoff", "2026-06-11")))
    if mode == "live" and live_fixtures is not None:
        live_dates = _completed_fixture_dates(live_fixtures)
        if live_dates.notna().any():
            cutoff = live_dates.max() + pd.Timedelta(days=1)
    elif mode == "pre_knockout" and live_fixtures is not None:
        if anchor_cutoff is not None:
            cutoff = anchor_cutoff
    train_frame = features[features["date"] < cutoff].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows before tournament cutoff: {cutoff.date()}")
    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    primary_model_name = str(model_config.get("primary_model") or next(iter(model_specs.keys())))
    if primary_model_name not in model_specs:
        raise ValueError(f"primary_model is not configured under models: {primary_model_name}")
    primary_model_spec = model_specs[primary_model_name]
    primary_model = train_model(
        _model_from_spec(
            primary_model_spec,
            primary_model_name,
            int(model_config.get("random_seed", RANDOM_SEED)),
        ),
        train_frame,
        feature_columns,
        target_column=str(model_config.get("target_column", "target")),
    )

    simulation_output_path: Path | None = None
    group_positions_output_path: Path | None = None
    bracket_output_path: Path | None = None
    simulation_interval_output_path: Path | None = None
    match_probabilities_output_path: Path | None = None
    knockout_comparison_output_path: Path | None = None
    forecast_registry_output_path: Path | None = None
    if teams_by_group:
        ratings = final_elo_ratings(matches_clean, cutoff=cutoff)
        rating_frame = matches_clean[pd.to_datetime(matches_clean["date"], errors="coerce") < cutoff]
        average_total_goals = float(
            tournament_config.get(
                "average_total_goals",
                (rating_frame["team_a_score"] + rating_frame["team_b_score"]).mean(),
            )
        )
        simulation_predictor = str(tournament_config.get("simulation_predictor", "elo_poisson"))
        forecast_metadata: dict[str, Any] | None = None
        if simulation_predictor == "elo_poisson":
            predictor = make_elo_poisson_predictor(ratings, average_total_goals=average_total_goals)
        elif simulation_predictor == "elo_baseline":
            predictor = make_elo_probability_predictor(
                ratings,
                draw_probability=float(tournament_config.get("draw_probability", 0.24)),
            )
        elif simulation_predictor == "ml_outcome":
            if mode == "live":
                predictor, forecast_metadata = _make_anchored_live_predictor(
                    primary_model,
                    anchor_matches_clean,
                    rankings_clean,
                    cutoff,
                    model_config,
                    model_specs,
                    primary_model_name,
                    feature_columns,
                    ratings,
                    matches_clean,
                    completed_knockout_matches,
                    tournament_config,
                    anchor_cutoff=anchor_cutoff,
                )
            else:
                predictor = make_ml_outcome_predictor(
                    primary_model,
                    ratings,
                    latest_ranking_snapshot(
                        rankings_clean,
                        cutoff,
                        inclusive=bool(tournament_config.get("ranking_cutoff_inclusive", False)),
                    ),
                    recent_form_snapshot(matches_clean, cutoff),
                    feature_columns,
                )
        else:
            raise ValueError(f"Unsupported simulation_predictor: {simulation_predictor}")
        simulation_outputs = simulate_tournament_detailed(
            teams_by_group,
            predictor,
            n_simulations=int(tournament_config.get("simulation_count", 10_000)),
            seed=int(model_config.get("random_seed", RANDOM_SEED)),
            third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
            knockout_bracket=knockout_bracket,
            completed_group_matches=completed_group_matches,
            completed_knockout_matches=completed_knockout_matches,
        )
        suffix = "_live" if mode == "live" else "_pre_knockout" if mode == "pre_knockout" else ""
        simulation_output_path = root / "outputs" / "simulations" / f"team_probabilities_2026{suffix}.csv"
        group_positions_output_path = root / "outputs" / "simulations" / f"group_position_probabilities_2026{suffix}.csv"
        bracket_output_path = root / "outputs" / "simulations" / f"predicted_knockout_bracket_2026{suffix}.csv"
        match_probabilities_output_path = root / "outputs" / "simulations" / f"match_probabilities_2026{suffix}.csv"
        knockout_comparison_output_path = (
            root / "outputs" / "simulations" / f"knockout_model_vs_monte_carlo_2026{suffix}.csv"
        )
        write_csv(simulation_outputs["team_probabilities"], simulation_output_path)
        write_csv(simulation_outputs["group_positions"], group_positions_output_path)
        write_csv(simulation_outputs["knockout_bracket"], bracket_output_path)
        knockout_comparison = knockout_model_vs_monte_carlo_frame(simulation_outputs["knockout_bracket"])
        write_csv(knockout_comparison, knockout_comparison_output_path)
        group_match_probabilities = match_probability_frame(teams_by_group, predictor)
        write_csv(group_match_probabilities, match_probabilities_output_path)

        interval_config = tournament_config.get("simulation_interval") or {}
        if bool(interval_config.get("enabled", True)):
            base_simulation_count = int(tournament_config.get("simulation_count", 10_000))
            interval_simulation_count = int(
                interval_config.get("simulations_per_seed", min(base_simulation_count, 2_000))
            )
            interval_seed_count = int(interval_config.get("seed_count", 5))
            interval_seeds = deterministic_interval_seeds(
                int(model_config.get("random_seed", RANDOM_SEED)),
                interval_seed_count,
            )
            if interval_seeds:
                simulation_interval_output_path = (
                    root / "outputs" / "simulations" / f"team_probabilities_2026{suffix}_with_ci.csv"
                )
                write_csv(
                    simulation_probability_intervals(
                        teams_by_group,
                        predictor,
                        n_simulations=interval_simulation_count,
                        seeds=interval_seeds,
                        third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
                        knockout_bracket=knockout_bracket,
                        completed_group_matches=completed_group_matches,
                        completed_knockout_matches=completed_knockout_matches,
                    ),
                    simulation_interval_output_path,
                )

        forecast_registry_output_path = write_forecast_registry(
            root,
            mode,
            cutoff,
            primary_model_name,
            simulation_predictor,
            int(tournament_config.get("simulation_count", 10_000)),
            feature_columns,
            {
                "simulation": simulation_output_path,
                "group_positions": group_positions_output_path,
                "knockout_bracket": bracket_output_path,
                "knockout_comparison": knockout_comparison_output_path,
                "simulation_interval": simulation_interval_output_path,
            },
            match_probabilities=group_match_probabilities,
            metadata={
                **(forecast_metadata or {}),
                "simulation_profile": tournament_config.get("simulation_profile", ""),
            },
        )

    return {
        "processed_matches": processed_matches_path,
        "processed_rankings": processed_rankings_path if rankings_clean is not None else None,
        "features": features_path,
        "backtest": backtest_output_path,
        "backtest_summary": backtest_summary_output_path,
        **evaluation_outputs,
        "simulation": simulation_output_path,
        "group_positions": group_positions_output_path,
        "knockout_bracket": bracket_output_path,
        "match_probabilities": match_probabilities_output_path,
        "knockout_comparison": knockout_comparison_output_path,
        "simulation_intervals": simulation_interval_output_path,
        "forecast_registry": forecast_registry_output_path,
    }
