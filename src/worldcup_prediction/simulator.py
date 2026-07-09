from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import exp, factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worldcup_prediction.utils import ensure_columns

MatchProbabilityFn = Callable[[str, str, Mapping[str, Any] | None], Mapping[str, float]]


@dataclass
class GroupRecord:
    team: str
    group: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class SimulatedMatch:
    group: str
    team_a: str
    team_b: str
    team_a_score: int
    team_b_score: int


@dataclass(frozen=True)
class ThirdPlaceSlot:
    match_id: int
    team_index: int
    candidates: tuple[str, ...]


THIRD_PLACE_WINNER_SLOTS = {
    "1A": (79, 1),
    "1B": (85, 1),
    "1D": (81, 1),
    "1E": (74, 1),
    "1G": (82, 1),
    "1I": (77, 1),
    "1K": (87, 1),
    "1L": (80, 1),
}


def normalize_match_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    aliases = {
        "team_a_win": ["team_a_win", "home_win", "a_win", "2"],
        "draw": ["draw", "1"],
        "team_b_win": ["team_b_win", "team_a_loss", "away_win", "b_win", "0"],
    }
    normalized: dict[str, float] = {}
    for target, names in aliases.items():
        normalized[target] = float(next((probabilities[name] for name in names if name in probabilities), 0.0))
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Match probabilities must have positive total mass")
    return {key: value / total for key, value in normalized.items()}


def generate_round_robin_matches(teams_by_group: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for group, teams in teams_by_group.items():
        teams = list(teams)
        for index, team_a in enumerate(teams):
            for team_b in teams[index + 1 :]:
                matches.append({"group": group, "team_a": team_a, "team_b": team_b})
    return matches


def group_match_key(match: Mapping[str, Any]) -> tuple[str, frozenset[str]]:
    return (
        str(match["group"]),
        frozenset([str(match["team_a"]), str(match["team_b"])]),
    )


def sample_scoreline(probabilities: Mapping[str, float], rng: np.random.Generator) -> tuple[int, int]:
    if "team_a_goals_lambda" in probabilities and "team_b_goals_lambda" in probabilities:
        lambda_a = max(float(probabilities["team_a_goals_lambda"]), 0.0)
        lambda_b = max(float(probabilities["team_b_goals_lambda"]), 0.0)
        return int(rng.poisson(lambda_a)), int(rng.poisson(lambda_b))

    probs = normalize_match_probabilities(probabilities)
    outcome = rng.choice(["team_a_win", "draw", "team_b_win"], p=[probs["team_a_win"], probs["draw"], probs["team_b_win"]])
    if outcome == "draw":
        score = int(rng.choice([0, 1, 2], p=[0.25, 0.55, 0.20]))
        return score, score
    if outcome == "team_a_win":
        return tuple(rng.choice([(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)], p=[0.30, 0.20, 0.30, 0.12, 0.08]))
    return tuple(rng.choice([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], p=[0.30, 0.20, 0.30, 0.12, 0.08]))


def poisson_outcome_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 10) -> dict[str, float]:
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("Poisson goal expectations must be non-negative")
    goals = range(max_goals + 1)
    probs_a = [exp(-lambda_a) * lambda_a**goal / factorial(goal) for goal in goals]
    probs_b = [exp(-lambda_b) * lambda_b**goal / factorial(goal) for goal in goals]

    team_a_win = 0.0
    draw = 0.0
    team_b_win = 0.0
    for goals_a, prob_a in zip(goals, probs_a, strict=False):
        for goals_b, prob_b in zip(goals, probs_b, strict=False):
            mass = prob_a * prob_b
            if goals_a > goals_b:
                team_a_win += mass
            elif goals_a == goals_b:
                draw += mass
            else:
                team_b_win += mass
    return normalize_match_probabilities(
        {
            "team_a_win": team_a_win,
            "draw": draw,
            "team_b_win": team_b_win,
        }
    )


def update_group_record(record_a: GroupRecord, record_b: GroupRecord, score_a: int, score_b: int) -> None:
    record_a.goals_for += score_a
    record_a.goals_against += score_b
    record_b.goals_for += score_b
    record_b.goals_against += score_a

    if score_a > score_b:
        record_a.points += 3
        record_a.wins += 1
    elif score_a < score_b:
        record_b.points += 3
        record_b.wins += 1
    else:
        record_a.points += 1
        record_b.points += 1


def apply_completed_group_matches(
    table: Mapping[str, Mapping[str, GroupRecord]],
    completed_matches: Sequence[Mapping[str, Any]],
) -> list[SimulatedMatch]:
    played_matches: list[SimulatedMatch] = []
    seen: set[tuple[str, frozenset[str]]] = set()
    for match in completed_matches:
        group = str(match["group"])
        team_a = str(match["team_a"])
        team_b = str(match["team_b"])
        key = group_match_key({"group": group, "team_a": team_a, "team_b": team_b})
        if key in seen:
            raise ValueError(f"Duplicate completed group match: {group} {team_a} vs {team_b}")
        seen.add(key)
        if group not in table or team_a not in table[group] or team_b not in table[group]:
            raise ValueError(f"Completed match is not in configured groups: {group} {team_a} vs {team_b}")
        score_a = int(match["team_a_score"])
        score_b = int(match["team_b_score"])
        update_group_record(table[group][team_a], table[group][team_b], score_a, score_b)
        played_matches.append(SimulatedMatch(group, team_a, team_b, score_a, score_b))
    return played_matches


def _head_to_head_stats(team: str, tied_teams: set[str], matches: Sequence[SimulatedMatch]) -> tuple[int, int, int]:
    points = 0
    goals_for = 0
    goals_against = 0
    for match in matches:
        if match.team_a == team and match.team_b in tied_teams:
            goals_for += match.team_a_score
            goals_against += match.team_b_score
            if match.team_a_score > match.team_b_score:
                points += 3
            elif match.team_a_score == match.team_b_score:
                points += 1
        elif match.team_b == team and match.team_a in tied_teams:
            goals_for += match.team_b_score
            goals_against += match.team_a_score
            if match.team_b_score > match.team_a_score:
                points += 3
            elif match.team_b_score == match.team_a_score:
                points += 1
    return points, goals_for - goals_against, goals_for


def rank_group(records: Sequence[GroupRecord], matches: Sequence[SimulatedMatch], rng: np.random.Generator | None = None) -> list[GroupRecord]:
    rng = rng or np.random.default_rng()
    random_tiebreakers = {record.team: rng.random() for record in records}

    base_groups: defaultdict[tuple[int, int, int], set[str]] = defaultdict(set)
    for record in records:
        base_groups[(record.points, record.goal_difference, record.goals_for)].add(record.team)

    def sort_key(record: GroupRecord) -> tuple[float, ...]:
        tied_teams = base_groups[(record.points, record.goal_difference, record.goals_for)]
        h2h_points, h2h_gd, h2h_gf = _head_to_head_stats(record.team, tied_teams, matches)
        return (
            record.points,
            record.goal_difference,
            record.goals_for,
            h2h_points,
            h2h_gd,
            h2h_gf,
            record.wins,
            random_tiebreakers[record.team],
        )

    return sorted(records, key=sort_key, reverse=True)


def simulate_group_stage(
    teams_by_group: Mapping[str, Sequence[str]],
    predict_match: MatchProbabilityFn,
    rng: np.random.Generator | None = None,
    group_matches: Sequence[Mapping[str, str]] | None = None,
    completed_matches: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[SimulatedMatch]]:
    rng = rng or np.random.default_rng()
    table = {
        group: {team: GroupRecord(team=team, group=group) for team in teams}
        for group, teams in teams_by_group.items()
    }
    played_matches = apply_completed_group_matches(table, completed_matches or [])
    completed_keys = {group_match_key(match.__dict__) for match in played_matches}
    matches = [
        match
        for match in list(group_matches or generate_round_robin_matches(teams_by_group))
        if group_match_key(match) not in completed_keys
    ]

    for match in matches:
        group = match["group"]
        team_a = match["team_a"]
        team_b = match["team_b"]
        probabilities = predict_match(team_a, team_b, match)
        score_a, score_b = sample_scoreline(probabilities, rng)
        update_group_record(table[group][team_a], table[group][team_b], score_a, score_b)
        played_matches.append(SimulatedMatch(group, team_a, team_b, score_a, score_b))

    rows: list[dict[str, Any]] = []
    for group, records_by_team in table.items():
        group_played = [match for match in played_matches if match.group == group]
        ranked = rank_group(list(records_by_team.values()), group_played, rng)
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
    return pd.DataFrame(rows), played_matches


def select_group_qualifiers(group_table: pd.DataFrame, third_place_count: int = 8) -> pd.DataFrame:
    top_two = group_table[group_table["position"] <= 2].copy()
    third_place = group_table[group_table["position"] == 3].copy()
    third_place = third_place.sort_values(
        ["points", "goal_difference", "goals_for", "wins"],
        ascending=[False, False, False, False],
    ).head(third_place_count)
    qualifiers = pd.concat([top_two, third_place], ignore_index=True)
    return qualifiers.reset_index(drop=True)


def advancement_probability(probabilities: Mapping[str, float]) -> float:
    probs = normalize_match_probabilities(probabilities)
    return probs["team_a_win"] + 0.5 * probs["draw"]


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    predict_match: MatchProbabilityFn,
    rng: np.random.Generator,
    context: Mapping[str, Any] | None = None,
) -> str:
    probability_a_advances = advancement_probability(predict_match(team_a, team_b, context))
    return team_a if rng.random() < probability_a_advances else team_b


def pair_seeded_qualifiers(qualifiers: Sequence[str]) -> list[tuple[str, str]]:
    teams = list(qualifiers)
    if len(teams) % 2 != 0:
        raise ValueError("Knockout qualifiers must be an even number")
    return [(teams[index], teams[-index - 1]) for index in range(len(teams) // 2)]


def pair_bracket_winners(winners: Sequence[str]) -> list[tuple[str, str]]:
    teams = list(winners)
    if len(teams) % 2 != 0:
        raise ValueError("Knockout winners must be an even number")
    return [(teams[index], teams[index + 1]) for index in range(0, len(teams), 2)]


def _normalize_group_label(value: Any) -> str:
    return str(value).strip().upper()


def _group_position_lookup(group_table: pd.DataFrame) -> dict[tuple[str, int], str]:
    ensure_columns(group_table, ["group", "position", "team"], "group_table")
    lookup: dict[tuple[str, int], str] = {}
    for row in group_table.itertuples(index=False):
        lookup[(_normalize_group_label(row.group), int(row.position))] = str(row.team)
    return lookup


def _slot_team(
    slot: Mapping[str, Any],
    position_lookup: Mapping[tuple[str, int], str],
    third_place_assignments: Mapping[tuple[int, int], str],
) -> str:
    if "group" in slot and "position" in slot:
        key = (_normalize_group_label(slot["group"]), int(slot["position"]))
        if key not in position_lookup:
            raise ValueError(f"Missing group position in simulated table: group={key[0]}, position={key[1]}")
        return position_lookup[key]
    if "third_place_from" in slot:
        third_place_key = (int(slot["match"]), int(slot["team_index"]))
        if third_place_key not in third_place_assignments:
            raise ValueError(f"Missing third-place assignment for match slot: {third_place_key}")
        assigned_group = third_place_assignments[third_place_key]
        key = (assigned_group, 3)
        if key not in position_lookup:
            raise ValueError(f"Missing third-place team for group: {assigned_group}")
        return position_lookup[key]
    raise ValueError(f"Unsupported knockout slot config: {slot}")


def assign_third_place_slots(
    slots: Sequence[ThirdPlaceSlot],
    third_place_teams_by_group: Mapping[str, str],
) -> dict[tuple[int, int], str]:
    normalized_thirds = {_normalize_group_label(group): team for group, team in third_place_teams_by_group.items()}
    if len(normalized_thirds) < len(slots):
        raise ValueError(
            f"Expected at least {len(slots)} qualified third-place teams for bracket slots, got {len(normalized_thirds)}"
        )

    ordered_slots = list(slots)

    def backtrack(slot_index: int, used_groups: set[str]) -> dict[tuple[int, int], str] | None:
        if slot_index == len(ordered_slots):
            return {}
        slot = ordered_slots[slot_index]
        for group in slot.candidates:
            if group in normalized_thirds and group not in used_groups:
                result = backtrack(slot_index + 1, used_groups | {group})
                if result is not None:
                    result[(slot.match_id, slot.team_index)] = group
                    return result
        return None

    assignments = backtrack(0, set())
    if assignments is None:
        qualified = ", ".join(sorted(normalized_thirds))
        raise ValueError(f"No valid third-place bracket assignment for qualified groups: {qualified}")
    return assignments


@lru_cache(maxsize=4)
def _official_third_place_mapping(mapping_path: str) -> dict[str, dict[tuple[int, int], str]]:
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Official third-place mapping table does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"qualified_third_groups", *[f"third_for_{winner_slot}" for winner_slot in THIRD_PLACE_WINNER_SLOTS]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Official third-place mapping table is missing columns: {missing}")

    mapping: dict[str, dict[tuple[int, int], str]] = {}
    for row in frame.itertuples(index=False):
        qualified_groups = "".join(sorted(_normalize_group_label(row.qualified_third_groups)))
        assignments: dict[tuple[int, int], str] = {}
        for winner_slot, match_slot in THIRD_PLACE_WINNER_SLOTS.items():
            assigned_group = _normalize_group_label(getattr(row, f"third_for_{winner_slot}"))
            if assigned_group.startswith("3"):
                assigned_group = assigned_group[1:]
            assignments[match_slot] = assigned_group
        mapping[qualified_groups] = assignments

    if len(mapping) != 495:
        raise ValueError(f"Official third-place mapping table should contain 495 combinations, got {len(mapping)}")
    return mapping


def assign_third_place_slots_from_official_table(
    slots: Sequence[ThirdPlaceSlot],
    third_place_teams_by_group: Mapping[str, str],
    mapping_path: str,
) -> dict[tuple[int, int], str]:
    normalized_thirds = {_normalize_group_label(group): team for group, team in third_place_teams_by_group.items()}
    qualified_key = "".join(sorted(normalized_thirds))
    official_mapping = _official_third_place_mapping(str(mapping_path))
    if qualified_key not in official_mapping:
        raise ValueError(f"No official third-place mapping for qualified groups: {qualified_key}")

    assignments: dict[tuple[int, int], str] = {}
    by_slot = {(slot.match_id, slot.team_index): slot for slot in slots}
    for match_slot, slot in by_slot.items():
        if match_slot not in official_mapping[qualified_key]:
            raise ValueError(f"Official third-place mapping does not define match slot: {match_slot}")
        assigned_group = official_mapping[qualified_key][match_slot]
        if assigned_group not in normalized_thirds:
            raise ValueError(
                f"Official third-place mapping assigned group {assigned_group}, "
                f"but qualified groups are {qualified_key}"
            )
        if assigned_group not in slot.candidates:
            raise ValueError(
                f"Official third-place mapping assigned group {assigned_group} "
                f"outside allowed candidates {slot.candidates} for match slot {match_slot}"
            )
        assignments[match_slot] = assigned_group
    return assignments


def build_round_of_32_bracket(
    group_table: pd.DataFrame,
    bracket_config: Mapping[str, Any],
    third_place_count: int = 8,
) -> list[dict[str, Any]]:
    round_config = list(bracket_config.get("round_of_32", []))
    if not round_config:
        raise ValueError("knockout_bracket.round_of_32 must contain match definitions")
    strategy = str(bracket_config.get("third_place_assignment_strategy", "first_valid"))

    position_lookup = _group_position_lookup(group_table)
    qualifiers = select_group_qualifiers(group_table, third_place_count=third_place_count)
    third_place = qualifiers[qualifiers["position"] == 3]
    third_place_teams_by_group = {
        _normalize_group_label(row.group): str(row.team)
        for row in third_place.itertuples(index=False)
    }

    third_place_slots: list[ThirdPlaceSlot] = []
    for match in round_config:
        teams = list(match.get("teams", []))
        if len(teams) != 2:
            raise ValueError(f"Round-of-32 match must define exactly two teams: {match}")
        for team_index, slot in enumerate(teams):
            if "third_place_from" not in slot:
                continue
            match_id = int(match["match"])
            candidates = tuple(_normalize_group_label(group) for group in slot["third_place_from"])
            third_place_slots.append(ThirdPlaceSlot(match_id, team_index, candidates))

    if strategy == "first_valid":
        third_place_assignments = assign_third_place_slots(third_place_slots, third_place_teams_by_group)
    elif strategy == "official_table":
        mapping_path = bracket_config.get("third_place_mapping_path")
        if not mapping_path:
            raise ValueError("official_table third-place assignment requires third_place_mapping_path")
        third_place_assignments = assign_third_place_slots_from_official_table(
            third_place_slots,
            third_place_teams_by_group,
            str(mapping_path),
        )
    else:
        raise ValueError(f"Unsupported third-place assignment strategy: {strategy}")
    matches: list[dict[str, Any]] = []
    for match in round_config:
        teams = list(match["teams"])
        match_id = int(match["match"])
        slots = [
            {**teams[0], "match": match_id, "team_index": 0},
            {**teams[1], "match": match_id, "team_index": 1},
        ]
        matches.append(
            {
                "match": match_id,
                "team_a": _slot_team(slots[0], position_lookup, third_place_assignments),
                "team_b": _slot_team(slots[1], position_lookup, third_place_assignments),
            }
        )
    return matches


def simulate_knockout_round(
    pairs: Sequence[tuple[str, str]],
    predict_match: MatchProbabilityFn,
    rng: np.random.Generator,
    round_name: str,
) -> list[str]:
    return [
        simulate_knockout_match(team_a, team_b, predict_match, rng, {"stage": round_name})
        for team_a, team_b in pairs
    ]


def _simulate_configured_knockout(
    group_table: pd.DataFrame,
    predict_match: MatchProbabilityFn,
    rng: np.random.Generator,
    counts: dict[str, dict[str, int]],
    bracket_config: Mapping[str, Any],
    third_place_count: int,
    bracket_counts: dict[tuple[str, int], dict[str, Counter[str]]] | None = None,
    completed_knockout_matches: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    winners_by_match: dict[int, str] = {}
    completed_by_match = {
        int(match["match"]): match
        for match in completed_knockout_matches or []
        if "match" in match
    }
    round_plan = [
        ("round_of_32", "reach_r16"),
        ("round_of_16", "reach_qf"),
        ("quarterfinals", "reach_sf"),
        ("semifinals", "reach_final"),
        ("final", "champion"),
    ]
    stage_names = {
        "round_of_32": "round_of_32",
        "round_of_16": "round_of_16",
        "quarterfinals": "quarterfinal",
        "semifinals": "semifinal",
        "final": "final",
    }

    for round_key, milestone in round_plan:
        if round_key == "round_of_32":
            match_records = build_round_of_32_bracket(
                group_table,
                bracket_config,
                third_place_count=third_place_count,
            )
        else:
            match_records = []
            for match in bracket_config.get(round_key, []):
                source_matches = [int(match_id) for match_id in match["winners_of"]]
                missing = [match_id for match_id in source_matches if match_id not in winners_by_match]
                if missing:
                    raise ValueError(f"Configured bracket references unknown winner match ids: {missing}")
                match_records.append(
                    {
                        "match": int(match["match"]),
                        "team_a": winners_by_match[source_matches[0]],
                        "team_b": winners_by_match[source_matches[1]],
                    }
                )

        if not match_records:
            raise ValueError(f"knockout_bracket.{round_key} must contain match definitions")

        for match in match_records:
            match_id = int(match["match"])
            completed_match = completed_by_match.get(match_id)
            team_a = str(completed_match.get("team_a") or match["team_a"]) if completed_match else str(match["team_a"])
            team_b = str(completed_match.get("team_b") or match["team_b"]) if completed_match else str(match["team_b"])
            if bracket_counts is not None:
                key = (round_key, match_id)
                bracket_counts[key]["team_a"][team_a] += 1
                bracket_counts[key]["team_b"][team_b] += 1
                bracket_counts[key]["appearance"][team_a] += 1
                bracket_counts[key]["appearance"][team_b] += 1
                bracket_counts[key]["matchup"][(team_a, team_b)] += 1
            if completed_match:
                winner = str(completed_match["winner"])
            else:
                winner = simulate_knockout_match(
                    team_a,
                    team_b,
                    predict_match,
                    rng,
                    {"stage": stage_names[round_key], "match": match["match"]},
                )
            winners_by_match[match_id] = winner
            if winner in counts:
                counts[winner][milestone] += 1
            if bracket_counts is not None:
                bracket_counts[(round_key, match_id)]["winner"][winner] += 1
                bracket_counts[(round_key, match_id)]["matchup_winner"][(team_a, team_b, winner)] += 1


def _probability_rows(counts: Mapping[str, Mapping[str, int]], n_simulations: int) -> pd.DataFrame:
    rows = [
        {"team": team, **{key: value / n_simulations for key, value in values.items()}}
        for team, values in counts.items()
    ]
    return pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)


def _group_position_rows(
    position_counts: Mapping[str, Mapping[int, int]],
    teams_by_group: Mapping[str, Sequence[str]],
    n_simulations: int,
) -> pd.DataFrame:
    groups_by_team = {team: group for group, teams in teams_by_group.items() for team in teams}
    rows: list[dict[str, Any]] = []
    for team, counts in position_counts.items():
        probabilities = {f"position_{position}": counts.get(position, 0) / n_simulations for position in range(1, 5)}
        expected_position = sum(
            position * probabilities[f"position_{position}"]
            for position in range(1, 5)
        )
        rows.append(
            {
                "group": groups_by_team[team],
                "team": team,
                **probabilities,
                "expected_position": expected_position,
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "expected_position", "team"]).reset_index(drop=True)


def _top_counter_value(counter: Counter[str], n_simulations: int) -> tuple[str | None, float]:
    if not counter:
        return None, 0.0
    team, count = counter.most_common(1)[0]
    return team, count / n_simulations


def _bracket_rows(
    bracket_counts: Mapping[tuple[str, int], Mapping[str, Counter[str]]],
    n_simulations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (round_name, match_id), counters in sorted(bracket_counts.items(), key=lambda item: item[0][1]):
        team_a, team_a_probability = _top_counter_value(counters["team_a"], n_simulations)
        team_b, team_b_probability = _top_counter_value(counters["team_b"], n_simulations)
        winner, winner_probability = _top_counter_value(counters["winner"], n_simulations)
        matchup_count = counters["matchup"].get((team_a, team_b), 0) if team_a and team_b else 0
        matchup_probability = matchup_count / n_simulations if matchup_count else 0.0
        winner_match_count = (
            counters["matchup_winner"].get((team_a, team_b, winner), 0)
            if team_a and team_b and winner
            else 0
        )
        winner_match_probability = winner_match_count / matchup_count if matchup_count else winner_probability
        rows.append(
            {
                "round": round_name,
                "match": match_id,
                "team_a_top": team_a,
                "team_a_probability": team_a_probability,
                "team_b_top": team_b,
                "team_b_probability": team_b_probability,
                "matchup_probability": matchup_probability,
                "winner_top": winner,
                "winner_probability": winner_probability,
                "winner_match_probability": winner_match_probability,
            }
        )
    return pd.DataFrame(rows)


def simulate_tournament_detailed(
    teams_by_group: Mapping[str, Sequence[str]],
    predict_match: MatchProbabilityFn,
    n_simulations: int = 1000,
    seed: int = 42,
    third_place_count: int = 8,
    knockout_bracket: Mapping[str, Any] | None = None,
    completed_group_matches: Sequence[Mapping[str, Any]] | None = None,
    completed_knockout_matches: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_teams = [team for teams in teams_by_group.values() for team in teams]
    milestones = [
        "group_win",
        "advance_from_group",
        "reach_r32",
        "reach_r16",
        "reach_qf",
        "reach_sf",
        "reach_final",
        "champion",
    ]
    counts = {team: {milestone: 0 for milestone in milestones} for team in all_teams}
    position_counts = {team: {position: 0 for position in range(1, 5)} for team in all_teams}
    bracket_counts: dict[tuple[str, int], dict[str, Counter[str]]] = defaultdict(
        lambda: {
            "team_a": Counter(),
            "team_b": Counter(),
            "appearance": Counter(),
            "winner": Counter(),
            "matchup": Counter(),
            "matchup_winner": Counter(),
        }
    )

    for _ in range(n_simulations):
        group_table, _ = simulate_group_stage(
            teams_by_group,
            predict_match,
            rng,
            completed_matches=completed_group_matches,
        )
        group_winners = set(group_table[group_table["position"] == 1]["team"])
        qualifiers = select_group_qualifiers(group_table, third_place_count=third_place_count)
        qualifier_teams = list(qualifiers["team"])
        for row in group_table.itertuples(index=False):
            position_counts[str(row.team)][int(row.position)] += 1

        for team in group_winners:
            counts[team]["group_win"] += 1
        for team in qualifier_teams:
            counts[team]["advance_from_group"] += 1
            counts[team]["reach_r32"] += 1

        if knockout_bracket:
            _simulate_configured_knockout(
                group_table,
                predict_match,
                rng,
                counts,
                knockout_bracket,
                third_place_count,
                bracket_counts,
                completed_knockout_matches,
            )
        else:
            pairs = pair_seeded_qualifiers(qualifier_teams)
            round_names = [
                ("round_of_32", "reach_r16"),
                ("round_of_16", "reach_qf"),
                ("quarterfinal", "reach_sf"),
                ("semifinal", "reach_final"),
                ("final", "champion"),
            ]
            current_pairs = pairs
            for round_name, milestone in round_names:
                winners = simulate_knockout_round(current_pairs, predict_match, rng, round_name)
                for team in winners:
                    counts[team][milestone] += 1
                if len(winners) == 1:
                    break
                current_pairs = pair_bracket_winners(winners)

    return {
        "team_probabilities": _probability_rows(counts, n_simulations),
        "group_positions": _group_position_rows(position_counts, teams_by_group, n_simulations),
        "knockout_bracket": _bracket_rows(bracket_counts, n_simulations),
    }


def simulate_tournament(
    teams_by_group: Mapping[str, Sequence[str]],
    predict_match: MatchProbabilityFn,
    n_simulations: int = 1000,
    seed: int = 42,
    third_place_count: int = 8,
    knockout_bracket: Mapping[str, Any] | None = None,
    completed_group_matches: Sequence[Mapping[str, Any]] | None = None,
    completed_knockout_matches: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    return simulate_tournament_detailed(
        teams_by_group,
        predict_match,
        n_simulations=n_simulations,
        seed=seed,
        third_place_count=third_place_count,
        knockout_bracket=knockout_bracket,
        completed_group_matches=completed_group_matches,
        completed_knockout_matches=completed_knockout_matches,
    )["team_probabilities"]
