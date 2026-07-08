from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worldcup_prediction.simulator import (
    GroupRecord,
    ThirdPlaceSlot,
    assign_third_place_slots,
    build_round_of_32_bracket,
    normalize_match_probabilities,
    pair_bracket_winners,
    poisson_outcome_probabilities,
    rank_group,
    select_group_qualifiers,
    simulate_group_stage,
    simulate_tournament,
    simulate_tournament_detailed,
)


def equal_predictor(team_a, team_b, context=None):
    return {"team_a_win": 0.45, "draw": 0.10, "team_b_win": 0.45}


def test_group_stage_produces_ranked_table():
    teams_by_group = {"A": ["A1", "A2", "A3", "A4"]}

    table, matches = simulate_group_stage(teams_by_group, equal_predictor, rng=np.random.default_rng(1))

    assert len(matches) == 6
    assert set(table["position"]) == {1, 2, 3, 4}
    assert set(table["team"]) == {"A1", "A2", "A3", "A4"}


def test_group_stage_locks_completed_matches_and_simulates_remaining_only():
    teams_by_group = {"A": ["A1", "A2", "A3", "A4"]}
    completed_matches = [
        {"group": "A", "team_a": "A1", "team_b": "A2", "team_a_score": 2, "team_b_score": 0}
    ]

    table, matches = simulate_group_stage(
        teams_by_group,
        lambda team_a, team_b, context=None: {"team_a_win": 1.0, "draw": 0.0, "team_b_win": 0.0},
        rng=np.random.default_rng(1),
        completed_matches=completed_matches,
    )

    assert len(matches) == 6
    assert sum(record.team_a == "A1" and record.team_b == "A2" for record in matches) == 1
    assert table["points"].sum() == 18


def test_normalize_match_probabilities_accepts_team_a_loss_alias():
    probabilities = normalize_match_probabilities({"team_a_loss": 0.30, "draw": 0.20, "team_a_win": 0.50})

    assert probabilities["team_a_win"] == pytest.approx(0.50)
    assert probabilities["draw"] == pytest.approx(0.20)
    assert probabilities["team_b_win"] == pytest.approx(0.30)


def test_poisson_outcome_probabilities_are_normalized_and_directional():
    probabilities = poisson_outcome_probabilities(2.0, 0.8)

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["team_a_win"] > probabilities["team_b_win"]


def test_best_third_place_selection_keeps_top_third_place_teams():
    table, _ = simulate_group_stage(
        {
            "A": ["A1", "A2", "A3", "A4"],
            "B": ["B1", "B2", "B3", "B4"],
            "C": ["C1", "C2", "C3", "C4"],
        },
        equal_predictor,
        rng=np.random.default_rng(2),
    )

    qualifiers = select_group_qualifiers(table, third_place_count=2)

    assert len(qualifiers) == 8
    assert (qualifiers["position"] <= 3).all()


def test_pair_bracket_winners_preserves_adjacent_bracket_paths():
    assert pair_bracket_winners(["A1", "A2", "B1", "B2"]) == [("A1", "A2"), ("B1", "B2")]


def test_assign_third_place_slots_finds_valid_candidate_mapping():
    assignments = assign_third_place_slots(
        [
            ThirdPlaceSlot(match_id=74, team_index=1, candidates=("A", "B")),
            ThirdPlaceSlot(match_id=77, team_index=1, candidates=("C", "D")),
        ],
        {"B": "B3", "C": "C3"},
    )

    assert assignments == {(74, 1): "B", (77, 1): "C"}


def test_build_round_of_32_bracket_uses_configured_slots():
    group_table = simulate_group_stage(
        {
            "A": ["A1", "A2", "A3", "A4"],
            "B": ["B1", "B2", "B3", "B4"],
        },
        equal_predictor,
        rng=np.random.default_rng(5),
    )[0]
    group_table.loc[(group_table["group"] == "A") & (group_table["position"] == 3), "points"] = 6
    group_table.loc[(group_table["group"] == "B") & (group_table["position"] == 3), "points"] = 1
    bracket_config = {
        "round_of_32": [
            {
                "match": 73,
                "teams": [
                    {"group": "A", "position": 1},
                    {"third_place_from": ["A", "B"]},
                ],
            }
        ]
    }

    bracket = build_round_of_32_bracket(group_table, bracket_config, third_place_count=1)

    assert bracket == [
        {
            "match": 73,
            "team_a": group_table[(group_table["group"] == "A") & (group_table["position"] == 1)].iloc[0]["team"],
            "team_b": group_table[(group_table["group"] == "A") & (group_table["position"] == 3)].iloc[0]["team"],
        }
    ]


def test_build_round_of_32_bracket_uses_official_third_place_table():
    rows = []
    for group in "ABCDEFGH":
        for position in range(1, 5):
            rows.append(
                {
                    "group": group,
                    "position": position,
                    "team": f"{group}{position}",
                    "points": 5 - position,
                    "goals_for": 5 - position,
                    "goal_difference": 5 - position,
                    "wins": 0,
                }
            )
    group_table = pd.DataFrame(rows)
    bracket_config = {
        "third_place_assignment_strategy": "official_table",
        "third_place_mapping_path": str(
            Path(__file__).resolve().parents[1] / "data" / "external" / "fwc2026_third_place_annex_c.csv"
        ),
        "round_of_32": [
            {
                "match": 79,
                "teams": [
                    {"group": "A", "position": 1},
                    {"third_place_from": ["C", "E", "F", "H", "I"]},
                ],
            }
        ],
    }

    bracket = build_round_of_32_bracket(group_table, bracket_config, third_place_count=8)

    assert bracket == [{"match": 79, "team_a": "A1", "team_b": "H3"}]


def test_simulate_tournament_keeps_fixed_knockout_bracket_between_rounds():
    teams_by_group = {group: [f"{group}{index}" for index in range(1, 5)] for group in ["A", "B", "C", "D"]}
    knockout_pairs = []

    def always_team_a_wins(team_a, team_b, context=None):
        if context and context.get("stage"):
            knockout_pairs.append((context["stage"], team_a, team_b))
        return {"team_a_win": 1.0, "draw": 0.0, "team_b_win": 0.0}

    simulate_tournament(
        teams_by_group,
        always_team_a_wins,
        n_simulations=1,
        seed=1,
        third_place_count=0,
    )

    round_of_16_pairs = [
        (team_a, team_b)
        for stage, team_a, team_b in knockout_pairs
        if stage == "round_of_16"
    ]
    assert round_of_16_pairs == [("A1", "A2"), ("B1", "B2")]


def test_simulate_tournament_detailed_returns_group_positions_and_bracket_rows():
    teams_by_group = {group: [f"{group}{index}" for index in range(1, 5)] for group in ["A", "B"]}
    bracket_config = {
        "round_of_32": [
            {"match": 73, "teams": [{"group": "A", "position": 1}, {"group": "B", "position": 2}]},
            {"match": 74, "teams": [{"group": "B", "position": 1}, {"group": "A", "position": 2}]},
        ],
        "round_of_16": [{"match": 89, "winners_of": [73, 74]}],
        "quarterfinals": [{"match": 97, "winners_of": [89, 89]}],
        "semifinals": [{"match": 101, "winners_of": [97, 97]}],
        "final": [{"match": 104, "winners_of": [101, 101]}],
    }

    details = simulate_tournament_detailed(
        teams_by_group,
        lambda team_a, team_b, context=None: {"team_a_win": 1.0, "draw": 0.0, "team_b_win": 0.0},
        n_simulations=1,
        seed=1,
        third_place_count=0,
        knockout_bracket=bracket_config,
    )

    assert {"team_probabilities", "group_positions", "knockout_bracket"} <= set(details)
    assert set(details["group_positions"]["team"]) == {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"}
    assert not details["knockout_bracket"].empty


def test_completed_knockout_match_winner_is_locked_into_later_rounds():
    teams_by_group = {group: [f"{group}{index}" for index in range(1, 5)] for group in ["A", "B"]}
    bracket_config = {
        "round_of_32": [
            {"match": 73, "teams": [{"group": "A", "position": 1}, {"group": "B", "position": 2}]},
            {"match": 74, "teams": [{"group": "B", "position": 1}, {"group": "A", "position": 2}]},
        ],
        "round_of_16": [{"match": 89, "winners_of": [73, 74]}],
        "quarterfinals": [{"match": 97, "winners_of": [89, 89]}],
        "semifinals": [{"match": 101, "winners_of": [97, 97]}],
        "final": [{"match": 104, "winners_of": [101, 101]}],
    }

    details = simulate_tournament_detailed(
        teams_by_group,
        lambda team_a, team_b, context=None: {"team_a_win": 1.0, "draw": 0.0, "team_b_win": 0.0},
        n_simulations=1,
        seed=1,
        third_place_count=0,
        knockout_bracket=bracket_config,
        completed_knockout_matches=[
            {
                "round": "round_of_32",
                "match": 73,
                "team_a": "A1",
                "team_b": "B2",
                "team_a_score": 0,
                "team_b_score": 1,
                "winner": "B2",
            }
        ],
    )

    bracket = details["knockout_bracket"].set_index("match")
    teams = details["team_probabilities"].set_index("team")
    assert bracket.loc[73, "winner_top"] == "B2"
    assert bracket.loc[89, "team_a_top"] == "B2"
    assert teams.loc["B2", "reach_r16"] == pytest.approx(1.0)
    assert teams.loc["A1", "reach_r16"] == pytest.approx(0.0)


def test_rank_group_orders_by_points_goal_difference_and_goals_for():
    records = [
        GroupRecord("A", "G", points=6, goals_for=3, goals_against=1),
        GroupRecord("B", "G", points=4, goals_for=4, goals_against=2),
        GroupRecord("C", "G", points=4, goals_for=3, goals_against=1),
    ]

    ranked = rank_group(records, matches=[], rng=np.random.default_rng(3))

    assert [record.team for record in ranked] == ["A", "B", "C"]
