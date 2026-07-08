import numpy as np
import pandas as pd
import pytest

from worldcup_prediction.pipeline import (
    apply_simulation_profile,
    completed_group_matches_from_fixture_frame,
    completed_knockout_matches_from_fixture_frame,
    final_elo_ratings,
    fixture_frame_for_reconstructed_round,
    group_table_from_completed_matches,
    latest_ranking_snapshot,
    make_elo_probability_predictor,
    make_elo_poisson_predictor,
    make_ml_outcome_predictor,
    merge_live_training_matches,
    recent_form_snapshot,
    summarize_backtests,
)


class DummyProbabilityModel:
    classes_ = [0, 1, 2]

    def __init__(self) -> None:
        self.last_features = pd.DataFrame()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.last_features = features.copy()
        return np.array([[0.2, 0.3, 0.5]])


def test_apply_simulation_profile_merges_nested_interval_config():
    config = {
        "simulation_count": 3000,
        "simulation_interval": {"enabled": True, "seed_count": 3, "simulations_per_seed": 500},
        "simulation_profiles": {
            "publication": {
                "simulation_count": 100000,
                "simulation_interval": {"seed_count": 30, "simulations_per_seed": 5000},
            }
        },
    }

    profiled = apply_simulation_profile(config, "publication")

    assert profiled["simulation_profile"] == "publication"
    assert profiled["simulation_count"] == 100000
    assert profiled["simulation_interval"] == {
        "enabled": True,
        "seed_count": 30,
        "simulations_per_seed": 5000,
    }
    assert config["simulation_count"] == 3000


def test_apply_simulation_profile_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown simulation profile"):
        apply_simulation_profile({"simulation_profiles": {"dev": {}}}, "publication")


def test_final_elo_ratings_respects_cutoff():
    matches = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 1,
                "team_b_score": 0,
                "tournament": "Friendly",
                "stage": "",
            },
            {
                "date": "2020-01-03",
                "team_a": "B",
                "team_b": "A",
                "team_a_score": 1,
                "team_b_score": 0,
                "tournament": "Friendly",
                "stage": "",
            },
        ]
    )

    ratings = final_elo_ratings(matches, cutoff=pd.Timestamp("2020-01-02"))

    assert ratings["A"] > 1500
    assert ratings["B"] < 1500


def test_elo_probability_predictor_returns_normalized_probabilities():
    predictor = make_elo_probability_predictor({"A": 1600, "B": 1400}, draw_probability=0.2)

    probabilities = predictor("A", "B")

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["team_a_win"] > probabilities["team_b_win"]
    assert probabilities["draw"] == pytest.approx(0.2)


def test_elo_poisson_predictor_returns_probabilities_and_goal_expectations():
    predictor = make_elo_poisson_predictor({"A": 1600, "B": 1400}, average_total_goals=2.6)

    probabilities = predictor("A", "B")

    assert probabilities["team_a_win"] > probabilities["team_b_win"]
    assert probabilities["team_a_goals_lambda"] > probabilities["team_b_goals_lambda"]
    assert probabilities["team_a_goals_lambda"] + probabilities["team_b_goals_lambda"] == pytest.approx(2.6)


def test_ml_outcome_predictor_builds_cutoff_safe_fixture_features():
    rankings = pd.DataFrame(
        [
            {"rank_date": "2020-01-01", "team": "A", "rank": 10, "points": 1600},
            {"rank_date": "2020-01-01", "team": "B", "rank": 20, "points": 1400},
            {"rank_date": "2020-02-01", "team": "A", "rank": 1, "points": 1900},
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 2,
                "team_b_score": 1,
            }
        ]
    )
    model = DummyProbabilityModel()
    feature_columns = [
        "elo_diff",
        "elo_abs_diff",
        "elo_expected_a",
        "fifa_rank_diff",
        "fifa_points_diff",
        "form_points_diff_5",
        "goal_diff_form_10",
        "is_world_cup_group",
    ]

    predictor = make_ml_outcome_predictor(
        model,
        {"A": 1600, "B": 1400},
        latest_ranking_snapshot(rankings, pd.Timestamp("2020-01-15")),
        recent_form_snapshot(matches, pd.Timestamp("2020-01-15")),
        feature_columns,
    )
    probabilities = predictor("A", "B", {"group": "A"})

    assert probabilities == pytest.approx({"team_a_loss": 0.2, "draw": 0.3, "team_a_win": 0.5})
    assert model.last_features.loc[0, "elo_diff"] == pytest.approx(200.0)
    assert model.last_features.loc[0, "fifa_rank_diff"] == pytest.approx(10.0)
    assert model.last_features.loc[0, "fifa_points_diff"] == pytest.approx(200.0)
    assert model.last_features.loc[0, "form_points_diff_5"] == pytest.approx(3.0)
    assert model.last_features.loc[0, "goal_diff_form_10"] == pytest.approx(2.0)
    assert model.last_features.loc[0, "is_world_cup_group"] == pytest.approx(1.0)


def test_completed_group_matches_from_fixture_frame_standardizes_names():
    fixtures = pd.DataFrame(
        [
            {
                "group": "Group A",
                "team_a": "Czech Republic",
                "team_b": "Korea Republic",
                "team_a_score": 1,
                "team_b_score": 2,
                "status": "completed",
            },
            {
                "group": "Group A",
                "team_a": "Mexico",
                "team_b": "South Africa",
                "team_a_score": None,
                "team_b_score": None,
                "status": "scheduled",
            },
        ]
    )

    matches = completed_group_matches_from_fixture_frame(
        fixtures,
        {"Czech Republic": "Czechia", "Korea Republic": "South Korea"},
    )

    assert matches == [
        {
            "group": "A",
            "team_a": "Czechia",
            "team_b": "South Korea",
            "team_a_score": 1,
            "team_b_score": 2,
        }
    ]


def test_completed_knockout_matches_from_fixture_frame_extracts_match_winner():
    fixtures = pd.DataFrame(
        [
            {
                "match": 89,
                "round": "Round of 16",
                "group": "",
                "team_a": "Czech Republic",
                "team_b": "Korea Republic",
                "team_a_score": 1,
                "team_b_score": 2,
                "status": "completed",
            },
            {
                "match": 90,
                "round": "Round of 16",
                "group": "",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": None,
                "team_b_score": None,
                "status": "scheduled",
            },
            {
                "match": 91,
                "round": "Round of 16",
                "group": "",
                "team_a": "W76",
                "team_b": "W78",
                "team_a_score": 1,
                "team_b_score": 0,
                "status": "completed",
            },
        ]
    )

    matches = completed_knockout_matches_from_fixture_frame(
        fixtures,
        {"round_of_16": [{"match": 89, "winners_of": [73, 75]}]},
        {"Czech Republic": "Czechia", "Korea Republic": "South Korea"},
    )

    assert matches == [
        {
            "round": "round_of_16",
            "match": 89,
            "team_a": "Czechia",
            "team_b": "South Korea",
            "team_a_score": 1,
            "team_b_score": 2,
            "winner": "South Korea",
            "winner_source": "score",
            "decided_by_penalties": False,
        }
    ]


def test_completed_knockout_matches_map_by_teams_and_infer_tied_winners():
    teams_by_group = {
        "A": ["A1", "A2"],
        "B": ["B1", "B2"],
        "C": ["C1", "C2"],
        "D": ["D1", "D2"],
        "E": ["E1", "E2"],
        "F": ["F1", "F2"],
    }
    group_rows = []
    for group, teams in teams_by_group.items():
        for position, team in enumerate(teams, start=1):
            group_rows.append(
                {
                    "group": group,
                    "position": position,
                    "team": team,
                    "points": 6 - position,
                    "goals_for": 6 - position,
                    "goal_difference": 6 - position,
                    "wins": 0,
                }
            )
    group_table = pd.DataFrame(group_rows)
    bracket_config = {
        "round_of_32": [
            {"match": 73, "teams": [{"group": "A", "position": 1}, {"group": "B", "position": 2}]},
            {"match": 74, "teams": [{"group": "E", "position": 1}, {"group": "D", "position": 2}]},
            {"match": 75, "teams": [{"group": "F", "position": 1}, {"group": "C", "position": 2}]},
            {"match": 76, "teams": [{"group": "C", "position": 1}, {"group": "F", "position": 2}]},
        ],
        "round_of_16": [
            {"match": 89, "winners_of": [73, 75]},
            {"match": 90, "winners_of": [74, 76]},
        ],
    }
    fixtures = pd.DataFrame(
        [
            {
                "round": "Round of 32",
                "team_a": "C1",
                "team_b": "F2",
                "team_a_score": 2,
                "team_b_score": 1,
                "status": "completed",
            },
            {
                "round": "Round of 32",
                "team_a": "E1",
                "team_b": "D2",
                "team_a_score": 1,
                "team_b_score": 1,
                "status": "completed",
            },
            {
                "round": "Round of 32",
                "team_a": "A1",
                "team_b": "B2",
                "team_a_score": 0,
                "team_b_score": 1,
                "status": "completed",
            },
            {
                "round": "Round of 32",
                "team_a": "F1",
                "team_b": "C2",
                "team_a_score": 0,
                "team_b_score": 1,
                "status": "completed",
            },
            {
                "round": "Round of 16",
                "team_a": "D2",
                "team_b": "C1",
                "team_a_score": None,
                "team_b_score": None,
                "status": "scheduled",
            },
        ]
    )

    matches = completed_knockout_matches_from_fixture_frame(
        fixtures,
        bracket_config,
        group_table=group_table,
    )
    by_match = {row["match"]: row for row in matches}

    assert by_match[73]["team_a"] == "A1"
    assert by_match[73]["winner"] == "B2"
    assert by_match[73]["winner_source"] == "score"
    assert by_match[73]["decided_by_penalties"] is False
    assert by_match[74]["team_a"] == "E1"
    assert by_match[74]["team_b"] == "D2"
    assert by_match[74]["winner"] == "D2"
    assert by_match[74]["winner_source"] == "next_round"
    assert by_match[74]["decided_by_penalties"] is True
    assert by_match[76]["team_a"] == "C1"
    assert by_match[76]["winner"] == "C1"


def test_fixture_frame_for_reconstructed_round_hides_future_rounds():
    fixtures = pd.DataFrame(
        [
            {
                "date": "2026-07-03",
                "round": "Round of 32",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 1,
                "team_b_score": 0,
                "status": "completed",
            },
            {
                "date": "2026-07-04",
                "round": "Round of 16",
                "team_a": "A",
                "team_b": "C",
                "team_a_score": 0,
                "team_b_score": 2,
                "status": "completed",
            },
            {
                "date": "2026-07-09",
                "round": "Quarter-final",
                "team_a": "C",
                "team_b": "D",
                "team_a_score": None,
                "team_b_score": None,
                "status": "scheduled",
            },
        ]
    )

    reconstructed = fixture_frame_for_reconstructed_round(
        fixtures,
        "round_of_16",
        pd.Timestamp("2026-07-04"),
    )

    assert list(reconstructed["round"]) == ["Round of 32", "Round of 16"]
    r16 = reconstructed[reconstructed["round"].eq("Round of 16")].iloc[0]
    assert r16["team_a"] == "A"
    assert r16["team_b"] == "C"
    assert r16["status"] == "scheduled"
    assert pd.isna(r16["team_a_score"])
    assert pd.isna(r16["team_b_score"])


def test_merge_live_training_matches_preserves_live_stage_metadata():
    raw = pd.DataFrame(
        [
            {
                "date": "2026-07-03",
                "home_team": "Argentina",
                "away_team": "Cape Verde",
                "home_score": 1,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "city": "",
                "country": "",
                "neutral": True,
                "team_a": "Argentina",
                "team_b": "Cape Verde",
                "team_a_score": 1,
                "team_b_score": 1,
                "stage": "",
                "group": "",
                "match_id": "raw",
            }
        ]
    )
    live = raw.copy()
    live.loc[0, "stage"] = "Round of 32"
    live.loc[0, "match_id"] = "live"

    merged = merge_live_training_matches(raw, live)

    assert len(merged) == 1
    assert merged.iloc[0]["stage"] == "Round of 32"
    assert merged.iloc[0]["match_id"] == "live"


def test_summarize_backtests_sorts_by_primary_metric():
    backtests = pd.DataFrame(
        [
            {"model": "a", "year": 1, "accuracy": 0.5, "top1_accuracy": 0.5, "log_loss": 1.0, "brier_score": 0.6},
            {"model": "a", "year": 2, "accuracy": 0.6, "top1_accuracy": 0.6, "log_loss": 0.9, "brier_score": 0.5},
            {"model": "b", "year": 1, "accuracy": 0.7, "top1_accuracy": 0.7, "log_loss": 1.2, "brier_score": 0.7},
        ]
    )

    summary = summarize_backtests(backtests, "log_loss")

    assert list(summary["model"]) == ["a", "b"]
    assert summary.loc[0, "accuracy_mean"] == pytest.approx(0.55)
