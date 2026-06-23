import numpy as np
import pandas as pd
import pytest

from worldcup_prediction.pipeline import (
    completed_group_matches_from_fixture_frame,
    final_elo_ratings,
    latest_ranking_snapshot,
    make_elo_probability_predictor,
    make_elo_poisson_predictor,
    make_ml_outcome_predictor,
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
