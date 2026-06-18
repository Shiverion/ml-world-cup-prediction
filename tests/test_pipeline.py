import pandas as pd
import pytest

from worldcup_prediction.pipeline import (
    completed_group_matches_from_fixture_frame,
    final_elo_ratings,
    make_elo_probability_predictor,
    make_elo_poisson_predictor,
    summarize_backtests,
)


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
