import numpy as np
import pandas as pd
import pytest

import worldcup_prediction.pipeline as pipeline
from worldcup_prediction.pipeline import (
    apply_simulation_profile,
    blend_match_probability_predictors,
    completed_group_matches_from_fixture_frame,
    completed_knockout_matches_from_fixture_frame,
    final_elo_ratings,
    fixture_frame_for_reconstructed_round,
    group_table_from_completed_matches,
    latest_ranking_snapshot,
    live_model_update_weight,
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


def test_live_model_update_weight_uses_prior_strength_and_cap():
    config = {"live_model_update": {"prior_strength": 80, "max_live_weight": 0.35}}

    assert live_model_update_weight([{}] * 16, config) == pytest.approx(16 / 96)
    assert live_model_update_weight([{}] * 200, config) == pytest.approx(0.35)


def test_blend_match_probability_predictors_mixes_anchor_and_live_probabilities():
    def anchor_predictor(team_a, team_b, context=None):
        return {"team_a_win": 0.7, "draw": 0.2, "team_a_loss": 0.1}

    def live_predictor(team_a, team_b, context=None):
        return {"team_a_win": 0.4, "draw": 0.1, "team_a_loss": 0.5}

    predictor = blend_match_probability_predictors(anchor_predictor, live_predictor, live_weight=0.25)

    probabilities = predictor("A", "B", {"stage": "Round of 16"})

    assert probabilities == pytest.approx(
        {
            "team_a_win": 0.625,
            "draw": 0.175,
            "team_a_loss": 0.2,
        }
    )


def test_anchored_live_predictor_freezes_anchor_snapshot(monkeypatch):
    live_cutoff = pd.Timestamp("2026-07-08")
    anchor_cutoff = pd.Timestamp("2026-06-28")
    feature_columns = ["elo_diff"]
    live_ratings = {"Spain": 1550.0, "Argentina": 1560.0}
    anchor_ratings = {"Spain": 1500.0, "Argentina": 1520.0}
    predictor_calls = []
    train_cutoffs = []
    rating_cutoffs = []
    snapshot_cutoffs = []

    def fake_make_ml_outcome_predictor(model, ratings, ranking_snapshot, form_snapshot, columns):
        predictor_calls.append(
            {
                "model": model,
                "ratings": ratings,
                "ranking_snapshot": ranking_snapshot,
                "form_snapshot": form_snapshot,
                "columns": columns,
            }
        )
        return lambda team_a, team_b, context=None: {"team_a_win": 0.6, "draw": 0.2, "team_a_loss": 0.2}

    def fake_build_feature_table(matches, rankings):
        return pd.DataFrame({"date": ["2026-06-01"], "target": [2], "elo_diff": [0.0]})

    def fake_train_primary_model(features, cutoff, model_config, model_specs, primary_model_name, columns, empty_message):
        train_cutoffs.append(pd.Timestamp(cutoff))
        return "anchor_model"

    def fake_final_elo_ratings(matches, cutoff):
        rating_cutoffs.append(pd.Timestamp(cutoff))
        return anchor_ratings

    def fake_latest_ranking_snapshot(rankings, cutoff, inclusive=False):
        snapshot_cutoffs.append(("rank", pd.Timestamp(cutoff)))
        return {"cutoff": pd.Timestamp(cutoff).isoformat()}

    def fake_recent_form_snapshot(matches, cutoff):
        snapshot_cutoffs.append(("form", pd.Timestamp(cutoff)))
        return {"cutoff": pd.Timestamp(cutoff).isoformat()}

    monkeypatch.setattr(pipeline, "make_ml_outcome_predictor", fake_make_ml_outcome_predictor)
    monkeypatch.setattr(pipeline, "add_elo_features", lambda matches: matches)
    monkeypatch.setattr(pipeline, "build_feature_table", fake_build_feature_table)
    monkeypatch.setattr(pipeline, "_train_primary_model_for_cutoff", fake_train_primary_model)
    monkeypatch.setattr(pipeline, "final_elo_ratings", fake_final_elo_ratings)
    monkeypatch.setattr(pipeline, "latest_ranking_snapshot", fake_latest_ranking_snapshot)
    monkeypatch.setattr(pipeline, "recent_form_snapshot", fake_recent_form_snapshot)

    predictor, metadata = pipeline._make_anchored_live_predictor(
        "live_model",
        pd.DataFrame({"date": ["2026-06-01", "2026-07-04"]}),
        None,
        live_cutoff,
        {"random_seed": 42},
        {"logistic": {"kind": "logistic"}},
        "logistic",
        feature_columns,
        live_ratings,
        pd.DataFrame({"date": ["2026-07-04"]}),
        [{"round": "round_of_32"}],
        {"live_model_update": {"prior_strength": 4, "max_live_weight": 0.35}},
        anchor_cutoff=anchor_cutoff,
    )

    probabilities = predictor("Spain", "Argentina", {"stage": "final"})

    assert probabilities == pytest.approx({"team_a_win": 0.6, "draw": 0.2, "team_a_loss": 0.2})
    assert predictor_calls[0]["model"] == "live_model"
    assert predictor_calls[0]["ratings"] == live_ratings
    assert predictor_calls[1]["model"] == "anchor_model"
    assert predictor_calls[1]["ratings"] == anchor_ratings
    assert train_cutoffs == [anchor_cutoff]
    assert rating_cutoffs == [anchor_cutoff]
    assert ("rank", live_cutoff) in snapshot_cutoffs
    assert ("form", live_cutoff) in snapshot_cutoffs
    assert ("rank", anchor_cutoff) in snapshot_cutoffs
    assert ("form", anchor_cutoff) in snapshot_cutoffs
    assert metadata["anchor_model_weight"] == pytest.approx(0.8)
    assert metadata["live_model_weight"] == pytest.approx(0.2)
    assert metadata["anchor_snapshot_cutoff"] == anchor_cutoff.isoformat()


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
                "team_a_penalties": None,
                "team_b_penalties": None,
                "winner": "South Korea",
                "winner_source": "score",
                "decided_by_penalties": False,
                "winner_method": "",
                "decided_after_extra_time": False,
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
                "match": 73,
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
    assert by_match[74]["decided_by_penalties"] is False
    assert by_match[76]["team_a"] == "C1"
    assert by_match[76]["winner"] == "C1"


def test_completed_knockout_matches_marks_penalties_only_when_explicit():
    bracket_config = {
        "round_of_32": [
            {"match": 73, "teams": [{"group": "A", "position": 1}, {"group": "B", "position": 2}]},
        ],
    }
    fixtures = pd.DataFrame(
        [
            {
                "match": 73,
                "round": "Round of 32",
                "team_a": "A1",
                "team_b": "B2",
                "team_a_score": 1,
                "team_b_score": 1,
                "team_a_penalties": 4,
                "team_b_penalties": 3,
                "status": "completed",
                "winner_method": "penalties",
            },
        ]
    )

    matches = completed_knockout_matches_from_fixture_frame(fixtures, bracket_config)

    assert matches[0]["winner"] == "A1"
    assert matches[0]["winner_source"] == "penalties"
    assert matches[0]["decided_by_penalties"] is True
    assert matches[0]["team_a_penalties"] == 4
    assert matches[0]["team_b_penalties"] == 3


def test_completed_knockout_matches_extracts_third_place_match():
    fixtures = pd.DataFrame(
        [
            {
                "match": 103,
                "round": "Third-place match",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 2,
                "team_b_score": 1,
                "status": "completed",
            }
        ]
    )

    matches = completed_knockout_matches_from_fixture_frame(
        fixtures,
        {"third_place": [{"match": 103, "losers_of": [101, 102]}]},
    )

    assert matches[0]["round"] == "third_place"
    assert matches[0]["match"] == 103
    assert matches[0]["winner"] == "A"


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
