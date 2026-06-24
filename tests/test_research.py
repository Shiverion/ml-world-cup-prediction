import pandas as pd
import pytest
import yaml

from worldcup_prediction.research import (
    deterministic_interval_seeds,
    elo_poisson_probability_frame,
    match_probability_frame,
    simulation_probability_intervals,
    write_forecast_registry,
)


def simple_predictor(team_a, team_b, context=None):
    return {
        "team_a_win": 0.5,
        "draw": 0.2,
        "team_b_win": 0.3,
        "team_a_goals_lambda": 1.4,
        "team_b_goals_lambda": 1.0,
    }


def test_match_probability_frame_exports_group_fixture_probabilities():
    frame = match_probability_frame({"A": ["A1", "A2", "A3", "A4"]}, simple_predictor)

    assert len(frame) == 6
    assert {"group", "team_a", "team_b", "team_a_win", "draw", "team_b_win"} <= set(frame.columns)
    assert frame.iloc[0][["team_a_win", "draw", "team_b_win"]].sum() == pytest.approx(1.0)


def test_elo_poisson_probability_frame_uses_evaluation_outcome_columns():
    frame = elo_poisson_probability_frame(pd.DataFrame({"elo_diff": [100.0, -100.0]}), average_total_goals=2.5)

    assert list(frame.columns) == ["team_a_loss", "draw", "team_a_win"]
    assert frame.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert frame.loc[0, "team_a_win"] > frame.loc[0, "team_a_loss"]


def test_simulation_probability_intervals_aggregates_seed_runs():
    intervals = simulation_probability_intervals(
        {
            "A": ["A1", "A2", "A3", "A4"],
            "B": ["B1", "B2", "B3", "B4"],
        },
        simple_predictor,
        n_simulations=2,
        seeds=[1, 2],
        third_place_count=0,
    )

    assert set(intervals["team"]) == {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"}
    assert {"champion_mean", "champion_p05", "champion_p50", "champion_p95"} <= set(intervals.columns)
    assert (intervals["seed_count"] == 2).all()


def test_deterministic_interval_seeds_are_stable_and_unique():
    seeds = deterministic_interval_seeds(42, 3)

    assert seeds == [42, 10015, 19988]
    assert len(seeds) == len(set(seeds))


def test_forecast_registry_config_uses_portable_paths(tmp_path):
    simulation_path = tmp_path / "outputs" / "simulations" / "team_probabilities.csv"
    simulation_path.parent.mkdir(parents=True)
    pd.DataFrame({"team": ["A"], "champion": [0.1]}).to_csv(simulation_path, index=False)

    outside_path = tmp_path.parent / "local_only_interval.csv"

    registry_dir = write_forecast_registry(
        root=tmp_path,
        mode="live",
        cutoff=pd.Timestamp("2026-06-23"),
        model_name="logistic_plain_c0_5",
        simulation_predictor="ml_outcome",
        simulation_count=3000,
        feature_columns=["elo_diff"],
        output_paths={
            "simulation": simulation_path,
            "simulation_interval": outside_path,
        },
    )

    config_text = (registry_dir / "config.yaml").read_text(encoding="utf-8")
    registry_config = yaml.safe_load(config_text)

    assert registry_config["outputs"]["simulation"] == "outputs/simulations/team_probabilities.csv"
    assert registry_config["outputs"]["simulation_interval"] == "${LOCAL_PATH}/local_only_interval.csv"
    assert str(tmp_path) not in config_text
