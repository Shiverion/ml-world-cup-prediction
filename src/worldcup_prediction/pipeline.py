from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from worldcup_prediction.backtest import DEFAULT_WORLDCUP_WINDOWS, WorldCupWindow, rolling_world_cup_backtest
from worldcup_prediction.cleaning import clean_matches, clean_rankings
from worldcup_prediction.config import CONFIG_DIR, PROJECT_ROOT, RANDOM_SEED
from worldcup_prediction.data_loader import read_csv, read_yaml, write_csv
from worldcup_prediction.elo import add_elo_features, default_k_factor, expected_score, match_result_score
from worldcup_prediction.features import build_feature_table
from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS, make_model, train_model
from worldcup_prediction.simulator import MatchProbabilityFn, poisson_outcome_probabilities, simulate_tournament_detailed
from worldcup_prediction.utils import ensure_columns, load_team_mapping, standardize_team_name


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


def completed_group_matches_from_fixture_frame(
    fixtures: pd.DataFrame,
    team_mapping: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    ensure_columns(fixtures, ["group", "team_a", "team_b", "team_a_score", "team_b_score", "status"], "fixtures")
    completed = fixtures[fixtures["status"].eq("completed")].copy()
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


def completed_fixture_matches_for_training(
    fixtures: pd.DataFrame,
    team_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    ensure_columns(fixtures, ["date", "team_a", "team_b", "team_a_score", "team_b_score", "status"], "fixtures")
    completed = fixtures[fixtures["status"].eq("completed")].copy()
    completed = completed.dropna(subset=["date", "team_a", "team_b", "team_a_score", "team_b_score"])
    if completed.empty:
        return pd.DataFrame()
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
            "stage": "Group",
            "group": completed["group"].astype(str).str.replace("Group ", "", regex=False).str.strip(),
        }
    )
    return clean_matches(raw, team_mapping)


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
    windows = parse_world_cup_windows(backtest_config)
    rows: list[pd.DataFrame] = []

    for model_name, spec in model_specs.items():
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


def summarize_backtests(backtests: pd.DataFrame, primary_metric: str = "log_loss") -> pd.DataFrame:
    if backtests.empty:
        return pd.DataFrame()
    required = {"model", "accuracy", "top1_accuracy", "log_loss", "brier_score"}
    missing = sorted(required - set(backtests.columns))
    if missing:
        raise ValueError(f"Backtest data is missing columns: {missing}")
    summary = (
        backtests.groupby("model", as_index=False)
        .agg(
            windows=("year", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_score_mean=("brier_score", "mean"),
            brier_score_std=("brier_score", "std"),
            top1_accuracy_mean=("top1_accuracy", "mean"),
        )
        .fillna(0.0)
    )
    ascending = primary_metric in {"log_loss", "brier_score"}
    sort_column = f"{primary_metric}_mean" if f"{primary_metric}_mean" in summary.columns else primary_metric
    return summary.sort_values(sort_column, ascending=ascending).reset_index(drop=True)


def _model_from_spec(spec: Mapping[str, Any], model_name: str, random_seed: int):
    kind = str(spec.get("kind", model_name))
    params = {key: value for key, value in spec.items() if key != "kind"}
    return make_model(kind, random_state=random_seed, **params)


def run_analysis(
    data_config_path: str | Path = CONFIG_DIR / "data_config.yaml",
    model_config_path: str | Path = CONFIG_DIR / "model_config.yaml",
    backtest_config_path: str | Path = CONFIG_DIR / "backtest_config.yaml",
    tournament_config_path: str | Path = CONFIG_DIR / "tournament_2026.yaml",
    root: Path = PROJECT_ROOT,
    live: bool = False,
) -> dict[str, Path | None]:
    data_config = read_yaml(resolve_project_path(data_config_path, root))
    model_config = read_yaml(resolve_project_path(model_config_path, root))
    backtest_config = read_yaml(resolve_project_path(backtest_config_path, root))
    tournament_config = read_yaml(resolve_project_path(tournament_config_path, root))

    raw_matches_path = resolve_project_path(data_config["raw_matches_path"], root)
    raw_rankings_path = resolve_project_path(data_config["raw_rankings_path"], root)
    if not raw_matches_path.exists():
        raise FileNotFoundError(f"Raw match data not found: {raw_matches_path}")

    team_mapping_path = data_config.get("team_mapping_path")
    team_mapping = load_team_mapping(str(resolve_project_path(team_mapping_path, root))) if team_mapping_path else {}
    mode = "live" if live else str(tournament_config.get("mode", "pre_tournament"))
    matches_clean = clean_matches(read_csv(raw_matches_path), team_mapping)
    completed_group_matches: list[dict[str, Any]] = []
    live_fixtures_path = resolve_project_path(
        tournament_config.get("live_results_path", "data/raw/world_cup_2026_matches.csv"),
        root,
    )
    if mode == "live":
        if not live_fixtures_path.exists():
            raise FileNotFoundError(f"Live fixture/results file not found: {live_fixtures_path}")
        live_fixtures = read_csv(live_fixtures_path)
        completed_group_matches = completed_group_matches_from_fixture_frame(live_fixtures, team_mapping)
        live_training_matches = completed_fixture_matches_for_training(live_fixtures, team_mapping)
        if not live_training_matches.empty:
            matches_clean = pd.concat([matches_clean, live_training_matches], ignore_index=True)
            matches_clean = matches_clean.drop_duplicates(
                subset=["date", "team_a", "team_b", "team_a_score", "team_b_score", "tournament"]
            ).sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)
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

    cutoff = pd.Timestamp(tournament_config.get("data_cutoff", data_config.get("pre_tournament_cutoff", "2026-06-11")))
    if mode == "live" and completed_group_matches:
        live_dates = pd.to_datetime(read_csv(live_fixtures_path).query("status == 'completed'")["date"], errors="coerce")
        if live_dates.notna().any():
            cutoff = live_dates.max() + pd.Timedelta(days=1)
    train_frame = features[features["date"] < cutoff].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows before tournament cutoff: {cutoff.date()}")
    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    primary_model_name = str(model_config.get("primary_model") or next(iter(model_specs.keys())))
    if primary_model_name not in model_specs:
        raise ValueError(f"primary_model is not configured under models: {primary_model_name}")
    primary_model_spec = model_specs[primary_model_name]
    train_model(
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
    teams_by_group = load_teams_by_group(tournament_config, root)
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
        if simulation_predictor == "elo_poisson":
            predictor = make_elo_poisson_predictor(ratings, average_total_goals=average_total_goals)
        elif simulation_predictor == "elo_baseline":
            predictor = make_elo_probability_predictor(
                ratings,
                draw_probability=float(tournament_config.get("draw_probability", 0.24)),
            )
        else:
            raise ValueError(f"Unsupported simulation_predictor: {simulation_predictor}")
        simulation_outputs = simulate_tournament_detailed(
            teams_by_group,
            predictor,
            n_simulations=int(tournament_config.get("simulation_count", 10_000)),
            seed=int(model_config.get("random_seed", RANDOM_SEED)),
            third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
            knockout_bracket=tournament_config.get("knockout_bracket"),
            completed_group_matches=completed_group_matches,
        )
        suffix = "_live" if mode == "live" else ""
        simulation_output_path = root / "outputs" / "simulations" / f"team_probabilities_2026{suffix}.csv"
        group_positions_output_path = root / "outputs" / "simulations" / f"group_position_probabilities_2026{suffix}.csv"
        bracket_output_path = root / "outputs" / "simulations" / f"predicted_knockout_bracket_2026{suffix}.csv"
        write_csv(simulation_outputs["team_probabilities"], simulation_output_path)
        write_csv(simulation_outputs["group_positions"], group_positions_output_path)
        write_csv(simulation_outputs["knockout_bracket"], bracket_output_path)

    return {
        "processed_matches": processed_matches_path,
        "processed_rankings": processed_rankings_path if rankings_clean is not None else None,
        "features": features_path,
        "backtest": backtest_output_path,
        "backtest_summary": backtest_summary_output_path,
        "simulation": simulation_output_path,
        "group_positions": group_positions_output_path,
        "knockout_bracket": bracket_output_path,
    }
