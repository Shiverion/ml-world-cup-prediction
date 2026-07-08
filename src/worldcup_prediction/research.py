from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from worldcup_prediction.backtest import (
    WorldCupWindow,
    rolling_world_cup_backtest,
    split_world_cup_backtest,
)
from worldcup_prediction.config import OUTCOME_COLUMNS, RANDOM_SEED
from worldcup_prediction.metrics import evaluate_probabilities
from worldcup_prediction.models import predict_probabilities
from worldcup_prediction.simulator import (
    MatchProbabilityFn,
    generate_round_robin_matches,
    normalize_match_probabilities,
    poisson_outcome_probabilities,
    simulate_tournament_detailed,
)
from worldcup_prediction.utils import ensure_columns

ELO_FEATURES = ["elo_diff", "elo_abs_diff", "elo_expected_a"]
FIFA_FEATURES = ["fifa_rank_diff", "fifa_points_diff"]
FORM_FEATURES = ["form_points_diff_5", "form_points_diff_10", "goal_diff_form_10"]
CONTEXT_FEATURES = [
    "is_neutral",
    "team_a_home_advantage",
    "is_friendly",
    "is_qualifier",
    "is_world_cup",
    "is_world_cup_group",
    "is_world_cup_knockout",
    "rest_days_diff",
]


def available_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def summarize_backtest_like_results(
    results: pd.DataFrame,
    group_column: str = "model",
    primary_metric: str = "log_loss",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    numeric_metrics = [
        "accuracy",
        "top1_accuracy",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
    ]
    aggregations: dict[str, tuple[str, str]] = {"windows": ("year", "count")}
    for metric in numeric_metrics:
        if metric in results.columns:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    summary = results.groupby(group_column, as_index=False).agg(**aggregations).fillna(0.0)
    sort_column = f"{primary_metric}_mean" if f"{primary_metric}_mean" in summary.columns else primary_metric
    ascending = primary_metric in {"log_loss", "brier_score", "ranked_probability_score"}
    if sort_column in summary.columns:
        summary = summary.sort_values(sort_column, ascending=ascending)
    return summary.reset_index(drop=True)


def uniform_probability_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(1.0 / 3.0, index=index, columns=OUTCOME_COLUMNS)


def elo_probability_frame(features: pd.DataFrame, draw_probability: float = 0.24) -> pd.DataFrame:
    ensure_columns(features, ["elo_expected_a"], "features")
    expected_a = pd.to_numeric(features["elo_expected_a"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    decisive_mass = 1.0 - draw_probability
    return pd.DataFrame(
        {
            "team_a_loss": decisive_mass * (1.0 - expected_a),
            "draw": draw_probability,
            "team_a_win": decisive_mass * expected_a,
        },
        index=features.index,
    )


def elo_poisson_probability_frame(features: pd.DataFrame, average_total_goals: float = 2.55) -> pd.DataFrame:
    ensure_columns(features, ["elo_diff"], "features")
    if average_total_goals <= 0:
        raise ValueError("average_total_goals must be positive")
    rows: list[dict[str, float]] = []
    for elo_diff in pd.to_numeric(features["elo_diff"], errors="coerce").fillna(0.0):
        strength_ratio = 10.0 ** (float(elo_diff) / 400.0)
        lambda_b = average_total_goals / (1.0 + strength_ratio)
        lambda_a = average_total_goals - lambda_b
        probabilities = poisson_outcome_probabilities(lambda_a, lambda_b)
        rows.append(
            {
                "team_a_loss": probabilities["team_b_win"],
                "draw": probabilities["draw"],
                "team_a_win": probabilities["team_a_win"],
            }
        )
    return pd.DataFrame(rows, index=features.index)[OUTCOME_COLUMNS]


def rolling_static_probability_backtest(
    features: pd.DataFrame,
    probability_factory: Callable[[pd.DataFrame], pd.DataFrame],
    windows: Sequence[WorldCupWindow],
    model_name: str,
    target_column: str = "target",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for window in windows:
        _, test = split_world_cup_backtest(features, window)
        if test.empty:
            continue
        probabilities = probability_factory(test)
        metrics = evaluate_probabilities(test[target_column], probabilities)
        rows.append(
            {
                "model": model_name,
                "year": window.year,
                "train_matches": 0,
                "test_matches": int(len(test)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def rolling_model_prediction_records(
    features: pd.DataFrame,
    model_factory: Callable[[], object],
    feature_columns: Sequence[str],
    windows: Sequence[WorldCupWindow],
    model_name: str,
    target_column: str = "target",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    columns = list(feature_columns)
    for window in windows:
        train, test = split_world_cup_backtest(features, window)
        if train.empty or test.empty:
            continue
        model = model_factory()
        model.fit(train[columns], train[target_column])
        probabilities = predict_probabilities(model, test, columns)
        frame = test[
            [
                "date",
                "team_a",
                "team_b",
                "tournament",
                target_column,
            ]
        ].copy()
        frame.insert(0, "model", model_name)
        frame.insert(1, "year", window.year)
        for column in OUTCOME_COLUMNS:
            frame[column] = probabilities[column].to_numpy()
        frame["predicted_label"] = probabilities[OUTCOME_COLUMNS].to_numpy().argmax(axis=1)
        frame["confidence"] = probabilities[OUTCOME_COLUMNS].max(axis=1).to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_baseline_comparison(
    features: pd.DataFrame,
    model_factory: Callable[[], object],
    feature_columns: Sequence[str],
    windows: Sequence[WorldCupWindow],
    average_total_goals: float = 2.55,
    draw_probability: float = 0.24,
    target_column: str = "target",
) -> pd.DataFrame:
    rows = [
        rolling_static_probability_backtest(
            features,
            lambda test: uniform_probability_frame(test.index),
            windows,
            "random_uniform",
            target_column=target_column,
        ),
        rolling_static_probability_backtest(
            features,
            lambda test: elo_probability_frame(test, draw_probability=draw_probability),
            windows,
            "elo_probability",
            target_column=target_column,
        ),
        rolling_static_probability_backtest(
            features,
            lambda test: elo_poisson_probability_frame(test, average_total_goals=average_total_goals),
            windows,
            "elo_poisson",
            target_column=target_column,
        ),
    ]

    model_benchmarks = {
        "elo_logistic": available_columns(features, ELO_FEATURES),
        "fifa_logistic": available_columns(features, FIFA_FEATURES),
        "full_primary_model": list(feature_columns),
    }
    for name, columns in model_benchmarks.items():
        if not columns:
            continue
        result = rolling_world_cup_backtest(
            features,
            model_factory=model_factory,
            feature_columns=columns,
            windows=windows,
            target_column=target_column,
        )
        if not result.empty:
            result.insert(0, "model", name)
            result["feature_count"] = len(columns)
        rows.append(result)

    return pd.concat([row for row in rows if not row.empty], ignore_index=True)


def feature_set_definitions(features: pd.DataFrame, full_feature_columns: Sequence[str]) -> dict[str, list[str]]:
    full = list(full_feature_columns)
    definitions = {
        "elo_only": available_columns(features, ELO_FEATURES),
        "fifa_only": available_columns(features, FIFA_FEATURES),
        "form_only": available_columns(features, FORM_FEATURES),
        "context_only": available_columns(features, CONTEXT_FEATURES),
        "full_features": full,
        "full_minus_elo": [column for column in full if column not in ELO_FEATURES],
        "full_minus_fifa": [column for column in full if column not in FIFA_FEATURES],
        "full_minus_form": [column for column in full if column not in FORM_FEATURES],
        "full_minus_context": [column for column in full if column not in CONTEXT_FEATURES],
    }
    return {name: columns for name, columns in definitions.items() if columns}


def run_ablation_study(
    features: pd.DataFrame,
    model_factory: Callable[[], object],
    full_feature_columns: Sequence[str],
    windows: Sequence[WorldCupWindow],
    target_column: str = "target",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for feature_set, columns in feature_set_definitions(features, full_feature_columns).items():
        result = rolling_world_cup_backtest(
            features,
            model_factory=model_factory,
            feature_columns=columns,
            windows=windows,
            target_column=target_column,
        )
        if result.empty:
            continue
        result.insert(0, "feature_set", feature_set)
        result["feature_count"] = len(columns)
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_nested_model_selection_backtest(
    features: pd.DataFrame,
    model_factories: Mapping[str, Callable[[], object]],
    feature_columns: Sequence[str],
    windows: Sequence[WorldCupWindow],
    primary_metric: str = "log_loss",
    target_column: str = "target",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    columns = list(feature_columns)
    ascending = primary_metric in {"log_loss", "brier_score", "ranked_probability_score"}

    for outer_window in windows:
        train, test = split_world_cup_backtest(features, outer_window)
        inner_windows = [window for window in windows if window.end_date < outer_window.start_date]
        if train.empty or test.empty or not inner_windows:
            continue

        candidate_rows: list[dict[str, float | str]] = []
        for model_name, factory in model_factories.items():
            inner_result = rolling_world_cup_backtest(
                train,
                model_factory=factory,
                feature_columns=columns,
                windows=inner_windows,
                target_column=target_column,
            )
            if inner_result.empty or primary_metric not in inner_result.columns:
                continue
            candidate_rows.append(
                {
                    "model": model_name,
                    "inner_windows": float(len(inner_result)),
                    f"inner_{primary_metric}_mean": float(inner_result[primary_metric].mean()),
                }
            )
        if not candidate_rows:
            continue

        candidates = pd.DataFrame(candidate_rows).sort_values(
            f"inner_{primary_metric}_mean",
            ascending=ascending,
        )
        selected_model = str(candidates.iloc[0]["model"])
        model = model_factories[selected_model]()
        model.fit(train[columns], train[target_column])
        probabilities = predict_probabilities(model, test, columns)
        metrics = evaluate_probabilities(test[target_column], probabilities)
        rows.append(
            {
                "year": outer_window.year,
                "selected_model": selected_model,
                "inner_windows": int(candidates.iloc[0]["inner_windows"]),
                f"inner_{primary_metric}_mean": float(candidates.iloc[0][f"inner_{primary_metric}_mean"]),
                "train_matches": int(len(train)),
                "test_matches": int(len(test)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def match_probability_frame(
    teams_by_group: Mapping[str, Sequence[str]],
    predict_match: MatchProbabilityFn,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in generate_round_robin_matches(teams_by_group):
        raw_probabilities = predict_match(match["team_a"], match["team_b"], match)
        probabilities = normalize_match_probabilities(raw_probabilities)
        rows.append(
            {
                **match,
                **probabilities,
                "team_a_goals_lambda": raw_probabilities.get("team_a_goals_lambda"),
                "team_b_goals_lambda": raw_probabilities.get("team_b_goals_lambda"),
            }
        )
    return pd.DataFrame(rows)


def simulation_probability_intervals(
    teams_by_group: Mapping[str, Sequence[str]],
    predict_match: MatchProbabilityFn,
    n_simulations: int,
    seeds: Sequence[int],
    third_place_count: int = 8,
    knockout_bracket: Mapping[str, Any] | None = None,
    completed_group_matches: Sequence[Mapping[str, Any]] | None = None,
    completed_knockout_matches: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        output = simulate_tournament_detailed(
            teams_by_group,
            predict_match,
            n_simulations=n_simulations,
            seed=int(seed),
            third_place_count=third_place_count,
            knockout_bracket=knockout_bracket,
            completed_group_matches=completed_group_matches,
            completed_knockout_matches=completed_knockout_matches,
        )["team_probabilities"].copy()
        output.insert(0, "seed", int(seed))
        rows.append(output)
    if not rows:
        return pd.DataFrame()

    samples = pd.concat(rows, ignore_index=True)
    milestone_columns = [column for column in samples.columns if column not in {"seed", "team"}]
    output_rows: list[dict[str, float | str]] = []
    for team, team_samples in samples.groupby("team", sort=False):
        row: dict[str, float | str] = {"team": str(team), "seed_count": float(team_samples["seed"].nunique())}
        for column in milestone_columns:
            values = team_samples[column].astype(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
            row[f"{column}_p05"] = float(values.quantile(0.05))
            row[f"{column}_p50"] = float(values.quantile(0.50))
            row[f"{column}_p95"] = float(values.quantile(0.95))
        output_rows.append(row)
    return pd.DataFrame(output_rows).sort_values("champion_mean", ascending=False).reset_index(drop=True)


def deterministic_interval_seeds(base_seed: int = RANDOM_SEED, seed_count: int = 5) -> list[int]:
    if seed_count <= 0:
        return []
    return [int(base_seed + 9973 * index) for index in range(seed_count)]


def git_commit_hash(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def registry_path_reference(path: Path, root: Path) -> str:
    """Return a portable path for forecast registry metadata."""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"${{LOCAL_PATH}}/{path.name}"


def forecast_registry_mode_slug(mode: str) -> str:
    return {
        "live": "live",
        "pre_knockout": "preknockout",
        "pre_tournament": "pretournament",
        "reconstructed_live": "reconstructed-live",
    }.get(mode, str(mode).replace("_", ""))


def _write_forecast_registry_metadata(
    registry_dir: Path,
    root: Path,
    mode: str,
    cutoff: pd.Timestamp,
    model_name: str,
    simulation_predictor: str,
    simulation_count: int,
    feature_columns: Sequence[str],
    outputs: Mapping[str, str],
    known_missing_data: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    commit = git_commit_hash(root)
    (registry_dir / "git_commit.txt").write_text(f"{commit}\n", encoding="utf-8")
    registry_config = {
        "forecast_date": pd.Timestamp.utcnow().date().isoformat(),
        "mode": mode,
        "data_cutoff": cutoff.isoformat(),
        "git_commit": commit,
        "model": model_name,
        "simulation_predictor": simulation_predictor,
        "simulation_count": simulation_count,
        "feature_columns": list(feature_columns),
        "outputs": dict(outputs),
    }
    if metadata:
        registry_config["metadata"] = dict(metadata)
    with (registry_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(registry_config, handle, sort_keys=False)

    missing_text = "\n".join(
        f"- {item}" for item in known_missing_data or ["No squad, injury, lineup, weather, travel, or odds features."]
    )
    card = f"""# Forecast Card

Forecast date: {registry_config["forecast_date"]}
Data cutoff: {cutoff.isoformat()}
Mode: {mode}
Git commit: {commit}
Model: {model_name}
Simulation predictor: {simulation_predictor}
Number of simulations: {simulation_count}
Calibration method: uncalibrated base probabilities; calibration diagnostics are reported separately

## Known Missing Data

{missing_text}
"""
    if metadata:
        card += "\n## Metadata\n\n" + "\n".join(f"- {key}: {value}" for key, value in metadata.items()) + "\n"
    (registry_dir / "model_card.md").write_text(card, encoding="utf-8")


def write_forecast_registry(
    root: Path,
    mode: str,
    cutoff: pd.Timestamp,
    model_name: str,
    simulation_predictor: str,
    simulation_count: int,
    feature_columns: Sequence[str],
    output_paths: Mapping[str, Path | None],
    match_probabilities: pd.DataFrame | None = None,
    known_missing_data: Sequence[str] | None = None,
) -> Path:
    mode_slug = forecast_registry_mode_slug(mode)
    registry_dir = root / "outputs" / "forecast_registry" / f"{cutoff.date()}_{mode_slug}"
    registry_dir.mkdir(parents=True, exist_ok=True)

    _write_forecast_registry_metadata(
        registry_dir,
        root,
        mode,
        cutoff,
        model_name,
        simulation_predictor,
        simulation_count,
        feature_columns,
        {name: registry_path_reference(path, root) for name, path in output_paths.items() if path is not None},
        known_missing_data=known_missing_data,
    )

    if output_paths.get("simulation"):
        pd.read_csv(output_paths["simulation"]).to_csv(registry_dir / "team_probabilities.csv", index=False)
    if output_paths.get("group_positions"):
        pd.read_csv(output_paths["group_positions"]).to_csv(
            registry_dir / "group_position_probabilities.csv",
            index=False,
        )
    if output_paths.get("knockout_bracket"):
        pd.read_csv(output_paths["knockout_bracket"]).to_csv(
            registry_dir / "predicted_knockout_bracket.csv",
            index=False,
        )
    if match_probabilities is not None:
        match_probabilities.to_csv(registry_dir / "match_probabilities.csv", index=False)

    return registry_dir


def write_forecast_registry_frames(
    root: Path,
    mode: str,
    cutoff: pd.Timestamp,
    model_name: str,
    simulation_predictor: str,
    simulation_count: int,
    feature_columns: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    match_probabilities: pd.DataFrame | None = None,
    known_missing_data: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    mode_slug = forecast_registry_mode_slug(mode)
    registry_dir = root / "outputs" / "forecast_registry" / f"{cutoff.date()}_{mode_slug}"
    registry_dir.mkdir(parents=True, exist_ok=True)

    file_names = {
        "simulation": "team_probabilities.csv",
        "group_positions": "group_position_probabilities.csv",
        "knockout_bracket": "predicted_knockout_bracket.csv",
        "simulation_interval": "team_probabilities_with_ci.csv",
    }
    outputs: dict[str, str] = {}
    for name, frame in frames.items():
        file_name = file_names.get(name, f"{name}.csv")
        frame.to_csv(registry_dir / file_name, index=False)
        outputs[name] = f"outputs/forecast_registry/{registry_dir.name}/{file_name}"
    if match_probabilities is not None:
        match_probabilities.to_csv(registry_dir / "match_probabilities.csv", index=False)
        outputs["match_probabilities"] = f"outputs/forecast_registry/{registry_dir.name}/match_probabilities.csv"

    _write_forecast_registry_metadata(
        registry_dir,
        root,
        mode,
        cutoff,
        model_name,
        simulation_predictor,
        simulation_count,
        feature_columns,
        outputs,
        known_missing_data=known_missing_data,
        metadata=metadata,
    )
    return registry_dir
