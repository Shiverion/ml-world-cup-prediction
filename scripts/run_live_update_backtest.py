from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldcup_prediction.cleaning import clean_matches, clean_rankings
from worldcup_prediction.data_loader import read_csv, read_yaml, write_csv
from worldcup_prediction.historical_knockout import load_historical_knockout_results
from worldcup_prediction.live_update_backtest import (
    default_live_update_candidates,
    rank_live_update_candidates,
    run_live_update_weight_backtest,
    summarize_live_update_backtest,
    walk_forward_live_update_selection,
)
from worldcup_prediction.utils import load_team_mapping


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the frozen-anchor/rolling-live ensemble using historical World Cup knockout ties."
    )
    parser.add_argument("--data-config", default=ROOT / "configs" / "data_config.yaml")
    parser.add_argument("--model-config", default=ROOT / "configs" / "model_config.yaml")
    parser.add_argument("--tournament-config", default=ROOT / "configs" / "tournament_2026.yaml")
    parser.add_argument(
        "--knockout-data",
        default=ROOT / "data" / "external" / "world_cup_knockout_results_2002_2022.csv",
    )
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "evaluation")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    data_config = read_yaml(resolve(args.data_config))
    model_config = read_yaml(resolve(args.model_config))
    tournament_config = read_yaml(resolve(args.tournament_config))
    mapping_path = data_config.get("team_mapping_path")
    team_mapping = load_team_mapping(str(resolve(mapping_path))) if mapping_path else {}

    matches_clean = clean_matches(read_csv(resolve(data_config["raw_matches_path"])), team_mapping)
    rankings_clean = clean_rankings(read_csv(resolve(data_config["raw_rankings_path"])), team_mapping)
    knockout_results = load_historical_knockout_results(str(resolve(args.knockout_data)), team_mapping)

    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    model_name = str(model_config.get("primary_model") or next(iter(model_specs)))
    if model_name not in model_specs:
        raise ValueError(f"Primary model is not configured: {model_name}")
    feature_columns = model_config.get("baseline_feature_columns")
    predictions = run_live_update_weight_backtest(
        matches_clean,
        rankings_clean,
        knockout_results,
        model_specs[model_name],
        model_name,
        candidates=default_live_update_candidates(),
        feature_columns=feature_columns,
        random_seed=int(model_config.get("random_seed", 42)),
        target_column=str(model_config.get("target_column", "target")),
        ranking_cutoff_inclusive=bool(tournament_config.get("ranking_cutoff_inclusive", False)),
    )
    candidate_summary, candidate_year_metrics = summarize_live_update_backtest(predictions)
    ranked_candidates = rank_live_update_candidates(candidate_summary)
    walk_forward, walk_forward_summary = walk_forward_live_update_selection(candidate_year_metrics)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(predictions, output_dir / "live_update_weight_backtest_predictions.csv")
    write_csv(candidate_summary, output_dir / "live_update_weight_backtest_candidate_summary.csv")
    write_csv(candidate_year_metrics, output_dir / "live_update_weight_backtest_by_year.csv")
    write_csv(walk_forward, output_dir / "live_update_weight_backtest_walk_forward.csv")
    write_csv(walk_forward_summary, output_dir / "live_update_weight_backtest_walk_forward_summary.csv")

    current_update = tournament_config.get("live_model_update") or {}
    current_prior = float(current_update.get("prior_strength", current_update.get("knockout_prior_matches", 80.0)))
    current_cap = float(current_update.get("max_live_weight", 0.35))
    current_rows = candidate_summary[
        candidate_summary["prior_strength"].eq(current_prior)
        & candidate_summary["max_live_weight"].eq(current_cap)
    ]
    recommendation = ranked_candidates.iloc[0].to_dict()
    report = {
        "primary_metric": "advance_log_loss",
        "target": "team advancing from a knockout tie, including extra time and penalties",
        "historical_world_cups": sorted(int(year) for year in knockout_results["year"].unique()),
        "historical_ties": int(len(knockout_results)),
        "recommended_candidate": recommendation,
        "current_2026_configuration": {
            "prior_strength": current_prior,
            "max_live_weight": current_cap,
            "backtest_result": current_rows.iloc[0].to_dict() if not current_rows.empty else None,
        },
        "walk_forward_selection": (
            walk_forward_summary.iloc[0].to_dict() if not walk_forward_summary.empty else None
        ),
    }
    with (output_dir / "live_update_weight_backtest_recommendation.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(report), handle, indent=2)

    print(f"Historical knockout ties evaluated: {len(knockout_results)}")
    print(f"Best pooled candidate: {recommendation['candidate']}")
    print(f"Advance log loss: {recommendation['advance_log_loss']:.4f}")
    print(f"Advance accuracy: {recommendation['advance_accuracy']:.1%}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
