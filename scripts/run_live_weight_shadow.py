from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldcup_prediction.cleaning import clean_matches
from worldcup_prediction.data_loader import read_csv, read_yaml, write_csv
from worldcup_prediction.elo import add_elo_features
from worldcup_prediction.features import build_feature_table
from worldcup_prediction.live_update_backtest import advancement_probability
from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS, train_model
from worldcup_prediction.pipeline import (
    _load_optional_rankings,
    _make_anchored_live_predictor,
    _model_from_spec,
    _pre_knockout_cutoff_from_fixture_frame,
    available_feature_columns,
    completed_fixture_matches_for_training,
    completed_group_matches_from_fixture_frame,
    completed_knockout_matches_from_fixture_frame,
    expected_group_match_count,
    final_elo_ratings,
    group_table_from_completed_matches,
    load_teams_by_group,
    merge_live_training_matches,
    resolve_knockout_bracket_config,
    resolve_project_path,
)
from worldcup_prediction.utils import load_team_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a candidate live-update weight against the current 2026 live configuration without writing production forecasts."
    )
    parser.add_argument("--data-config", default=ROOT / "configs" / "data_config.yaml")
    parser.add_argument("--model-config", default=ROOT / "configs" / "model_config.yaml")
    parser.add_argument("--tournament-config", default=ROOT / "configs" / "tournament_2026.yaml")
    parser.add_argument("--prior-strength", type=float, required=True)
    parser.add_argument("--max-live-weight", type=float, required=True)
    parser.add_argument("--output", default=ROOT / "outputs" / "evaluation" / "live_update_2026_weight_shadow.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = read_yaml(resolve_project_path(args.data_config, ROOT))
    model_config = read_yaml(resolve_project_path(args.model_config, ROOT))
    tournament_config = read_yaml(resolve_project_path(args.tournament_config, ROOT))
    team_mapping_path = data_config.get("team_mapping_path")
    team_mapping = load_team_mapping(str(resolve_project_path(team_mapping_path, ROOT))) if team_mapping_path else {}
    fixtures_path = resolve_project_path(
        tournament_config.get("live_results_path", "data/raw/world_cup_2026_matches.csv"),
        ROOT,
    )
    fixtures = read_csv(fixtures_path)
    completed_dates = pd.to_datetime(fixtures.loc[fixtures["status"].eq("completed"), "date"], errors="coerce")
    cutoff = pd.Timestamp(completed_dates.max()) + pd.Timedelta(days=1)

    raw_matches = read_csv(resolve_project_path(data_config["raw_matches_path"], ROOT))
    matches_clean = clean_matches(raw_matches, team_mapping)
    anchor_matches_clean = matches_clean.copy()
    teams_by_group = load_teams_by_group(tournament_config, ROOT)
    knockout_bracket = resolve_knockout_bracket_config(tournament_config, ROOT)
    completed_group_matches = completed_group_matches_from_fixture_frame(fixtures, team_mapping)
    expected_groups = expected_group_match_count(teams_by_group)
    if len(completed_group_matches) < expected_groups:
        raise ValueError(f"Expected {expected_groups} completed group matches, found {len(completed_group_matches)}")
    anchor_cutoff = _pre_knockout_cutoff_from_fixture_frame(fixtures) or cutoff
    group_table = group_table_from_completed_matches(teams_by_group, completed_group_matches)
    completed_knockout_matches = completed_knockout_matches_from_fixture_frame(
        fixtures,
        knockout_bracket,
        team_mapping,
        group_table=group_table,
        third_place_count=int(tournament_config.get("third_place_qualifiers", 8)),
    )
    live_training_matches = completed_fixture_matches_for_training(fixtures, team_mapping, include_knockout=True)
    anchor_training_matches = completed_fixture_matches_for_training(fixtures, team_mapping, include_knockout=False)
    if not anchor_training_matches.empty:
        anchor_matches_clean = merge_live_training_matches(anchor_matches_clean, anchor_training_matches)
    if not live_training_matches.empty:
        matches_clean = merge_live_training_matches(matches_clean, live_training_matches)

    rankings_clean = _load_optional_rankings(
        resolve_project_path(data_config["raw_rankings_path"], ROOT),
        team_mapping,
    )
    features = build_feature_table(add_elo_features(matches_clean), rankings_clean)
    requested_columns = model_config.get("baseline_feature_columns") or DEFAULT_FEATURE_COLUMNS
    feature_columns = available_feature_columns(features, requested_columns)
    model_specs = model_config.get("models") or {"logistic": {"kind": "logistic"}}
    model_name = str(model_config.get("primary_model") or next(iter(model_specs)))
    train_frame = features[pd.to_datetime(features["date"], errors="coerce") < cutoff].copy()
    live_model = train_model(
        _model_from_spec(model_specs[model_name], model_name, int(model_config.get("random_seed", 42))),
        train_frame,
        feature_columns,
        target_column=str(model_config.get("target_column", "target")),
    )
    ratings = final_elo_ratings(matches_clean, cutoff=cutoff)
    candidate_config = deepcopy(tournament_config)
    candidate_config["live_model_update"] = {
        "enabled": True,
        "prior_strength": args.prior_strength,
        "max_live_weight": args.max_live_weight,
    }
    final_fixtures = fixtures[
        fixtures["round"].fillna("").str.strip().str.lower().eq("final") & ~fixtures["status"].eq("completed")
    ].copy()
    if final_fixtures.empty:
        raise ValueError("No scheduled final is available for the shadow comparison")

    rows: list[dict[str, object]] = []
    scenarios = [("current", tournament_config), ("shadow", candidate_config)]
    for scenario, config in scenarios:
        predictor, metadata = _make_anchored_live_predictor(
            live_model,
            anchor_matches_clean,
            rankings_clean,
            cutoff,
            model_config,
            model_specs,
            model_name,
            feature_columns,
            ratings,
            matches_clean,
            completed_knockout_matches,
            config,
            anchor_cutoff=anchor_cutoff,
        )
        for fixture in final_fixtures.itertuples(index=False):
            probabilities = predictor(
                str(fixture.team_a),
                str(fixture.team_b),
                {"stage": "final", "match": int(float(fixture.match))},
            )
            probability_a_advances = advancement_probability(probabilities)
            rows.append(
                {
                    "scenario": scenario,
                    "cutoff": cutoff.date().isoformat(),
                    "match": int(float(fixture.match)),
                    "team_a": str(fixture.team_a),
                    "team_b": str(fixture.team_b),
                    "team_a_advance_probability": probability_a_advances,
                    "team_b_advance_probability": 1.0 - probability_a_advances,
                    "predicted_winner": str(fixture.team_a)
                    if probability_a_advances >= 0.5
                    else str(fixture.team_b),
                    "live_weight": metadata["live_model_weight"] if metadata else 1.0,
                    "anchor_weight": metadata["anchor_model_weight"] if metadata else 0.0,
                    "prior_strength": config["live_model_update"]["prior_strength"],
                    "max_live_weight": config["live_model_update"]["max_live_weight"],
                    "completed_knockout_matches": len(completed_knockout_matches),
                }
            )

    output = Path(args.output)
    write_csv(pd.DataFrame(rows), output)
    for row in rows:
        print(
            f"{row['scenario']}: {row['team_a']} {row['team_a_advance_probability']:.1%} vs "
            f"{row['team_b']} {row['team_b_advance_probability']:.1%} -> {row['predicted_winner']} "
            f"(live weight {row['live_weight']:.1%})"
        )
    print(f"Shadow comparison: {output}")


if __name__ == "__main__":
    main()
