from __future__ import annotations

import os
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldcup_prediction.data_loader import read_yaml
from worldcup_prediction.pipeline import (
    completed_group_matches_from_fixture_frame,
    completed_knockout_matches_from_fixture_frame,
    expected_group_match_count,
    group_table_from_completed_matches,
    load_teams_by_group,
    resolve_knockout_bracket_config,
    run_reconstructed_live_snapshot,
)
from worldcup_prediction.simulator import GroupRecord, SimulatedMatch, rank_group, select_group_qualifiers
from worldcup_prediction.utils import load_team_mapping

st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Prediction Engine")

st.caption("Match-level probabilities, time-aware backtesting, and Monte Carlo tournament simulation.")

simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026.csv"
live_simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_live.csv"
pre_knockout_simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_pre_knockout.csv"
simulation_interval_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_with_ci.csv"
live_simulation_interval_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_live_with_ci.csv"
pre_knockout_simulation_interval_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_pre_knockout_with_ci.csv"
group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026.csv"
live_group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026_live.csv"
pre_knockout_group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026_pre_knockout.csv"
bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026.csv"
live_bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026_live.csv"
pre_knockout_bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026_pre_knockout.csv"
match_probabilities_path = ROOT / "outputs" / "simulations" / "match_probabilities_2026.csv"
live_match_probabilities_path = ROOT / "outputs" / "simulations" / "match_probabilities_2026_live.csv"
pre_knockout_match_probabilities_path = ROOT / "outputs" / "simulations" / "match_probabilities_2026_pre_knockout.csv"
backtest_path = ROOT / "outputs" / "backtest_results" / "model_backtest.csv"
backtest_summary_path = ROOT / "outputs" / "backtest_results" / "model_backtest_summary.csv"
evaluation_dir = ROOT / "outputs" / "evaluation"
forecast_registry_dir = ROOT / "outputs" / "forecast_registry"
baseline_summary_path = evaluation_dir / "baseline_comparison_summary.csv"
baseline_path = evaluation_dir / "baseline_comparison.csv"
ablation_summary_path = evaluation_dir / "ablation_summary.csv"
ablation_path = evaluation_dir / "ablation_results.csv"
nested_backtest_path = evaluation_dir / "nested_backtest_results.csv"
calibration_summary_path = evaluation_dir / "calibration_summary.csv"
calibration_by_world_cup_path = evaluation_dir / "calibration_by_world_cup.csv"
calibration_table_path = evaluation_dir / "calibration_table_by_probability_bin.csv"
sharpness_path = evaluation_dir / "probability_sharpness_report.csv"
data_config_path = ROOT / "configs" / "data_config.yaml"
tournament_config_path = ROOT / "configs" / "tournament_2026.yaml"


def modified_time(path: Path) -> str:
    if not path.exists():
        return "not generated"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def numeric_frame(frame: pd.DataFrame, skip_columns: set[str] | None = None) -> pd.DataFrame:
    skip_columns = skip_columns or set()
    output = frame.copy()
    for column in output.columns:
        if column in skip_columns:
            continue
        converted = pd.to_numeric(output[column], errors="coerce")
        if converted.notna().any():
            output[column] = converted
    return output


def probability_column_config(columns: list[str]) -> dict[str, st.column_config.NumberColumn]:
    return {
        column: st.column_config.NumberColumn(column, format="%.2f")
        for column in columns
    }


def actual_score_display(actual: dict[str, object] | pd.Series) -> str:
    score = f"{actual['team_a_score']}-{actual['team_b_score']}"
    if bool(actual.get("decided_by_penalties", False)):
        penalties_a = actual.get("team_a_penalties")
        penalties_b = actual.get("team_b_penalties")
        if not pd.isna(penalties_a) and not pd.isna(penalties_b):
            return f"{score} (pens {int(penalties_a)}-{int(penalties_b)})"
        return f"{score} (pens)"
    if bool(actual.get("decided_after_extra_time", False)) or str(actual.get("winner_method", "")) == "extra_time":
        return f"{score} (aet)"
    return score


def actual_status_lines(row: pd.Series | dict[str, object]) -> list[str]:
    actual_winner = str(row.get("actual_winner", "") or "")
    actual_score = str(row.get("actual_score", "") or "")
    if actual_winner:
        return [f"Winner: {actual_winner}", f"Score: {actual_score}"]
    return ["Pending"]


@st.cache_data(show_spinner=False)
def load_tournament_context() -> dict[str, object]:
    data_config = read_yaml(data_config_path)
    tournament_config = read_yaml(tournament_config_path)
    team_mapping_path = data_config.get("team_mapping_path")
    team_mapping = load_team_mapping(str(ROOT / team_mapping_path)) if team_mapping_path else {}
    teams_by_group = load_teams_by_group(tournament_config, ROOT)
    bracket_config = resolve_knockout_bracket_config(tournament_config, ROOT)
    live_results_path = ROOT / tournament_config.get("live_results_path", "data/raw/world_cup_2026_matches.csv")
    return {
        "team_mapping": team_mapping,
        "teams_by_group": teams_by_group,
        "bracket_config": bracket_config,
        "live_results_path": live_results_path,
    }


def actual_group_table(
    teams_by_group: dict[str, list[str]],
    completed_matches: list[dict[str, object]],
) -> pd.DataFrame:
    if not teams_by_group or not completed_matches:
        return pd.DataFrame()
    table = {
        group: {team: GroupRecord(team=team, group=group) for team in teams}
        for group, teams in teams_by_group.items()
    }
    played: list[SimulatedMatch] = []
    for match in completed_matches:
        group = str(match["group"])
        team_a = str(match["team_a"])
        team_b = str(match["team_b"])
        if group not in table or team_a not in table[group] or team_b not in table[group]:
            continue
        score_a = int(match["team_a_score"])
        score_b = int(match["team_b_score"])
        record_a = table[group][team_a]
        record_b = table[group][team_b]
        record_a.goals_for += score_a
        record_a.goals_against += score_b
        record_b.goals_for += score_b
        record_b.goals_against += score_a
        if score_a > score_b:
            record_a.points += 3
            record_a.wins += 1
        elif score_b > score_a:
            record_b.points += 3
            record_b.wins += 1
        else:
            record_a.points += 1
            record_b.points += 1
        played.append(SimulatedMatch(group, team_a, team_b, score_a, score_b))

    rows: list[dict[str, object]] = []
    for group, records_by_team in table.items():
        group_played = [match for match in played if match.group == group]
        ranked = rank_group(list(records_by_team.values()), group_played, np.random.default_rng(0))
        for position, record in enumerate(ranked, start=1):
            rows.append(
                {
                    "group": group,
                    "position": position,
                    "team": record.team,
                    "points": record.points,
                    "goals_for": record.goals_for,
                    "goals_against": record.goals_against,
                    "goal_difference": record.goal_difference,
                    "wins": record.wins,
                }
            )
    return pd.DataFrame(rows)


def forecast_group_order(group_positions: pd.DataFrame) -> pd.DataFrame:
    if group_positions.empty:
        return pd.DataFrame()
    frame = numeric_frame(group_positions, {"group", "team"})
    rows: list[dict[str, object]] = []
    for group, group_frame in frame.groupby("group", sort=True):
        ordered = group_frame.sort_values(["expected_position", "team"]).reset_index(drop=True)
        for position, row in enumerate(ordered.itertuples(index=False), start=1):
            rows.append({"group": group, "predicted_position": position, "team": row.team})
    return pd.DataFrame(rows)


def group_stage_accuracy(
    pre_tournament_groups: pd.DataFrame | None,
    pre_tournament_teams: pd.DataFrame | None,
    actual_groups: pd.DataFrame,
    third_place_count: int = 8,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if pre_tournament_groups is None or pre_tournament_teams is None or actual_groups.empty:
        return pd.DataFrame(), {}
    predicted_order = forecast_group_order(pre_tournament_groups)
    if predicted_order.empty:
        return pd.DataFrame(), {}

    actual_positions = actual_groups[["group", "position", "team"]].copy()
    actual_positions = actual_positions.rename(columns={"position": "actual_position"})
    comparison = predicted_order.merge(actual_positions, on=["group", "team"], how="left")
    comparison["position_correct"] = comparison["predicted_position"].eq(comparison["actual_position"])

    actual_qualifiers = set(select_group_qualifiers(actual_groups, third_place_count=third_place_count)["team"])
    team_probabilities = numeric_frame(pre_tournament_teams, {"team"})
    if "advance_from_group" in team_probabilities.columns:
        predicted_qualifiers = set(
            team_probabilities.sort_values("advance_from_group", ascending=False)
            .head(len(actual_qualifiers))["team"]
            .astype(str)
        )
    else:
        predicted_qualifiers = set(comparison[comparison["predicted_position"] <= 2]["team"].astype(str))

    predicted_top_two_by_group = (
        predicted_order[predicted_order["predicted_position"] <= 2]
        .groupby("group")["team"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    comparison["top_two_team_correct"] = comparison.apply(
        lambda row: bool(
            row["actual_position"] <= 2
            and str(row["team"]) in predicted_top_two_by_group.get(str(row["group"]), set())
        ),
        axis=1,
    )
    comparison["top_two_slot_correct"] = comparison["actual_position"].le(2) & comparison["position_correct"]

    winner_rows = comparison[comparison["actual_position"] == 1]
    runner_up_rows = comparison[comparison["actual_position"] == 2]
    top_two_rows = comparison[comparison["actual_position"] <= 2]
    group_winners_correct = int((winner_rows["predicted_position"] == 1).sum()) if not winner_rows.empty else 0
    runner_ups_correct = int((runner_up_rows["predicted_position"] == 2).sum()) if not runner_up_rows.empty else 0
    top_two_team_correct = int(top_two_rows["top_two_team_correct"].sum()) if not top_two_rows.empty else 0
    top_two_slot_correct = int(top_two_rows["top_two_slot_correct"].sum()) if not top_two_rows.empty else 0
    top_two_total = int(len(top_two_rows))
    metrics = {
        "qualifier_accuracy": len(predicted_qualifiers & actual_qualifiers) / max(len(actual_qualifiers), 1),
        "exact_position_accuracy": float(comparison["position_correct"].mean()),
        "group_winner_accuracy": group_winners_correct / max(len(winner_rows), 1),
        "runner_up_accuracy": runner_ups_correct / max(len(runner_up_rows), 1),
        "top_two_team_accuracy": top_two_team_correct / max(top_two_total, 1),
        "top_two_slot_accuracy": top_two_slot_correct / max(top_two_total, 1),
        "qualifiers_correct": float(len(predicted_qualifiers & actual_qualifiers)),
        "qualifiers_total": float(len(actual_qualifiers)),
        "top_two_team_correct": float(top_two_team_correct),
        "top_two_slot_correct": float(top_two_slot_correct),
        "top_two_total": float(top_two_total),
        "group_winners_correct": float(group_winners_correct),
        "group_winners_total": float(len(winner_rows)),
        "runner_ups_correct": float(runner_ups_correct),
        "runner_ups_total": float(len(runner_up_rows)),
    }
    return comparison.sort_values(["group", "actual_position", "team"]), metrics


def bracket_prediction_status(
    bracket: pd.DataFrame,
    completed_knockout_matches: list[dict[str, object]],
    prediction_bracket: pd.DataFrame | None = None,
    prediction_brackets_by_round: dict[str, pd.DataFrame] | None = None,
    snapshot_labels_by_round: dict[str, str] | None = None,
) -> pd.DataFrame:
    if bracket.empty:
        return bracket
    actual_by_match = {int(match["match"]): match for match in completed_knockout_matches}
    prediction_by_match = (
        {int(row["match"]): row for _, row in prediction_bracket.iterrows()}
        if prediction_bracket is not None and not prediction_bracket.empty
        else {}
    )
    prediction_brackets_by_round = prediction_brackets_by_round or {}
    snapshot_labels_by_round = snapshot_labels_by_round or {}
    rows: list[dict[str, object]] = []
    for row in bracket.to_dict("records"):
        match_id = int(row["match"])
        round_key = str(row.get("round", ""))
        actual = actual_by_match.get(match_id)
        round_prediction = prediction_brackets_by_round.get(round_key)
        if round_prediction is not None and not round_prediction.empty:
            round_prediction_by_match = {int(item["match"]): item for _, item in round_prediction.iterrows()}
            prediction = round_prediction_by_match.get(match_id)
        else:
            prediction = prediction_by_match.get(match_id) if not prediction_brackets_by_round else None
        if prediction is None:
            prediction = row if actual is None else {}
        predicted_winner = str(prediction.get("winner_top", ""))
        predicted_probability = prediction.get("winner_match_probability", prediction.get("winner_probability", ""))
        if actual is None:
            status = "Ongoing"
            actual_winner = ""
            actual_score = ""
        elif not predicted_winner:
            actual_winner = str(actual["winner"])
            actual_score = actual_score_display(actual)
            status = "No round snapshot"
        else:
            actual_winner = str(actual["winner"])
            actual_score = actual_score_display(actual)
            status = "Successfully predicted" if predicted_winner == actual_winner else "False predicted"
        rows.append(
            {
                **row,
                "prediction_winner_top": predicted_winner,
                "prediction_winner_probability": predicted_probability,
                "prediction_winner_match_probability": prediction.get("winner_match_probability", ""),
                "prediction_team_a_top": prediction.get("team_a_top", ""),
                "prediction_team_b_top": prediction.get("team_b_top", ""),
                "prediction_snapshot": snapshot_labels_by_round.get(round_key, ""),
                "actual_winner": actual_winner,
                "actual_score": actual_score,
                "actual_winner_source": actual.get("winner_source", "") if actual is not None else "",
                "actual_decided_by_penalties": actual.get("decided_by_penalties", False) if actual is not None else False,
                "prediction_status": status,
            }
        )
    return pd.DataFrame(rows)


def status_summary(status_frame: pd.DataFrame) -> dict[str, int]:
    if status_frame.empty or "prediction_status" not in status_frame.columns:
        return {}
    counts = status_frame["prediction_status"].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def fixture_round_key(value: object) -> str | None:
    text = str(value or "").strip().lower().replace("-", " ")
    if "round of 32" in text:
        return "round_of_32"
    if "round of 16" in text:
        return "round_of_16"
    if "quarter" in text:
        return "quarterfinals"
    if "semi" in text:
        return "semifinals"
    if text == "final" or " final" in text:
        return "final"
    return None


def knockout_round_dates(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty or "round" not in fixtures or "date" not in fixtures:
        return pd.DataFrame(columns=["round", "start_date", "end_date", "match_count"])
    frame = fixtures.copy()
    frame["round_key"] = frame["round"].map(fixture_round_key)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["round_key", "date"])
    if frame.empty:
        return pd.DataFrame(columns=["round", "start_date", "end_date", "match_count"])
    return (
        frame.groupby("round_key", as_index=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"), match_count=("date", "size"))
        .rename(columns={"round_key": "round"})
    )


def group_stage_end_date(fixtures: pd.DataFrame) -> pd.Timestamp | None:
    if fixtures.empty or "round" not in fixtures or "date" not in fixtures:
        return None
    frame = fixtures.copy()
    frame["round_key"] = frame["round"].map(fixture_round_key)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    group_rows = frame["round_key"].isna() & frame["date"].notna()
    if "status" in frame:
        group_rows &= frame["status"].eq("completed")
    if not group_rows.any():
        return None
    return pd.Timestamp(frame.loc[group_rows, "date"].max())


def forecast_snapshot_label(snapshot: dict[str, object]) -> str:
    mode = str(snapshot.get("mode", "")).replace("_", "-")
    cutoff = snapshot.get("cutoff")
    if isinstance(cutoff, pd.Timestamp) and not pd.isna(cutoff):
        return f"{cutoff.date()} {mode}"
    return mode or "Unknown"


def forecast_snapshots() -> list[dict[str, object]]:
    if not forecast_registry_dir.exists():
        return []
    snapshots: list[dict[str, object]] = []
    for directory in sorted(path for path in forecast_registry_dir.iterdir() if path.is_dir()):
        config_path = directory / "config.yaml"
        bracket_path = directory / "predicted_knockout_bracket.csv"
        if not config_path.exists() or not bracket_path.exists():
            continue
        try:
            config = read_yaml(config_path)
        except Exception:
            continue
        mode = str(config.get("mode", ""))
        if mode not in {"pre_knockout", "live", "reconstructed_live"}:
            continue
        cutoff = pd.to_datetime(config.get("data_cutoff"), errors="coerce")
        if pd.isna(cutoff):
            continue
        snapshots.append(
            {
                "directory": directory,
                "mode": mode,
                "cutoff": pd.Timestamp(cutoff),
                "bracket_path": bracket_path,
                "metadata": config.get("metadata") or {},
            }
        )
    return sorted(snapshots, key=lambda item: (pd.Timestamp(item["cutoff"]), str(item["directory"])))


def latest_snapshot_for_round(
    round_key: str,
    round_dates: pd.DataFrame,
    snapshots: list[dict[str, object]],
    group_end: pd.Timestamp | None,
) -> dict[str, object] | None:
    if round_dates.empty:
        return None
    round_row = round_dates[round_dates["round"].eq(round_key)]
    if round_row.empty:
        return None
    round_start = pd.Timestamp(round_row.iloc[0]["start_date"])
    round_index = ROUND_SEQUENCE.index(round_key)
    if round_index == 0:
        previous_end = group_end
        required_modes = {"pre_knockout"}
    else:
        previous_round = ROUND_SEQUENCE[round_index - 1]
        previous_row = round_dates[round_dates["round"].eq(previous_round)]
        previous_end = pd.Timestamp(previous_row.iloc[0]["end_date"]) if not previous_row.empty else None
        required_modes = {"live", "reconstructed_live"}

    eligible: list[dict[str, object]] = []
    for snapshot in snapshots:
        if str(snapshot.get("mode", "")) not in required_modes:
            continue
        cutoff = pd.Timestamp(snapshot["cutoff"])
        if cutoff > round_start:
            continue
        if previous_end is not None and cutoff <= pd.Timestamp(previous_end):
            continue
        eligible.append(snapshot)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            pd.Timestamp(item["cutoff"]),
            1 if str(item.get("mode", "")) == "live" else 0,
        ),
    )


def round_prediction_brackets(
    live_results: pd.DataFrame | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    if live_results is None or live_results.empty:
        return {}, {}
    round_dates = knockout_round_dates(live_results)
    if round_dates.empty:
        return {}, {}
    snapshots = forecast_snapshots()
    group_end = group_stage_end_date(live_results)
    brackets: dict[str, pd.DataFrame] = {}
    labels: dict[str, str] = {}
    for round_key in ROUND_SEQUENCE:
        snapshot = latest_snapshot_for_round(round_key, round_dates, snapshots, group_end)
        if snapshot is None:
            continue
        brackets[round_key] = numeric_frame(
            pd.read_csv(snapshot["bracket_path"]),
            {"round", "team_a_top", "team_b_top", "winner_top"},
        )
        labels[round_key] = forecast_snapshot_label(snapshot)
    return brackets, labels


def round_by_round_accuracy(
    completed_knockout_matches: list[dict[str, object]],
    live_results: pd.DataFrame | None,
) -> pd.DataFrame:
    if live_results is None or live_results.empty:
        return pd.DataFrame()
    round_dates = knockout_round_dates(live_results)
    if round_dates.empty:
        return pd.DataFrame()
    snapshots = forecast_snapshots()
    actual_by_round: dict[str, list[dict[str, object]]] = {round_key: [] for round_key in ROUND_SEQUENCE}
    for match in completed_knockout_matches:
        round_key = str(match.get("round", ""))
        if round_key in actual_by_round:
            actual_by_round[round_key].append(match)

    group_end = group_stage_end_date(live_results)
    rows: list[dict[str, object]] = []
    for round_key in ROUND_SEQUENCE:
        round_row = round_dates[round_dates["round"].eq(round_key)]
        if round_row.empty:
            continue
        round_total = int(round_row.iloc[0]["match_count"])
        actual_matches = actual_by_round.get(round_key, [])
        snapshot = latest_snapshot_for_round(round_key, round_dates, snapshots, group_end)
        status = "Pending"
        correct = 0
        evaluated = 0
        avg_confidence = np.nan
        brier_score = np.nan
        snapshot_label = "No snapshot"

        if snapshot is None:
            status = "No round snapshot" if actual_matches else "Pending snapshot"
        else:
            snapshot_label = forecast_snapshot_label(snapshot)
            bracket = numeric_frame(pd.read_csv(snapshot["bracket_path"]), {"round", "team_a_top", "team_b_top", "winner_top"})
            prediction_by_match = {int(row["match"]): row for _, row in bracket.iterrows()}
            confidences: list[float] = []
            brier_values: list[float] = []
            for actual in actual_matches:
                prediction = prediction_by_match.get(int(actual["match"]))
                if prediction is None:
                    continue
                predicted_winner = str(prediction.get("winner_top", ""))
                confidence = float(
                    prediction.get("winner_match_probability", prediction.get("winner_probability", 0.0)) or 0.0
                )
                is_correct = predicted_winner == str(actual.get("winner", ""))
                correct += int(is_correct)
                evaluated += 1
                confidences.append(confidence)
                brier_values.append((1.0 - confidence) ** 2 if is_correct else confidence**2)
            if evaluated:
                status = "Evaluated"
                avg_confidence = float(np.mean(confidences))
                brier_score = float(np.mean(brier_values))
            elif actual_matches:
                status = "Missing predictions"

        rows.append(
            {
                "round": ROUND_LABELS.get(round_key, round_key),
                "forecast_snapshot": snapshot_label,
                "completed_matches": f"{len(actual_matches)}/{round_total}",
                "evaluated_predictions": f"{evaluated}/{len(actual_matches)}" if actual_matches else "",
                "correct": f"{correct}/{evaluated}" if evaluated else "",
                "winner_accuracy": correct / evaluated if evaluated else np.nan,
                "avg_pick_share": avg_confidence,
                "brier_score": brier_score,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def earliest_snapshot_after_round(
    round_key: str,
    round_dates: pd.DataFrame,
    snapshots: list[dict[str, object]],
) -> dict[str, object] | None:
    if round_dates.empty:
        return None
    round_row = round_dates[round_dates["round"].eq(round_key)]
    if round_row.empty:
        return None
    round_end = pd.Timestamp(round_row.iloc[0]["end_date"])
    eligible = [
        snapshot
        for snapshot in snapshots
        if str(snapshot.get("mode", "")) == "live" and pd.Timestamp(snapshot["cutoff"]) > round_end
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: pd.Timestamp(item["cutoff"]))


def round_by_round_live_sync(
    completed_knockout_matches: list[dict[str, object]],
    live_results: pd.DataFrame | None,
) -> pd.DataFrame:
    if live_results is None or live_results.empty:
        return pd.DataFrame()
    round_dates = knockout_round_dates(live_results)
    if round_dates.empty:
        return pd.DataFrame()
    snapshots = forecast_snapshots()
    actual_by_round: dict[str, list[dict[str, object]]] = {round_key: [] for round_key in ROUND_SEQUENCE}
    for match in completed_knockout_matches:
        round_key = str(match.get("round", ""))
        if round_key in actual_by_round:
            actual_by_round[round_key].append(match)

    rows: list[dict[str, object]] = []
    for round_key in ROUND_SEQUENCE:
        round_row = round_dates[round_dates["round"].eq(round_key)]
        if round_row.empty:
            continue
        round_total = int(round_row.iloc[0]["match_count"])
        actual_matches = actual_by_round.get(round_key, [])
        snapshot = earliest_snapshot_after_round(round_key, round_dates, snapshots)
        snapshot_label = "No post-round snapshot"
        synced = 0
        checked = 0
        status = "Pending" if not actual_matches else "No post-round snapshot"
        if snapshot is not None:
            snapshot_label = forecast_snapshot_label(snapshot)
            bracket = numeric_frame(pd.read_csv(snapshot["bracket_path"]), {"round", "team_a_top", "team_b_top", "winner_top"})
            prediction_by_match = {int(row["match"]): row for _, row in bracket.iterrows()}
            for actual in actual_matches:
                prediction = prediction_by_match.get(int(actual["match"]))
                if prediction is None:
                    continue
                checked += 1
                synced += int(str(prediction.get("winner_top", "")) == str(actual.get("winner", "")))
            if checked:
                status = "Synced" if synced == checked else "Mismatch"
            elif actual_matches:
                status = "Missing bracket rows"

        rows.append(
            {
                "round": ROUND_LABELS.get(round_key, round_key),
                "live_snapshot": snapshot_label,
                "completed_matches": f"{len(actual_matches)}/{round_total}",
                "synced_winners": f"{synced}/{len(actual_matches)}" if actual_matches else "",
                "sync_rate": synced / len(actual_matches) if actual_matches else np.nan,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def completed_knockout_counts_by_round(completed_knockout_matches: list[dict[str, object]]) -> dict[str, int]:
    counts = {round_key: 0 for round_key in ROUND_SEQUENCE}
    for match in completed_knockout_matches:
        round_key = str(match.get("round", ""))
        if round_key in counts:
            counts[round_key] += 1
    return counts


@st.cache_data(show_spinner=False)
def ensure_reconstructed_live_snapshots(
    live_results_mtime: float,
    simulation_profile: str = "dev",
) -> list[str]:
    del live_results_mtime
    live_results = pd.read_csv(context_live_results_path())
    round_dates = knockout_round_dates(live_results)
    if round_dates.empty:
        return []
    context = load_tournament_context()
    teams_by_group = context["teams_by_group"]
    bracket_config = context["bracket_config"]
    team_mapping = context["team_mapping"]
    third_place_total = int(read_yaml(tournament_config_path).get("third_place_qualifiers", 8))
    group_matches = completed_group_matches_from_fixture_frame(live_results, team_mapping)
    if len(group_matches) < expected_group_match_count(teams_by_group):
        return []
    group_table = group_table_from_completed_matches(teams_by_group, group_matches)
    knockout_matches = completed_knockout_matches_from_fixture_frame(
        live_results,
        bracket_config,
        team_mapping,
        group_table=group_table,
        third_place_count=third_place_total,
    )
    completed_counts = completed_knockout_counts_by_round(knockout_matches)
    generated: list[str] = []

    for round_key in ROUND_SEQUENCE[1:]:
        round_row = round_dates[round_dates["round"].eq(round_key)]
        if round_row.empty:
            continue
        round_index = ROUND_SEQUENCE.index(round_key)
        previous_round = ROUND_SEQUENCE[round_index - 1]
        previous_row = round_dates[round_dates["round"].eq(previous_round)]
        if previous_row.empty:
            continue
        previous_total = int(previous_row.iloc[0]["match_count"])
        if completed_counts.get(previous_round, 0) < previous_total:
            continue
        snapshots = forecast_snapshots()
        group_end = group_stage_end_date(live_results)
        if latest_snapshot_for_round(round_key, round_dates, snapshots, group_end) is not None:
            continue
        cutoff = pd.Timestamp(previous_row.iloc[0]["end_date"]) + pd.Timedelta(days=1)
        registry_dir = run_reconstructed_live_snapshot(
            round_key,
            cutoff,
            root=ROOT,
            simulation_profile=simulation_profile,
        )
        generated.append(f"{ROUND_LABELS.get(round_key, round_key)}: {registry_dir.name}")
    return generated


def context_live_results_path() -> Path:
    context = load_tournament_context()
    return Path(context["live_results_path"])


def latest_registry_dir() -> Path | None:
    if not forecast_registry_dir.exists():
        return None
    candidates = [path for path in forecast_registry_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_update_step(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )


def update_forecast_data_with_progress(forecast_name: str, analysis_args: list[str]) -> tuple[bool, str]:
    steps = [
        ("Downloading latest scores", [sys.executable, "scripts/download_data.py"]),
        (f"Rebuilding {forecast_name} forecast", [sys.executable, "scripts/run_analysis.py", *analysis_args]),
    ]
    progress = st.sidebar.progress(0, text=f"Starting {forecast_name} update...")
    status = st.sidebar.empty()
    details = st.sidebar.empty()
    start_time = time.monotonic()

    for index, (label, command) in enumerate(steps, start=1):
        percent = int(((index - 1) / len(steps)) * 100)
        progress.progress(percent, text=label)
        status.info(f"{label} ({index}/{len(steps)})")
        result = run_update_step(command)
        elapsed = int(time.monotonic() - start_time)
        if result.returncode != 0:
            progress.progress(percent, text="Update failed")
            output = (result.stderr or result.stdout or "Unknown update failure").strip()
            details.code(output[-1500:] or "No process output")
            return False, f"{forecast_name.title()} update failed after {elapsed}s."
        output = (result.stdout or result.stderr or "").strip()
        if output:
            details.code(output[-1200:])

    elapsed = int(time.monotonic() - start_time)
    progress.progress(100, text=f"{forecast_name.title()} update complete")
    status.success(f"{forecast_name.title()} data updated in {elapsed}s.")
    st.cache_data.clear()
    return True, f"{forecast_name.title()} data updated in {elapsed}s."


def update_live_data_with_progress() -> tuple[bool, str]:
    return update_forecast_data_with_progress("live", ["--live"])


def update_pre_knockout_data_with_progress() -> tuple[bool, str]:
    return update_forecast_data_with_progress("pre-knockout", ["--pre-knockout"])

ROUND_SEQUENCE = ["round_of_32", "round_of_16", "quarterfinals", "semifinals", "final"]
ROUND_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarterfinals": "Quarterfinals",
    "semifinals": "Semifinals",
    "final": "Final",
}


def format_probability(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def format_probability_or_blank(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return format_probability(float(value))
    except (TypeError, ValueError):
        return ""


def heatmap_color(rate: float) -> tuple[str, str, str]:
    rate = max(0.0, min(1.0, float(rate)))
    if rate >= 0.8:
        return "#14532d", "#bbf7d0", "#dcfce7"
    if rate >= 0.65:
        return "#3f6212", "#d9f99d", "#f0fdf4"
    if rate >= 0.5:
        return "#854d0e", "#fde68a", "#fffbeb"
    return "#7f1d1d", "#fecaca", "#fef2f2"


def group_accuracy_card(label: str, correct: int, total: int) -> str:
    rate = correct / total if total else 0.0
    background, accent, foreground = heatmap_color(rate)
    return f"""
      <div class="group-accuracy-card" style="background:{background}; border-color:{accent}; color:{foreground};">
        <div class="group-accuracy-label">{escape(label)}</div>
        <div class="group-accuracy-count">{correct}/{total}</div>
        <div class="group-accuracy-rate">{format_probability(rate)}</div>
      </div>
    """


def render_group_accuracy_cards(items: list[tuple[str, int, int]]) -> None:
    cards = "\n".join(group_accuracy_card(label, correct, total) for label, correct, total in items)
    components.html(
        f"""
        <style>
          html,
          body {{
            margin: 0;
            background: transparent;
            font-family: "Source Sans Pro", Arial, sans-serif;
          }}
          * {{
            box-sizing: border-box;
          }}
          .group-accuracy-grid {{
            display: flex;
            flex-direction: row;
            flex-wrap: nowrap;
            gap: 12px;
            margin: 0;
            padding: 2px 2px 10px;
            overflow-x: auto;
            overflow-y: hidden;
            scrollbar-width: thin;
          }}
          .group-accuracy-card {{
            flex: 0 0 190px;
            border: 1.5px solid;
            border-radius: 8px;
            padding: 12px 14px;
            min-height: 112px;
            box-shadow: 0 1px 5px rgba(15, 23, 42, 0.18);
          }}
          .group-accuracy-label {{
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.2;
            min-height: 2.1em;
            margin-bottom: 8px;
          }}
          .group-accuracy-count {{
            font-size: 1.75rem;
            font-weight: 750;
            line-height: 1;
            letter-spacing: 0;
          }}
          .group-accuracy-rate {{
            margin-top: 7px;
            font-size: 0.95rem;
            font-weight: 700;
            opacity: 0.92;
          }}
        </style>
        <div class="group-accuracy-grid">{cards}</div>
        """,
        height=142,
        scrolling=False,
    )


def card_winner_display(row: pd.Series | dict[str, object]) -> tuple[str, str, str]:
    get_value = row.get if isinstance(row, dict) else row.get
    actual_winner = str(get_value("actual_winner", "") or "")
    predicted_winner_value = get_value("prediction_winner_top", get_value("winner_top", ""))
    display_prediction = bool(str(predicted_winner_value or ""))
    if actual_winner and not display_prediction:
        return "Prediction", "No snapshot", ""
    try:
        slot_known = (
            float(get_value("team_a_probability", 0.0) or 0.0) >= 0.999
            and float(get_value("team_b_probability", 0.0) or 0.0) >= 0.999
        )
    except (TypeError, ValueError):
        slot_known = False
    winner_label = "Prediction" if display_prediction else "Head-to-head" if slot_known else "Top winner"
    winner_value = (
        get_value("prediction_winner_top", get_value("winner_top", ""))
        if display_prediction
        else get_value("winner_top", "")
    )
    if display_prediction:
        winner_probability_value = get_value(
            "prediction_winner_match_probability",
            get_value(
                "prediction_winner_probability",
                get_value("winner_match_probability", get_value("winner_probability", "")),
            ),
        )
    else:
        winner_probability_value = get_value("winner_match_probability", get_value("winner_probability", ""))
    return winner_label, str(winner_value), format_probability_or_blank(winner_probability_value)


def svg_text_lines(text: object, max_chars: int = 22, max_lines: int = 2) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break
    if current and len(lines) < max_lines:
        remaining_words = words[sum(len(line.split()) for line in lines):]
        line = " ".join(remaining_words) if len(lines) == max_lines - 1 else current
        if len(line) > max_chars:
            line = f"{line[: max_chars - 1]}..."
        lines.append(line)
    return lines[:max_lines]


def match_card(row: pd.Series, left: float, top: float) -> str:
    team_a = escape(str(row["team_a_top"]))
    team_b = escape(str(row["team_b_top"]))
    team_a_probability = format_probability(row["team_a_probability"])
    team_b_probability = format_probability(row["team_b_probability"])
    match_id = int(row["match"])
    status = str(row.get("prediction_status", "") or "")
    status_class = {
        "Successfully predicted": "is-success",
        "False predicted": "is-false",
        "Ongoing": "is-ongoing",
        "No round snapshot": "is-missing",
    }.get(status, "")
    winner_label, winner_value, winner_probability = card_winner_display(row)
    winner = escape(winner_value)
    status_html = ""
    if status:
        detail_html = "".join(f"<strong>{escape(line)}</strong>" for line in actual_status_lines(row))
        status_html = (
            f'<div class="bracket-status {status_class}">'
            f'<span>{escape(status)}</span>{detail_html}'
            "</div>"
        )
    return f"""
    <div class="bracket-card-wrap" style="left: {left:.1f}px; top: {top:.1f}px;">
      <div class="bracket-card {status_class}">
        <div class="bracket-match">Match {match_id}</div>
        <div class="bracket-team">
          <span>{team_a}</span><strong>Slot {team_a_probability}</strong>
        </div>
        <div class="bracket-team">
          <span>{team_b}</span><strong>Slot {team_b_probability}</strong>
        </div>
        <div class="bracket-winner"><span><small>{escape(winner_label)}</small>{winner}</span><strong>{winner_probability}</strong></div>
      </div>
      {status_html}
    </div>
    """


def render_bracket_chart(bracket: pd.DataFrame, zoom: float = 1.0) -> None:
    match_lookup = {int(row["match"]): row for _, row in bracket.iterrows()}
    show_status = "prediction_status" in bracket.columns
    card_width = 202
    card_height = 132
    status_gap = 6
    status_height = 52 if show_status else 0
    match_block_height = card_height + (status_gap + status_height if show_status else 0)
    board_width = 2100
    board_height = 1620 if show_status else 1200
    zoom = max(0.5, min(float(zoom), 1.6))
    zoomed_width = board_width * zoom
    zoomed_height = board_height * zoom
    column_x = {
        "left_r32": 20,
        "left_r16": 252,
        "left_qf": 484,
        "left_sf": 716,
        "final": 949,
        "right_sf": 1182,
        "right_qf": 1414,
        "right_r16": 1646,
        "right_r32": 1878,
    }
    base_y = 62
    step_y = 194 if show_status else 146
    positions: dict[int, tuple[float, float]] = {}
    for index, match_id in enumerate([73, 75, 74, 77, 83, 84, 81, 82]):
        positions[match_id] = (column_x["left_r32"], base_y + index * step_y)
    for index, match_id in enumerate([76, 78, 79, 80, 86, 88, 85, 87]):
        positions[match_id] = (column_x["right_r32"], base_y + index * step_y)

    source_pairs = {
        89: (73, 75),
        90: (74, 77),
        91: (76, 78),
        92: (79, 80),
        93: (83, 84),
        94: (81, 82),
        95: (86, 88),
        96: (85, 87),
        97: (89, 90),
        98: (93, 94),
        99: (91, 92),
        100: (95, 96),
        101: (97, 98),
        102: (99, 100),
        104: (101, 102),
    }
    x_by_match = {
        89: column_x["left_r16"],
        90: column_x["left_r16"],
        93: column_x["left_r16"],
        94: column_x["left_r16"],
        97: column_x["left_qf"],
        98: column_x["left_qf"],
        101: column_x["left_sf"],
        104: column_x["final"],
        102: column_x["right_sf"],
        99: column_x["right_qf"],
        100: column_x["right_qf"],
        91: column_x["right_r16"],
        92: column_x["right_r16"],
        95: column_x["right_r16"],
        96: column_x["right_r16"],
    }
    for match_id in [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104]:
        source_a, source_b = source_pairs[match_id]
        source_centers = [
            positions[source_a][1] + card_height / 2,
            positions[source_b][1] + card_height / 2,
        ]
        positions[match_id] = (x_by_match[match_id], sum(source_centers) / 2 - card_height / 2)

    def card_center(match_id: int) -> tuple[float, float]:
        left, top = positions[match_id]
        return left + card_width / 2, top + card_height / 2

    def connector_rectangles(source_id: int, target_id: int) -> list[tuple[float, float, float, float]]:
        source_left, source_top = positions[source_id]
        target_left, target_top = positions[target_id]
        source_y = source_top + card_height / 2
        target_y = target_top + card_height / 2
        if source_left < target_left:
            start_x = source_left + card_width
            end_x = target_left
        else:
            start_x = source_left
            end_x = target_left + card_width
        mid_x = (start_x + end_x) / 2
        thickness = 4
        top_band = source_y - thickness / 2
        bottom_band = target_y - thickness / 2
        left_band = min(start_x, mid_x)
        right_band = min(mid_x, end_x)
        vertical_top = min(source_y, target_y)
        vertical_height = max(1.0, abs(target_y - source_y))
        return [
            (left_band, top_band, max(1.0, abs(mid_x - start_x)), thickness),
            (mid_x - thickness / 2, vertical_top, thickness, vertical_height),
            (right_band, bottom_band, max(1.0, abs(end_x - mid_x)), thickness),
        ]

    def connector_segments(source_id: int, target_id: int) -> str:
        return "\n".join(
            f'<div class="bracket-connector" style="left: {left:.1f}px; top: {top:.1f}px; width: {width:.1f}px; height: {height:.1f}px;"></div>'
            for left, top, width, height in connector_rectangles(source_id, target_id)
        )

    connector_html = "\n".join(
        connector_segments(source_id, target_id)
        for target_id, sources in source_pairs.items()
        for source_id in sources
        if target_id in match_lookup and source_id in match_lookup
    )
    cards = "\n".join(
        match_card(match_lookup[match_id], *positions[match_id])
        for match_id in sorted(match_lookup)
        if match_id in positions
    )
    labels = [
        ("Round of 32", column_x["left_r32"]),
        ("Round of 16", column_x["left_r16"]),
        ("Quarterfinals", column_x["left_qf"]),
        ("Semifinal", column_x["left_sf"]),
        ("Final", column_x["final"]),
        ("Semifinal", column_x["right_sf"]),
        ("Quarterfinals", column_x["right_qf"]),
        ("Round of 16", column_x["right_r16"]),
        ("Round of 32", column_x["right_r32"]),
    ]
    label_html = "\n".join(
        f'<div class="bracket-label" style="left: {left}px;">{escape(label)}</div>'
        for label, left in labels
    )
    connector_svg = "\n".join(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" rx="2" fill="#334155" />'
        for target_id, sources in source_pairs.items()
        for source_id in sources
        if target_id in match_lookup and source_id in match_lookup
        for left, top, width, height in connector_rectangles(source_id, target_id)
    )
    label_svg = "\n".join(
        f'<text x="{left + card_width / 2:.1f}" y="34" text-anchor="middle" class="svg-round-label">{escape(label)}</text>'
        for label, left in labels
    )

    def svg_text_block(lines: list[str], x: float, y: float, class_name: str, anchor: str = "start") -> str:
        tspans = "\n".join(
            f'<tspan x="{x:.1f}" dy="{0 if index == 0 else 13}">{escape(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        return f'<text text-anchor="{anchor}" y="{y:.1f}" class="{class_name}">{tspans}</text>'

    def svg_card(row: pd.Series, left: float, top: float) -> str:
        status = str(row.get("prediction_status", "") or "")
        stroke = {
            "Successfully predicted": "#16a34a",
            "False predicted": "#dc2626",
            "Ongoing": "#94a3b8",
            "No round snapshot": "#f59e0b",
        }.get(status, "#d7dde6")
        status_fill = {
            "Successfully predicted": "#15803d",
            "False predicted": "#b91c1c",
            "Ongoing": "#f8fafc",
            "No round snapshot": "#92400e",
        }.get(status, "#f8fafc")
        status_text_color = {
            "Successfully predicted": "#ffffff",
            "False predicted": "#ffffff",
            "Ongoing": "#475569",
            "No round snapshot": "#ffffff",
        }.get(status, "#475569")
        winner_label, winner_value, winner_probability = card_winner_display(row)
        status_lines = actual_status_lines(row)
        team_a_probability = format_probability(row["team_a_probability"])
        team_b_probability = format_probability(row["team_b_probability"])
        team_a_slot_probability = f"Slot {team_a_probability}"
        team_b_slot_probability = f"Slot {team_b_probability}"
        match_y = top + 21
        team_a_y = top + 47
        team_b_y = top + 72
        winner_label_y = top + 99
        winner_y = top + 116
        status_top = top + card_height + status_gap
        status_y = status_top + 13
        status_detail_y = status_top + 30
        status_score_y = status_top + 44
        status_detail = status_lines[0]
        status_score = status_lines[1] if len(status_lines) > 1 else ""
        return "\n".join(
            [
                f'<rect x="{left:.1f}" y="{top:.1f}" width="{card_width}" height="{card_height}" rx="8" fill="#ffffff" stroke="{stroke}" stroke-width="1.5" />',
                f'<text x="{left + 12:.1f}" y="{match_y:.1f}" class="svg-match">Match {int(row["match"])}</text>',
                svg_text_block(svg_text_lines(row["team_a_top"], 19, 2), left + 12, team_a_y, "svg-team"),
                f'<text x="{left + card_width - 12:.1f}" y="{team_a_y:.1f}" text-anchor="end" class="svg-prob">{team_a_slot_probability}</text>',
                svg_text_block(svg_text_lines(row["team_b_top"], 19, 2), left + 12, team_b_y, "svg-team"),
                f'<text x="{left + card_width - 12:.1f}" y="{team_b_y:.1f}" text-anchor="end" class="svg-prob">{team_b_slot_probability}</text>',
                f'<line x1="{left + 12:.1f}" y1="{top + 88:.1f}" x2="{left + card_width - 12:.1f}" y2="{top + 88:.1f}" stroke="#edf1f5" />',
                f'<text x="{left + 12:.1f}" y="{winner_label_y:.1f}" class="svg-winner-label">{escape(winner_label)}</text>',
                svg_text_block(svg_text_lines(winner_value, 21, 2), left + 12, winner_y, "svg-winner"),
                f'<text x="{left + card_width - 12:.1f}" y="{winner_y:.1f}" text-anchor="end" class="svg-prob svg-winner-prob">{winner_probability}</text>',
                f'<rect x="{left:.1f}" y="{status_top:.1f}" width="{card_width}" height="{status_height}" rx="7" fill="{status_fill}" stroke="{stroke}" stroke-width="1.5" />',
                f'<text x="{left + card_width / 2:.1f}" y="{status_y:.1f}" text-anchor="middle" fill="{status_text_color}" class="svg-status">{escape(status)}</text>',
                f'<text x="{left + card_width / 2:.1f}" y="{status_detail_y:.1f}" text-anchor="middle" fill="{status_text_color}" class="svg-status-detail">{escape(status_detail)}</text>',
                f'<text x="{left + card_width / 2:.1f}" y="{status_score_y:.1f}" text-anchor="middle" fill="{status_text_color}" class="svg-status-detail">{escape(status_score)}</text>',
            ]
        )

    card_svg = "\n".join(
        svg_card(match_lookup[match_id], *positions[match_id])
        for match_id in sorted(match_lookup)
        if match_id in positions
    )
    download_svg = f"""
        <svg id="bracketDownloadSvg" xmlns="http://www.w3.org/2000/svg" width="{board_width}" height="{board_height}" viewBox="0 0 {board_width} {board_height}" style="position:absolute;width:0;height:0;overflow:hidden;">
          <style>
            .svg-round-label {{ font: 700 14px Arial, sans-serif; fill: #1f2937; }}
            .svg-match {{ font: 12px Arial, sans-serif; fill: #64748b; }}
            .svg-team {{ font: 13px Arial, sans-serif; fill: #111827; }}
            .svg-prob {{ font: 700 12px Arial, sans-serif; fill: #334155; }}
            .svg-winner-label {{ font: 700 10px Arial, sans-serif; fill: #64748b; }}
            .svg-winner {{ font: 700 12px Arial, sans-serif; fill: #0f766e; }}
            .svg-winner-prob {{ fill: #0f766e; }}
            .svg-status {{ font: 700 11px Arial, sans-serif; }}
            .svg-status-detail {{ font: 700 11px Arial, sans-serif; }}
          </style>
          <rect width="{board_width}" height="{board_height}" rx="10" fill="#f8fafc" stroke="#cbd5e1" />
          {label_svg}
          {connector_svg}
          {card_svg}
        </svg>
        """
    scroll_height = "height: 760px; max-height: 760px;"
    fullscreen_controls = f"""
        <div class="bracket-toolbar">
          <span id="bracketZoomReadout">{int(zoom * 100)}%</span>
          <button id="bracketDownloadButton" type="button">Download PNG</button>
          <button id="bracketFullscreenButton" type="button">Fullscreen</button>
        </div>
        """
    fullscreen_script = """
        <script>
          const scrollArea = document.querySelector(".bracket-scroll");
          const zoomFrame = document.querySelector(".bracket-zoom-frame");
          const board = document.querySelector(".bracket-board");
          const fullscreenButton = document.getElementById("bracketFullscreenButton");
          const downloadButton = document.getElementById("bracketDownloadButton");
          const zoomReadout = document.getElementById("bracketZoomReadout");
          const boardWidth = BOARD_WIDTH;
          const boardHeight = BOARD_HEIGHT;
          let currentZoom = INITIAL_ZOOM;
          let isDragging = false;
          let startX = 0;
          let startY = 0;
          let scrollLeft = 0;
          let scrollTop = 0;

          function clamp(value, min, max) {
            return Math.min(Math.max(value, min), max);
          }

          function applyZoom(nextZoom, anchorX, anchorY) {
            if (!scrollArea || !zoomFrame || !board) return;
            const previousZoom = currentZoom;
            currentZoom = clamp(nextZoom, 0.5, 1.8);
            if (Math.abs(currentZoom - previousZoom) < 0.001) return;

            const rect = scrollArea.getBoundingClientRect();
            const localX = anchorX - rect.left;
            const localY = anchorY - rect.top;
            const contentX = (scrollArea.scrollLeft + localX) / previousZoom;
            const contentY = (scrollArea.scrollTop + localY) / previousZoom;

            zoomFrame.style.width = (boardWidth * currentZoom) + "px";
            zoomFrame.style.height = (boardHeight * currentZoom) + "px";
            board.style.transform = "scale(" + currentZoom + ")";
            if (zoomReadout) {
              zoomReadout.textContent = Math.round(currentZoom * 100) + "%";
            }

            scrollArea.scrollLeft = contentX * currentZoom - localX;
            scrollArea.scrollTop = contentY * currentZoom - localY;
          }

          if (scrollArea) {
            scrollArea.addEventListener("mousedown", (event) => {
              if (event.target.closest("button")) return;
              isDragging = true;
              scrollArea.classList.add("is-dragging");
              startX = event.pageX - scrollArea.offsetLeft;
              startY = event.pageY - scrollArea.offsetTop;
              scrollLeft = scrollArea.scrollLeft;
              scrollTop = scrollArea.scrollTop;
            });
            window.addEventListener("mouseup", () => {
              isDragging = false;
              scrollArea.classList.remove("is-dragging");
            });
            scrollArea.addEventListener("mouseleave", () => {
              isDragging = false;
              scrollArea.classList.remove("is-dragging");
            });
            scrollArea.addEventListener("mousemove", (event) => {
              if (!isDragging) return;
              event.preventDefault();
              const x = event.pageX - scrollArea.offsetLeft;
              const y = event.pageY - scrollArea.offsetTop;
              scrollArea.scrollLeft = scrollLeft - (x - startX);
              scrollArea.scrollTop = scrollTop - (y - startY);
            });
            scrollArea.addEventListener("wheel", (event) => {
              event.preventDefault();
              const zoomFactor = event.deltaY > 0 ? 0.92 : 1.08;
              applyZoom(currentZoom * zoomFactor, event.clientX, event.clientY);
            }, { passive: false });
          }

          if (fullscreenButton) {
            fullscreenButton.addEventListener("click", async () => {
              const root = document.documentElement;
              if (document.fullscreenElement) {
                await document.exitFullscreen();
              } else if (root.requestFullscreen) {
                await root.requestFullscreen();
              }
            });
          }

          if (downloadButton) {
            downloadButton.addEventListener("click", () => {
              const svg = document.getElementById("bracketDownloadSvg");
              if (!svg) return;
              const serializer = new XMLSerializer();
              const source = serializer.serializeToString(svg);
              const svgBlob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
              const svgUrl = URL.createObjectURL(svgBlob);
              const image = new Image();
              image.onload = () => {
                const scale = 2;
                const canvas = document.createElement("canvas");
                canvas.width = boardWidth * scale;
                canvas.height = boardHeight * scale;
                const context = canvas.getContext("2d");
                context.fillStyle = "#ffffff";
                context.fillRect(0, 0, canvas.width, canvas.height);
                context.drawImage(image, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(svgUrl);
                canvas.toBlob((pngBlob) => {
                  if (!pngBlob) return;
                  const pngUrl = URL.createObjectURL(pngBlob);
                  const link = document.createElement("a");
                  link.href = pngUrl;
                  link.download = "world-cup-knockout-bracket.png";
                  document.body.appendChild(link);
                  link.click();
                  link.remove();
                  URL.revokeObjectURL(pngUrl);
                }, "image/png");
              };
              image.src = svgUrl;
            });
          }
        </script>
        """.replace("BOARD_WIDTH", str(board_width)).replace("BOARD_HEIGHT", str(board_height)).replace("INITIAL_ZOOM", f"{zoom:.3f}")

    bracket_html = f"""
        <style>
          body {{
            margin: 0;
            background: #ffffff;
            font-family: "Source Sans Pro", Arial, sans-serif;
          }}
          .bracket-shell {{
            position: relative;
            background: #ffffff;
          }}
          .bracket-toolbar {{
            position: sticky;
            top: 0;
            z-index: 5;
            display: flex;
            gap: 8px;
            align-items: center;
            justify-content: flex-end;
            padding: 8px 10px;
            background: rgba(255, 255, 255, 0.94);
            border-bottom: 1px solid #e2e8f0;
          }}
          .bracket-toolbar span {{
            color: #334155;
            font-size: 13px;
            font-weight: 650;
            min-width: 42px;
            text-align: right;
          }}
          .bracket-toolbar button {{
            appearance: none;
            border: 1px solid #94a3b8;
            border-radius: 6px;
            background: #ffffff;
            color: #0f172a;
            font-size: 13px;
            font-weight: 650;
            padding: 7px 10px;
            cursor: pointer;
          }}
          .bracket-toolbar button:hover {{
            background: #f8fafc;
          }}
          .bracket-scroll {{
            overflow: auto;
            padding: 6px 0 16px;
            {scroll_height}
            cursor: grab;
            user-select: none;
          }}
          .bracket-scroll.is-dragging {{
            cursor: grabbing;
          }}
          :fullscreen .bracket-toolbar {{
            background: #ffffff;
          }}
          :fullscreen .bracket-scroll {{
            height: calc(100vh - 54px);
            max-height: none;
          }}
          .bracket-zoom-frame {{
            position: relative;
            width: {zoomed_width:.1f}px;
            height: {zoomed_height:.1f}px;
          }}
          .bracket-board {{
            position: relative;
            width: {board_width}px;
            height: {board_height}px;
            background: linear-gradient(180deg, #fbfdff 0%, #f5f7fb 100%);
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            box-sizing: border-box;
            transform: scale({zoom:.3f});
            transform-origin: top left;
          }}
          .bracket-label {{
            position: absolute;
            top: 16px;
            width: {card_width}px;
            font-size: 0.84rem;
            font-weight: 700;
            color: #1f2937;
            text-align: center;
          }}
          .bracket-connectors {{
            position: absolute;
            inset: 0;
            z-index: 1;
            pointer-events: none;
          }}
          .bracket-connector {{
            position: absolute;
            background: #334155;
            border-radius: 999px;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.5);
          }}
          .bracket-card-wrap {{
            position: absolute;
            width: {card_width}px;
            height: {match_block_height}px;
            z-index: 2;
          }}
          .bracket-card {{
            position: relative;
            width: {card_width}px;
            height: {card_height}px;
            box-sizing: border-box;
            border: 1px solid #d7dde6;
            border-radius: 8px;
            background: #ffffff;
            padding: 12px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.10);
          }}
          .bracket-card.is-success {{
            border-color: #16a34a;
            box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.14);
          }}
          .bracket-card.is-false {{
            border-color: #dc2626;
            box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.14);
          }}
          .bracket-card.is-ongoing {{
            border-color: #94a3b8;
          }}
          .bracket-card.is-missing {{
            border-color: #f59e0b;
            box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.14);
          }}
          .bracket-match {{
            color: #64748b;
            font-size: 0.72rem;
            margin-bottom: 6px;
          }}
          .bracket-team {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 8px;
            align-items: start;
            padding: 3px 0;
            font-size: 0.81rem;
            line-height: 1.12;
            color: #111827;
          }}
          .bracket-team span {{
            min-width: 0;
            overflow-wrap: anywhere;
          }}
          .bracket-team strong {{
            color: #334155;
            font-weight: 650;
            white-space: nowrap;
            text-align: right;
            font-size: 0.70rem;
          }}
          .bracket-winner small {{
            display: block;
            color: #64748b;
            font-size: 0.58rem;
            font-weight: 700;
            line-height: 1;
            text-transform: uppercase;
          }}
          .bracket-winner {{
            margin-top: 8px;
            padding-top: 8px;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 8px;
            align-items: start;
            border-top: 1px solid #edf1f5;
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1.12;
          }}
          .bracket-winner span {{
            min-width: 0;
            overflow-wrap: anywhere;
          }}
          .bracket-winner strong {{
            white-space: nowrap;
            text-align: right;
            align-self: end;
          }}
          .bracket-status {{
            margin-top: {status_gap}px;
            width: {card_width}px;
            height: {status_height}px;
            box-sizing: border-box;
            border: 1.5px solid #94a3b8;
            border-radius: 7px;
            background: #ffffff;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 1px;
            font-size: 0.66rem;
            line-height: 1.1;
            color: #475569;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
          }}
          .bracket-status span {{
            min-width: 0;
            max-width: calc({card_width}px - 16px);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-weight: 700;
          }}
          .bracket-status strong {{
            max-width: calc({card_width}px - 16px);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-weight: 650;
            font-size: 0.64rem;
          }}
          .bracket-status.is-success {{
            border-color: #16a34a;
            background: #15803d;
            color: #ffffff;
          }}
          .bracket-status.is-false {{
            border-color: #dc2626;
            background: #b91c1c;
            color: #ffffff;
          }}
          .bracket-status.is-ongoing {{
            border-color: #94a3b8;
            background: #f8fafc;
            color: #475569;
          }}
          .bracket-status.is-missing {{
            border-color: #f59e0b;
            background: #92400e;
            color: #ffffff;
          }}
          .bracket-status.is-success span,
          .bracket-status.is-success strong {{
            color: #ffffff;
          }}
          .bracket-status.is-false span,
          .bracket-status.is-false strong {{
            color: #ffffff;
          }}
          .bracket-status.is-missing span,
          .bracket-status.is-missing strong {{
            color: #ffffff;
          }}
        </style>
        <div class="bracket-shell">
          {download_svg}
          {fullscreen_controls}
          <div class="bracket-scroll">
            <div class="bracket-zoom-frame">
              <div class="bracket-board">
                {label_html}
                <div class="bracket-connectors" aria-hidden="true">
                  {connector_html}
                </div>
                {cards}
              </div>
            </div>
          </div>
        </div>
        {fullscreen_script}
        """
    components.html(bracket_html, height=820, scrolling=False)

forecast_options = {
    "Live": {
        "team": live_simulation_path,
        "team_ci": live_simulation_interval_path,
        "groups": live_group_positions_path,
        "bracket": live_bracket_path,
        "prediction_bracket": pre_knockout_bracket_path,
        "matches": live_match_probabilities_path,
    },
    "Pre-knockout": {
        "team": pre_knockout_simulation_path,
        "team_ci": pre_knockout_simulation_interval_path,
        "groups": pre_knockout_group_positions_path,
        "bracket": pre_knockout_bracket_path,
        "prediction_bracket": pre_knockout_bracket_path,
        "matches": pre_knockout_match_probabilities_path,
    },
    "Pre-tournament": {
        "team": simulation_path,
        "team_ci": simulation_interval_path,
        "groups": group_positions_path,
        "bracket": bracket_path,
        "prediction_bracket": bracket_path,
        "matches": match_probabilities_path,
    },
}
forecast_descriptions = {
    "Live": (
        "Uses the latest downloaded 2026 fixture/results feed. Completed group-stage "
        "matches are locked into the standings, then remaining group matches and the "
        "knockout bracket are simulated. It changes only after a live update rebuilds "
        "the generated files."
    ),
    "Pre-knockout": (
        "Frozen after the group stage is complete and before knockout results are applied. "
        "The bracket uses actual group-stage qualifiers and standings, then simulates the "
        "knockout path from that fixed point."
    ),
    "Pre-tournament": (
        "Frozen before-kickoff forecast. No 2026 completed-match results are locked; "
        "the full tournament is simulated from the configured groups, bracket, team "
        "strength ratings, and model assumptions."
    ),
}
available_options = dict(forecast_options)

if "last_forecast_update" in st.session_state:
    status, message = st.session_state.pop("last_forecast_update")
    if status == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
if st.sidebar.button("Update live data", type="primary"):
    success, message = update_live_data_with_progress()
    if success:
        st.session_state["last_forecast_update"] = ("success", message)
        st.rerun()
    else:
        st.session_state["last_forecast_update"] = ("error", message)
        st.rerun()
if st.sidebar.button("Build pre-knockout snapshot"):
    success, message = update_pre_knockout_data_with_progress()
    if success:
        st.session_state["last_forecast_update"] = ("success", message)
        st.rerun()
    else:
        st.session_state["last_forecast_update"] = ("error", message)
        st.rerun()
if st.sidebar.button("Reload generated files"):
    st.rerun()

if available_options:
    default_index = 0 if "Live" in available_options else len(available_options) - 1
    selected_label = st.sidebar.radio("Forecast", list(available_options), index=default_index)
    selected_paths = available_options[selected_label]
    st.sidebar.caption(f"Showing {selected_label.lower()} outputs")
    st.sidebar.info(forecast_descriptions[selected_label])
    st.sidebar.caption(f"Last generated: {modified_time(selected_paths['team'])}")
    if selected_label == "Pre-knockout" and not selected_paths["team"].exists():
        st.sidebar.warning("Pre-knockout snapshot has not been generated yet.")
else:
    selected_label = "None"
    selected_paths = {}
    st.sidebar.warning("No generated forecast files found. Click Update live data to build live outputs.")

actual_group_matches: list[dict[str, object]] = []
actual_knockout_matches: list[dict[str, object]] = []
actual_groups = pd.DataFrame()
group_accuracy_table = pd.DataFrame()
group_accuracy_metrics: dict[str, float] = {}
live_results_frame: pd.DataFrame | None = None
try:
    context = load_tournament_context()
    live_results_path = Path(context["live_results_path"])
    teams_by_group = context["teams_by_group"]
    bracket_config = context["bracket_config"]
    team_mapping = context["team_mapping"]
    third_place_total = int(read_yaml(tournament_config_path).get("third_place_qualifiers", 8))
    if live_results_path.exists():
        live_results_frame = pd.read_csv(live_results_path)
        actual_group_matches = completed_group_matches_from_fixture_frame(live_results_frame, team_mapping)
        actual_groups = group_table_from_completed_matches(teams_by_group, actual_group_matches)
        actual_knockout_matches = completed_knockout_matches_from_fixture_frame(
            live_results_frame,
            bracket_config,
            team_mapping,
            group_table=actual_groups,
            third_place_count=third_place_total,
        )
        expected_group_matches = expected_group_match_count(teams_by_group)
        if actual_group_matches:
            st.sidebar.caption(f"Completed group matches: {len(actual_group_matches)}/{expected_group_matches}")
        if actual_knockout_matches:
            st.sidebar.caption(f"Completed knockout matches: {len(actual_knockout_matches)}")
        pre_tournament_groups = read_csv_if_exists(group_positions_path)
        pre_tournament_teams = read_csv_if_exists(simulation_path)
        group_accuracy_table, group_accuracy_metrics = group_stage_accuracy(
            pre_tournament_groups,
            pre_tournament_teams,
            actual_groups,
            third_place_total,
        )
except Exception as exc:
    st.sidebar.warning(f"Actual-results evaluation unavailable: {exc}")

if selected_label == "Live":
    try:
        live_results_path_for_reconstruction = context_live_results_path()
        if live_results_path_for_reconstruction.exists():
            with st.spinner("Checking rolling round snapshots..."):
                generated_snapshots = ensure_reconstructed_live_snapshots(
                    live_results_path_for_reconstruction.stat().st_mtime,
                    simulation_profile="dev",
                )
            if generated_snapshots:
                st.sidebar.info("Built reconstructed snapshots: " + ", ".join(generated_snapshots))
    except Exception as exc:
        st.sidebar.warning(f"Reconstructed rolling snapshots unavailable: {exc}")

prob_tab, match_tab, group_tab, bracket_tab, research_tab, registry_tab, backtest_tab = st.tabs(
    [
        "Probabilities",
        "Match Probabilities",
        "Group Standings",
        "Knockout Bracket",
        "Research Evaluation",
        "Forecast Registry",
        "Backtests",
    ]
)

with prob_tab:
    st.subheader("Tournament Probabilities")
    if selected_paths and selected_paths["team"].exists():
        probabilities = numeric_frame(pd.read_csv(selected_paths["team"]), {"team"})
        leader = probabilities.sort_values("champion", ascending=False).iloc[0]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Top Champion", str(leader["team"]))
        metric_cols[1].metric("Champion Probability", format_probability(leader["champion"]))
        metric_cols[2].metric("Final Probability", format_probability(leader["reach_final"]))
        metric_cols[3].metric("Advance From Group", format_probability(leader["advance_from_group"]))

        ci_path = selected_paths.get("team_ci")
        ci_frame = read_csv_if_exists(ci_path) if ci_path else None
        if ci_frame is not None:
            ci_frame = numeric_frame(ci_frame, {"team"})
            ci_columns = ["team", "champion_mean", "champion_p05", "champion_p50", "champion_p95"]
            st.subheader("Champion Probability With Simulation Interval")
            st.dataframe(
                ci_frame[ci_columns],
                width="stretch",
                column_config=probability_column_config(ci_columns[1:]),
            )
        st.subheader("Progression Probabilities")
        st.dataframe(probabilities, width="stretch")
    else:
        st.info("Run the tournament simulation to populate team probabilities.")

with match_tab:
    st.subheader("Match-Level Group Probabilities")
    match_path = selected_paths.get("matches") if selected_paths else None
    if match_path and match_path.exists():
        match_probabilities = numeric_frame(pd.read_csv(match_path), {"group", "team_a", "team_b"})
        groups = sorted(match_probabilities["group"].dropna().unique())
        selected_group = st.selectbox("Group", groups, key="match_probability_group")
        group_matches = match_probabilities[match_probabilities["group"] == selected_group].copy()
        display_columns = [
            "team_a",
            "team_b",
            "team_a_win",
            "draw",
            "team_b_win",
            "team_a_goals_lambda",
            "team_b_goals_lambda",
        ]
        st.dataframe(
            group_matches[display_columns],
            width="stretch",
            column_config=probability_column_config(["team_a_win", "draw", "team_b_win"]),
        )
        labels = [f"{row.team_a} vs {row.team_b}" for row in group_matches.itertuples(index=False)]
        if labels:
            selected_match = st.selectbox("Inspect match", labels, key="inspect_match_probability")
            match_index = labels.index(selected_match)
            row = group_matches.iloc[match_index]
            outcome_frame = pd.DataFrame(
                {
                    "probability": {
                        f"{row['team_a']} win": row["team_a_win"],
                        "Draw": row["draw"],
                        f"{row['team_b']} win": row["team_b_win"],
                    }
                }
            )
            st.bar_chart(outcome_frame)
    else:
        st.info("Run the tournament simulation to populate match-level probabilities.")

with group_tab:
    st.subheader("Group Position Probabilities")
    if group_accuracy_metrics:
        st.subheader("Pre-tournament vs Actual Group Stage")
        st.caption(
            "This evaluates the frozen pre-tournament forecast against the completed group stage and the actual "
            "pre-knockout field. Top-2 exact slots matter because group winners and runner-ups feed fixed bracket slots."
        )
        qualifiers_correct = int(group_accuracy_metrics["qualifiers_correct"])
        qualifiers_total = int(group_accuracy_metrics["qualifiers_total"])
        top_two_team_correct = int(group_accuracy_metrics["top_two_team_correct"])
        top_two_slot_correct = int(group_accuracy_metrics["top_two_slot_correct"])
        top_two_total = int(group_accuracy_metrics["top_two_total"])
        group_winners_correct = int(group_accuracy_metrics["group_winners_correct"])
        group_winners_total = int(group_accuracy_metrics["group_winners_total"])
        runner_ups_correct = int(group_accuracy_metrics["runner_ups_correct"])
        runner_ups_total = int(group_accuracy_metrics["runner_ups_total"])
        render_group_accuracy_cards(
            [
                ("Knockout Teams Correct", qualifiers_correct, qualifiers_total),
                ("Top-2 Teams Correct", top_two_team_correct, top_two_total),
                ("Top-2 Exact Slots", top_two_slot_correct, top_two_total),
                ("Group Winners", group_winners_correct, group_winners_total),
                ("Runner-ups", runner_ups_correct, runner_ups_total),
            ]
        )
        if not group_accuracy_table.empty:
            with st.expander("Show pre-tournament group-stage comparison"):
                st.dataframe(
                    group_accuracy_table,
                    width="stretch",
                    column_config={
                        "position_correct": st.column_config.CheckboxColumn("position_correct"),
                        "top_two_team_correct": st.column_config.CheckboxColumn("top_two_team_correct"),
                        "top_two_slot_correct": st.column_config.CheckboxColumn("top_two_slot_correct"),
                    },
                )
        if not actual_groups.empty:
            with st.expander("Show actual group-stage standings"):
                st.dataframe(
                    actual_groups.sort_values(["group", "position"]),
                    width="stretch",
                )
    group_path = selected_paths.get("groups") if selected_paths else None
    if group_path and group_path.exists():
        group_positions = numeric_frame(pd.read_csv(group_path), {"group", "team"})
        group = st.selectbox("Group", sorted(group_positions["group"].unique()))
        group_frame = group_positions[group_positions["group"] == group].copy()
        display_columns = ["team", "position_1", "position_2", "position_3", "position_4", "expected_position"]
        st.dataframe(
            group_frame[display_columns],
            width="stretch",
            column_config=probability_column_config(["position_1", "position_2", "position_3", "position_4"]),
        )
        st.bar_chart(group_frame.set_index("team")[["position_1", "position_2", "position_3", "position_4"]])
    else:
        st.info("Run the tournament simulation to populate group standings.")

with bracket_tab:
    st.subheader("Predicted Knockout Bracket")
    selected_bracket_path = selected_paths.get("bracket") if selected_paths else None
    if selected_bracket_path and selected_bracket_path.exists():
        prediction_bracket_path = selected_paths.get("prediction_bracket") if selected_paths else None
        prediction_bracket = read_csv_if_exists(prediction_bracket_path) if prediction_bracket_path else None
        source_bracket = pd.read_csv(selected_bracket_path)
        if selected_label == "Pre-knockout" and live_bracket_path.exists():
            source_bracket = pd.read_csv(live_bracket_path)
        baseline_bracket = None
        if selected_label == "Live":
            round_prediction_map, round_snapshot_labels = round_prediction_brackets(live_results_frame)
            bracket = bracket_prediction_status(
                source_bracket,
                actual_knockout_matches,
                prediction_brackets_by_round=round_prediction_map,
                snapshot_labels_by_round=round_snapshot_labels,
            )
            if prediction_bracket is not None:
                baseline_bracket = bracket_prediction_status(source_bracket, actual_knockout_matches, prediction_bracket)
            st.caption(
                "Live locks completed knockout results into the bracket path; prediction status is evaluated against "
                "the latest valid snapshot before each round starts."
            )
            live_snapshots = [
                snapshot
                for snapshot in forecast_snapshots()
                if str(snapshot.get("mode", "")) == "live"
            ]
            if live_snapshots:
                latest_live_snapshot = max(
                    live_snapshots,
                    key=lambda snapshot: pd.Timestamp(snapshot["cutoff"]),
                )
                live_metadata = latest_live_snapshot.get("metadata") or {}
                if live_metadata.get("anchored_live_update"):
                    anchor_weight = float(live_metadata.get("anchor_model_weight", 0.0))
                    live_weight = float(live_metadata.get("live_model_weight", 0.0))
                    knockout_count = int(live_metadata.get("live_knockout_training_matches", 0))
                    st.caption(
                        "Live model blend: "
                        f"{format_probability(anchor_weight)} pre-knockout anchor + "
                        f"{format_probability(live_weight)} knockout-updated model "
                        f"from {knockout_count} completed knockout matches."
                    )
        else:
            bracket = bracket_prediction_status(
                source_bracket,
                actual_knockout_matches,
                prediction_bracket,
            )
            if selected_label == "Pre-knockout" and selected_bracket_path != live_bracket_path and live_bracket_path.exists():
                st.caption(
                    "Pre-knockout shows the actual/live bracket path for placement, while prediction labels compare "
                    "against the frozen pre-knockout forecast."
                )
        summary = status_summary(bracket)
        if summary:
            completed = summary.get("Successfully predicted", 0) + summary.get("False predicted", 0)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Completed Evaluated", completed)
            metric_cols[1].metric("Successfully Predicted", summary.get("Successfully predicted", 0))
            metric_cols[2].metric("False Predicted", summary.get("False predicted", 0))
            if completed:
                accuracy_label = (
                    "Rolling Live Accuracy"
                    if selected_label == "Live"
                    else "Pre-KO Baseline Accuracy"
                    if selected_label == "Pre-knockout"
                    else f"{selected_label} Accuracy"
                )
                metric_cols[3].metric(
                    accuracy_label,
                    format_probability(summary.get("Successfully predicted", 0) / completed),
                )
            else:
                metric_cols[3].metric("Ongoing", summary.get("Ongoing", 0))
            missing_snapshots = summary.get("No round snapshot", 0)
            if selected_label == "Live" and missing_snapshots:
                st.warning(f"{missing_snapshots} completed match(es) do not have a saved round-specific snapshot yet.")
            if selected_label == "Live" and baseline_bracket is not None:
                baseline_summary = status_summary(baseline_bracket)
                baseline_completed = baseline_summary.get("Successfully predicted", 0) + baseline_summary.get("False predicted", 0)
                if baseline_completed:
                    baseline_accuracy = baseline_summary.get("Successfully predicted", 0) / baseline_completed
                    st.caption(
                        "Pre-KO baseline across completed knockout matches: "
                        f"{format_probability(baseline_accuracy)} "
                        f"({baseline_summary.get('Successfully predicted', 0)}/{baseline_completed})."
                    )
        if selected_label == "Live":
            rolling_accuracy = round_by_round_accuracy(actual_knockout_matches, live_results_frame)
            if not rolling_accuracy.empty:
                st.subheader("Round-by-round Forecast Accuracy")
                st.caption(
                    "Round of 32 uses the pre-knockout snapshot. Later rounds require a live snapshot after the "
                    "previous round finished and before the evaluated round started."
                )
                rolling_display = rolling_accuracy.copy()
                for probability_column in ["winner_accuracy", "avg_pick_share"]:
                    rolling_display[probability_column] = rolling_display[probability_column] * 100
                st.dataframe(
                    rolling_display,
                    width="stretch",
                    column_config={
                        "completed_matches": st.column_config.TextColumn("completed_matches"),
                        "evaluated_predictions": st.column_config.TextColumn("evaluated_predictions"),
                        "winner_accuracy": st.column_config.NumberColumn("winner_accuracy", format="%.1f%%"),
                        "avg_pick_share": st.column_config.NumberColumn("avg_pick_share", format="%.1f%%"),
                        "brier_score": st.column_config.NumberColumn("brier_score", format="%.3f"),
                    },
                )
            else:
                st.info("No saved round-by-round forecast snapshots are available yet.")
            live_sync = round_by_round_live_sync(actual_knockout_matches, live_results_frame)
            if not live_sync.empty:
                st.subheader("Completed-round Live Sync")
                st.caption(
                    "This checks whether a post-round live snapshot has incorporated completed winners. It is not "
                    "forecast accuracy because the result is already known."
                )
                sync_display = live_sync.copy()
                sync_display["sync_rate"] = sync_display["sync_rate"] * 100
                st.dataframe(
                    sync_display,
                    width="stretch",
                    column_config={
                        "completed_matches": st.column_config.TextColumn("completed_matches"),
                        "synced_winners": st.column_config.TextColumn("synced_winners"),
                        "sync_rate": st.column_config.NumberColumn("sync_rate", format="%.1f%%"),
                    },
                )
        if "bracket_zoom" not in st.session_state:
            st.session_state["bracket_zoom"] = 85
        zoom_out, zoom_slider, zoom_in, zoom_reset = st.columns([1, 8, 1, 1])
        with zoom_out:
            if st.button("-", key="bracket_zoom_out", help="Zoom out"):
                st.session_state["bracket_zoom"] = max(50, int(st.session_state["bracket_zoom"]) - 10)
        with zoom_in:
            if st.button("+", key="bracket_zoom_in", help="Zoom in"):
                st.session_state["bracket_zoom"] = min(160, int(st.session_state["bracket_zoom"]) + 10)
        with zoom_reset:
            if st.button("100", key="bracket_zoom_reset", help="Reset zoom to 100%"):
                st.session_state["bracket_zoom"] = 100
        with zoom_slider:
            st.slider(
                "Zoom",
                min_value=50,
                max_value=160,
                step=5,
                key="bracket_zoom",
            )
        st.caption(
            "Slot % is the chance a team occupies that bracket slot. Top winner % is the largest simulated winner share; "
            "Head-to-head appears when both slots are fixed."
        )
        render_bracket_chart(
            bracket,
            zoom=int(st.session_state["bracket_zoom"]) / 100.0,
        )
        round_names = list(bracket["round"].drop_duplicates())
        round_name = st.selectbox("Inspect round", round_names)
        round_frame = bracket[bracket["round"] == round_name].copy()
        with st.expander("Show bracket data"):
            st.dataframe(round_frame, width="stretch")
    else:
        st.info("Run the configured-bracket simulation to populate knockout predictions.")

with research_tab:
    st.subheader("Research Evaluation")
    research_views = st.tabs(["Baselines", "Calibration", "Ablation", "Nested Selection"])

    with research_views[0]:
        baseline_summary = read_csv_if_exists(baseline_summary_path)
        baseline_results = read_csv_if_exists(baseline_path)
        if baseline_summary is not None:
            baseline_summary = numeric_frame(baseline_summary, {"model"})
            st.caption(f"Last generated: {modified_time(baseline_summary_path)}")
            st.dataframe(baseline_summary, width="stretch")
            chart_columns = [column for column in ["log_loss_mean", "brier_score_mean", "ranked_probability_score_mean"] if column in baseline_summary.columns]
            if chart_columns:
                st.bar_chart(baseline_summary.set_index("model")[chart_columns])
        else:
            st.info("Run the analysis pipeline to populate baseline comparison reports.")
        if baseline_results is not None:
            with st.expander("Window-level baseline results"):
                st.dataframe(numeric_frame(baseline_results, {"model"}), width="stretch")

    with research_views[1]:
        calibration_summary = read_csv_if_exists(calibration_summary_path)
        calibration_by_world_cup = read_csv_if_exists(calibration_by_world_cup_path)
        calibration_table = read_csv_if_exists(calibration_table_path)
        sharpness = read_csv_if_exists(sharpness_path)
        if calibration_summary is not None:
            summary = numeric_frame(calibration_summary).iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric("ECE", f"{summary['expected_calibration_error']:.3f}")
            metric_cols[1].metric("MCE", f"{summary['maximum_calibration_error']:.3f}")
            metric_cols[2].metric("Mean Confidence", f"{summary['mean_confidence']:.3f}")
            metric_cols[3].metric("Top-1 Accuracy", f"{summary['top1_accuracy']:.3f}")
        else:
            st.info("Run the analysis pipeline to populate calibration diagnostics.")
        if calibration_by_world_cup is not None:
            st.subheader("Calibration By World Cup")
            st.dataframe(numeric_frame(calibration_by_world_cup), width="stretch")
        if calibration_table is not None:
            st.subheader("Reliability Table")
            calibration_table = numeric_frame(calibration_table, {"outcome"})
            outcome = st.selectbox("Outcome", sorted(calibration_table["outcome"].unique()))
            st.dataframe(calibration_table[calibration_table["outcome"] == outcome], width="stretch")
        if sharpness is not None:
            st.subheader("Sharpness")
            st.dataframe(numeric_frame(sharpness, {"metric"}), width="stretch")

    with research_views[2]:
        ablation_summary = read_csv_if_exists(ablation_summary_path)
        ablation_results = read_csv_if_exists(ablation_path)
        if ablation_summary is not None:
            ablation_summary = numeric_frame(ablation_summary, {"feature_set"})
            st.dataframe(ablation_summary, width="stretch")
            chart_columns = [column for column in ["log_loss_mean", "brier_score_mean", "ranked_probability_score_mean"] if column in ablation_summary.columns]
            if chart_columns:
                st.bar_chart(ablation_summary.set_index("feature_set")[chart_columns])
        else:
            st.info("Run the analysis pipeline to populate ablation reports.")
        if ablation_results is not None:
            with st.expander("Window-level ablation results"):
                st.dataframe(numeric_frame(ablation_results, {"feature_set"}), width="stretch")

    with research_views[3]:
        nested = read_csv_if_exists(nested_backtest_path)
        if nested is not None:
            nested = numeric_frame(nested, {"selected_model"})
            st.dataframe(nested, width="stretch")
            if "log_loss" in nested.columns:
                st.line_chart(nested.set_index("year")[["log_loss", "brier_score"]])
        else:
            st.info("Run the analysis pipeline to populate nested model-selection results.")

with registry_tab:
    st.subheader("Forecast Registry")
    registry = latest_registry_dir()
    if registry is None:
        st.info("Run the analysis pipeline to populate forecast registry outputs.")
    else:
        st.caption(f"Latest registry: `{registry.name}`")
        model_card_path = registry / "model_card.md"
        if model_card_path.exists():
            st.markdown(model_card_path.read_text(encoding="utf-8"))
        files = []
        for path in sorted(registry.iterdir()):
            if path.is_file():
                files.append(
                    {
                        "file": path.name,
                        "size_bytes": path.stat().st_size,
                        "modified": modified_time(path),
                    }
                )
        st.dataframe(pd.DataFrame(files), width="stretch")
        config_path = registry / "config.yaml"
        if config_path.exists():
            with st.expander("Forecast config"):
                st.code(config_path.read_text(encoding="utf-8"), language="yaml")

with backtest_tab:
    st.subheader("Backtest Results")
    if backtest_path.exists():
        backtests = pd.read_csv(backtest_path)
        if backtest_summary_path.exists():
            model_summary = pd.read_csv(backtest_summary_path)
            st.subheader("Average By Model")
            st.dataframe(model_summary, width="stretch")
        st.subheader("Window Results")
        st.dataframe(backtests, width="stretch")
    else:
        st.info("Run rolling World Cup backtests to populate model metrics.")
