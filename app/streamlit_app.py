from __future__ import annotations

import os
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Prediction Engine")

st.caption("Match-level probabilities, time-aware backtesting, and Monte Carlo tournament simulation.")

simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026.csv"
live_simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_live.csv"
simulation_interval_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_with_ci.csv"
live_simulation_interval_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_live_with_ci.csv"
group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026.csv"
live_group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026_live.csv"
bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026.csv"
live_bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026_live.csv"
match_probabilities_path = ROOT / "outputs" / "simulations" / "match_probabilities_2026.csv"
live_match_probabilities_path = ROOT / "outputs" / "simulations" / "match_probabilities_2026_live.csv"
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


def update_live_data_with_progress() -> tuple[bool, str]:
    steps = [
        ("Downloading latest scores", [sys.executable, "scripts/download_data.py"]),
        ("Rebuilding live forecast", [sys.executable, "scripts/run_analysis.py", "--live"]),
    ]
    progress = st.sidebar.progress(0, text="Starting live update...")
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
            return False, f"Live update failed after {elapsed}s."
        output = (result.stdout or result.stderr or "").strip()
        if output:
            details.code(output[-1200:])

    elapsed = int(time.monotonic() - start_time)
    progress.progress(100, text="Live update complete")
    status.success(f"Live data updated in {elapsed}s.")
    return True, f"Live data updated in {elapsed}s."

ROUND_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarterfinals": "Quarterfinals",
    "semifinals": "Semifinals",
    "final": "Final",
}


def format_probability(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def match_card(row: pd.Series, left: float, top: float) -> str:
    team_a = escape(str(row["team_a_top"]))
    team_b = escape(str(row["team_b_top"]))
    winner = escape(str(row["winner_top"]))
    team_a_probability = format_probability(row["team_a_probability"])
    team_b_probability = format_probability(row["team_b_probability"])
    winner_probability = format_probability(row["winner_probability"])
    match_id = int(row["match"])
    return f"""
    <div class="bracket-card" style="left: {left:.1f}px; top: {top:.1f}px;">
      <div class="bracket-match">Match {match_id}</div>
      <div class="bracket-team">
        <span>{team_a}</span><strong>{team_a_probability}</strong>
      </div>
      <div class="bracket-team">
        <span>{team_b}</span><strong>{team_b_probability}</strong>
      </div>
      <div class="bracket-winner">Winner: {winner} <strong>{winner_probability}</strong></div>
    </div>
    """


def render_bracket_chart(bracket: pd.DataFrame) -> None:
    match_lookup = {int(row["match"]): row for _, row in bracket.iterrows()}
    card_width = 178
    card_height = 94
    board_width = 2060
    board_height = 1110
    column_x = {
        "left_r32": 20,
        "left_r16": 250,
        "left_qf": 480,
        "left_sf": 710,
        "final": 940,
        "right_sf": 1170,
        "right_qf": 1400,
        "right_r16": 1630,
        "right_r32": 1860,
    }
    base_y = 62
    step_y = 122
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

    def connector_path(source_id: int, target_id: int) -> str:
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
        return f"M {start_x:.1f} {source_y:.1f} H {mid_x:.1f} V {target_y:.1f} H {end_x:.1f}"

    connector_paths = "\n".join(
        f'<path d="{connector_path(source_id, target_id)}" />'
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

    bracket_html = f"""
        <style>
          .bracket-scroll {{
            overflow-x: auto;
            padding: 6px 0 16px;
          }}
          .bracket-board {{
            position: relative;
            width: {board_width}px;
            height: {board_height}px;
            background: #f8fafc;
            border: 1px solid #d8dee9;
            border-radius: 8px;
            box-sizing: border-box;
          }}
          .bracket-label {{
            position: absolute;
            top: 18px;
            width: {card_width}px;
            font-size: 0.88rem;
            font-weight: 700;
            color: #1f2937;
            text-align: center;
          }}
          .bracket-lines {{
            position: absolute;
            inset: 0;
            width: {board_width}px;
            height: {board_height}px;
            z-index: 1;
            pointer-events: none;
          }}
          .bracket-lines path {{
            fill: none;
            stroke: #1d4ed8;
            stroke-width: 3;
            stroke-linecap: square;
            stroke-linejoin: round;
            opacity: 0.9;
          }}
          .bracket-card {{
            position: absolute;
            width: {card_width}px;
            height: {card_height}px;
            z-index: 2;
            box-sizing: border-box;
            border: 1px solid #d7dde6;
            border-radius: 8px;
            background: #ffffff;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
          }}
          .bracket-match {{
            color: #64748b;
            font-size: 0.74rem;
            margin-bottom: 7px;
          }}
          .bracket-team {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 4px 0;
            font-size: 0.84rem;
            color: #111827;
          }}
          .bracket-team strong {{
            color: #334155;
            font-weight: 650;
            white-space: nowrap;
          }}
          .bracket-winner {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #edf1f5;
            color: #0f766e;
            font-size: 0.8rem;
            font-weight: 650;
          }}
          .bracket-winner strong {{
            float: right;
          }}
        </style>
        <div class="bracket-scroll">
          <div class="bracket-board">
            {label_html}
            <svg class="bracket-lines" viewBox="0 0 {board_width} {board_height}" aria-hidden="true">
              {connector_paths}
            </svg>
            {cards}
          </div>
        </div>
        """
    st.html(bracket_html)

forecast_options = {
    "Live": {
        "team": live_simulation_path,
        "team_ci": live_simulation_interval_path,
        "groups": live_group_positions_path,
        "bracket": live_bracket_path,
        "matches": live_match_probabilities_path,
    },
    "Pre-tournament": {
        "team": simulation_path,
        "team_ci": simulation_interval_path,
        "groups": group_positions_path,
        "bracket": bracket_path,
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
    "Pre-tournament": (
        "Frozen before-kickoff forecast. No 2026 completed-match results are locked; "
        "the full tournament is simulated from the configured groups, bracket, team "
        "strength ratings, and model assumptions."
    ),
}
available_options = {
    label: paths
    for label, paths in forecast_options.items()
    if paths["team"].exists()
}

if "last_live_update" in st.session_state:
    status, message = st.session_state.pop("last_live_update")
    if status == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
if st.sidebar.button("Update live data", type="primary"):
    success, message = update_live_data_with_progress()
    if success:
        st.session_state["last_live_update"] = ("success", message)
        st.rerun()
    else:
        st.session_state["last_live_update"] = ("error", message)
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
else:
    selected_label = "None"
    selected_paths = {}
    st.sidebar.warning("No generated forecast files found. Click Update live data to build live outputs.")

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
        bracket = pd.read_csv(selected_bracket_path)
        render_bracket_chart(bracket)
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
