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
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Prediction Engine")

st.caption("Match-level probabilities, time-aware backtesting, and Monte Carlo tournament simulation.")

simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026.csv"
live_simulation_path = ROOT / "outputs" / "simulations" / "team_probabilities_2026_live.csv"
group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026.csv"
live_group_positions_path = ROOT / "outputs" / "simulations" / "group_position_probabilities_2026_live.csv"
bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026.csv"
live_bracket_path = ROOT / "outputs" / "simulations" / "predicted_knockout_bracket_2026_live.csv"
backtest_path = ROOT / "outputs" / "backtest_results" / "model_backtest.csv"
backtest_summary_path = ROOT / "outputs" / "backtest_results" / "model_backtest_summary.csv"


def modified_time(path: Path) -> str:
    if not path.exists():
        return "not generated"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


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
    components.html(bracket_html, height=1350, scrolling=True)

forecast_options = {
    "Live": {
        "team": live_simulation_path,
        "groups": live_group_positions_path,
        "bracket": live_bracket_path,
    },
    "Pre-tournament": {
        "team": simulation_path,
        "groups": group_positions_path,
        "bracket": bracket_path,
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

prob_tab, group_tab, bracket_tab, backtest_tab = st.tabs(
    ["Probabilities", "Group Standings", "Knockout Bracket", "Backtests"]
)

with prob_tab:
    st.subheader("Tournament Probabilities")
    if selected_paths and selected_paths["team"].exists():
        probabilities = pd.read_csv(selected_paths["team"])
        st.dataframe(probabilities, use_container_width=True)
    else:
        st.info("Run the tournament simulation to populate team probabilities.")

with group_tab:
    st.subheader("Group Position Probabilities")
    group_path = selected_paths.get("groups") if selected_paths else None
    if group_path and group_path.exists():
        group_positions = pd.read_csv(group_path)
        group = st.selectbox("Group", sorted(group_positions["group"].unique()))
        group_frame = group_positions[group_positions["group"] == group].copy()
        display_columns = ["team", "position_1", "position_2", "position_3", "position_4", "expected_position"]
        st.dataframe(group_frame[display_columns], use_container_width=True)
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
            st.dataframe(round_frame, use_container_width=True)
    else:
        st.info("Run the configured-bracket simulation to populate knockout predictions.")

with backtest_tab:
    st.subheader("Backtest Results")
    if backtest_path.exists():
        backtests = pd.read_csv(backtest_path)
        if backtest_summary_path.exists():
            model_summary = pd.read_csv(backtest_summary_path)
            st.subheader("Average By Model")
            st.dataframe(model_summary, use_container_width=True)
        st.subheader("Window Results")
        st.dataframe(backtests, use_container_width=True)
    else:
        st.info("Run rolling World Cup backtests to populate model metrics.")
