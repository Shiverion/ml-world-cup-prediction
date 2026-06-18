from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RANKINGS_URL = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/ranking_fifa_historical.csv"
WORLD_CUP_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"


def completed_match_results(url: str = RESULTS_URL) -> pd.DataFrame:
    results = pd.read_csv(url)
    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    missing = sorted(set(required) - set(results.columns))
    if missing:
        raise ValueError(f"Results source is missing columns: {missing}")

    completed = results.dropna(subset=["home_score", "away_score"]).copy()
    completed["home_score"] = completed["home_score"].astype(int)
    completed["away_score"] = completed["away_score"].astype(int)
    completed["date"] = pd.to_datetime(completed["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return completed[required].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def normalized_fifa_rankings(url: str = RANKINGS_URL) -> pd.DataFrame:
    rankings = pd.read_csv(url)
    required = ["team", "total_points", "date"]
    missing = sorted(set(required) - set(rankings.columns))
    if missing:
        raise ValueError(f"Rankings source is missing columns: {missing}")

    rankings = rankings.dropna(subset=["team", "total_points", "date"]).copy()
    rankings["rank_date"] = pd.to_datetime(rankings["date"], errors="raise")
    rankings["points"] = pd.to_numeric(rankings["total_points"], errors="raise")
    rankings = rankings.sort_values(["rank_date", "points", "team"], ascending=[True, False, True])
    rankings["rank"] = rankings.groupby("rank_date")["points"].rank(method="first", ascending=False).astype(int)
    output = rankings[["rank_date", "team", "rank", "points"]].copy()
    output["rank_date"] = output["rank_date"].dt.strftime("%Y-%m-%d")
    return output.reset_index(drop=True)


def world_cup_2026_matches(url: str = WORLD_CUP_2026_URL) -> pd.DataFrame:
    with urlopen(url, timeout=30) as response:
        payload: dict[str, Any] = json.load(response)

    rows: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        score = match.get("score") or {}
        full_time = score.get("ft") or [None, None]
        group = str(match.get("group", "")).replace("Group ", "").strip()
        rows.append(
            {
                "date": match.get("date"),
                "round": match.get("round", ""),
                "group": group,
                "team_a": match.get("team1"),
                "team_b": match.get("team2"),
                "team_a_score": full_time[0],
                "team_b_score": full_time[1],
                "status": "completed" if full_time[0] is not None and full_time[1] is not None else "scheduled",
                "ground": match.get("ground", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "group", "team_a", "team_b"]).reset_index(drop=True)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize raw data for the World Cup prediction pipeline.")
    parser.add_argument("--matches-out", default=ROOT / "data" / "raw" / "international_matches.csv")
    parser.add_argument("--rankings-out", default=ROOT / "data" / "raw" / "fifa_rankings.csv")
    parser.add_argument("--world-cup-2026-out", default=ROOT / "data" / "raw" / "world_cup_2026_matches.csv")
    parser.add_argument("--results-url", default=RESULTS_URL)
    parser.add_argument("--rankings-url", default=RANKINGS_URL)
    parser.add_argument("--world-cup-2026-url", default=WORLD_CUP_2026_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches_path = Path(args.matches_out)
    rankings_path = Path(args.rankings_out)
    world_cup_2026_path = Path(args.world_cup_2026_out)

    matches = completed_match_results(args.results_url)
    rankings = normalized_fifa_rankings(args.rankings_url)
    world_cup_matches = world_cup_2026_matches(args.world_cup_2026_url)

    write_frame(matches, matches_path)
    write_frame(rankings, rankings_path)
    write_frame(world_cup_matches, world_cup_2026_path)

    print(f"matches: {len(matches):,} rows -> {matches_path}")
    print(f"match dates: {matches['date'].min()} to {matches['date'].max()}")
    print(f"rankings: {len(rankings):,} rows -> {rankings_path}")
    print(f"ranking dates: {rankings['rank_date'].min()} to {rankings['rank_date'].max()}")
    print(f"world cup 2026 matches: {len(world_cup_matches):,} rows -> {world_cup_2026_path}")
    print(f"world cup 2026 completed: {(world_cup_matches['status'] == 'completed').sum():,}")


if __name__ == "__main__":
    main()
