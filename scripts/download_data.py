from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RANKINGS_URL = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/ranking_fifa_historical.csv"
FIFA_RANKING_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
FIFA_RANKING_API_URL = "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
WORLD_CUP_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"


def read_json_url(url: str) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://inside.fifa.com/",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def read_text_url(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


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


def latest_official_fifa_ranking_schedule(page_url: str = FIFA_RANKING_PAGE_URL) -> tuple[str, str]:
    page = read_text_url(page_url)
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', page, re.DOTALL)
    if not match:
        raise ValueError("Could not find FIFA ranking page metadata")
    payload = json.loads(html.unescape(match.group(1)))
    ranking_data = payload["props"]["pageProps"]["pageData"]["ranking"]
    latest = ranking_data["allAvailableDates"][0]
    schedule_id = str(latest["id"])
    ranking_date = str(latest.get("matchWindowEndDate") or latest["date"])
    return schedule_id, ranking_date


def _team_description(team_names: list[dict[str, Any]]) -> str:
    for item in team_names:
        if item.get("Locale") in {"en-GB", "en"} and item.get("Description"):
            return str(item["Description"])
    for item in team_names:
        if item.get("Description"):
            return str(item["Description"])
    raise ValueError("FIFA ranking row is missing a team description")


def official_fifa_rankings(
    api_url: str = FIFA_RANKING_API_URL,
    page_url: str = FIFA_RANKING_PAGE_URL,
    schedule_id: str | None = None,
    ranking_date: str | None = None,
) -> pd.DataFrame:
    if schedule_id is None or ranking_date is None:
        detected_schedule_id, detected_ranking_date = latest_official_fifa_ranking_schedule(page_url)
        schedule_id = schedule_id or detected_schedule_id
        ranking_date = ranking_date or detected_ranking_date

    query = urlencode({"rankingScheduleId": schedule_id, "language": "en-GB"})
    payload = read_json_url(f"{api_url}?{query}")
    results = payload.get("Results") or []
    if not results:
        raise ValueError(f"FIFA rankings API returned no rows for schedule: {schedule_id}")

    rows: list[dict[str, Any]] = []
    for row in results:
        rows.append(
            {
                "rank_date": ranking_date,
                "team": _team_description(row.get("TeamName") or []),
                "rank": int(row["Rank"]),
                "points": float(row["TotalPoints"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["rank", "team"]).reset_index(drop=True)


def ranking_source_frame(
    source: str,
    historical_url: str = RANKINGS_URL,
    official_api_url: str = FIFA_RANKING_API_URL,
    official_page_url: str = FIFA_RANKING_PAGE_URL,
    official_schedule_id: str | None = None,
    official_ranking_date: str | None = None,
) -> pd.DataFrame:
    if source == "historical":
        return normalized_fifa_rankings(historical_url)
    if source == "official":
        return official_fifa_rankings(
            official_api_url,
            official_page_url,
            schedule_id=official_schedule_id,
            ranking_date=official_ranking_date,
        )
    if source == "historical-plus-official":
        historical = normalized_fifa_rankings(historical_url)
        official = official_fifa_rankings(
            official_api_url,
            official_page_url,
            schedule_id=official_schedule_id,
            ranking_date=official_ranking_date,
        )
        combined = pd.concat([historical, official], ignore_index=True)
        combined["rank_date"] = pd.to_datetime(combined["rank_date"], errors="raise").dt.strftime("%Y-%m-%d")
        return combined.drop_duplicates(subset=["rank_date", "team"], keep="last").reset_index(drop=True)
    raise ValueError(f"Unsupported ranking source: {source}")


def _score_pair(score: dict[str, Any], key: str) -> list[Any]:
    values = score.get(key) or [None, None]
    return list(values) if isinstance(values, list) else [None, None]


def _score_winner(team_a: str | None, team_b: str | None, score_a: Any, score_b: Any) -> str | None:
    if score_a is None or score_b is None:
        return None
    if score_a > score_b:
        return team_a
    if score_b > score_a:
        return team_b
    return None


def world_cup_2026_matches_from_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        score = match.get("score") or {}
        full_time = _score_pair(score, "ft")
        extra_time = _score_pair(score, "et")
        penalties = _score_pair(score, "p")
        final_score = extra_time if extra_time[0] is not None and extra_time[1] is not None else full_time
        team_a = match.get("team1")
        team_b = match.get("team2")
        penalty_winner = _score_winner(team_a, team_b, penalties[0], penalties[1])
        score_winner = _score_winner(team_a, team_b, final_score[0], final_score[1])
        winner = penalty_winner or score_winner
        winner_method = (
            "penalties"
            if penalty_winner is not None
            else "extra_time"
            if extra_time[0] is not None and extra_time[1] is not None
            else "full_time"
            if score_winner is not None or (full_time[0] is not None and full_time[1] is not None)
            else ""
        )
        group = str(match.get("group", "")).replace("Group ", "").strip()
        rows.append(
            {
                "match": match.get("num"),
                "date": match.get("date"),
                "round": match.get("round", ""),
                "group": group,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_score": final_score[0],
                "team_b_score": final_score[1],
                "team_a_score_ft": full_time[0],
                "team_b_score_ft": full_time[1],
                "team_a_score_et": extra_time[0],
                "team_b_score_et": extra_time[1],
                "team_a_penalties": penalties[0],
                "team_b_penalties": penalties[1],
                "winner": winner,
                "winner_method": winner_method,
                "status": "completed" if final_score[0] is not None and final_score[1] is not None else "scheduled",
                "ground": match.get("ground", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "group", "team_a", "team_b"]).reset_index(drop=True)


def world_cup_2026_matches(url: str = WORLD_CUP_2026_URL) -> pd.DataFrame:
    with urlopen(url, timeout=30) as response:
        payload: dict[str, Any] = json.load(response)
    return world_cup_2026_matches_from_payload(payload)


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
    parser.add_argument(
        "--ranking-source",
        choices=["historical", "official", "historical-plus-official"],
        default="historical-plus-official",
        help="Use the historical GitHub feed, the latest official FIFA snapshot, or both.",
    )
    parser.add_argument("--official-rankings-api-url", default=FIFA_RANKING_API_URL)
    parser.add_argument("--official-rankings-page-url", default=FIFA_RANKING_PAGE_URL)
    parser.add_argument("--official-ranking-schedule-id", default=None)
    parser.add_argument("--official-ranking-date", default=None)
    parser.add_argument("--world-cup-2026-url", default=WORLD_CUP_2026_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches_path = Path(args.matches_out)
    rankings_path = Path(args.rankings_out)
    world_cup_2026_path = Path(args.world_cup_2026_out)

    matches = completed_match_results(args.results_url)
    rankings = ranking_source_frame(
        args.ranking_source,
        historical_url=args.rankings_url,
        official_api_url=args.official_rankings_api_url,
        official_page_url=args.official_rankings_page_url,
        official_schedule_id=args.official_ranking_schedule_id,
        official_ranking_date=args.official_ranking_date,
    )
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
