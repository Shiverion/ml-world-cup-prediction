from __future__ import annotations

import argparse
from datetime import datetime
from datetime import time
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_now(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=ZoneInfo("UTC"))
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").to_pydatetime()


def has_score(row: pd.Series) -> bool:
    return not pd.isna(row.get("team_a_score")) and not pd.isna(row.get("team_b_score"))


def row_datetime(row: pd.Series, schedule_timezone: ZoneInfo, date_only_ready_hour: int) -> datetime | None:
    for column in ["kickoff_utc", "utc_datetime", "datetime_utc"]:
        if column in row and not pd.isna(row[column]):
            parsed = pd.Timestamp(row[column])
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize("UTC")
            return parsed.tz_convert("UTC").to_pydatetime()

    date_value = row.get("date")
    if pd.isna(date_value):
        return None

    parsed_date = pd.Timestamp(date_value)
    time_value = None
    for column in ["time", "kickoff_time", "local_time"]:
        if column in row and not pd.isna(row[column]) and str(row[column]).strip():
            time_value = str(row[column]).strip()
            break

    if time_value:
        parsed = pd.Timestamp(f"{parsed_date.date()} {time_value}")
    elif parsed_date.hour or parsed_date.minute or parsed_date.second:
        parsed = parsed_date
    else:
        ready_time = time(hour=max(0, min(23, date_only_ready_hour)), minute=0)
        parsed = pd.Timestamp(datetime.combine(parsed_date.date(), ready_time))

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(schedule_timezone)
    return parsed.tz_convert("UTC").to_pydatetime()


def update_gate(
    fixtures: pd.DataFrame,
    now_utc: datetime,
    schedule_timezone: ZoneInfo,
    finish_buffer_minutes: int,
    lookback_hours: int,
    date_only_ready_hour: int,
) -> tuple[bool, str]:
    if fixtures.empty:
        return False, "fixture file is empty"

    lookback_start = now_utc - timedelta(hours=lookback_hours)
    recent_completed = 0
    recently_due = 0

    for _, row in fixtures.iterrows():
        kickoff_utc = row_datetime(row, schedule_timezone, date_only_ready_hour)
        if kickoff_utc is None:
            continue
        estimated_ready = kickoff_utc + timedelta(minutes=finish_buffer_minutes)
        status = str(row.get("status", "")).lower()

        if status == "completed" and has_score(row) and kickoff_utc >= lookback_start:
            recent_completed += 1
        elif lookback_start <= estimated_ready <= now_utc:
            recently_due += 1

    if recent_completed:
        return True, f"{recent_completed} completed fixture(s) inside the lookback window"
    if recently_due:
        return True, f"{recently_due} fixture(s) passed the estimated completion buffer"
    return False, "no fixture is newly completed or due for refresh"


def write_github_output(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether a scheduled live forecast update should run.")
    parser.add_argument("--fixtures", default=ROOT / "data" / "raw" / "world_cup_2026_matches.csv")
    parser.add_argument("--now-utc", default=None, help="Override current time, for tests/manual checks.")
    parser.add_argument("--schedule-timezone", default="UTC", help="Timezone for date/time columns without tz info.")
    parser.add_argument(
        "--finish-buffer-minutes",
        type=int,
        default=210,
        help="Wait this long after kickoff before considering a match safe to refresh.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=14,
        help="Refresh when a completed/due fixture falls within this recent window.",
    )
    parser.add_argument(
        "--date-only-ready-hour",
        type=int,
        default=23,
        help="Conservative local hour used when a fixture source has a date but no kickoff time.",
    )
    parser.add_argument("--github-output", default=None, help="Optional GitHub Actions output file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = pd.read_csv(args.fixtures)
    now_utc = parse_now(args.now_utc)
    schedule_timezone = ZoneInfo(args.schedule_timezone)
    eligible, reason = update_gate(
        fixtures,
        now_utc,
        schedule_timezone,
        args.finish_buffer_minutes,
        args.lookback_hours,
        args.date_only_ready_hour,
    )
    print(f"eligible={str(eligible).lower()}")
    print(f"reason={reason}")
    print(f"now_utc={now_utc.isoformat()}")
    write_github_output(
        args.github_output,
        {
            "eligible": str(eligible).lower(),
            "reason": reason,
            "now_utc": now_utc.isoformat(),
        },
    )


if __name__ == "__main__":
    main()
