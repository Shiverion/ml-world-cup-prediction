from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.live_update_gate import update_gate


def test_gate_waits_until_after_completion_buffer_for_scheduled_fixture():
    fixtures = pd.DataFrame(
        [
            {
                "date": "2026-07-09 19:00",
                "team_a": "A",
                "team_b": "B",
                "status": "scheduled",
                "team_a_score": None,
                "team_b_score": None,
            }
        ]
    )

    eligible, reason = update_gate(
        fixtures,
        datetime(2026, 7, 9, 20, 0, tzinfo=ZoneInfo("UTC")),
        ZoneInfo("UTC"),
        finish_buffer_minutes=210,
        lookback_hours=14,
        date_only_ready_hour=23,
    )

    assert eligible is False
    assert "no fixture" in reason


def test_gate_allows_recent_completed_fixture():
    fixtures = pd.DataFrame(
        [
            {
                "date": "2026-07-09 19:00",
                "team_a": "A",
                "team_b": "B",
                "status": "completed",
                "team_a_score": 2,
                "team_b_score": 1,
            }
        ]
    )

    eligible, reason = update_gate(
        fixtures,
        datetime(2026, 7, 9, 23, 0, tzinfo=ZoneInfo("UTC")),
        ZoneInfo("UTC"),
        finish_buffer_minutes=210,
        lookback_hours=14,
        date_only_ready_hour=23,
    )

    assert eligible is True
    assert "completed fixture" in reason
