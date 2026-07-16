import pandas as pd
import pytest

from worldcup_prediction.live_update_backtest import (
    LiveUpdateCandidate,
    advancement_probability,
    walk_forward_live_update_selection,
)


def test_live_update_candidate_uses_the_same_capped_weight_formula_as_live_mode():
    candidate = LiveUpdateCandidate("current", prior_strength=80.0, max_live_weight=0.35)

    assert candidate.live_weight(0) == 0.0
    assert candidate.live_weight(30) == pytest.approx(30 / 110)
    assert candidate.live_weight(200) == pytest.approx(0.35)


def test_advancement_probability_splits_a_draw_evenly_between_the_two_teams():
    assert advancement_probability({"team_a_win": 0.4, "draw": 0.2, "team_a_loss": 0.4}) == pytest.approx(0.5)


def test_walk_forward_selection_only_uses_earlier_world_cups():
    rows = []
    values = {
        2002: {"anchor_only": 0.20, "prior_20_cap_0.35": 0.40},
        2006: {"anchor_only": 0.30, "prior_20_cap_0.35": 0.20},
        2010: {"anchor_only": 0.40, "prior_20_cap_0.35": 0.10},
    }
    for year, candidates in values.items():
        for candidate, loss in candidates.items():
            rows.append(
                {
                    "candidate": candidate,
                    "year": year,
                    "candidate_type": "anchor_only" if candidate == "anchor_only" else "anchored_live_ensemble",
                    "prior_strength": None if candidate == "anchor_only" else 20.0,
                    "max_live_weight": None if candidate == "anchor_only" else 0.35,
                    "fixed_live_weight": 0.0 if candidate == "anchor_only" else None,
                    "eligible_for_selection": True,
                    "matches": 15,
                    "advance_accuracy": 0.6,
                    "advance_log_loss": loss,
                    "advance_brier_score": loss / 2,
                    "mean_live_weight": 0.0 if candidate == "anchor_only" else 0.2,
                }
            )

    selection, summary = walk_forward_live_update_selection(pd.DataFrame(rows), min_prior_world_cups=2)

    assert list(selection["holdout_year"]) == [2010]
    assert selection.iloc[0]["selected_candidate"] == "anchor_only"
    assert summary.iloc[0]["holdout_matches"] == 15
