import numpy as np
import pandas as pd
import pytest

from worldcup_prediction.calibration import (
    calibration_table_by_probability_bin,
    top_label_calibration_summary,
)


def test_calibration_table_reports_outcome_bins():
    table = calibration_table_by_probability_bin(
        np.array([0, 1, 2, 2]),
        pd.DataFrame(
            {
                "team_a_loss": [0.8, 0.2, 0.1, 0.1],
                "draw": [0.1, 0.6, 0.2, 0.2],
                "team_a_win": [0.1, 0.2, 0.7, 0.7],
            }
        ),
        n_bins=2,
    )

    assert set(table["outcome"]) == {"team_a_loss", "draw", "team_a_win"}
    assert len(table) == 6
    assert table["count"].sum() == 12


def test_top_label_calibration_summary_is_zero_for_perfect_confident_predictions():
    summary = top_label_calibration_summary(
        np.array([0, 1, 2]),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        n_bins=5,
    )

    assert summary["expected_calibration_error"] == pytest.approx(0.0)
    assert summary["maximum_calibration_error"] == pytest.approx(0.0)
