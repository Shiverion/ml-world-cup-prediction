import pytest
import numpy as np

from worldcup_prediction.metrics import evaluate_probabilities, multiclass_brier_score, ranked_probability_score


def test_multiclass_brier_score_perfect_predictions_are_zero():
    y_true = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert multiclass_brier_score(y_true, probabilities) == 0.0


def test_evaluate_probabilities_returns_expected_keys():
    metrics = evaluate_probabilities(
        np.array([0, 1, 2]),
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
    )

    assert {"log_loss", "brier_score", "ranked_probability_score", "accuracy", "top1_accuracy"} <= set(metrics)


def test_ranked_probability_score_perfect_predictions_are_zero():
    score = ranked_probability_score(
        np.array([0, 1, 2]),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    assert score == pytest.approx(0.0)
