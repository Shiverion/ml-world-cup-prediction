from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from worldcup_prediction.config import OUTCOME_LABELS


def as_probability_array(probabilities: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = probabilities.to_numpy() if isinstance(probabilities, pd.DataFrame) else np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected probability array with shape (n_samples, 3)")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1")
    return values


def multiclass_brier_score(y_true: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> float:
    probs = as_probability_array(probabilities)
    y = np.asarray(y_true, dtype=int)
    encoded = np.zeros_like(probs)
    encoded[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - encoded) ** 2, axis=1)))


def ranked_probability_score(y_true: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> float:
    probs = as_probability_array(probabilities)
    y = np.asarray(y_true, dtype=int)
    encoded = np.zeros_like(probs)
    encoded[np.arange(len(y)), y] = 1.0
    cumulative_error = (np.cumsum(probs, axis=1) - np.cumsum(encoded, axis=1)) ** 2
    return float(np.mean(np.sum(cumulative_error[:, :-1], axis=1) / (probs.shape[1] - 1)))


def evaluate_probabilities(y_true: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> dict[str, float]:
    probs = as_probability_array(probabilities)
    y = np.asarray(y_true, dtype=int)
    preds = probs.argmax(axis=1)
    return {
        "log_loss": float(log_loss(y, probs, labels=OUTCOME_LABELS)),
        "brier_score": multiclass_brier_score(y, probs),
        "ranked_probability_score": ranked_probability_score(y, probs),
        "accuracy": float(accuracy_score(y, preds)),
        "top1_accuracy": float(accuracy_score(y, preds)),
    }
