from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from worldcup_prediction.config import OUTCOME_COLUMNS
from worldcup_prediction.metrics import as_probability_array, evaluate_probabilities
from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS


def calibrate_prefit_model(
    model,
    calibration_data: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "target",
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    calibrator = CalibratedClassifierCV(model, cv="prefit", method=method)
    calibrator.fit(calibration_data[columns], calibration_data[target_column])
    return calibrator


def _probability_bins(n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_index(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(bins, values, side="right") - 1, 0, len(bins) - 2)


def calibration_table_by_probability_bin(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.DataFrame | np.ndarray,
    n_bins: int = 10,
    outcome_columns: Sequence[str] = OUTCOME_COLUMNS,
) -> pd.DataFrame:
    probs = as_probability_array(probabilities)
    y = np.asarray(y_true, dtype=int)
    bins = _probability_bins(n_bins)
    rows: list[dict[str, float | int | str]] = []

    for outcome_index, outcome_name in enumerate(outcome_columns):
        outcome_probabilities = probs[:, outcome_index]
        observed = (y == outcome_index).astype(float)
        indices = _bin_index(outcome_probabilities, bins)
        for bin_id in range(n_bins):
            mask = indices == bin_id
            count = int(mask.sum())
            if count == 0:
                rows.append(
                    {
                        "outcome": str(outcome_name),
                        "bin": bin_id + 1,
                        "bin_lower": float(bins[bin_id]),
                        "bin_upper": float(bins[bin_id + 1]),
                        "count": 0,
                        "mean_predicted_probability": np.nan,
                        "observed_frequency": np.nan,
                        "absolute_error": np.nan,
                    }
                )
                continue
            mean_predicted = float(outcome_probabilities[mask].mean())
            observed_frequency = float(observed[mask].mean())
            rows.append(
                {
                    "outcome": str(outcome_name),
                    "bin": bin_id + 1,
                    "bin_lower": float(bins[bin_id]),
                    "bin_upper": float(bins[bin_id + 1]),
                    "count": count,
                    "mean_predicted_probability": mean_predicted,
                    "observed_frequency": observed_frequency,
                    "absolute_error": abs(mean_predicted - observed_frequency),
                }
            )

    return pd.DataFrame(rows)


def top_label_calibration_summary(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.DataFrame | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    probs = as_probability_array(probabilities)
    y = np.asarray(y_true, dtype=int)
    bins = _probability_bins(n_bins)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y).astype(float)
    indices = _bin_index(confidence, bins)

    ece = 0.0
    mce = 0.0
    non_empty_bins = 0
    for bin_id in range(n_bins):
        mask = indices == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        non_empty_bins += 1
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        error = abs(bin_accuracy - bin_confidence)
        ece += (count / len(y)) * error
        mce = max(mce, error)

    entropy = -np.sum(np.clip(probs, 1e-15, 1.0) * np.log(np.clip(probs, 1e-15, 1.0)), axis=1)
    return {
        "samples": float(len(y)),
        "bins": float(n_bins),
        "non_empty_bins": float(non_empty_bins),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "mean_confidence": float(confidence.mean()),
        "top1_accuracy": float(correct.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "mean_entropy": float(entropy.mean()),
    }


def probability_sharpness_report(probabilities: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    probs = as_probability_array(probabilities)
    confidence = probs.max(axis=1)
    entropy = -np.sum(np.clip(probs, 1e-15, 1.0) * np.log(np.clip(probs, 1e-15, 1.0)), axis=1)
    rows = [
        {"metric": "mean_max_probability", "value": float(confidence.mean())},
        {"metric": "median_max_probability", "value": float(np.median(confidence))},
        {"metric": "p90_max_probability", "value": float(np.quantile(confidence, 0.90))},
        {"metric": "mean_entropy", "value": float(entropy.mean())},
        {"metric": "median_entropy", "value": float(np.median(entropy))},
    ]
    for outcome_index, outcome_name in enumerate(OUTCOME_COLUMNS):
        rows.append(
            {
                "metric": f"mean_probability_{outcome_name}",
                "value": float(probs[:, outcome_index].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_by_group(
    predictions: pd.DataFrame,
    group_column: str,
    target_column: str = "target",
    probability_columns: Sequence[str] = OUTCOME_COLUMNS,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    required = {group_column, target_column, *probability_columns}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")

    for group_value, group_frame in predictions.groupby(group_column, sort=True):
        metrics = evaluate_probabilities(group_frame[target_column], group_frame[list(probability_columns)])
        calibration = top_label_calibration_summary(
            group_frame[target_column],
            group_frame[list(probability_columns)],
            n_bins=n_bins,
        )
        rows.append(
            {
                group_column: group_value,
                "samples": int(len(group_frame)),
                **metrics,
                "expected_calibration_error": calibration["expected_calibration_error"],
                "maximum_calibration_error": calibration["maximum_calibration_error"],
                "mean_confidence": calibration["mean_confidence"],
                "confidence_minus_accuracy": calibration["confidence_minus_accuracy"],
            }
        )
    return pd.DataFrame(rows)
