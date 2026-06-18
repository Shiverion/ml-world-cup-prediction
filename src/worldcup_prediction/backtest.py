from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from worldcup_prediction.metrics import evaluate_probabilities
from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS, predict_probabilities


@dataclass(frozen=True)
class WorldCupWindow:
    year: int
    start: str
    end: str

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)


DEFAULT_WORLDCUP_WINDOWS = [
    WorldCupWindow(2002, "2002-05-31", "2002-06-30"),
    WorldCupWindow(2006, "2006-06-09", "2006-07-09"),
    WorldCupWindow(2010, "2010-06-11", "2010-07-11"),
    WorldCupWindow(2014, "2014-06-12", "2014-07-13"),
    WorldCupWindow(2018, "2018-06-14", "2018-07-15"),
    WorldCupWindow(2022, "2022-11-20", "2022-12-18"),
]


def split_world_cup_backtest(features: pd.DataFrame, window: WorldCupWindow) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = features[features["date"] < window.start_date].copy()
    test = features[
        (features["date"] >= window.start_date)
        & (features["date"] <= window.end_date)
        & (features["tournament"] == "FIFA World Cup")
    ].copy()
    if not test.empty and not train.empty and train["date"].max() >= test["date"].min():
        raise AssertionError("Backtest split leaked future data into training set")
    return train, test


def rolling_world_cup_backtest(
    features: pd.DataFrame,
    model_factory: Callable[[], object],
    feature_columns: Sequence[str] | None = None,
    windows: Sequence[WorldCupWindow] = DEFAULT_WORLDCUP_WINDOWS,
    target_column: str = "target",
) -> pd.DataFrame:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    rows: list[dict[str, float | int]] = []

    for window in windows:
        train, test = split_world_cup_backtest(features, window)
        if train.empty or test.empty:
            continue
        model = model_factory()
        model.fit(train[columns], train[target_column])
        probabilities = predict_probabilities(model, test, columns)
        metrics = evaluate_probabilities(test[target_column], probabilities)
        rows.append(
            {
                "year": window.year,
                "train_matches": int(len(train)),
                "test_matches": int(len(test)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
