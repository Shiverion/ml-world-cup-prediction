import pandas as pd

from worldcup_prediction.backtest import WorldCupWindow, split_world_cup_backtest


def test_backtest_split_uses_only_pre_tournament_training_data():
    features = pd.DataFrame(
        [
            {"date": pd.Timestamp("2018-06-01"), "tournament": "Friendly"},
            {"date": pd.Timestamp("2018-06-15"), "tournament": "FIFA World Cup"},
            {"date": pd.Timestamp("2018-07-16"), "tournament": "Friendly"},
        ]
    )
    window = WorldCupWindow(2018, "2018-06-14", "2018-07-15")

    train, test = split_world_cup_backtest(features, window)

    assert list(train["date"]) == [pd.Timestamp("2018-06-01")]
    assert list(test["date"]) == [pd.Timestamp("2018-06-15")]
