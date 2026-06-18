import pandas as pd

from worldcup_prediction.features import add_rolling_form_features, merge_latest_rankings


def test_rolling_form_features_do_not_use_current_match():
    matches = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 3,
                "team_b_score": 0,
                "tournament": "Friendly",
                "neutral": True,
            },
            {
                "date": "2020-01-02",
                "team_a": "A",
                "team_b": "C",
                "team_a_score": 0,
                "team_b_score": 1,
                "tournament": "Friendly",
                "neutral": True,
            },
        ]
    )

    features = add_rolling_form_features(matches, windows=(5,))

    assert pd.isna(features.loc[0, "team_a_points_per_game_last_5"])
    assert features.loc[1, "team_a_points_per_game_last_5"] == 3.0


def test_ranking_merge_uses_only_rankings_before_match_date():
    matches = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-01-10"), "team_a": "A", "team_b": "B"},
        ]
    )
    rankings = pd.DataFrame(
        [
            {"rank_date": pd.Timestamp("2020-01-01"), "team": "A", "rank": 10, "points": 1500},
            {"rank_date": pd.Timestamp("2020-01-11"), "team": "A", "rank": 1, "points": 1900},
            {"rank_date": pd.Timestamp("2020-01-01"), "team": "B", "rank": 20, "points": 1300},
        ]
    )

    features = merge_latest_rankings(matches, rankings)

    assert features.loc[0, "team_a_fifa_rank"] == 10
    assert features.loc[0, "team_a_fifa_points"] == 1500
    assert features.loc[0, "fifa_rank_diff"] == 10
