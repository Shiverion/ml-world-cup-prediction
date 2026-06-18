import pandas as pd

from worldcup_prediction.elo import add_elo_features, default_k_factor, expected_score


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == 0.5


def test_world_cup_qualification_uses_qualifier_k_factor():
    assert default_k_factor("FIFA World Cup qualification", "First round") == 30.0
    assert default_k_factor("FIFA World Cup qualification", None) == 30.0


def test_world_cup_semifinals_do_not_use_final_k_factor():
    assert default_k_factor("FIFA World Cup", "Semi-finals") == 60.0
    assert default_k_factor("FIFA World Cup", "Quarter-finals") == 60.0
    assert default_k_factor("FIFA World Cup", "Final") == 70.0


def test_elo_uses_pre_match_ratings_and_updates_after_match():
    matches = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 1,
                "team_b_score": 0,
                "tournament": "Friendly",
                "stage": "",
            },
            {
                "date": "2020-01-02",
                "team_a": "A",
                "team_b": "B",
                "team_a_score": 0,
                "team_b_score": 0,
                "tournament": "Friendly",
                "stage": "",
            },
        ]
    )

    result = add_elo_features(matches, initial_rating=1500)

    assert result.loc[0, "team_a_elo"] == 1500
    assert result.loc[0, "team_b_elo"] == 1500
    assert result.loc[1, "team_a_elo"] > 1500
    assert result.loc[1, "team_b_elo"] < 1500
