import pytest

from worldcup_prediction.historical_knockout import world_cup_knockout_results_from_payload


def test_historical_knockout_parser_keeps_decisive_ties_and_preserves_penalties():
    payload = {
        "matches": [
            {
                "round": "Matchday 1",
                "date": "2022-11-20",
                "team1": "Group A",
                "team2": "Group B",
                "score": {"ft": [1, 0]},
            },
            {
                "round": "Match for third place",
                "date": "2022-12-17",
                "team1": "Third A",
                "team2": "Third B",
                "score": {"ft": [2, 1]},
            },
            {
                "round": "Round of 16",
                "date": "2022-12-05",
                "team1": "Japan",
                "team2": "Croatia",
                "score": {"ft": [1, 1], "et": [1, 1], "p": [1, 3]},
            },
            {
                "round": "Quarter-finals",
                "date": "2022-12-09",
                "team1": "Netherlands",
                "team2": "Argentina",
                "score": {"ft": [2, 2], "et": [2, 3]},
            },
        ]
    }

    frame = world_cup_knockout_results_from_payload(payload, 2022)

    assert len(frame) == 2
    japan = frame.set_index("team_a").loc["Japan"]
    assert japan["round_key"] == "round_of_16"
    assert japan["team_a_score"] == 1
    assert japan["team_b_score"] == 1
    assert japan["team_a_penalties"] == 1
    assert japan["team_b_penalties"] == 3
    assert japan["winner"] == "Croatia"
    assert japan["winner_method"] == "penalties"

    netherlands = frame.set_index("team_a").loc["Netherlands"]
    assert netherlands["team_a_score"] == 2
    assert netherlands["team_b_score"] == 3
    assert netherlands["winner"] == "Argentina"
    assert netherlands["winner_method"] == "extra_time"


def test_historical_knockout_parser_rejects_unresolved_decisive_tie():
    payload = {
        "matches": [
            {
                "round": "Final",
                "date": "2022-12-18",
                "team1": "A",
                "team2": "B",
                "score": {"ft": [1, 1], "et": [1, 1]},
            }
        ]
    }

    with pytest.raises(ValueError, match="no decisive winner"):
        world_cup_knockout_results_from_payload(payload, 2022)
