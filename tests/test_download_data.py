from scripts import download_data


def test_latest_official_fifa_ranking_schedule_reads_page_metadata(monkeypatch):
    page = """
    <script id="__NEXT_DATA__" type="application/json">
    {"props":{"pageProps":{"pageData":{"ranking":{"allAvailableDates":[
      {"id":"FRS_Male_Football_20260401","date":"2026-06-11","matchWindowEndDate":"2026-06-11"}
    ]}}}}}
    </script>
    """
    monkeypatch.setattr(download_data, "read_text_url", lambda _: page)

    schedule_id, ranking_date = download_data.latest_official_fifa_ranking_schedule()

    assert schedule_id == "FRS_Male_Football_20260401"
    assert ranking_date == "2026-06-11"


def test_official_fifa_rankings_normalizes_api_payload(monkeypatch):
    payload = {
        "Results": [
            {
                "TeamName": [{"Locale": "en-GB", "Description": "Argentina"}],
                "Rank": 1,
                "TotalPoints": 1877.266571,
            },
            {
                "TeamName": [{"Locale": "en-GB", "Description": "France"}],
                "Rank": 2,
                "TotalPoints": 1870.919823,
            },
        ]
    }
    monkeypatch.setattr(download_data, "read_json_url", lambda _: payload)

    rankings = download_data.official_fifa_rankings(
        schedule_id="FRS_Male_Football_20260401",
        ranking_date="2026-06-11",
    )

    assert list(rankings.columns) == ["rank_date", "team", "rank", "points"]
    assert rankings.to_dict(orient="records") == [
        {"rank_date": "2026-06-11", "team": "Argentina", "rank": 1, "points": 1877.266571},
        {"rank_date": "2026-06-11", "team": "France", "rank": 2, "points": 1870.919823},
    ]


def test_world_cup_2026_matches_preserves_extra_time_and_penalties():
    payload = {
        "matches": [
            {
                "num": 86,
                "date": "2026-07-03",
                "round": "Round of 32",
                "team1": "Argentina",
                "team2": "Cape Verde",
                "score": {"ft": [1, 1], "et": [3, 2], "ht": [1, 0]},
                "ground": "Miami",
            },
            {
                "num": 96,
                "date": "2026-07-07",
                "round": "Round of 16",
                "team1": "Switzerland",
                "team2": "Colombia",
                "score": {"ft": [0, 0], "et": [0, 0], "p": [4, 3], "ht": [0, 0]},
                "ground": "New York",
            },
        ]
    }

    matches = download_data.world_cup_2026_matches_from_payload(payload)
    argentina = matches[matches["match"].eq(86)].iloc[0]
    switzerland = matches[matches["match"].eq(96)].iloc[0]

    assert argentina["team_a_score"] == 3
    assert argentina["team_b_score"] == 2
    assert argentina["team_a_score_ft"] == 1
    assert argentina["team_b_score_ft"] == 1
    assert argentina["winner"] == "Argentina"
    assert argentina["winner_method"] == "extra_time"
    assert switzerland["team_a_score"] == 0
    assert switzerland["team_b_score"] == 0
    assert switzerland["team_a_penalties"] == 4
    assert switzerland["team_b_penalties"] == 3
    assert switzerland["winner"] == "Switzerland"
    assert switzerland["winner_method"] == "penalties"
