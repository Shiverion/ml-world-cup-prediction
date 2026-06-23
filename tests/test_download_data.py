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
