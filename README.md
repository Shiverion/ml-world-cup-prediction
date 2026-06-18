# World Cup 2026 Prediction Engine

Time-aware machine learning, backtesting, and Monte Carlo tournament simulation for international football.

The project predicts match-level probabilities first, then aggregates those probabilities through tournament simulation. This keeps model evaluation grounded in historical matches instead of trying to learn directly from the tiny sample of past World Cup winners.

## Current Status

This repository contains an MVP foundation:

- Data loading and cleaning helpers
- Time-aware Elo ratings
- Time-aware rolling-form features
- FIFA ranking merge without future leakage
- Baseline model factories
- World Cup rolling backtest utilities
- Match probability metrics
- 48-team tournament simulation primitives
- Focused unit tests for leakage-sensitive logic

The 2026 FIFA World Cup started on June 11, 2026. To produce a true pre-tournament forecast, configure a data cutoff before that date. If you include matches after kickoff, treat the output as a live-updating forecast instead.

## Project Structure

```text
data/
  raw/                 # downloaded source data, ignored by git
  processed/           # generated datasets, ignored by git
  external/            # team mapping and manual lookup files
configs/               # YAML configs for data, models, backtests, tournament setup
src/worldcup_prediction/
  cleaning.py          # schema checks and standardization
  elo.py               # pre-match Elo features
  features.py          # targets, ranking merge, rolling form, context features
  models.py            # sklearn model factories
  metrics.py           # log loss, Brier score, accuracy
  backtest.py          # rolling World Cup validation
  simulator.py         # group and knockout Monte Carlo simulation
app/
  streamlit_app.py     # lightweight dashboard entry point
tests/                 # unit tests
```

## Data Inputs

Expected historical match columns:

```text
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

Expected FIFA ranking columns:

```text
rank_date, team, rank, points
```

Optional team mapping file:

```csv
raw_team_name,standard_team_name
USA,United States
United States of America,United States
Korea Republic,South Korea
IR Iran,Iran
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Run The Pipeline

Download the baseline public data:

```powershell
python scripts/download_data.py
```

Then run:

```powershell
python scripts/run_analysis.py
```

For live tournament updates, refresh the public feeds and run:

```powershell
python scripts/download_data.py
python scripts/run_analysis.py --live
```

The pipeline writes cleaned data and features to `data/processed/`, rolling backtests to `outputs/backtest_results/model_backtest.csv`, and simulation outputs under `outputs/simulations/`.

Simulation outputs include:

```text
team_probabilities_2026.csv
group_position_probabilities_2026.csv
predicted_knockout_bracket_2026.csv
```

Live mode writes the same files with a `_live` suffix.

Tournament simulation currently uses an Elo-scaled independent Poisson scoreline model for future fixtures. The match-level ML backtests and training table are still produced from the configured feature set.

Current downloader sources:

- Match results: `martj42/international_results`, filtered to completed matches only.
- FIFA rankings: `Dato-Futbol/fifa-ranking`, normalized to `rank_date, team, rank, points`.

As of the last checked download, match results run through 2026-06-16 and FIFA rankings run through 2024-09-19.

## Example Pipeline

```python
from worldcup_prediction.cleaning import clean_matches, clean_rankings
from worldcup_prediction.elo import add_elo_features
from worldcup_prediction.features import build_feature_table
from worldcup_prediction.models import make_model, train_model

matches = clean_matches(raw_matches)
rankings = clean_rankings(raw_rankings)
matches = add_elo_features(matches)
features = build_feature_table(matches, rankings)

feature_cols = [
    "elo_diff",
    "fifa_rank_diff",
    "fifa_points_diff",
    "form_points_diff_5",
    "goal_diff_form_10",
    "is_neutral",
    "is_world_cup",
    "rest_days_diff",
]

model = train_model(make_model("logistic"), features, feature_cols)
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the reviewed roadmap and implementation notes.

## Key Guardrails

- Use time-based validation only.
- Do not random-split international match data.
- Do not merge FIFA rankings released after the match date.
- Do not compute rolling form from the current or future match.
- Separate pre-tournament forecasts from live forecasts with an explicit cutoff.
