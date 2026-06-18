# World Cup 2026 Prediction Engine

Time-aware machine learning, backtesting, and Monte Carlo tournament simulation for international football.

The project predicts match-level probabilities first, then aggregates those probabilities through tournament simulation. This keeps model evaluation grounded in historical matches instead of trying to learn directly from the tiny sample of past World Cup winners.

## Current Status

This repository contains an end-to-end Streamlit app and prediction pipeline:

- Data loading and cleaning helpers
- Time-aware Elo ratings
- Time-aware rolling-form features
- FIFA ranking merge without future leakage
- Conservative model comparison and backtesting
- World Cup rolling backtest utilities
- Match probability metrics
- 48-team tournament simulation with fixed 2026 knockout bracket
- Elo-scaled Poisson scoreline simulation
- Live group-result locking for in-tournament updates
- Streamlit dashboard with probabilities, group standings, and bracket views
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
  streamlit_app.py     # dashboard entry point
scripts/
  download_data.py     # downloads public input feeds
  run_analysis.py      # builds backtests and simulation outputs
  update_live.py       # one-command live refresh
tests/                 # unit tests
```

## Streamlit Cloud Deploy

Use these settings in Streamlit Community Cloud:

```text
Repository: Shiverion/ml-world-cup-prediction
Branch: main
Main file path: app/streamlit_app.py
```

The repo commits small bootstrap forecast CSVs so the deployed dashboard can render immediately. Raw downloaded data and processed feature tables are not committed.

To refresh the deployed forecast, open the app and click:

```text
Update live data
```

That button downloads public data, rebuilds live predictions, and reloads the generated CSV outputs. The update can take a few minutes because it reruns data preparation, backtests, and simulations.

For local development:

```powershell
streamlit run app/streamlit_app.py
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

For live tournament updates from the command line:

```powershell
python scripts/update_live.py
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

## Forecast Modes

The app separates the frozen baseline forecast from the refreshed in-tournament forecast.

### Pre-Tournament

Pre-tournament mode is the before-kickoff snapshot. It is meant to answer: "What did the model expect before any 2026 match results were known?"

In this mode:

- No completed 2026 World Cup results are locked into the standings.
- Every group and knockout match is simulated.
- Team strength comes from the configured historical data cutoff, Elo ratings, rankings, and feature/model configuration.
- Outputs use the base filenames such as `team_probabilities_2026.csv`.
- The result is useful as a stable benchmark against later live updates.

This mode should use a cutoff before the tournament starts. For the 2026 World Cup, that means before June 11, 2026.

### Live

Live mode is the refreshed forecast after the tournament has started. It is meant to answer: "Given the latest completed matches we have downloaded, what happens from here?"

In this mode:

- The app downloads the public 2026 fixture/results feed.
- Completed group-stage matches are locked into simulated group tables with their actual scores.
- Unplayed group matches are simulated with the Elo-scaled Poisson score model.
- The knockout bracket is simulated from the resulting group qualifiers and fixed 2026 bracket path.
- Outputs use `_live` filenames such as `team_probabilities_2026_live.csv`.

Live mode is not automatic minute-by-minute score tracking. It changes when `python scripts/update_live.py` runs or when the Streamlit **Update live data** button completes. If the public source feed has not posted a result yet, the app cannot include that result.

The dashboard includes:

- team tournament probabilities
- group position probabilities
- FIFA-style knockout bracket chart
- rolling World Cup backtest comparison
- one-click live data refresh

Current downloader sources:

- Match results: `martj42/international_results`, filtered to completed matches only.
- FIFA rankings: `Dato-Futbol/fifa-ranking`, normalized to `rank_date, team, rank, points`.
- 2026 fixtures/results: `openfootball/worldcup.json`.

As of the last checked download, match results run through 2026-06-16 and FIFA rankings run through 2024-09-19.

## Model Notes

The current primary model is configured in `configs/model_config.yaml`:

```yaml
primary_model: logistic_plain_c0_5
primary_metric: log_loss
```

The selected model is the conservative winner on average log loss across rolling World Cup windows. Accuracy is tracked, but model selection prioritizes probability quality because tournament simulation depends on calibrated probabilities.

## Methodology

The project separates match prediction from tournament simulation:

1. Historical matches are cleaned and standardized.
2. Pre-match Elo ratings and form features are generated without future leakage.
3. Match outcome models are validated on rolling historical World Cup windows.
4. Team strength is converted into match probabilities and scoreline simulations.
5. Monte Carlo simulations aggregate match probabilities into group, knockout, finalist, and champion probabilities.

### Match Outcome Models

The model comparison currently includes:

- `logistic`: multinomial logistic regression with balanced class weights
- `logistic_plain`: multinomial logistic regression without class weighting
- `logistic_plain_c0_5`: unweighted logistic regression with stronger regularization
- `logistic_plain_c2`: unweighted logistic regression with weaker regularization
- `logistic_balanced_c0_5`: balanced logistic regression with stronger regularization
- `logistic_balanced_c2`: balanced logistic regression with weaker regularization
- `random_forest`: random forest baseline
- `hist_gradient_boosting`: histogram gradient boosting classifier
- `hist_gradient_boosting_l2_1`: histogram gradient boosting with stronger L2 regularization

The current primary model is `logistic_plain_c0_5`. It was selected because it produced the best average log loss in the rolling World Cup validation set. `logistic_plain` had nearly identical performance and slightly higher average accuracy, but the lower log loss is preferred for probability-driven tournament simulation.

### Feature Set

The configured baseline features include:

- Elo difference, absolute Elo difference, and expected Elo score
- FIFA ranking and ranking-points differences
- rolling form over recent matches
- recent goal-difference form
- rest-days difference
- neutral-site and home-advantage flags
- tournament context flags such as friendly, qualifier, World Cup group, and World Cup knockout

All time-dependent features are computed before the match being predicted.

### Validation

Validation uses rolling World Cup windows rather than random splits. For each historical World Cup from 2002 through 2022:

- Train on matches before the tournament starts.
- Test only on matches from that World Cup.
- Report log loss, Brier score, accuracy, and top-1 accuracy.

This avoids future leakage and better matches the real forecasting workflow. The downside is that each World Cup test window is small, so single-tournament accuracy can be noisy. Average log loss and Brier score are more important than one-off accuracy spikes.

Current conservative-tuning result:

```text
Best average log loss: logistic_plain_c0_5
Average accuracy:      about 56%
Best single window:    about 64%
```

These numbers are reasonable for three-way football outcome prediction. They should not be interpreted like binary classification accuracy.

### Tournament Simulation

Tournament simulation uses:

- official 48-team group configuration in `configs/tournament_2026.yaml`
- fixed 2026 knockout bracket match IDs
- top-two plus best-third-place qualification
- simulated group tables with points, goals for, goal difference, wins, and head-to-head tie logic
- fixed knockout paths after Round of 32 assignment

Future scorelines are simulated with an Elo-scaled independent Poisson model:

- stronger team gets the larger expected-goals share
- total expected goals are calibrated from historical data unless overridden
- simulated scorelines drive group points, goal difference, and goals-for tiebreakers

Live mode locks completed group-stage results from the 2026 fixture feed, then simulates only the remaining matches.

### Research Basis

The implementation follows common findings from football prediction literature:

- Elo-style team strength is a strong baseline for international football.
- Poisson score models are useful because football tournament standings depend on goals, not just win/draw/loss labels.
- Dixon-Coles style models improve on plain independent Poisson by adjusting low-score dependence and time decay.
- Probability calibration matters more than raw accuracy when predictions feed a tournament simulator.
- Time-aware validation is necessary because random splits leak future team strength into historical predictions.

Relevant references and background:

- Dixon and Coles, "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
- Zeileis/Groll-style World Cup forecasting work using rankings, ensembles, and simulated tournament paths
- Recent international-football modeling work comparing Elo/ranking systems and machine-learning classifiers
- Applied Poisson regression work for football scoreline prediction

### Implemented

- public data downloader
- data cleaning and team-name standardization
- Elo feature generation
- rolling form features
- FIFA ranking merge without future leakage
- rolling World Cup backtests
- conservative model grid
- independent Poisson scoreline simulation
- fixed 2026 knockout bracket
- live group-result locking
- Streamlit dashboard with one-click live refresh
- group-position probability output
- predicted knockout bracket output

### Limitations

- FIFA rankings source currently ends at 2024-09-19.
- The live score feed is public and may lag real match events.
- The app is not true minute-by-minute in-match modeling.
- The current Poisson model is independent Poisson, not Dixon-Coles or bivariate Poisson.
- Player injuries, squad strength, betting odds, weather, travel, and lineup data are not included.
- Third-place bracket assignment currently uses a deterministic first-valid solver over configured candidate groups.

### Next Improvements

- Add fresher FIFA rankings or a second ranking source.
- Add Dixon-Coles or bivariate Poisson score modeling.
- Add recency-weighted Elo and tune K-factors through time-aware validation.
- Add market odds or squad value features if reliable public data is available.
- Cache live updates so Streamlit Cloud refreshes faster.
- Add scheduled refresh outside the Streamlit request cycle.

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
