# World Cup 2026 Prediction Engine

Time-aware machine learning, backtesting, and Monte Carlo tournament simulation for international football.

The project predicts match-level probabilities first, then aggregates those probabilities through tournament simulation. This keeps model evaluation grounded in historical matches instead of trying to learn directly from the tiny sample of past World Cup winners.

> **TL;DR:** Time-aware ML pipeline for World Cup match-outcome forecasting using Elo strength, rolling form, FIFA ranking, and leakage-aware feature generation. Rolling World Cup backtests from 2002-2022 show about 56.0% average accuracy for the primary ML model, matching strong Elo-only baselines while improving probability quality versus Elo Poisson (log loss 0.973 vs. 0.985; Brier 0.573 vs. 0.581) and substantially beating random uniform forecasting (33.6% accuracy, 1.099 log loss). The model is selected on log loss rather than raw accuracy because match probabilities feed a 48-team Monte Carlo tournament simulator with official FIFA Annex C bracket logic.

## What This Demonstrates

This repository is built to show a forecasting workflow, not only a football dashboard:

- Leakage-aware validation with rolling 2002-2022 World Cup test windows instead of random train/test splits.
- Probability-first model selection using log loss, Brier score, ranked probability score, calibration diagnostics, and sharpness reports.
- Baseline discipline against random, Elo probability, Elo Poisson, Elo-only logistic, FIFA-only logistic, and full-feature ML models.
- Forecast accountability through registry snapshots containing config, git commit, model card, match probabilities, and tournament outputs.
- Live-vs-frozen forecast separation, where completed 2026 results can be locked for live forecasts without rewriting the pre-tournament forecast story.

## Honest Limitations

- The default `ml_outcome` simulator uses the primary ML model for win/draw/loss probabilities, then samples scorelines conditionally from outcome templates. It does not yet estimate team-specific expected goals (`team_a_goals_lambda`, `team_b_goals_lambda`) in that path.
- The Elo-scaled independent Poisson simulator remains available as a baseline, but it is not a Dixon-Coles or bivariate Poisson score model.
- FIFA ranking freshness depends on the latest downloaded ranking snapshot. If ranking data is stale relative to match results, ranking features should be interpreted cautiously.
- Live forecasts are not pure pre-tournament predictions. They lock completed group-stage results and resimulate the remaining tournament from the current cutoff.

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
  metrics.py           # log loss, Brier score, RPS, accuracy
  backtest.py          # rolling World Cup validation
  calibration.py       # calibration tables, ECE/MCE, sharpness reports
  research.py          # baselines, ablations, nested backtests, forecast registry
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

Simulation runtime can be selected with profiles from `configs/tournament_2026.yaml`:

```powershell
python scripts/run_analysis.py --profile dev
python scripts/run_analysis.py --profile local
python scripts/run_analysis.py --profile publication
python scripts/run_analysis.py --live --profile dev
```

Profile intent:

| Profile | Main simulations | Interval seeds | Simulations per interval seed | Use case |
| --- | ---: | ---: | ---: | --- |
| `dev` | 3,000 | 3 | 500 | Streamlit and quick local refresh |
| `local` | 20,000 | 10 | 2,000 | normal local analysis |
| `publication` | 100,000 | 30 | 5,000 | paper-style forecast artifacts |

For live tournament updates from the command line:

```powershell
python scripts/update_live.py
```

The Streamlit live-update button and `scripts/update_live.py` should stay on the `dev` profile. Use `publication` manually from the CLI when you intentionally want a long-running, high-precision artifact.

The pipeline writes cleaned data and features to `data/processed/`, rolling backtests to `outputs/backtest_results/model_backtest.csv`, research evaluation reports to `outputs/evaluation/`, simulation outputs under `outputs/simulations/`, and a reproducible forecast snapshot under `outputs/forecast_registry/`.

Research evaluation outputs include:

```text
baseline_comparison.csv
baseline_comparison_summary.csv
ablation_results.csv
ablation_summary.csv
nested_backtest_results.csv
calibration_table_by_probability_bin.csv
calibration_summary.csv
calibration_by_world_cup.csv
probability_sharpness_report.csv
rolling_prediction_records.csv
```

Simulation outputs include:

```text
team_probabilities_2026.csv
group_position_probabilities_2026.csv
predicted_knockout_bracket_2026.csv
match_probabilities_2026.csv
team_probabilities_2026_with_ci.csv
```

Live mode writes the same files with a `_live` suffix, including `match_probabilities_2026_live.csv` and `team_probabilities_2026_live_with_ci.csv`.

Tournament simulation now uses the configured primary ML model as the default match-outcome probability engine (`simulation_predictor: ml_outcome`). Scorelines needed for group tables are sampled conditionally from the predicted win/draw/loss outcome. The Elo-scaled independent Poisson simulator remains available as a baseline via `simulation_predictor: elo_poisson`.

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
- Unplayed group matches are simulated with the configured match-outcome probability engine.
- The knockout bracket is simulated from the resulting group qualifiers and fixed 2026 bracket path.
- Outputs use `_live` filenames such as `team_probabilities_2026_live.csv`.

Live mode is not automatic minute-by-minute score tracking. It changes when `python scripts/update_live.py` runs or when the Streamlit **Update live data** button completes. If the public source feed has not posted a result yet, the app cannot include that result.

The dashboard includes:

- team tournament probabilities
- champion probability simulation intervals
- match-level group probability explorer
- group position probabilities
- FIFA-style knockout bracket chart
- baseline, calibration, ablation, and nested-selection evaluation views
- forecast registry model card and config viewer
- rolling World Cup backtest comparison
- one-click live data refresh

Current downloader sources:

- Match results: `martj42/international_results`, filtered to completed matches only.
- FIFA rankings: `Dato-Futbol/fifa-ranking` for the historical feed plus the official FIFA ranking API for the latest snapshot, normalized to `rank_date, team, rank, points`.
- 2026 fixtures/results: `openfootball/worldcup.json`.

As of the last checked download, match results run through 2026-06-22 and FIFA rankings run through the official 2026-06-11 snapshot. The tournament config keeps match training strict-before the 2026-06-11 kickoff cutoff while allowing the same-day official ranking snapshot for simulation strength features.

## Model Notes

The current primary model is configured in `configs/model_config.yaml`:

```yaml
primary_model: logistic_plain_c0_5
primary_metric: log_loss
backtest_model_candidates:
  - logistic_plain
  - logistic_plain_c0_5
  - logistic_plain_c2
  - logistic_balanced_c0_5
  - logistic_balanced_c2
```

The selected model is the conservative winner on average log loss across rolling World Cup windows. Accuracy is tracked, but model selection prioritizes probability quality because tournament simulation depends on calibrated probabilities.

## Methodology

The project separates match prediction from tournament simulation:

1. Historical matches are cleaned and standardized.
2. Pre-match Elo ratings and form features are generated without future leakage.
3. Match outcome models are validated on rolling historical World Cup windows.
4. Baselines, ablations, nested model selection, and calibration diagnostics are written as reproducible CSV reports.
5. The primary ML model converts team-strength, ranking, form, and context features into match-outcome probabilities.
6. Monte Carlo simulations aggregate match probabilities into group, knockout, finalist, and champion probabilities.
7. Forecast snapshots are versioned by cutoff, config, git commit, match probabilities, and tournament probabilities.

The same leakage discipline, calibration-first model selection, and forecast registry pattern applies to operational forecasting systems beyond football, especially when predictions feed downstream simulations or decisions.

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

Default pipeline backtests are intentionally limited to the configured logistic candidates so local and Streamlit refreshes stay practical. Heavier candidates remain in `configs/model_config.yaml` for manual experiments.

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
- Report log loss, Brier score, ranked probability score, accuracy, and top-1 accuracy.

This avoids future leakage and better matches the real forecasting workflow. The downside is that each World Cup test window is small, so single-tournament accuracy can be noisy. Average log loss and Brier score are more important than one-off accuracy spikes.

The pipeline also writes:

- `baseline_comparison.csv` for random, Elo probability, Elo Poisson, Elo-only logistic, FIFA-only logistic, and full primary-model comparisons.
- `ablation_results.csv` for feature-set contribution checks.
- `nested_backtest_results.csv` for leakage-aware model-selection estimates.
- `calibration_table_by_probability_bin.csv` and `calibration_by_world_cup.csv` for reliability diagnostics.
- `probability_sharpness_report.csv` for confidence and entropy summaries.

Current conservative-tuning result:

```text
Best average log loss: logistic_plain_c0_5
Average accuracy:      about 56%
Best single window:    about 64%
```

These numbers are reasonable for three-way football outcome prediction. They should not be interpreted like binary classification accuracy.

Baseline comparison summary from the generated rolling World Cup reports:

| Model | Avg accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: |
| Full primary ML | 56.0% | 0.973 | 0.573 |
| Elo Poisson | 56.0% | 0.985 | 0.581 |
| Elo probability | 56.0% | 0.992 | 0.588 |
| Elo-only logistic | 52.6% | 1.012 | 0.599 |
| FIFA-only logistic | 49.5% | 1.014 | 0.606 |
| Random uniform | 33.6% | 1.099 | 0.667 |

The primary ML model should therefore be read as a probability-quality improvement over strong Elo baselines, not as a headline accuracy jump.

### Tournament Simulation

Tournament simulation uses:

- official 48-team group configuration in `configs/tournament_2026.yaml`
- fixed 2026 knockout bracket match IDs
- top-two plus best-third-place qualification
- simulated group tables with points, goals for, goal difference, wins, and head-to-head tie logic
- FIFA Annex C mapping for the 495 possible best-third-place Round-of-32 assignments
- fixed knockout paths after Round-of-32 assignment

By default, match outcome probabilities are produced by the configured primary ML model:

- Elo, FIFA ranking, rolling form, and context features are built for each simulated fixture
- the trained primary model returns three-way win/draw/loss probabilities
- simulated scorelines are sampled conditionally from the predicted outcome so group points, goal difference, and goals-for tiebreakers remain available

The Elo-scaled independent Poisson simulator remains available as `simulation_predictor: elo_poisson` for scoreline-model baselines. Live mode locks completed group-stage results from the 2026 fixture feed, then simulates only the remaining matches.

Simulation uncertainty is estimated by repeating Monte Carlo runs across deterministic seeds. The default `dev` profile uses a lighter configuration for dashboard speed; use `--profile local` or `--profile publication` for higher-precision forecast artifacts.

### Forecast Registry

Each pipeline run writes a forecast registry directory such as:

```text
outputs/forecast_registry/2026-06-11_pretournament/
```

The registry contains:

```text
model_card.md
config.yaml
git_commit.txt
team_probabilities.csv
group_position_probabilities.csv
predicted_knockout_bracket.csv
match_probabilities.csv
```

`config.yaml` stores output references as project-relative paths where possible, using portable `/` separators instead of machine-specific absolute paths. Absolute local paths outside the project root are masked as `${LOCAL_PATH}/...` so registry metadata does not depend on one Windows user directory.

Because the current date is after the June 11, 2026 tournament kickoff, this registry is only a true pre-tournament forecast if it was generated before kickoff from a pre-kickoff commit and data snapshot. Later runs should be treated as reproducible cutoff-based snapshots, not original pre-kickoff predictions.

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
- official FIFA latest-ranking snapshot downloader
- data cleaning and team-name standardization
- Elo feature generation
- rolling form features
- FIFA ranking merge without future leakage
- rolling World Cup backtests
- conservative model grid
- baseline comparison reports
- ablation study reports
- calibration diagnostics and sharpness report
- nested model-selection backtest
- ML-driven tournament simulation
- independent Poisson baseline simulation
- fixed 2026 knockout bracket
- official Annex C third-place assignment table
- simulation uncertainty intervals across seeds
- forecast registry and model card
- live group-result locking
- Streamlit dashboard with one-click live refresh
- Streamlit knockout bracket zoom controls
- group-position probability output
- predicted knockout bracket output

### Limitations

- The live score feed is public and may lag real match events.
- The app is not true minute-by-minute in-match modeling.
- The default ML simulation uses a simple conditional scoreline sampler, not a fitted expected-goals, Dixon-Coles, or bivariate Poisson score model.
- Player injuries, squad strength, betting odds, weather, travel, and lineup data are not included.

### Next Improvements

- Add Dixon-Coles or bivariate Poisson score modeling.
- Add a hybrid ML outcome plus fitted expected-goals scoreline layer.
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
