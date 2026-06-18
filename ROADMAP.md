# World Cup 2026 Prediction Model Roadmap

## Goal

Build a machine learning system that predicts match outcome probabilities, scoreline distributions, and tournament progression probabilities for the Men's World Cup.

The core strategy remains match-level prediction:

1. Train models on historical international matches.
2. Backtest on previous World Cups using only data available before each tournament.
3. Select a calibrated model using probability-quality metrics.
4. Simulate the tournament many times.
5. Aggregate group, knockout, finalist, and champion probabilities.

## Current Status

The project now has an end-to-end working app and pipeline.

Implemented:

- Public data downloader for historical match results, FIFA rankings, and 2026 fixture/results feed.
- Team-name normalization through `data/external/team_mapping.csv`.
- Cleaned match and ranking data generation.
- Time-aware Elo features.
- Rolling form, rest-day, neutral-site, home-advantage, and tournament-context features.
- FIFA ranking merge without future leakage.
- Rolling World Cup backtests from 2002 through 2022.
- Conservative model comparison grid.
- Primary model selection by average log loss.
- Elo-scaled independent Poisson scoreline simulation.
- Fixed 2026 Round of 32 through final bracket config.
- Best-third-place qualification simulation.
- Live mode that locks completed 2026 group-stage results.
- Team progression probability outputs.
- Group-position probability outputs.
- Predicted knockout bracket outputs.
- Streamlit dashboard with one-click live refresh.
- FIFA-style knockout bracket visualization.
- Unit tests for leakage-sensitive and simulator-critical behavior.

Current primary model:

```yaml
primary_model: logistic_plain_c0_5
primary_metric: log_loss
```

Current simulation model:

```yaml
simulation_predictor: elo_poisson
```

## Forecast Modes

### Pre-Tournament

Uses the configured cutoff before the tournament:

```yaml
data_cutoff: "2026-06-11"
```

This is the mode for a frozen forecast made before kickoff.

### Live

Uses the 2026 fixture/results feed to lock completed group-stage results, then simulates only the remaining matches.

Run locally:

```powershell
python scripts/update_live.py
```

In the Streamlit app, use:

```text
Update live data
```

Live mode is not minute-by-minute in-match modeling. It updates when the public completed-match feed updates.

## Validation Protocol

Validation uses rolling World Cup windows:

```text
Train: all matches before World Cup kickoff
Test: World Cup matches only
```

Historical test windows:

- 2002
- 2006
- 2010
- 2014
- 2018
- 2022

Primary metrics:

- Log loss
- Multiclass Brier score
- Accuracy
- Top-1 accuracy

Model selection prioritizes log loss and Brier score because tournament simulation needs useful probabilities, not just the most likely class.

## Completed Phases

### Phase 1: Data Foundation

Status: complete for baseline.

- International results are downloaded from a public CSV source.
- FIFA rankings are downloaded and normalized.
- 2026 fixtures/results are downloaded from a public JSON feed.
- Team names are standardized.
- Cleaned data and deterministic match IDs are generated.

Remaining:

- Add fresher FIFA rankings.
- Add a second source for cross-checking 2026 live results.

### Phase 2: Time-Aware Features

Status: complete for baseline.

- Pre-match Elo ratings.
- FIFA ranking merge using only rankings available before each match.
- Rolling form before each match.
- Rest-day, neutral-site, home-advantage, tournament-context features.

Remaining:

- Add recency-weighted Elo.
- Tune Elo K-factors through time-aware validation.

### Phase 3: Baselines and ML

Status: complete for conservative baseline.

Compared models:

- balanced multinomial logistic regression
- unweighted multinomial logistic regression
- lightly tuned logistic regularization variants
- random forest
- histogram gradient boosting

Current result:

- `logistic_plain_c0_5` has the best average log loss.
- `logistic_plain` has very similar performance and slightly higher average accuracy.

Remaining:

- Add calibration curves and reliability plots.
- Consider isotonic or Platt calibration only if it improves rolling-window log loss.
- Avoid broad hyperparameter search until more validation data or stronger features are available.

### Phase 4: Backtesting

Status: complete.

Outputs:

```text
outputs/backtest_results/model_backtest.csv
outputs/backtest_results/model_backtest_summary.csv
```

Remaining:

- Add charts for metric trends by World Cup.
- Add baseline comparisons against naive ranking-only and Elo-only predictors.

### Phase 5: Scoreline Modeling

Status: baseline implemented.

Implemented:

- Elo-scaled independent Poisson expected goals.
- Poisson scoreline sampling for group-stage goal difference and goals-for behavior.

Remaining:

- Implement Dixon-Coles adjustment.
- Test bivariate Poisson or correlated goal models.
- Validate scoreline distribution against historical World Cup scorelines.

### Phase 6: Tournament Simulation

Status: complete for baseline.

Implemented:

- Group-stage simulation.
- Locked completed group matches in live mode.
- Top-two plus eight best third-place qualification.
- Fixed 2026 knockout match IDs and winner paths.
- Group-position probability output.
- Team progression probability output.
- Predicted knockout bracket output.

Remaining:

- Replace deterministic first-valid third-place slot assignment with the exact official mapping table if FIFA publishes a combination table.
- Add support for completed knockout matches once the tournament reaches that stage.

### Phase 7: Dashboard

Status: complete for first deploy.

Implemented Streamlit tabs:

- Tournament probabilities.
- Group position probabilities.
- FIFA-style knockout bracket.
- Backtest comparison.

Implemented controls:

- Forecast mode selection.
- One-click live data update.
- Generated-file reload.

Remaining:

- Add match-level fixture explorer.
- Add team detail page.
- Add data freshness badges per feed.
- Cache expensive live update steps.

## Research Basis

The current implementation is based on established football prediction practices:

- Elo-style ratings are a strong baseline for international football.
- Poisson score models are useful because group standings depend on goals, goal difference, and goals for.
- Dixon-Coles style models are a known improvement over independent Poisson for low-scoring football matches.
- Probability quality matters more than raw accuracy when predictions are fed into a tournament simulator.
- Time-aware validation is required to avoid future leakage.

Research and background areas considered:

- Dixon and Coles association football score modeling.
- Zeileis/Groll-style World Cup forecasting using rankings, ensembles, and simulated tournament paths.
- International-football ranking and Elo variants.
- Applied Poisson regression for scoreline prediction.
- Machine-learning classifiers for three-way match outcomes.

## Known Limitations

- FIFA rankings source currently ends at 2024-09-19.
- Live score feed may lag real match events.
- No minute-by-minute in-match state.
- No player injuries, lineup, market odds, squad value, travel, or weather features.
- Independent Poisson ignores low-score dependence.
- Historical World Cup validation has only six 64-match test windows, so single-window accuracy is noisy.
- Streamlit live update runs inside the app request cycle and can take several minutes.

## Next Priorities

1. Add fresher FIFA rankings or a second ranking source.
2. Add exact third-place bracket mapping if available from official tournament rules.
3. Add Dixon-Coles scoreline model and compare against independent Poisson.
4. Add calibration diagnostics and reliability plots.
5. Add naive baseline comparisons for ranking-only and Elo-only predictors.
6. Cache Streamlit live update outputs and surface data freshness by feed.
7. Add support for completed knockout matches in live mode.
8. Add team-level detail pages and match fixture explorer.

## Success Criteria

- Produces calibrated match probabilities, not only winners.
- Beats simple ranking/Elo baselines on log loss or Brier score.
- Backtests previous World Cups without future data.
- Simulates 48-team World Cup group and knockout logic.
- Supports live updates without contaminating pre-tournament forecasts.
- Documents model limitations clearly.
