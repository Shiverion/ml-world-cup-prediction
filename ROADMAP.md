# World Cup 2026 Prediction Model Roadmap

## Goal

Build a machine learning system that predicts match outcome probabilities, scoreline distributions, and tournament progression probabilities for the Men's World Cup.

The core strategy is match-level prediction:

1. Train models on historical international matches.
2. Backtest on previous World Cups using only data available before each tournament.
3. Select a calibrated model using probability-quality metrics.
4. Simulate the tournament many times.
5. Aggregate group, knockout, finalist, and champion probabilities.

## Review Notes

The roadmap is directionally strong. The biggest practical issue is data leakage, especially because the current date is June 17, 2026 and the tournament has already started. The project should define two modes:

- `pre_tournament`: data cutoff before June 11, 2026.
- `live`: include completed 2026 matches up to a configured timestamp.

The second issue is tournament bracket fidelity. The 2026 format sends 12 group winners, 12 runners-up, and 8 third-place teams into the Round of 32. The exact third-place bracket paths are configurable tournament rules, so the simulator should load bracket mappings from config rather than hard-code assumptions.

The third issue is model scope. Start with Elo, rolling form, FIFA rankings, logistic regression, and calibrated probabilities. XGBoost, Poisson score models, ensembles, and player-level data should come after the leakage-safe baseline works.

## MVP Phases

### Phase 1: Data Foundation

- Collect international results.
- Collect historical FIFA rankings.
- Create `team_mapping.csv`.
- Standardize teams, dates, tournaments, score columns, and neutral flags.
- Generate deterministic match IDs.

### Phase 2: Time-Aware Features

- Add pre-match Elo ratings.
- Merge latest available FIFA ranking before each match.
- Add rolling form before each match.
- Add tournament context and rest-day features.

### Phase 3: Baselines and ML

- FIFA ranking baseline.
- Elo logistic baseline.
- Multinomial logistic regression.
- Optional random forest and gradient boosting after the first backtest is working.

### Phase 4: Backtesting

Use rolling World Cup validation:

```text
Train: all matches before World Cup kickoff
Test: World Cup matches only
```

Primary metrics:

- Log loss
- Multiclass Brier score
- Accuracy

### Phase 5: Calibration

Calibrate the selected model before simulation. A model that is directionally accurate but overconfident will produce poor champion probabilities.

### Phase 6: Tournament Simulation

- Simulate group stage tables.
- Advance top two from each group.
- Advance eight best third-place teams.
- Simulate Round of 32 through final.
- Aggregate progression probabilities.

### Phase 7: Dashboard

Build a small Streamlit dashboard:

- Match predictor
- Backtest comparison
- Tournament probabilities
- Team explorer

## Success Criteria

- Produces match probabilities, not only winners.
- Beats simple ranking/Elo baselines on log loss or Brier score.
- Backtests previous World Cups without future data.
- Simulates 48-team World Cup logic.
- Documents model limitations clearly.
