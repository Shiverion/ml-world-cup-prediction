-- DuckDB source script for the live-update weight backtest report.
-- Each statement exposes a reviewed report dataset from the reproducible CSV outputs.

SELECT *
FROM read_csv_auto('outputs/evaluation/live_update_weight_backtest_candidate_summary.csv');

SELECT *
FROM read_csv_auto('outputs/evaluation/live_update_weight_backtest_by_year.csv');

SELECT *
FROM read_csv_auto('outputs/evaluation/live_update_weight_backtest_walk_forward.csv');

SELECT *
FROM read_csv_auto('outputs/evaluation/live_update_2026_weight_shadow.csv');
