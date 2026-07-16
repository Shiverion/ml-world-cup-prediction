from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from worldcup_prediction.backtest import DEFAULT_WORLDCUP_WINDOWS, WorldCupWindow
from worldcup_prediction.elo import add_elo_features
from worldcup_prediction.features import build_feature_table
from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS, make_model, train_model
from worldcup_prediction.pipeline import (
    final_elo_ratings,
    latest_ranking_snapshot,
    live_model_update_weight,
    make_ml_outcome_predictor,
    recent_form_snapshot,
)


ROUND_ORDER = ("round_of_16", "quarterfinals", "semifinals", "final")
ROUND_CONTEXT = {
    "round_of_16": "round_of_16",
    "quarterfinals": "quarterfinal",
    "semifinals": "semifinal",
    "final": "final",
}
ROUND_MATCH_COUNTS = {
    "round_of_16": 8,
    "quarterfinals": 4,
    "semifinals": 2,
    "final": 1,
}


@dataclass(frozen=True)
class LiveUpdateCandidate:
    name: str
    prior_strength: float | None = None
    max_live_weight: float | None = None
    fixed_live_weight: float | None = None
    eligible_for_selection: bool = True

    def live_weight(self, completed_knockout_matches: int) -> float:
        if self.fixed_live_weight is not None:
            return float(self.fixed_live_weight)
        if self.prior_strength is None or self.max_live_weight is None:
            raise ValueError(f"Candidate {self.name} must define update parameters or a fixed weight")
        return live_model_update_weight(
            completed_knockout_matches,
            {
                "live_model_update": {
                    "enabled": True,
                    "prior_strength": self.prior_strength,
                    "max_live_weight": self.max_live_weight,
                }
            },
        )

    @property
    def candidate_type(self) -> str:
        if self.fixed_live_weight == 0.0:
            return "anchor_only"
        if self.fixed_live_weight == 1.0:
            return "live_only_diagnostic"
        return "anchored_live_ensemble"


def default_live_update_candidates() -> list[LiveUpdateCandidate]:
    candidates = [
        LiveUpdateCandidate("anchor_only", fixed_live_weight=0.0),
        LiveUpdateCandidate(
            "live_only_diagnostic",
            fixed_live_weight=1.0,
            eligible_for_selection=False,
        ),
    ]
    for prior_strength in (20.0, 40.0, 60.0, 80.0, 120.0):
        for max_live_weight in (0.15, 0.25, 0.35):
            candidates.append(
                LiveUpdateCandidate(
                    name=f"prior_{int(prior_strength)}_cap_{max_live_weight:.2f}",
                    prior_strength=prior_strength,
                    max_live_weight=max_live_weight,
                )
            )
    return candidates


def _candidate_fields(candidate: LiveUpdateCandidate) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "candidate_type": candidate.candidate_type,
        "prior_strength": candidate.prior_strength,
        "max_live_weight": candidate.max_live_weight,
        "fixed_live_weight": candidate.fixed_live_weight,
        "eligible_for_selection": candidate.eligible_for_selection,
    }


def _model_from_spec(spec: Mapping[str, Any], model_name: str, random_seed: int) -> object:
    kind = str(spec.get("kind", model_name))
    params = {key: value for key, value in spec.items() if key != "kind"}
    return make_model(kind, random_state=random_seed, **params)


def _window_mask(matches: pd.DataFrame, window: WorldCupWindow) -> pd.Series:
    dates = pd.to_datetime(matches["date"], errors="coerce")
    return (
        matches["tournament"].eq("FIFA World Cup")
        & dates.ge(window.start_date)
        & dates.le(window.end_date)
    )


def _historical_fixture_rows(knockout_results: pd.DataFrame) -> pd.DataFrame:
    rows = knockout_results.copy()
    rows["tournament"] = "FIFA World Cup"
    rows["neutral"] = True
    rows["stage"] = rows["round"]
    rows["group"] = ""
    rows["city"] = ""
    rows["country"] = ""
    return rows


def _replace_window_matches(
    matches_clean: pd.DataFrame,
    window: WorldCupWindow,
    replacement_rows: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Remove all known rows for one historical World Cup before adding only known fixtures."""
    dates = pd.to_datetime(matches_clean["date"], errors="coerce")
    known_matches = matches_clean.loc[dates.lt(cutoff)].copy()
    retained = known_matches.loc[~_window_mask(known_matches, window)].copy()
    replacement_dates = pd.to_datetime(replacement_rows["date"], errors="coerce")
    known_replacements = replacement_rows.loc[replacement_dates.lt(cutoff)].copy()
    frame = pd.concat([retained, known_replacements], ignore_index=True, sort=False)
    return frame.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)


def _group_stage_rows(
    matches_clean: pd.DataFrame,
    window: WorldCupWindow,
    anchor_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(matches_clean["date"], errors="coerce")
    rows = matches_clean.loc[_window_mask(matches_clean, window) & dates.lt(anchor_cutoff)].copy()
    if len(rows) != 48:
        raise ValueError(
            f"World Cup {window.year} should have 48 group-stage rows before {anchor_cutoff.date()}, found {len(rows)}"
        )
    rows["stage"] = "Group"
    rows["group"] = ""
    return rows


def _available_feature_columns(matches: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    return [column for column in requested if column in matches.columns]


def _fit_model_for_cutoff(
    matches: pd.DataFrame,
    rankings: pd.DataFrame | None,
    cutoff: pd.Timestamp,
    model_spec: Mapping[str, Any],
    model_name: str,
    random_seed: int,
    requested_feature_columns: Sequence[str],
    target_column: str,
) -> tuple[object, list[str]]:
    known_matches = matches[pd.to_datetime(matches["date"], errors="coerce") < cutoff].copy()
    if known_matches.empty:
        raise ValueError(f"No training data before historical cutoff {cutoff.date()}")
    features = build_feature_table(add_elo_features(known_matches), rankings)
    feature_columns = _available_feature_columns(features, requested_feature_columns)
    missing_columns = sorted(set(requested_feature_columns) - set(feature_columns))
    if missing_columns:
        raise ValueError(f"Live-update backtest features are missing columns: {missing_columns}")
    model = train_model(
        _model_from_spec(model_spec, model_name, random_seed),
        features,
        feature_columns,
        target_column=target_column,
    )
    return model, feature_columns


def _make_predictor(
    model: object,
    matches: pd.DataFrame,
    rankings: pd.DataFrame | None,
    cutoff: pd.Timestamp,
    feature_columns: Sequence[str],
    ranking_cutoff_inclusive: bool,
):
    known_matches = matches[pd.to_datetime(matches["date"], errors="coerce") < cutoff].copy()
    return make_ml_outcome_predictor(
        model,
        final_elo_ratings(known_matches),
        latest_ranking_snapshot(rankings, cutoff, inclusive=ranking_cutoff_inclusive),
        recent_form_snapshot(known_matches, cutoff),
        feature_columns,
    )


def advancement_probability(probabilities: Mapping[str, float]) -> float:
    return float(probabilities["team_a_win"] + 0.5 * probabilities["draw"])


def _blend_advance_probability(
    anchor_probabilities: Mapping[str, float],
    live_probabilities: Mapping[str, float],
    live_weight: float,
) -> float:
    anchor_weight = 1.0 - live_weight
    return float(
        anchor_weight * advancement_probability(anchor_probabilities)
        + live_weight * advancement_probability(live_probabilities)
    )


def _validate_knockout_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    year_frame = frame[frame["year"].eq(year)].copy()
    expected_counts = pd.Series(ROUND_MATCH_COUNTS)
    counts = year_frame.groupby("round_key").size().reindex(expected_counts.index, fill_value=0)
    if not counts.eq(expected_counts).all():
        raise ValueError(
            f"World Cup {year} archive does not contain the expected knockout rounds: {counts.to_dict()}"
        )
    if len(year_frame) != int(expected_counts.sum()):
        raise ValueError(f"World Cup {year} archive should contain 15 decisive knockout ties")
    return year_frame


def run_live_update_weight_backtest(
    matches_clean: pd.DataFrame,
    rankings_clean: pd.DataFrame | None,
    knockout_results: pd.DataFrame,
    model_spec: Mapping[str, Any],
    model_name: str,
    candidates: Sequence[LiveUpdateCandidate] | None = None,
    feature_columns: Sequence[str] | None = None,
    random_seed: int = 42,
    target_column: str = "target",
    ranking_cutoff_inclusive: bool = True,
    windows: Sequence[WorldCupWindow] = DEFAULT_WORLDCUP_WINDOWS,
) -> pd.DataFrame:
    """Run a round-by-round, no-future-information backtest of the live ensemble weight."""
    candidates = list(candidates or default_live_update_candidates())
    requested_feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []

    for window in windows:
        year_knockout = _validate_knockout_year(knockout_results, window.year)
        anchor_cutoff = pd.Timestamp(year_knockout["date"].min())
        group_rows = _group_stage_rows(matches_clean, window, anchor_cutoff)
        anchor_matches = _replace_window_matches(matches_clean, window, group_rows, anchor_cutoff)
        anchor_model, available_columns = _fit_model_for_cutoff(
            anchor_matches,
            rankings_clean,
            anchor_cutoff,
            model_spec,
            model_name,
            random_seed,
            requested_feature_columns,
            target_column,
        )
        anchor_predictor = _make_predictor(
            anchor_model,
            anchor_matches,
            rankings_clean,
            anchor_cutoff,
            available_columns,
            ranking_cutoff_inclusive,
        )

        for round_index, round_key in enumerate(ROUND_ORDER):
            round_fixtures = year_knockout[year_knockout["round_key"].eq(round_key)].copy()
            cutoff = pd.Timestamp(round_fixtures["date"].min())
            completed = year_knockout[
                year_knockout["round_key"].map(ROUND_ORDER.index).lt(round_index)
            ].copy()
            if completed.empty:
                # Before the first knockout tie, the rolling model and frozen anchor share the same data.
                live_predictor = anchor_predictor
            else:
                replacement_rows = pd.concat(
                    [group_rows, _historical_fixture_rows(completed)],
                    ignore_index=True,
                    sort=False,
                )
                live_matches = _replace_window_matches(matches_clean, window, replacement_rows, cutoff)
                live_model, live_columns = _fit_model_for_cutoff(
                    live_matches,
                    rankings_clean,
                    cutoff,
                    model_spec,
                    model_name,
                    random_seed,
                    requested_feature_columns,
                    target_column,
                )
                if live_columns != available_columns:
                    raise ValueError("Anchor and live models resolved different feature columns")
                live_predictor = _make_predictor(
                    live_model,
                    live_matches,
                    rankings_clean,
                    cutoff,
                    available_columns,
                    ranking_cutoff_inclusive,
                )

            completed_count = len(completed)
            for fixture in round_fixtures.itertuples(index=False):
                context = {"stage": ROUND_CONTEXT[round_key]}
                anchor_probabilities = anchor_predictor(str(fixture.team_a), str(fixture.team_b), context)
                live_probabilities = live_predictor(str(fixture.team_a), str(fixture.team_b), context)
                anchor_advance_probability = advancement_probability(anchor_probabilities)
                live_advance_probability = advancement_probability(live_probabilities)
                actual_team_a_advances = int(str(fixture.winner) == str(fixture.team_a))
                for candidate in candidates:
                    live_weight = candidate.live_weight(completed_count)
                    team_a_advance_probability = _blend_advance_probability(
                        anchor_probabilities,
                        live_probabilities,
                        live_weight,
                    )
                    predicted_winner = (
                        str(fixture.team_a)
                        if team_a_advance_probability >= 0.5
                        else str(fixture.team_b)
                    )
                    rows.append(
                        {
                            **_candidate_fields(candidate),
                            "year": window.year,
                            "round": round_key,
                            "date": pd.Timestamp(fixture.date),
                            "team_a": str(fixture.team_a),
                            "team_b": str(fixture.team_b),
                            "actual_winner": str(fixture.winner),
                            "winner_method": str(fixture.winner_method),
                            "actual_team_a_advances": actual_team_a_advances,
                            "predicted_winner": predicted_winner,
                            "correct": int(predicted_winner == str(fixture.winner)),
                            "completed_knockout_matches": completed_count,
                            "live_weight": live_weight,
                            "anchor_team_a_advance_probability": anchor_advance_probability,
                            "live_team_a_advance_probability": live_advance_probability,
                            "team_a_advance_probability": team_a_advance_probability,
                        }
                    )

    return pd.DataFrame(rows).sort_values(["candidate", "year", "date", "team_a", "team_b"]).reset_index(drop=True)


def _advance_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "matches": 0,
            "advance_accuracy": np.nan,
            "advance_log_loss": np.nan,
            "advance_brier_score": np.nan,
            "mean_live_weight": np.nan,
            "max_observed_live_weight": np.nan,
        }
    actual = frame["actual_team_a_advances"].to_numpy(dtype=float)
    probabilities = np.clip(frame["team_a_advance_probability"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    return {
        "matches": int(len(frame)),
        "advance_accuracy": float(frame["correct"].mean()),
        "advance_log_loss": float(-np.mean(actual * np.log(probabilities) + (1.0 - actual) * np.log(1.0 - probabilities))),
        "advance_brier_score": float(np.mean((probabilities - actual) ** 2)),
        "mean_live_weight": float(frame["live_weight"].mean()),
        "max_observed_live_weight": float(frame["live_weight"].max()),
    }


def summarize_live_update_backtest(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_rows: list[dict[str, Any]] = []
    candidate_year_rows: list[dict[str, Any]] = []
    for candidate, frame in predictions.groupby("candidate", sort=False):
        metadata = frame.iloc[0][
            [
                "candidate_type",
                "prior_strength",
                "max_live_weight",
                "fixed_live_weight",
                "eligible_for_selection",
            ]
        ].to_dict()
        candidate_rows.append({"candidate": candidate, **metadata, **_advance_metrics(frame)})
        for year, year_frame in frame.groupby("year", sort=True):
            candidate_year_rows.append(
                {"candidate": candidate, "year": int(year), **metadata, **_advance_metrics(year_frame)}
            )

    summary = pd.DataFrame(candidate_rows)
    summary = summary.sort_values(
        ["advance_log_loss", "advance_brier_score", "advance_accuracy", "mean_live_weight", "candidate"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    by_year = pd.DataFrame(candidate_year_rows).sort_values(["year", "candidate"]).reset_index(drop=True)
    return summary, by_year


def rank_live_update_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary[summary["eligible_for_selection"].astype(bool)].copy()
    return eligible.sort_values(
        ["advance_log_loss", "advance_brier_score", "advance_accuracy", "mean_live_weight", "candidate"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)


def _aggregate_candidate_year_metrics(by_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = ["advance_accuracy", "advance_log_loss", "advance_brier_score", "mean_live_weight"]
    for candidate, frame in by_year.groupby("candidate", sort=False):
        metadata = frame.iloc[0][
            [
                "candidate_type",
                "prior_strength",
                "max_live_weight",
                "fixed_live_weight",
                "eligible_for_selection",
            ]
        ].to_dict()
        matches = frame["matches"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate": candidate,
                **metadata,
                "matches": int(matches.sum()),
                **{
                    column: float(np.average(frame[column].to_numpy(dtype=float), weights=matches))
                    for column in metric_columns
                },
            }
        )
    return pd.DataFrame(rows)


def walk_forward_live_update_selection(
    candidate_year_metrics: pd.DataFrame,
    min_prior_world_cups: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune only on earlier World Cups, then evaluate the selected candidate on the next one."""
    rows: list[dict[str, Any]] = []
    years = sorted(int(year) for year in candidate_year_metrics["year"].unique())
    for holdout_year in years:
        prior_years = [year for year in years if year < holdout_year]
        if len(prior_years) < min_prior_world_cups:
            continue
        training_metrics = _aggregate_candidate_year_metrics(
            candidate_year_metrics[candidate_year_metrics["year"].isin(prior_years)]
        )
        selected = rank_live_update_candidates(training_metrics).iloc[0]
        holdout = candidate_year_metrics[
            (candidate_year_metrics["candidate"] == selected["candidate"])
            & (candidate_year_metrics["year"] == holdout_year)
        ].iloc[0]
        rows.append(
            {
                "holdout_year": holdout_year,
                "training_years": ",".join(str(year) for year in prior_years),
                "selected_candidate": selected["candidate"],
                "selected_prior_strength": selected["prior_strength"],
                "selected_max_live_weight": selected["max_live_weight"],
                "training_advance_log_loss": selected["advance_log_loss"],
                "training_advance_accuracy": selected["advance_accuracy"],
                "holdout_matches": int(holdout["matches"]),
                "holdout_advance_log_loss": holdout["advance_log_loss"],
                "holdout_advance_brier_score": holdout["advance_brier_score"],
                "holdout_advance_accuracy": holdout["advance_accuracy"],
            }
        )
    selection = pd.DataFrame(rows)
    if selection.empty:
        return selection, pd.DataFrame()
    weights = selection["holdout_matches"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "holdout_world_cups": int(len(selection)),
                "holdout_matches": int(weights.sum()),
                "advance_log_loss": float(np.average(selection["holdout_advance_log_loss"], weights=weights)),
                "advance_brier_score": float(np.average(selection["holdout_advance_brier_score"], weights=weights)),
                "advance_accuracy": float(np.average(selection["holdout_advance_accuracy"], weights=weights)),
            }
        ]
    )
    return selection, summary
