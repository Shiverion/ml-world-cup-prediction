from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_COLUMNS = [
    "elo_diff",
    "elo_abs_diff",
    "elo_expected_a",
    "fifa_rank_diff",
    "fifa_points_diff",
    "form_points_diff_5",
    "form_points_diff_10",
    "goal_diff_form_10",
    "is_neutral",
    "team_a_home_advantage",
    "is_friendly",
    "is_qualifier",
    "is_world_cup",
    "is_world_cup_group",
    "is_world_cup_knockout",
    "rest_days_diff",
]


def make_model(kind: str = "logistic", random_state: int = 42, **kwargs: Any) -> Pipeline:
    if kind == "logistic":
        class_weight = kwargs.pop("class_weight", "balanced")
        estimator = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state, **kwargs)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if kind == "random_forest":
        class_weight = kwargs.pop("class_weight", "balanced_subsample")
        n_estimators = kwargs.pop("n_estimators", 300)
        min_samples_leaf = kwargs.pop("min_samples_leaf", 5)
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    if kind == "hist_gradient_boosting":
        max_iter = kwargs.pop("max_iter", 250)
        learning_rate = kwargs.pop("learning_rate", 0.05)
        l2_regularization = kwargs.pop("l2_regularization", 0.1)
        estimator = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            random_state=random_state,
            **kwargs,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install the optional boosting extra: pip install -e .[boosting]") from exc
        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            **kwargs,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unknown model kind: {kind}")


def train_model(
    model: Pipeline,
    features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "target",
) -> Pipeline:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    missing = sorted(set(columns + [target_column]) - set(features.columns))
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    model.fit(features[columns], features[target_column])
    return model


def predict_probabilities(
    model: Pipeline,
    features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    probabilities = model.predict_proba(features[columns])
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
    output = pd.DataFrame(0.0, index=features.index, columns=[0, 1, 2])
    for class_index, class_label in enumerate(classes):
        output[int(class_label)] = probabilities[:, class_index]
    output.columns = ["team_a_loss", "draw", "team_a_win"]
    return output
