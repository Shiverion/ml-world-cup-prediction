from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from worldcup_prediction.models import DEFAULT_FEATURE_COLUMNS


def calibrate_prefit_model(
    model,
    calibration_data: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "target",
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    calibrator = CalibratedClassifierCV(model, cv="prefit", method=method)
    calibrator.fit(calibration_data[columns], calibration_data[target_column])
    return calibrator
