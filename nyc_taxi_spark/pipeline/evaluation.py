"""Model evaluation.

Real responsibility: evaluate trained models and compute RMSE, MAE, and R2,
plus prediction samples and feature importance when the model supports it.

Skeleton behaviour: returns zeroed metrics, an empty prediction sample, and (if
the model advertises importance) a zeroed importance table, tagged placeholder.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from config import PLACEHOLDER_MODE
from pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from pipeline.mock import PLACEHOLDER_LABEL, zero_metrics


def evaluate(model: Any, predictions: Any, spec: Any = None) -> dict:
    """Evaluate a trained model's predictions.

    Returns
    -------
    dict
        Metrics, a prediction-sample frame, and optionally feature importance.
    """
    if PLACEHOLDER_MODE:
        result: dict[str, Any] = {
            "status": PLACEHOLDER_LABEL,
            "metrics": zero_metrics("RMSE", "MAE", "R2"),
            "prediction_sample": _empty_prediction_sample(),
        }
        if spec is not None and getattr(spec, "supports_feature_importance", False):
            result["feature_importance"] = _zero_importance()
        return result

    raise NotImplementedError("Real evaluation not yet implemented.")


def _empty_prediction_sample(n: int = 10) -> pd.DataFrame:
    """A prediction-sample frame with the right columns and zeroed values."""
    return pd.DataFrame(
        {
            "actual": [0] * n,
            "predicted": [0] * n,
            "error": [0] * n,
        }
    )


def _zero_importance() -> pd.DataFrame:
    """A feature-importance table across the known feature set, all zeros."""
    features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    return pd.DataFrame({"feature": features, "importance": [0] * len(features)})
