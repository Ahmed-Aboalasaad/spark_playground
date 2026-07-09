"""Feature engineering.

Real responsibility: select ML features, encode categorical variables, assemble
feature vectors, and split into training and testing datasets. Output is the
train/test pair the modeling stage consumes. The regression target is
``config.ML_TARGET_COLUMN`` (fare_amount).

Skeleton behaviour: returns ``None`` datasets and a placeholder description of
the feature set so the Modeling page can render its configuration UI.
"""
from __future__ import annotations

from typing import Any

from config import ML_TARGET_COLUMN, PLACEHOLDER_MODE
from pipeline.mock import PLACEHOLDER_LABEL

# The feature set the real implementation is expected to assemble. Listed here
# so the Modeling page can show "features used" before training exists.
NUMERIC_FEATURES = ("trip_distance", "passenger_count", "pickup_hour", "pickup_weekday")
CATEGORICAL_FEATURES = ("PULocationID", "DOLocationID", "RatecodeID")


def build_features(df: Any, test_fraction: float = 0.2, seed: int = 42) -> dict:
    """Assemble features and split into train/test sets.

    Returns
    -------
    dict
        ``{"train": <df|None>, "test": <df|None>, "info": {...}}``.
    """
    if PLACEHOLDER_MODE:
        return {
            "train": None,
            "test": None,
            "info": {
                "status": PLACEHOLDER_LABEL,
                "target": ML_TARGET_COLUMN,
                "numeric_features": list(NUMERIC_FEATURES),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "test_fraction": test_fraction,
                "n_train": 0,
                "n_test": 0,
            },
        }

    raise NotImplementedError("Real feature engineering not yet implemented.")
