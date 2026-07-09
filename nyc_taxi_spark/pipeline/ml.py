"""Machine learning.

Real responsibility: train Spark MLlib regression models against the engineered
features, save/load models, and generate predictions. Target is
``config.ML_TARGET_COLUMN`` (fare_amount).

Like the analysis engine, models live in a **registry**. Each entry declares
the model's display name and its configurable hyperparameters (name, type,
default, range). The Modeling page reads this to render the right controls, and
adding a fifth model later is a single registry entry -- no existing code
changes. This satisfies the "add new ML models without modifying existing
components" requirement.

Training is stubbed: it returns a placeholder handle and does no real fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import ML_TARGET_COLUMN, PLACEHOLDER_MODE
from pipeline.mock import PLACEHOLDER_LABEL


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    CHOICE = "choice"


@dataclass(frozen=True)
class HyperParam:
    """One configurable hyperparameter for a model."""

    name: str            # Spark param name, e.g. "maxDepth"
    label: str           # UI label, e.g. "Max tree depth"
    ptype: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple = ()


@dataclass(frozen=True)
class ModelSpec:
    """A registered model: how to name it, configure it, and (later) build it."""

    key: str
    name: str
    # Fully-qualified MLlib class the real implementation will instantiate.
    mllib_class: str
    params: tuple[HyperParam, ...] = field(default_factory=tuple)
    supports_feature_importance: bool = False


_MODELS: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    if spec.key in _MODELS:
        raise ValueError(f"Duplicate model key: {spec.key}")
    _MODELS[spec.key] = spec
    return spec


def get_model_spec(key: str) -> ModelSpec:
    return _MODELS[key]


def list_models() -> list[ModelSpec]:
    return list(_MODELS.values())


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

register_model(ModelSpec(
    key="linear",
    name="Linear Regression",
    mllib_class="pyspark.ml.regression.LinearRegression",
    params=(
        HyperParam("regParam", "Regularization", ParamType.FLOAT, 0.0, 0.0, 1.0),
        HyperParam("elasticNetParam", "Elastic net mix", ParamType.FLOAT, 0.0, 0.0, 1.0),
        HyperParam("maxIter", "Max iterations", ParamType.INT, 100, 1, 1000),
    ),
))

register_model(ModelSpec(
    key="dtree",
    name="Decision Tree Regressor",
    mllib_class="pyspark.ml.regression.DecisionTreeRegressor",
    params=(
        HyperParam("maxDepth", "Max tree depth", ParamType.INT, 5, 1, 30),
        HyperParam("maxBins", "Max bins", ParamType.INT, 32, 2, 256),
    ),
    supports_feature_importance=True,
))

register_model(ModelSpec(
    key="rforest",
    name="Random Forest Regressor",
    mllib_class="pyspark.ml.regression.RandomForestRegressor",
    params=(
        HyperParam("numTrees", "Number of trees", ParamType.INT, 20, 1, 200),
        HyperParam("maxDepth", "Max tree depth", ParamType.INT, 5, 1, 30),
        HyperParam("subsamplingRate", "Subsampling rate", ParamType.FLOAT, 1.0, 0.1, 1.0),
    ),
    supports_feature_importance=True,
))

register_model(ModelSpec(
    key="gbt",
    name="GBT Regressor",
    mllib_class="pyspark.ml.regression.GBTRegressor",
    params=(
        HyperParam("maxIter", "Boosting iterations", ParamType.INT, 20, 1, 200),
        HyperParam("maxDepth", "Max tree depth", ParamType.INT, 5, 1, 30),
        HyperParam("stepSize", "Step size", ParamType.FLOAT, 0.1, 0.01, 1.0),
    ),
    supports_feature_importance=True,
))


# --------------------------------------------------------------------------- #
# Training (stubbed)
# --------------------------------------------------------------------------- #

def train_model(spec: ModelSpec, params: dict, train_df: Any) -> dict:
    """Train a model. Placeholder: returns a mock handle, fits nothing.

    Returns
    -------
    dict
        ``{"model": <handle|None>, "info": {...}}``.
    """
    if PLACEHOLDER_MODE:
        return {
            "model": None,
            "info": {
                "status": PLACEHOLDER_LABEL,
                "model_key": spec.key,
                "model_name": spec.name,
                "target": ML_TARGET_COLUMN,
                "params": params,
            },
        }

    raise NotImplementedError("Real training not yet implemented.")


def predict(model: Any, test_df: Any) -> Any:
    """Generate predictions. Stubbed to return None."""
    if PLACEHOLDER_MODE:
        return None
    raise NotImplementedError("Real prediction not yet implemented.")
