"""Machine learning.

Trains regression models on the engineered :data:`pipeline.features.MODEL_FEATURES`
against either target in ``config.ML_TARGETS`` (``fare_amount`` or
``trip_duration_min``), evaluates them, and saves/loads them for inference.

Models live in a **registry**. Each entry declares its display name, family
(GPU XGBoost vs Spark MLlib), configurable hyperparameters, and how to build the
estimator. Adding a model is one registry entry -- the Modeling page renders its
controls automatically. This is the "add models without touching existing code"
requirement.

The star model is XGBoost via ``SparkXGBRegressor`` which trains on the **GPU**
(``device="cuda"``), mirroring the modeling notebook (R²≈0.97 on fare). The four
Spark MLlib regressors are CPU baselines.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pipeline.features import MODEL_FEATURES

FEATURES_COL = "features"
PREDICTION_COL = "prediction"


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    CHOICE = "choice"


class Family(str, Enum):
    XGBOOST = "XGBoost (GPU)"
    MLLIB = "Spark MLlib (CPU)"


@dataclass(frozen=True)
class HyperParam:
    """One configurable hyperparameter for a model."""

    name: str            # estimator kwarg, e.g. "maxDepth" / "n_estimators"
    label: str           # UI label
    ptype: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple = ()


@dataclass(frozen=True)
class ModelSpec:
    """A registered model: how to name, configure, and build it."""

    key: str
    name: str
    family: Family
    # For MLlib models, the fully-qualified estimator class. Ignored for XGBoost.
    mllib_class: str = ""
    params: tuple[HyperParam, ...] = field(default_factory=tuple)
    supports_feature_importance: bool = False
    gpu_capable: bool = False
    blurb: str = ""


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
    key="xgboost",
    name="XGBoost Regressor",
    family=Family.XGBOOST,
    params=(
        HyperParam("n_estimators", "Boosting rounds", ParamType.INT, 100, 10, 1000),
        HyperParam("max_depth", "Max tree depth", ParamType.INT, 6, 2, 15),
        HyperParam("learning_rate", "Learning rate", ParamType.FLOAT, 0.3, 0.01, 1.0),
        HyperParam("subsample", "Row subsample", ParamType.FLOAT, 1.0, 0.1, 1.0),
    ),
    supports_feature_importance=True,
    gpu_capable=True,
    blurb="Gradient-boosted trees on the GPU. The notebook's best model "
          "(R²≈0.97 on fare). Handles the non-linear distance→fare relationship.",
))

register_model(ModelSpec(
    key="linear",
    name="Linear Regression",
    family=Family.MLLIB,
    mllib_class="pyspark.ml.regression.LinearRegression",
    params=(
        HyperParam("regParam", "Regularization", ParamType.FLOAT, 0.0, 0.0, 1.0),
        HyperParam("elasticNetParam", "Elastic net mix", ParamType.FLOAT, 0.0, 0.0, 1.0),
        HyperParam("maxIter", "Max iterations", ParamType.INT, 100, 1, 1000),
    ),
    blurb="A fast linear baseline. Weak here (fare is non-linear in distance) "
          "but a useful yardstick.",
))

register_model(ModelSpec(
    key="dtree",
    name="Decision Tree Regressor",
    family=Family.MLLIB,
    mllib_class="pyspark.ml.regression.DecisionTreeRegressor",
    params=(
        HyperParam("maxDepth", "Max tree depth", ParamType.INT, 5, 1, 30),
        HyperParam("maxBins", "Max bins", ParamType.INT, 32, 2, 256),
    ),
    supports_feature_importance=True,
    blurb="A single CART tree — interpretable, but prone to overfit alone.",
))

register_model(ModelSpec(
    key="rforest",
    name="Random Forest Regressor",
    family=Family.MLLIB,
    mllib_class="pyspark.ml.regression.RandomForestRegressor",
    params=(
        HyperParam("numTrees", "Number of trees", ParamType.INT, 20, 1, 200),
        HyperParam("maxDepth", "Max tree depth", ParamType.INT, 5, 1, 30),
        HyperParam("subsamplingRate", "Subsampling rate", ParamType.FLOAT, 1.0, 0.1, 1.0),
    ),
    supports_feature_importance=True,
    blurb="An ensemble of trees on CPU. Robust, but slow on Spark at this scale.",
))

register_model(ModelSpec(
    key="gbt",
    name="GBT Regressor",
    family=Family.MLLIB,
    mllib_class="pyspark.ml.regression.GBTRegressor",
    params=(
        HyperParam("maxIter", "Boosting iterations", ParamType.INT, 20, 1, 200),
        HyperParam("maxDepth", "Max tree depth", ParamType.INT, 5, 1, 30),
        HyperParam("stepSize", "Step size", ParamType.FLOAT, 0.1, 0.01, 1.0),
    ),
    supports_feature_importance=True,
    blurb="Spark's native gradient boosting (CPU). Accurate but the slowest "
          "option locally.",
))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def prepare_model_frame(df: Any, target: str) -> Any:
    """Engineer the model features and make the frame safe to assemble.

    Fills the only nullable model feature (``RatecodeID``) and drops rows with a
    null target, so both XGBoost and MLlib can consume the assembled vector.
    """
    from pyspark.sql import functions as F

    from pipeline import features as feat

    out = feat.engineer_basic_features(df).fillna({"RatecodeID": 1.0})
    return out.filter(F.col(target).isNotNull())


def _build_estimator(spec: ModelSpec, params: dict, target: str, device: str) -> Any:
    """Instantiate the estimator for ``spec`` with the given params and device."""
    if spec.family is Family.XGBOOST:
        from xgboost.spark import SparkXGBRegressor

        return SparkXGBRegressor(
            features_col=FEATURES_COL,
            label_col=target,
            prediction_col=PREDICTION_COL,
            device=device,           # "cuda" or "cpu"
            num_workers=1,           # single local node / single GPU
            missing=float("nan"),
            **params,
        )

    module_name, cls_name = spec.mllib_class.rsplit(".", 1)
    estimator_cls = getattr(importlib.import_module(module_name), cls_name)
    return estimator_cls(featuresCol=FEATURES_COL, labelCol=target,
                         predictionCol=PREDICTION_COL, **params)


def train_pipeline(spec: ModelSpec, params: dict, train_df: Any, target: str,
                   device: str) -> Any:
    """Assemble ``MODEL_FEATURES`` and fit ``spec`` -> a Spark ``PipelineModel``.

    Returning the assembler + estimator as one PipelineModel means prediction
    (and inference) is a single ``transform`` on raw engineered rows.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler(
        inputCols=list(MODEL_FEATURES), outputCol=FEATURES_COL, handleInvalid="keep",
    )
    estimator = _build_estimator(spec, params, target, device)
    return Pipeline(stages=[assembler, estimator]).fit(train_df)


def predict(model: Any, df: Any) -> Any:
    """Run a fitted PipelineModel over ``df``; adds a ``prediction`` column."""
    return model.transform(df)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def model_dir(model_id: str) -> Any:
    from config import MODELS_DIR

    return MODELS_DIR / model_id


def save_model(model: Any, meta: dict) -> str:
    """Save a PipelineModel + its metadata under ``MODELS_DIR/<model_id>/``."""
    import json

    from config import MODELS_DIR

    model_id = meta["model_id"]
    dest = MODELS_DIR / model_id
    dest.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str((dest / "pipeline").as_posix()))
    with open(dest / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return model_id


def load_model(model_id: str) -> Any:
    """Load a saved PipelineModel by id."""
    from pyspark.ml import PipelineModel

    import xgboost.spark  # noqa: F401 -- registers the XGBoost stage for loading

    return PipelineModel.load(str((model_dir(model_id) / "pipeline").as_posix()))


def load_meta(model_id: str) -> dict:
    import json

    with open(model_dir(model_id) / "meta.json", encoding="utf-8") as fh:
        return json.load(fh)


def list_saved_models() -> list[dict]:
    """Every saved model's metadata, newest first."""
    from config import MODELS_DIR

    if not MODELS_DIR.exists():
        return []
    metas = []
    for child in MODELS_DIR.iterdir():
        meta_path = child / "meta.json"
        if meta_path.exists():
            try:
                metas.append(load_meta(child.name))
            except Exception:
                continue
    return sorted(metas, key=lambda m: m.get("trained_at", ""), reverse=True)


def delete_saved_model(model_id: str) -> None:
    import shutil

    d = model_dir(model_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
