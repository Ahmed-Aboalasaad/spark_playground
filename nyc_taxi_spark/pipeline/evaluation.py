"""Model evaluation.

Computes RMSE, MAE, and R² for a fitted model's predictions, a small
prediction-sample frame (actual vs predicted vs error), and feature importance
for the models that expose it (XGBoost via gain, MLlib trees via
``featureImportances``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features import MODEL_FEATURES
from pipeline.ml import PREDICTION_COL, Family


def evaluate(model: Any, predictions: Any, target: str, spec: Any = None,
             sample_n: int = 200) -> dict:
    """Evaluate a fitted model's predictions on the test set.

    Returns metrics, a prediction-sample frame, and (when supported) a
    feature-importance table.
    """
    from pyspark.ml.evaluation import RegressionEvaluator

    ev = RegressionEvaluator(labelCol=target, predictionCol=PREDICTION_COL)
    metrics = {
        "RMSE": float(ev.evaluate(predictions, {ev.metricName: "rmse"})),
        "MAE": float(ev.evaluate(predictions, {ev.metricName: "mae"})),
        "R2": float(ev.evaluate(predictions, {ev.metricName: "r2"})),
    }

    sample = predictions.select(target, PREDICTION_COL).limit(sample_n).toPandas()
    sample.columns = ["actual", "predicted"]
    sample["error"] = sample["predicted"] - sample["actual"]

    result: dict[str, Any] = {"metrics": metrics, "prediction_sample": sample}

    if spec is not None and getattr(spec, "supports_feature_importance", False):
        fi = feature_importance(model, spec)
        if fi is not None:
            result["feature_importance"] = fi
    return result


def feature_importance(model: Any, spec: Any) -> pd.DataFrame | None:
    """Feature-importance table for the last stage of a fitted PipelineModel."""
    estimator = model.stages[-1]
    feats = list(MODEL_FEATURES)

    if spec.family is Family.XGBOOST:
        try:
            booster = estimator.get_booster()
            score = booster.get_score(importance_type="gain")  # {"f0": .., "f3": ..}
            imp = [float(score.get(f"f{i}", 0.0)) for i in range(len(feats))]
        except Exception:
            return None
    else:
        try:
            vec = estimator.featureImportances
            imp = [float(vec[i]) for i in range(len(feats))]
        except Exception:
            return None

    pdf = pd.DataFrame({"feature": feats, "importance": imp})
    return pdf.sort_values("importance", ascending=False, ignore_index=True)
