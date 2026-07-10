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


def evaluate_saved(model: Any, df: Any, target: str, spec: Any = None,
                   test_fraction: float = 0.2, sample_n: int = 200) -> dict:
    """Re-evaluate an already-trained model on a fresh hold-out from ``df``.

    Rebuilds the same time-based hold-out the model was trained against
    (``prepare_model_frame`` + :func:`pipeline.features.time_split`), scores the
    loaded ``PipelineModel`` on it, and returns the same report shape as
    :func:`evaluate`. Powers the Modeling page's "re-evaluate a loaded model".
    """
    from pipeline import features as feat
    from pipeline import ml

    frame = ml.prepare_model_frame(df, target)
    _, test = feat.time_split(frame, test_fraction)
    preds = ml.predict(model, test).select(target, PREDICTION_COL)
    report = evaluate(model, preds, target, spec, sample_n=sample_n)
    report["n_test"] = test.count()
    return report


def feature_importance(model: Any, spec: Any) -> pd.DataFrame | None:
    """Feature-importance table for a fitted model.

    XGBoost (a :class:`~pipeline.ml.TrainedXGBoostModel`) exposes its own
    ``feature_importance()``; MLlib models are a Spark ``PipelineModel`` whose
    last stage carries ``featureImportances``.
    """
    feats = list(MODEL_FEATURES)

    if spec.family is Family.XGBOOST:
        score = model.feature_importance()
        if score is None:
            return None
        imp = [score.get(f"f{i}", 0.0) for i in range(len(feats))]
    else:
        try:
            vec = model.stages[-1].featureImportances
            imp = [float(vec[i]) for i in range(len(feats))]
        except Exception:
            return None

    pdf = pd.DataFrame({"feature": feats, "importance": imp})
    return pdf.sort_values("importance", ascending=False, ignore_index=True)
