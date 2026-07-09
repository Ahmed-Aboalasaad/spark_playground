"""Application service layer.

Services are the seam between the Streamlit pages and the Spark pipeline. Per
ARCHITECTURE.md they coordinate workflows, validate input, invoke pipeline
components, collect execution metrics, and return results to the UI -- and they
contain **no business-specific data transformations** themselves. All the real
work lives in ``pipeline`` modules; services just orchestrate and time.

Every service returns a :class:`~services.timing.Result`, so the UI gets the
value, the real elapsed time, and the placeholder flag uniformly.
"""
from __future__ import annotations

from typing import Any

from config import PLACEHOLDER_MODE
from pipeline import analysis as analysis_mod
from pipeline import cleaning, evaluation, features, loader, ml
from services.state import AppState
from services.timing import Result, timer


class DataService:
    """Loading, cleaning, and quality reporting workflows."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def load(self, spark: Any, data_dir: str | None = None) -> Result:
        with timer() as t:
            out = loader.load_raw_dataset(spark, data_dir)
        info = out["info"]
        self._state.raw_df = out["df"]
        self._state.load_info = {**info, "elapsed": t.elapsed}
        self._state.record_metric("Dataset load", t.elapsed, info.get("n_records"))
        return Result(value=info, elapsed=t.elapsed,
                      n_records=info.get("n_records"),
                      placeholder=PLACEHOLDER_MODE)

    def clean(self) -> Result:
        with timer() as t:
            out = cleaning.clean_dataset(self._state.raw_df)
        self._state.cleaned_df = out["df"]
        self._state.cleaning_summary = out["summary"]
        self._state.record_metric("Dataset cleaning", t.elapsed,
                                  out["summary"].get("n_output"))
        return Result(value=out["summary"], elapsed=t.elapsed,
                      placeholder=PLACEHOLDER_MODE)

    def quality_report(self) -> Result:
        with timer() as t:
            report = cleaning.data_quality_report(self._state.active_df())
        self._state.record_metric("Quality report", t.elapsed)
        return Result(value=report, elapsed=t.elapsed, placeholder=PLACEHOLDER_MODE)


class AnalysisService:
    """Runs a registered analysis against the currently selected dataset."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def run(self, key: str) -> Result:
        entry = analysis_mod.get_analysis(key)
        df = self._state.active_df()
        with timer() as t:
            frame, metrics = entry.producer(df)
        self._state.record_metric(f"Analysis: {entry.title}", t.elapsed, len(frame))
        payload = {"analysis": entry, "frame": frame, "metrics": metrics}
        return Result(value=payload, elapsed=t.elapsed, n_records=len(frame),
                      placeholder=PLACEHOLDER_MODE)


class ModelingService:
    """Feature engineering, training, and evaluation workflows."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def prepare_features(self, test_fraction: float = 0.2) -> Result:
        # Modeling always uses the cleaned dataset per the architecture.
        df = self._state.cleaned_df or self._state.raw_df
        with timer() as t:
            out = features.build_features(df, test_fraction=test_fraction)
        self._state.record_metric("Feature engineering", t.elapsed)
        return Result(value=out, elapsed=t.elapsed, placeholder=PLACEHOLDER_MODE)

    def train(self, model_key: str, params: dict, train_df: Any = None) -> Result:
        spec = ml.get_model_spec(model_key)
        with timer() as t:
            out = ml.train_model(spec, params, train_df)
        if out["model"] is not None:
            self._state.add_model(model_key, out["model"])
        self._state.record_metric(f"Train: {spec.name}", t.elapsed)
        return Result(value=out, elapsed=t.elapsed, placeholder=PLACEHOLDER_MODE)

    def evaluate(self, model_key: str, model: Any = None, predictions: Any = None) -> Result:
        spec = ml.get_model_spec(model_key)
        with timer() as t:
            report = evaluation.evaluate(model, predictions, spec)
        self._state.record_metric(f"Evaluate: {spec.name}", t.elapsed)
        return Result(value=report, elapsed=t.elapsed, placeholder=PLACEHOLDER_MODE)
