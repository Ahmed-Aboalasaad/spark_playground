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

from pipeline import analysis as analysis_mod
from pipeline import cleaning, dataset_manager, evaluation, features, loader, ml
from services.state import AppState
from services.timing import Result, timer


class DataService:
    """Loading, cleaning, and quality reporting workflows."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def load(
        self,
        spark: Any,
        data_dir: str | None = None,
        months: list[tuple[int, int]] | None = None,
    ) -> Result:
        """Load the raw dataset. Real from here down -- loading is implemented
        regardless of ``PLACEHOLDER_MODE``, which still gates cleaning/analysis/ml."""
        with timer() as t:
            out = loader.load_raw_dataset(spark, data_dir, months)
        info = out["info"]
        self._state.raw_df = out["df"]
        self._state.load_info = {**info, "elapsed": t.elapsed}
        self._state.record_metric("Dataset load", t.elapsed, info.get("n_records"))
        return Result(value=info, elapsed=t.elapsed,
                      n_records=info.get("n_records"),
                      placeholder=False)

    def clean(self) -> Result:
        with timer() as t:
            out = cleaning.clean_dataset(self._state.raw_df)
        self._state.cleaned_df = out["df"]
        self._state.cleaning_summary = out["summary"]
        self._state.record_metric("Dataset cleaning", t.elapsed,
                                  out["summary"].get("n_output"))
        return Result(value=out["summary"], elapsed=t.elapsed,
                      n_records=out["summary"].get("n_output"), placeholder=False)

    def quality_report(self) -> Result:
        with timer() as t:
            report = cleaning.data_quality_report(self._state.active_df())
        self._state.record_metric("Quality report", t.elapsed, report.get("n_records"))
        return Result(value=report, elapsed=t.elapsed,
                      n_records=report.get("n_records"), placeholder=False)

    def cleaning_distributions(self, fraction: float = 0.01) -> Result:
        """Sampled (before, after) frames for the cleaning-effect distribution plots."""
        with timer() as t:
            before = cleaning.sample_for_dist(self._state.raw_df, fraction)
            after = (cleaning.sample_for_dist(self._state.cleaned_df, fraction)
                     if self._state.cleaned_df is not None else None)
        self._state.record_metric("Cleaning distributions", t.elapsed)
        return Result(value={"before": before, "after": after},
                      elapsed=t.elapsed, placeholder=False)


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
                      placeholder=False)


class ModelingService:
    """Feature engineering, training, and evaluation workflows."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def prepare_features(self, test_fraction: float = 0.2) -> Result:
        # Modeling always uses the cleaned dataset per the architecture.
        df = self._state.cleaned_df or self._state.raw_df
        with timer() as t:
            out = features.build_features(df, test_fraction=test_fraction)
        self._state.record_metric("Feature engineering", t.elapsed,
                                  out["info"].get("n_train"))
        return Result(value=out, elapsed=t.elapsed,
                      n_records=out["info"].get("n_train"), placeholder=False)

    def feature_signals(self, distance_fraction: float = 0.001) -> Result:
        """Aggregated frames that visually justify the engineered features.

        Runs on the cleaned dataset (falls back to raw). Each entry is a small
        pandas frame; the Spark aggregation happens inside the pipeline helpers.
        """
        from config import DEFAULT_TEST_FRACTION
        df = self._state.cleaned_df if self._state.cleaned_df is not None else self._state.raw_df
        with timer() as t:
            out = {
                "hourly": features.signal_by_hour(df),
                "weekday": features.signal_by_weekday(df),
                "airport": features.signal_airport(df),
                "distance_fare": features.distance_fare_sample(df, fraction=distance_fraction),
                "daily": features.daily_trip_series(df),
                "split_date": features._split_date(
                    features.add_derived_columns(df), DEFAULT_TEST_FRACTION),
            }
        self._state.record_metric("Feature signals", t.elapsed)
        return Result(value=out, elapsed=t.elapsed, placeholder=False)

    # -- Training (real, GPU-capable, background) ------------------------- #
    def training_frame(self) -> Any:
        """The DataFrame models train on: always the Cleaned dataset per the
        architecture, falling back to Raw if cleaning hasn't run yet."""
        return (self._state.cleaned_df
                if self._state.cleaned_df is not None else self._state.raw_df)

    def start_training(self, *, model_key: str, target: str, device: str,
                       max_rows: int, params: dict, test_fraction: float = 0.2) -> str:
        """Kick off a background training run and return its job id.

        The heavy lifting (feature engineering, fit, evaluate, save) happens on a
        daemon thread inside :mod:`services.training`; the page polls the job.
        """
        from services import training

        spark = self._state.spark
        df = self.training_frame()
        if spark is None or df is None:
            raise RuntimeError(
                "No active Spark session or dataset. Load data on the Home page "
                "first — that starts Spark and populates the raw/cleaned frames.")
        return training.start_training(
            spark, df, model_key=model_key, target=target, device=device,
            max_rows=max_rows, params=params, test_fraction=test_fraction)

    # -- Saved-model registry -------------------------------------------- #
    def saved_models(self) -> list[dict]:
        """Metadata for every model saved on disk, newest first."""
        return ml.list_saved_models()

    def load_saved(self, model_id: str) -> tuple[Any, dict]:
        """Load a saved PipelineModel and its metadata by id (expensive)."""
        return ml.load_model(model_id), ml.load_meta(model_id)

    def delete_saved(self, model_id: str) -> None:
        ml.delete_saved_model(model_id)

    def reevaluate(self, model: Any, meta: dict, test_fraction: float = 0.2) -> Result:
        """Score a loaded model on a fresh hold-out from the cleaned dataset."""
        spec = ml.get_model_spec(meta["model_key"])
        df = self.training_frame()
        if df is None:
            raise RuntimeError("Load a dataset on the Home page before re-evaluating.")
        with timer() as t:
            report = evaluation.evaluate_saved(
                model, df, meta["target"], spec, test_fraction=test_fraction)
        self._state.record_metric(f"Re-evaluate: {spec.name}", t.elapsed,
                                  report.get("n_test"))
        return Result(value=report, elapsed=t.elapsed, n_records=report.get("n_test"))

    # -- Single-row inference -------------------------------------------- #
    def predict(self, model: Any, inputs: dict) -> Result:
        """Predict the target for one user-supplied trip. Value is a float."""
        from spark.session import get_spark

        spark = self._state.spark or get_spark()
        with timer() as t:
            frame = features.inference_frame(spark, inputs)
            row = ml.predict(model, frame).select(ml.PREDICTION_COL).first()
        self._state.record_metric("Inference", t.elapsed, 1)
        return Result(value=float(row[0]), elapsed=t.elapsed, n_records=1)


class DatasetManagerService:
    """Inventory, download, and deletion of the on-disk monthly parquet files.

    Real, and independent of ``PLACEHOLDER_MODE``: this manages files on disk,
    a step upstream of (and unrelated to) the Spark pipeline stages that are
    still stubbed.
    """

    def __init__(self, state: AppState) -> None:
        self._state = state

    def inventory(self) -> dict:
        """What's on disk right now. Not timed -- a directory scan, not a
        long-running operation, so it doesn't belong in the execution-metrics log."""
        return dataset_manager.coverage_summary()

    def download(self, months: list[tuple[int, int]], on_progress=None) -> Result:
        with timer() as t:
            summary = dataset_manager.download_months(months, on_progress=on_progress)
        n_ok = len(summary["downloaded"])
        self._state.record_metric("Dataset download", t.elapsed, n_ok)
        return Result(value=summary, elapsed=t.elapsed, n_records=n_ok, placeholder=False)

    def delete(self, months: list[tuple[int, int]]) -> Result:
        with timer() as t:
            summary = dataset_manager.delete_months(months)
        self._state.record_metric("Dataset delete", t.elapsed, len(summary["deleted"]))
        return Result(value=summary, elapsed=t.elapsed, placeholder=False)

    def delete_all(self) -> Result:
        with timer() as t:
            summary = dataset_manager.delete_all()
        self._state.record_metric("Dataset delete all", t.elapsed, len(summary["deleted"]))
        return Result(value=summary, elapsed=t.elapsed, placeholder=False)
