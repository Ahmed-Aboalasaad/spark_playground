"""Background training manager.

Training a model is a long, blocking Spark call. To keep the Streamlit UI live —
showing stage, elapsed time, and GPU VRAM while it runs, and letting the user
cancel — each run happens on a **background thread**. The thread never touches
Streamlit (no ScriptRunContext); it only mutates a plain :class:`TrainingJob`
held in a module-level registry, which the page polls from a fragment.

Safety is a first-class concern: an out-of-memory error (easy to hit on a 4 GB
laptop GPU) is caught and surfaced as a failed job with a helpful hint — the app
never crashes, and the user can load fewer months or lower the row cap and retry.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pipeline import evaluation, ml
from pipeline import features as feat

# Job states, in order. The first four are "running".
RUNNING_STATES = ("preparing", "training", "evaluating", "saving")


@dataclass
class TrainingJob:
    """Mutable handle shared between the worker thread and the polling UI."""

    id: str
    model_key: str
    model_name: str
    target: str
    device: str            # "cuda" | "cpu"
    max_rows: int
    params: dict
    status: str = "preparing"
    stage_msg: str = "Starting…"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict | None = None       # metrics, prediction_sample, importance, ids…
    error: str | None = None          # one-line, user-facing
    error_detail: str | None = None   # full traceback for the expander
    hint: str | None = None
    _cancel_group: str = ""
    _spark: Any = None
    _cancelled: bool = False

    @property
    def running(self) -> bool:
        return self.status in RUNNING_STATES

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


_JOBS: dict[str, TrainingJob] = {}


def get_job(job_id: str | None) -> TrainingJob | None:
    return _JOBS.get(job_id) if job_id else None


def active_job() -> TrainingJob | None:
    for job in _JOBS.values():
        if job.running:
            return job
    return None


def start_training(spark: Any, df: Any, *, model_key: str, target: str, device: str,
                   max_rows: int, params: dict, test_fraction: float = 0.2) -> str:
    """Kick off a training run on a daemon thread; returns the job id."""
    spec = ml.get_model_spec(model_key)
    job = TrainingJob(
        id=uuid.uuid4().hex[:8], model_key=model_key, model_name=spec.name,
        target=target, device=device, max_rows=max_rows, params=params,
    )
    _JOBS[job.id] = job
    threading.Thread(target=_run, args=(job, spark, df, test_fraction),
                     daemon=True).start()
    return job.id


def cancel_training(job_id: str) -> None:
    """Request cancellation: flag the job and cancel its Spark job group."""
    job = _JOBS.get(job_id)
    if not job or not job.running:
        return
    job._cancelled = True
    job.stage_msg = "Cancelling…"
    if job._spark is not None and job._cancel_group:
        try:
            job._spark.sparkContext.cancelJobGroup(job._cancel_group)
        except Exception:
            pass


def _run(job: TrainingJob, spark: Any, df: Any, test_fraction: float) -> None:
    sc = spark.sparkContext
    group = f"train-{job.id}"
    job._cancel_group = group
    job._spark = spark
    train_cached = preds_cached = None
    try:
        sc.setJobGroup(group, f"Training {job.model_name}", interruptOnCancel=True)

        job.status, job.stage_msg = "preparing", "Engineering features & splitting by time…"
        frame = ml.prepare_model_frame(df, job.target)
        train, test = feat.time_split(frame, test_fraction)

        n_train_full = train.count()
        if job.max_rows and n_train_full > job.max_rows:
            train = train.sample(False, min(1.0, job.max_rows / n_train_full), seed=42)
        train_cached = train.cache()
        n_train = train_cached.count()
        if job._cancelled:
            raise _Cancelled()

        job.status, job.stage_msg = "training", (
            f"Fitting {job.model_name} on {'GPU' if job.device == 'cuda' else 'CPU'} "
            f"({n_train:,} rows)…")
        spec = ml.get_model_spec(job.model_key)
        model = ml.train_pipeline(spec, job.params, train_cached, job.target, job.device)
        if job._cancelled:
            raise _Cancelled()

        job.status, job.stage_msg = "evaluating", "Scoring on the held-out test set…"
        preds_cached = model.transform(test).select(job.target, ml.PREDICTION_COL).cache()
        n_test = preds_cached.count()
        report = evaluation.evaluate(model, preds_cached, job.target, spec)

        job.status, job.stage_msg = "saving", "Saving trained model…"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_id = f"{job.model_key}_{job.target}_{stamp}"
        meta = {
            "model_id": model_id,
            "model_key": job.model_key,
            "model_name": job.model_name,
            "target": job.target,
            "device": job.device,
            "params": job.params,
            "features": list(feat.MODEL_FEATURES),
            "metrics": report["metrics"],
            "n_train": n_train,
            "n_test": n_test,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(job.elapsed, 1),
        }
        ml.save_model(model, meta)

        job.result = {**report, "n_train": n_train, "n_test": n_test,
                      "model_id": model_id, "meta": meta}
        job.status, job.stage_msg = "done", "Completed."

    except _Cancelled:
        job.status, job.stage_msg = "cancelled", "Training cancelled."
    except Exception as exc:  # noqa: BLE001 - deliberately catch-all: never crash
        if job._cancelled:
            job.status, job.stage_msg = "cancelled", "Training cancelled."
        else:
            job.status, job.stage_msg = "failed", "Training failed."
            job.error, job.hint = _describe_error(exc)
            job.error_detail = traceback.format_exc()
    finally:
        job.finished_at = time.time()
        for cached in (preds_cached, train_cached):
            try:
                if cached is not None:
                    cached.unpersist()
            except Exception:
                pass
        try:
            sc.clearJobGroup()
        except Exception:
            pass


class _Cancelled(Exception):
    """Internal signal that the user cancelled the run."""


def _describe_error(exc: Exception) -> tuple[str, str]:
    """Turn an exception into (one-line message, actionable hint)."""
    text = str(exc).lower()
    oom_markers = ("out of memory", "outofmemory", "bad_alloc", "cudaerror",
                   "cuda error", "memory_status", "std::bad_alloc", "oom")
    if any(m in text for m in oom_markers):
        return (
            "Out of memory during training.",
            "The GPU/driver ran out of memory. Lower **Max training rows**, switch "
            "to **CPU**, or load fewer months on the Home page, then try again.",
        )
    if "cuda" in text or "gpu" in text or "device" in text and "cuda" in text:
        return (
            "GPU training error.",
            "CUDA wasn't usable for this run. Switch to **CPU** and retry — the "
            "result will be identical, just slower.",
        )
    return ("Training failed.", "See the details below. You can adjust settings and retry.")
