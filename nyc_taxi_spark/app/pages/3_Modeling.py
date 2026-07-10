"""Modeling — train, evaluate, manage, and serve Spark regression models.

Three tabs, all driven by the model registry in ``pipeline.ml`` (add a model
there and it shows up here automatically):

* **Train & Evaluate** — pick a target + model + device, tune hyperparameters,
  and run a real, cancellable, background training job with live progress. The
  star is GPU XGBoost (the notebook's R²≈0.97 on fare); the four Spark MLlib
  regressors are CPU baselines. Every finished run is saved to disk.
* **Saved Models** — compare, re-evaluate, load, or delete saved models.
* **Inference** — enter one trip's features and get a live prediction.

Modeling always trains on the **cleaned** dataset (per ARCHITECTURE.md).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from datetime import date, datetime, time as dtime

import streamlit as st

import app.modeling_ui as mui
from app.sidebar import render_sidebar
from app.ui import page_header, require_dataset, show_execution_time
from config import DEFAULT_TEST_FRACTION, ML_TARGETS
from pipeline import ml, zones
from services import gpu, training
from services.services import ModelingService
from services.state import AppState

st.set_page_config(page_title="Modeling", page_icon="🤖", layout="wide")

state = AppState()
render_sidebar(state)

page_header(
    "Modeling",
    "Train GPU-accelerated regressors, compare them, and predict a trip's fare or duration.",
    icon="🤖",
)
mui.inject_style()

if not require_dataset(state):
    st.stop()

service = ModelingService(state)


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _ensure_spark():
    """Start (once) and return the shared Spark session for load/predict paths."""
    from spark.session import get_spark

    if state.spark is None:
        state.spark = get_spark()
    return state.spark


@st.cache_resource(show_spinner="Loading saved model…")
def _load_model_cached(model_id: str):
    """Load and cache a saved PipelineModel + meta (expensive; keyed by id)."""
    _ensure_spark()
    return ModelingService(AppState()).load_saved(model_id)


@st.cache_data(show_spinner=False)
def _zone_options() -> list[int]:
    df = zones.load_zone_lookup()
    return sorted(int(i) for i in df["LocationID"].tolist())


def _zone_fmt(location_id: int) -> str:
    return f"{location_id} — {zones.zone_name(location_id)}"


# Training data readiness banner (modeling always uses the cleaned dataset).
_using_cleaned = state.cleaned_df is not None
if _using_cleaned:
    st.markdown(
        "Training on the :green-badge[Cleaned dataset] — the notebook's cleaning "
        "(fares $2.5–250, distance 0–100 mi, passengers > 0) is what lifts XGBoost "
        "from R²≈0.78 to **R²≈0.97** on fare.")
else:
    st.warning(
        "No **Cleaned** dataset yet — modeling will fall back to the raw data, which "
        "trains a much weaker model. Run cleaning on the **Data Preprocessing** page "
        "first for the best results.",
        icon=":material/cleaning_services:")

train_tab, saved_tab, infer_tab = st.tabs(
    ["🚀 Train & Evaluate", "💾 Saved Models", "🔮 Inference"])


# =========================================================================== #
# TAB 1 — Train & Evaluate
# =========================================================================== #
with train_tab:
    gpu_info = gpu.gpu_info()
    if gpu_info:
        st.markdown(
            mui.chip(f"GPU · {gpu_info['name']}", "gpu")
            + f" &nbsp; {gpu_info['memory_total_mb']:,} MB VRAM · driver "
              f"{gpu_info['driver_version']}", unsafe_allow_html=True)
    else:
        st.caption("No CUDA GPU detected on this machine — XGBoost will train on CPU "
                   "(same result, slower). MLlib models are CPU-only by design.")

    st.subheader("1 · Configure")
    cfg_left, cfg_right = st.columns([1, 1])

    with cfg_left:
        target = st.selectbox(
            "Target to predict", ML_TARGETS, format_func=mui.target_label,
            help="Fare is the notebook's strong target; duration is a second option.")
        spec = st.selectbox(
            "Model", ml.list_models(), format_func=lambda s: s.name,
            help="XGBoost (GPU) is the recommended star model.")

        fam_chip = (mui.chip("XGBoost · GPU-capable", "xgb") if spec.gpu_capable
                    else mui.chip(spec.family.value, "mllib"))
        star = mui.chip("★ Recommended", "star") if spec.key == "xgboost" else ""
        st.markdown(fam_chip + " " + star, unsafe_allow_html=True)
        st.caption(spec.blurb)

    with cfg_right:
        # Device: XGBoost can use the GPU; MLlib is always CPU.
        if spec.gpu_capable and gpu_info:
            dev_label = st.radio("Compute device", ["GPU (CUDA)", "CPU"],
                                 horizontal=True, index=0)
            device = "cuda" if dev_label.startswith("GPU") else "cpu"
        elif spec.gpu_capable:
            device = "cpu"
            st.caption("Device: **CPU** (no GPU available).")
        else:
            device = "cpu"
            st.caption("Device: **CPU** — Spark MLlib trains on the cluster CPUs.")

        test_fraction = st.slider(
            "Held-out test fraction", 0.1, 0.5, DEFAULT_TEST_FRACTION, 0.05,
            help="The most recent slice of trips, held out as the 'future' test set.")
        max_rows = st.number_input(
            "Max training rows (0 = all)", min_value=0, value=1_000_000,
            step=100_000,
            help="Caps the training sample so a large multi-month load fits in "
                 "GPU/driver memory. 0 trains on every row.")

        if device == "cuda" and max_rows:
            risk = gpu.estimate_oom_risk(int(max_rows))
            if risk and risk["risky"]:
                st.caption(f":orange[⚠ ~{risk['est_mb']:,.0f} MB estimated vs "
                           f"{risk['budget_mb']:,.0f} MB budget — consider fewer rows "
                           f"or CPU if training OOMs.]")

    # Hyperparameters, generated from the registry.
    with st.expander("Hyperparameters", icon=":material/tune:", expanded=False):
        params: dict = {}
        if spec.params:
            hp_cols = st.columns(min(3, len(spec.params)))
            for i, hp in enumerate(spec.params):
                with hp_cols[i % len(hp_cols)]:
                    if hp.ptype is ml.ParamType.INT:
                        params[hp.name] = st.number_input(
                            hp.label, value=int(hp.default),
                            min_value=int(hp.min) if hp.min is not None else None,
                            max_value=int(hp.max) if hp.max is not None else None,
                            step=1, key=f"hp_{spec.key}_{hp.name}")
                    elif hp.ptype is ml.ParamType.FLOAT:
                        params[hp.name] = st.number_input(
                            hp.label, value=float(hp.default),
                            min_value=float(hp.min) if hp.min is not None else None,
                            max_value=float(hp.max) if hp.max is not None else None,
                            key=f"hp_{spec.key}_{hp.name}")
                    elif hp.ptype is ml.ParamType.CHOICE:
                        params[hp.name] = st.selectbox(
                            hp.label, hp.choices, key=f"hp_{spec.key}_{hp.name}")
        else:
            st.caption("This model exposes no configurable hyperparameters.")

    st.subheader("2 · Train")
    active = training.active_job()
    start = st.button(
        "Train model", type="primary", icon=":material/rocket_launch:",
        disabled=active is not None,
        help="Runs in the background — progress updates live and you can cancel.")
    if active is not None:
        st.caption("A training job is already running below — let it finish or cancel it.")

    if start:
        try:
            _ensure_spark()
            job_id = service.start_training(
                model_key=spec.key, target=target, device=device,
                max_rows=int(max_rows), params=params, test_fraction=test_fraction)
            st.session_state["train_job_id"] = job_id
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surface, never crash the page
            st.error(f"Could not start training: {exc}")

    # ---- Live monitor / results ------------------------------------------ #
    job = training.get_job(st.session_state.get("train_job_id"))

    def _render_progress(j) -> None:
        st.progress(mui.stage_fraction(j.status),
                    text=f"**{j.status.title()}** — {j.stage_msg}")
        cols = st.columns([1, 1, 1])
        cols[0].metric("Elapsed", f"{j.elapsed:.0f}s")
        cols[1].metric("Model", j.model_name)
        cols[2].metric("Device", "GPU" if j.device == "cuda" else "CPU")
        if j.device == "cuda":
            vram = gpu.vram_usage()
            if vram:
                st.progress(min(vram["pct"] / 100.0, 1.0),
                            text=f"GPU VRAM {vram['used_mb']:,} / {vram['total_mb']:,} MB "
                                 f"({vram['pct']:.0f}%)")
        if st.button("Cancel training", icon=":material/cancel:", key="cancel_job"):
            training.cancel_training(j.id)
            st.rerun()

    if job is not None and job.running:
        with st.container(border=True):
            @st.fragment(run_every=1.2)
            def _monitor():
                live = training.get_job(job.id)
                if live is None:
                    return
                _render_progress(live)
                if not live.running:
                    st.rerun()  # exit the fragment; main script renders the result
            _monitor()

    elif job is not None:
        if job.status == "done" and job.result:
            # Record the run's timing once, on first render of the finished job.
            recorded = st.session_state.setdefault("_recorded_jobs", set())
            if job.id not in recorded:
                state.record_metric(f"Train: {job.model_name}", job.elapsed,
                                    job.result.get("n_train"))
                recorded.add(job.id)

            report = job.result
            st.success(f"Trained and saved **{job.model_name}** → "
                       f"`{report['model_id']}`", icon=":material/check_circle:")
            with st.container(border=True):
                st.markdown(f"**Evaluation on the held-out test set** · target "
                            f"{mui.target_label(job.target)}")
                mui.render_metric_cards(
                    report["metrics"], elapsed=job.elapsed,
                    n_train=report.get("n_train"), n_test=report.get("n_test"))
                mui.render_eval_plots(report, job.target)
                st.caption("Saved to the models directory · use the **Inference** tab "
                           "to predict with it.")
        elif job.status == "cancelled":
            st.info("Last training run was cancelled.", icon=":material/cancel:")
        elif job.status == "failed":
            st.error(job.error or "Training failed.", icon=":material/error:")
            if job.hint:
                st.warning(job.hint, icon=":material/lightbulb:")
            if job.error_detail:
                with st.expander("Error details"):
                    st.code(job.error_detail, language="text")


# =========================================================================== #
# TAB 2 — Saved Models
# =========================================================================== #
with saved_tab:
    metas = service.saved_models()
    if not metas:
        st.info("No models saved yet. Train one in the **Train & Evaluate** tab — "
                "every finished run is saved automatically.", icon=":material/inventory_2:")
    else:
        st.subheader("Comparison")
        mui.render_saved_comparison(metas)

        st.subheader("Manage")

        @st.dialog("Delete model")
        def _confirm_delete(model_id: str, name: str) -> None:
            st.write(f"Permanently delete **{name}** (`{model_id}`)? This removes the "
                     "saved pipeline and metadata from disk.")
            c1, c2 = st.columns(2)
            if c1.button("Cancel", use_container_width=True):
                st.rerun()
            if c2.button("Delete", type="primary", use_container_width=True,
                         icon=":material/delete_forever:"):
                service.delete_saved(model_id)
                _load_model_cached.clear()
                st.toast(f"Deleted {name}", icon=":material/delete:")
                st.rerun()

        for m in metas:
            mid = m["model_id"]
            met = m.get("metrics", {})
            with st.container(border=True):
                head, actions = st.columns([2.4, 1])
                with head:
                    dev_chip = mui.chip("GPU", "gpu") if m.get("device") == "cuda" else mui.chip("CPU", "cpu")
                    fam_chip = mui.chip("XGBoost", "xgb") if m.get("model_key") == "xgboost" else mui.chip("MLlib", "mllib")
                    st.markdown(f"**{m.get('model_name', mid)}** &nbsp; {fam_chip} {dev_chip}",
                                unsafe_allow_html=True)
                    st.caption(f"Target: {mui.target_label(m.get('target',''))}  ·  "
                               f"trained {m.get('trained_at','—')}  ·  "
                               f"{m.get('n_train',0):,} train / {m.get('n_test',0):,} test rows")
                    mc = st.columns(3)
                    mc[0].metric("RMSE", f"{met.get('RMSE',0):.3f}")
                    mc[1].metric("MAE", f"{met.get('MAE',0):.3f}")
                    mc[2].metric("R²", f"{met.get('R2',0):.4f}")
                with actions:
                    if st.button("Use for inference", key=f"use_{mid}",
                                 icon=":material/bolt:", use_container_width=True):
                        st.session_state["infer_model_id"] = mid
                        st.toast("Selected for inference — open the Inference tab.",
                                 icon=":material/bolt:")
                    if st.button("Re-evaluate", key=f"re_{mid}",
                                 icon=":material/analytics:", use_container_width=True):
                        st.session_state["reeval_id"] = mid
                    if st.button("Delete", key=f"del_{mid}",
                                 icon=":material/delete:", use_container_width=True):
                        _confirm_delete(mid, m.get("model_name", mid))

                if st.session_state.get("reeval_id") == mid:
                    try:
                        model, meta = _load_model_cached(mid)
                        with st.spinner("Re-evaluating on a fresh hold-out…"):
                            res = service.reevaluate(model, meta, test_fraction=DEFAULT_TEST_FRACTION)
                        st.markdown("**Fresh evaluation**")
                        mui.render_metric_cards(res.value["metrics"], n_test=res.value.get("n_test"))
                        mui.render_eval_plots(res.value, meta["target"])
                        show_execution_time(res)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Re-evaluation failed: {exc}")

        with st.expander("Parameters & metadata (raw)", icon=":material/data_object:"):
            st.json(metas)


# =========================================================================== #
# TAB 3 — Inference
# =========================================================================== #
with infer_tab:
    metas = service.saved_models()
    if not metas:
        st.info("Train and save a model first — then predict a single trip here.",
                icon=":material/model_training:")
    else:
        ids = [m["model_id"] for m in metas]
        default_id = st.session_state.get("infer_model_id")
        index = ids.index(default_id) if default_id in ids else 0
        meta_by_id = {m["model_id"]: m for m in metas}

        chosen_id = st.selectbox(
            "Model", ids, index=index,
            format_func=lambda i: f"{meta_by_id[i].get('model_name', i)} · "
                                  f"{mui.TARGET_META.get(meta_by_id[i].get('target',''), {}).get('label','')} "
                                  f"· R²={meta_by_id[i].get('metrics',{}).get('R2',0):.3f}")
        meta = meta_by_id[chosen_id]
        target = meta.get("target", "fare_amount")

        st.caption(f"Predicting {mui.target_label(target)} · "
                   f"{'GPU' if meta.get('device')=='cuda' else 'CPU'}-trained "
                   f"{meta.get('model_name','')} (R²={meta.get('metrics',{}).get('R2',0):.3f})")

        zone_ids = _zone_options()
        pu_default = zone_ids.index(161) if 161 in zone_ids else 0   # Midtown Center
        do_default = zone_ids.index(132) if 132 in zone_ids else 0   # JFK Airport

        with st.form("inference_form"):
            r1 = st.columns(3)
            trip_distance = r1[0].number_input("Trip distance (mi)", 0.0, 100.0, 3.0, 0.1)
            passenger_count = r1[1].number_input("Passengers", 1, 6, 1, 1)
            ratecode = r1[2].selectbox("Rate code", list(mui.RATECODE_LABELS),
                                       format_func=lambda c: f"{c} · {mui.RATECODE_LABELS[c]}")
            r2 = st.columns(2)
            pu = r2[0].selectbox("Pickup zone", zone_ids, index=pu_default, format_func=_zone_fmt)
            do = r2[1].selectbox("Dropoff zone", zone_ids, index=do_default, format_func=_zone_fmt)
            r3 = st.columns(2)
            pickup_day = r3[0].date_input("Pickup date", value=date(2023, 6, 15))
            pickup_time = r3[1].time_input("Pickup time", value=dtime(8, 30))
            submitted = st.form_submit_button("Predict", type="primary",
                                              icon=":material/online_prediction:")

        if submitted:
            pickup_dt = datetime.combine(pickup_day, pickup_time)
            inputs = {
                "pickup_datetime": pickup_dt,
                "passenger_count": passenger_count,
                "trip_distance": trip_distance,
                "RatecodeID": float(ratecode),
                "PULocationID": int(pu),
                "DOLocationID": int(do),
            }
            try:
                model, _ = _load_model_cached(chosen_id)
                with st.spinner("Scoring in Spark…"):
                    res = service.predict(model, inputs)

                st.markdown(
                    f'<div class="mdl-pred"><div class="lbl">Predicted '
                    f'{mui.TARGET_META.get(target,{}).get("label", target)}</div>'
                    f'<div class="val">{mui.format_target_value(res.value, target)}</div></div>',
                    unsafe_allow_html=True)

                # A transparent "what the model saw" summary of the derived flags.
                airports = {1, 132, 138}
                is_airport = int(pu in airports or do in airports)
                is_weekend = int(pickup_dt.isoweekday() in (6, 7))
                hour = pickup_dt.hour
                is_rush = int((not is_weekend) and (7 <= hour <= 9 or 16 <= hour <= 19))
                st.caption("What the model saw (derived from your inputs):")
                with st.container(border=True):
                    f = st.columns(4)
                    f[0].metric("Hour", hour)
                    f[1].metric("Weekend", "Yes" if is_weekend else "No")
                    f[2].metric("Rush hour", "Yes" if is_rush else "No")
                    f[3].metric("Airport trip", "Yes" if is_airport else "No")
                st.caption(f"{_zone_fmt(pu)} → {_zone_fmt(do)}")
                show_execution_time(res)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
