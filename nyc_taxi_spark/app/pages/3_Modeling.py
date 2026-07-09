"""Modeling — configure, train, and evaluate Spark MLlib regressors.

Models and their hyperparameters come from the registry in ``pipeline.ml``, so
the configuration controls are generated from each model's declared params.
Target is fare_amount. Training and evaluation are placeholder-backed.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import streamlit as st

from app.sidebar import render_sidebar
from app.ui import (
    placeholder_badge,
    placeholder_banner,
    render_metrics,
    require_dataset,
    show_execution_time,
)
from config import ML_TARGET_COLUMN, PLACEHOLDER_MODE
from pipeline import ml
from pipeline.ml import ParamType
from services.services import ModelingService
from services.state import AppState

st.set_page_config(page_title="Modeling", page_icon="🤖", layout="wide")

state = AppState()
render_sidebar(state)

st.title("🤖 Modeling")
if PLACEHOLDER_MODE:
    placeholder_banner()

if not require_dataset(state):
    st.stop()

st.caption(f"Regression target: **{ML_TARGET_COLUMN}**. Models train on the "
           f"cleaned dataset.")

service = ModelingService(state)

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
st.header("1 · Features")
test_fraction = st.slider("Test set fraction", 0.1, 0.5, 0.2, 0.05)
if st.button("Prepare features"):
    result = service.prepare_features(test_fraction)
    info = result.value["info"]
    placeholder_badge(result)
    c1, c2 = st.columns(2)
    c1.metric("Train records", f"{info.get('n_train', 0):,}")
    c2.metric("Test records", f"{info.get('n_test', 0):,}")
    st.caption("Numeric: " + ", ".join(info.get("numeric_features", [])))
    st.caption("Categorical: " + ", ".join(info.get("categorical_features", [])))
    show_execution_time(result)

st.divider()

# --------------------------------------------------------------------------- #
# Model selection & hyperparameters
# --------------------------------------------------------------------------- #
st.header("2 · Model")
specs = ml.list_models()
spec = st.selectbox("Model", specs, format_func=lambda s: s.name)

st.subheader("Hyperparameters")
params: dict = {}
if spec.params:
    cols = st.columns(min(3, len(spec.params)))
    for i, hp in enumerate(spec.params):
        col = cols[i % len(cols)]
        with col:
            if hp.ptype is ParamType.INT:
                params[hp.name] = st.number_input(
                    hp.label, value=int(hp.default),
                    min_value=int(hp.min) if hp.min is not None else None,
                    max_value=int(hp.max) if hp.max is not None else None,
                    step=1,
                )
            elif hp.ptype is ParamType.FLOAT:
                params[hp.name] = st.number_input(
                    hp.label, value=float(hp.default),
                    min_value=float(hp.min) if hp.min is not None else None,
                    max_value=float(hp.max) if hp.max is not None else None,
                )
            elif hp.ptype is ParamType.CHOICE:
                params[hp.name] = st.selectbox(hp.label, hp.choices)
else:
    st.caption("This model exposes no configurable hyperparameters.")

st.divider()

# --------------------------------------------------------------------------- #
# Train & evaluate
# --------------------------------------------------------------------------- #
st.header("3 · Train & evaluate")

if st.button("Train model", type="primary"):
    with st.spinner(f"Training {spec.name}..."):
        train_result = service.train(spec.key, params)
    st.success("Training routine completed.")
    placeholder_badge(train_result)
    show_execution_time(train_result)

    with st.spinner("Evaluating..."):
        eval_result = service.evaluate(spec.key)
    report = eval_result.value

    st.subheader("Evaluation metrics")
    placeholder_badge(eval_result)
    render_metrics(report.get("metrics", {}))

    st.subheader("Prediction sample")
    st.dataframe(report.get("prediction_sample"), use_container_width=True)

    if "feature_importance" in report:
        st.subheader("Feature importance")
        fi = report["feature_importance"]
        st.bar_chart(fi.set_index("feature"))

    show_execution_time(eval_result)
