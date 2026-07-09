"""Home — project overview, dataset loading, and Spark session info.

This is the Streamlit entrypoint (run with ``streamlit run app/Home.py``).
Sibling pages live in ``app/pages/`` and Streamlit wires up navigation
automatically.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

import streamlit as st

from app.sidebar import render_sidebar
from app.ui import placeholder_banner, placeholder_badge, show_execution_time
from config import PLACEHOLDER_MODE
from services.services import DataService
from services.state import AppState
from spark.session import get_spark, session_info

st.set_page_config(page_title="NYC Taxi Analytics", page_icon="🚕", layout="wide")

state = AppState()
render_sidebar(state)

st.title("🚕 NYC Taxi Analytics with Apache Spark")

if PLACEHOLDER_MODE:
    placeholder_banner()

# --------------------------------------------------------------------------- #
# Project overview
# --------------------------------------------------------------------------- #
st.markdown(
    """
An interactive analytics and machine-learning application built on **Apache
Spark**, processing the NYC Yellow Taxi trip dataset. Use the pages in the
sidebar to explore the data, inspect preprocessing, run analyses, train MLlib
models, and look under the hood at Spark execution details.
"""
)

with st.expander("Project objectives & technology stack"):
    st.markdown(
        """
**Objectives:** distributed loading from Parquet, large-scale cleaning,
analytics via the DataFrame API and Spark SQL, feature engineering, MLlib
model training and evaluation, and Spark execution monitoring.

**Stack:** Apache Spark · Python · Spark MLlib · Streamlit · Plotly · Parquet.
"""
    )

st.divider()

# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
st.header("Dataset")

col_load, col_reload = st.columns([1, 1])
data_service = DataService(state)

with col_load:
    if st.button("Load dataset", type="primary", use_container_width=True):
        # In placeholder mode the loader ignores the session entirely, so we
        # skip starting Spark — the app must stay fully navigable on mock data
        # even where no JDK/Spark is available. The real loader needs a session.
        spark = None if PLACEHOLDER_MODE else get_spark()
        state.spark = spark
        with st.spinner("Loading NYC Taxi dataset..."):
            result = data_service.load(spark)
        st.success("Load routine completed.")
        placeholder_badge(result)
        show_execution_time(result)

with col_reload:
    if st.button("Reload dataset", use_container_width=True, disabled=not state.is_loaded):
        spark = None if PLACEHOLDER_MODE else (state.spark or get_spark())
        with st.spinner("Reloading..."):
            result = data_service.load(spark)
        st.success("Reloaded.")
        show_execution_time(result)

# --------------------------------------------------------------------------- #
# Dataset information & statistics
# --------------------------------------------------------------------------- #
info = state.load_info
if info:
    st.subheader("Dataset information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{info.get('n_records', 0):,}")
    c2.metric("Columns", info.get("n_columns", 0))
    c3.metric("Parquet files", info.get("n_files", 0))
    c4.metric("Size (bytes)", f"{info.get('size_bytes', 0):,}")

    start, end = info.get("date_range", (None, None))
    st.caption(f"Date range: {start or '—'} → {end or '—'}")

    with st.expander("Schema (expected columns)"):
        st.write(info.get("columns", []))
else:
    st.info("Load the dataset to see its information and statistics.")

st.divider()

# --------------------------------------------------------------------------- #
# Spark session information
# --------------------------------------------------------------------------- #
st.header("Spark session")
if state.spark is not None:
    try:
        for k, v in session_info(state.spark).items():
            st.text(f"{k}: {v}")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Could not read Spark session info: {exc}")
else:
    st.caption("Spark session starts when you first load the dataset.")
