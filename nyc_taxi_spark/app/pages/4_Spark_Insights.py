"""Spark Insights — visibility into Spark execution.

Surfaces the running execution-metrics log collected by the services, plus
partition, cache, and storage details and (where available) logical/physical
query plans for the active dataset. Plan and partition details need a real
Spark DataFrame; while placeholder mode is on they show guidance instead.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st

from app.sidebar import render_sidebar
from app.ui import placeholder_banner, require_dataset
from config import PLACEHOLDER_MODE
from services.state import AppState

st.set_page_config(page_title="Spark Insights", page_icon="⚙️", layout="wide")

state = AppState()
render_sidebar(state)

st.title("⚙️ Spark Insights")
if PLACEHOLDER_MODE:
    placeholder_banner()

# --------------------------------------------------------------------------- #
# Execution metrics log (always available — timings are real)
# --------------------------------------------------------------------------- #
st.header("Execution metrics")
metrics = state.metrics
if metrics:
    frame = pd.DataFrame(metrics)
    frame["elapsed"] = frame["elapsed"].map(lambda s: f"{s:.3f} s")
    frame = frame.rename(columns={"label": "Operation", "elapsed": "Time",
                                  "n_records": "Records"})
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("These execution times are real, even in placeholder mode.")
else:
    st.info("No operations have run yet. Run a load, cleaning, analysis, or "
            "training step and its timing will appear here.")

st.divider()

# --------------------------------------------------------------------------- #
# Partition / cache / storage details for the active dataset
# --------------------------------------------------------------------------- #
st.header("Dataset execution details")

if not require_dataset(state):
    st.stop()

df = state.active_df()

if df is None or PLACEHOLDER_MODE:
    st.caption("Partition count, cache status, storage level, and query plans "
               "will populate here once a real Spark DataFrame is loaded. In "
               "placeholder mode there is no live DataFrame to inspect.")
else:  # pragma: no cover - exercised only with a real Spark DataFrame
    c1, c2, c3 = st.columns(3)
    c1.metric("Partitions", df.rdd.getNumPartitions())
    c2.metric("Cached", str(df.is_cached))
    c3.metric("Storage level", str(df.storageLevel))

    with st.expander("Logical & physical plan"):
        st.code(df._jdf.queryExecution().toString(), language="text")
