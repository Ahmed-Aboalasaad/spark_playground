"""Data Preprocessing — data quality and the cleaning pipeline.

Shows quality metrics for the active dataset, lets the user run cleaning, and
reports what the cleaning pipeline changed. All values are placeholder zeros
until the real cleaning lands.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import streamlit as st

from app.sidebar import render_sidebar
from app.ui import (
    placeholder_badge,
    placeholder_banner,
    require_dataset,
    show_execution_time,
)
from config import PLACEHOLDER_MODE
from services.services import DataService
from services.state import AppState

st.set_page_config(page_title="Data Preprocessing", page_icon="🧹", layout="wide")

state = AppState()
render_sidebar(state)

st.title("🧹 Data Preprocessing")
if PLACEHOLDER_MODE:
    placeholder_banner()

if not require_dataset(state):
    st.stop()

service = DataService(state)

# --------------------------------------------------------------------------- #
# Data quality report
# --------------------------------------------------------------------------- #
st.header("Data quality")
st.caption(f"Reporting on the **{state.selection.value}**.")

if st.button("Run quality report"):
    result = service.quality_report()
    report = result.value
    placeholder_badge(result)

    c1, c2 = st.columns(2)
    c1.metric("Records", f"{report.get('n_records', 0):,}")
    c2.metric("Duplicate records", f"{report.get('n_duplicates', 0):,}")

    missing = report.get("missing_per_column", {})
    if missing:
        st.subheader("Missing values per column")
        st.bar_chart(missing)
    else:
        st.caption("Missing-value breakdown will appear here (currently empty).")

    show_execution_time(result)

st.divider()

# --------------------------------------------------------------------------- #
# Cleaning pipeline
# --------------------------------------------------------------------------- #
st.header("Cleaning pipeline")

if st.button("Run cleaning", type="primary"):
    with st.spinner("Cleaning dataset..."):
        result = service.clean()
    summary = result.value
    st.success("Cleaning routine completed.")
    placeholder_badge(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Input records", f"{summary.get('n_input', 0):,}")
    c2.metric("Output records", f"{summary.get('n_output', 0):,}")
    c3.metric("Removed", f"{summary.get('n_removed', 0):,}",
              delta=f"{summary.get('pct_removed', 0.0):.1f}%")

    st.subheader("Operations applied")
    for op in summary.get("operations", []):
        st.markdown(f"- {op}")

    show_execution_time(result)

# Show the effect once cleaning has run (the cleaned frame may be a placeholder).
if state.cleaning_summary:
    st.divider()
    st.subheader("Effect of preprocessing")
    st.caption("Select the **Cleaned Dataset** in the sidebar to run analyses "
               "on the cleaned data.")
