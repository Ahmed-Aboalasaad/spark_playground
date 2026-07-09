"""Analysis — run registered analytical queries and visualize results.

Analyses come from the registry in ``pipeline.analysis``, grouped by family.
Because the page is driven by the registry, adding a new analysis there makes
it show up here automatically — no page changes required.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import streamlit as st

from app.sidebar import render_sidebar
from app.ui import placeholder_banner, render_analysis_result, require_dataset
from config import PLACEHOLDER_MODE
from pipeline import analysis as analysis_mod
from services.services import AnalysisService
from services.state import AppState

st.set_page_config(page_title="Analysis", page_icon="📊", layout="wide")

state = AppState()
render_sidebar(state)

st.title("📊 Analysis")
if PLACEHOLDER_MODE:
    placeholder_banner()

if not require_dataset(state):
    st.stop()

st.caption(f"Analyzing the **{state.selection.value}**.")

# Family → analysis selectors, both driven by the registry.
families = analysis_mod.all_families()
family = st.selectbox("Analysis family", families, format_func=lambda f: f.value)

entries = analysis_mod.analyses_in(family)
entry = st.selectbox(
    "Analysis",
    entries,
    format_func=lambda a: a.title,
)

service = AnalysisService(state)

if st.button("Run analysis", type="primary"):
    with st.spinner(f"Running: {entry.title}..."):
        result = service.run(entry.key)
    render_analysis_result(result)
