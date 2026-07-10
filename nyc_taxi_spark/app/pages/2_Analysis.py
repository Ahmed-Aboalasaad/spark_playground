"""Analysis — run registered analytical queries and visualize results.

Analyses come from the registry in ``pipeline.analysis``, grouped by family.
Because the page is driven by the registry, adding a new analysis there makes
it show up here automatically — no page changes required. Each run aggregates
in Spark and returns a visualization, summary metrics, and its real execution
time, on whichever dataset (Raw or Cleaned) is currently selected.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import streamlit as st

from app.sidebar import render_sidebar
from app.ui import page_header, render_analysis_result, require_dataset
from pipeline import analysis as analysis_mod
from services.services import AnalysisService
from services.state import AppState, DatasetChoice

st.set_page_config(page_title="Analysis", page_icon=":material/analytics:", layout="wide")

state = AppState()
render_sidebar(state)

page_header(
    "Analysis",
    "Demand, geography, trips, revenue, passengers and airports — aggregated live in Spark.",
    icon="📊",
)

if not require_dataset(state):
    st.stop()

# Active-dataset indicator.
is_clean = state.selection is DatasetChoice.CLEANED and state.cleaned_df is not None
badge = ":green-badge[Cleaned dataset]" if is_clean else ":blue-badge[Raw dataset]"
st.markdown(f"Analyzing the {badge}. "
            + ("" if is_clean else "Tip: run cleaning on the **Data Preprocessing** "
               "page and switch to the Cleaned dataset for outlier-free results."))

# --------------------------------------------------------------------------- #
# Family → analysis selection, both driven by the registry.
# --------------------------------------------------------------------------- #
FAMILY_ICONS = {
    "Demand": ":material/schedule:",
    "Geographic": ":material/map:",
    "Trip": ":material/route:",
    "Revenue": ":material/payments:",
    "Passenger": ":material/groups:",
    "Airport": ":material/flight:",
}

families = analysis_mod.all_families()
family = st.segmented_control(
    "Analysis family",
    families,
    format_func=lambda f: f.value,
    default=families[0],
    selection_mode="single",
)
if family is None:
    family = families[0]

entries = analysis_mod.analyses_in(family)
entry = st.selectbox(
    "Analysis",
    entries,
    format_func=lambda a: a.title,
    help="Pick a query to run against the active dataset.",
)

with st.container(border=True):
    st.markdown(f"**{FAMILY_ICONS.get(family.value, '')} {entry.title}**")
    st.caption(entry.description or "—")
    run = st.button("Run analysis", type="primary", icon=":material/play_arrow:")

service = AnalysisService(state)

# Persist the last result so it survives incidental reruns (e.g. sidebar clicks).
if run:
    with st.spinner(f"Running “{entry.title}” in Spark…"):
        state.cache_analysis("last", service.run(entry.key))

last = state.analysis_cache.get("last")
if last is not None:
    with st.container(border=True):
        render_analysis_result(last)
else:
    st.caption("Choose an analysis and press **Run analysis** to see results.")
