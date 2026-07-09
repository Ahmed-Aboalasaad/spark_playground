"""Shared sidebar: dataset selector and current-selection indicator.

FEATURES.md requires a sidebar selector to switch between the Raw and Cleaned
datasets, with the current choice clearly indicated everywhere. Centralizing it
here means every page shows an identical, consistent control.
"""
from __future__ import annotations

from services.state import AppState, DatasetChoice


def render_sidebar(state: AppState) -> None:
    """Render the dataset selector and status into the sidebar."""
    import streamlit as st

    with st.sidebar:
        st.markdown("### Dataset")

        # "Ready" means cleaning has run, not that a concrete DataFrame exists:
        # in placeholder mode the cleaner returns df=None but records a summary,
        # so keying on the summary lets the Cleaned option be selected there too.
        cleaned_ready = state.cleaning_summary is not None
        options = [DatasetChoice.RAW, DatasetChoice.CLEANED]

        current = state.selection
        choice = st.radio(
            "Active dataset",
            options,
            index=options.index(current),
            format_func=lambda c: c.value,
            help="All analyses and visualizations use the active dataset.",
        )

        # If the user picks Cleaned before cleaning has run, warn and hold Raw.
        if choice is DatasetChoice.CLEANED and not cleaned_ready:
            st.caption("Cleaned dataset not produced yet — run cleaning on the "
                       "Data Preprocessing page. Showing Raw for now.")
            choice = DatasetChoice.RAW

        state.selection = choice

        badge = "🟢 Cleaned" if choice is DatasetChoice.CLEANED else "🔵 Raw"
        st.markdown(f"**Active:** {badge}")

        if not state.is_loaded:
            st.caption("No data loaded. Start on the Home page.")
