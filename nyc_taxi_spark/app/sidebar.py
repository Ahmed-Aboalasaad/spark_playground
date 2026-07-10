"""Shared sidebar: the app's control rail.

FEATURES.md requires a global selector to switch between the **Raw** and
**Cleaned** datasets, with the active choice clearly indicated on every page.
This renders that selector plus a compact pipeline-status readout, and gives the
sidebar its own visual identity — a cool teal→indigo palette distinct from the
blue hero used in the page bodies — so the "control rail" reads as a separate
surface from the content.

The brand mark ("NYC Taxi Analytics") is rendered via ``st.logo()`` rather than
inline HTML. Streamlit always renders the auto-generated page navigation into a
fixed slot at the very top of the sidebar, *before* any custom content added
inside ``with st.sidebar:`` -- no amount of call-ordering (including switching
to ``st.navigation``/``st.Page``, verified) changes that. ``st.logo()`` is the
one thing Streamlit renders into the genuinely-first header slot, above the
nav, which is what puts the brand mark at the literal top of the sidebar.

Centralizing it here means every page shows an identical, consistent control.
Streamlit/HTML is emitted lazily inside the function so the module stays
importable outside a Streamlit runtime.
"""
from __future__ import annotations

from pathlib import Path

from services.state import AppState, DatasetChoice

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_BRAND_LOGO = _ASSETS_DIR / "brand_logo.svg"
_BRAND_ICON = _ASSETS_DIR / "brand_icon.svg"

# Scoped to the sidebar only, so the page bodies keep their own look. A new
# palette (mint/teal → soft indigo) plus branded header and status chips.
_SIDEBAR_STYLE = """
<style>
[data-testid="stSidebarContent"] {
  background:
    radial-gradient(600px 300px at 20% -5%, #CCFBF1 0%, rgba(204,251,241,0) 60%),
    linear-gradient(180deg, #F0FDFA 0%, #EEF2FF 100%);
}
.sb-label {
  text-transform: uppercase; letter-spacing: .08em; font-size: .68rem;
  font-weight: 800; color: #0F766E; margin: 6px 2px 2px 2px;
}
.sb-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 999px; font-size: .8rem; font-weight: 700;
  width: 100%; box-sizing: border-box;
}
.sb-chip.raw     { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
.sb-chip.cleaned { background: #D1FAE5; color: #047857; border: 1px solid #A7F3D0; }
.sb-step {
  display: flex; align-items: center; gap: 9px; font-size: .84rem;
  color: #334155; padding: 3px 2px;
}
.sb-step .dot {
  width: 16px; height: 16px; border-radius: 50%; flex: 0 0 auto;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: .7rem; color: #fff; font-weight: 900;
}
.sb-step .dot.on   { background: #059669; }
.sb-step .dot.off  { background: #CBD5E1; }
.sb-step.off { color: #94A3B8; }
</style>
"""


def render_sidebar(state: AppState) -> None:
    """Render the branded selector + pipeline status into the sidebar."""
    import streamlit as st

    # The very top of the sidebar, above the page nav -- see module docstring
    # for why this has to be st.logo() rather than inline HTML.
    st.logo(str(_BRAND_LOGO), size="large", icon_image=str(_BRAND_ICON))

    with st.sidebar:
        st.markdown(_SIDEBAR_STYLE, unsafe_allow_html=True)

        loaded = state.is_loaded
        cleaned_ready = state.cleaning_summary is not None

        # ---- Active dataset selector ------------------------------------- #
        st.markdown('<div class="sb-label">Active dataset</div>', unsafe_allow_html=True)
        options = [DatasetChoice.RAW, DatasetChoice.CLEANED]
        icons = {DatasetChoice.RAW: "🟡 Raw", DatasetChoice.CLEANED: "🟢 Cleaned"}
        choice = st.segmented_control(
            "Active dataset", options, format_func=lambda c: icons[c],
            default=state.selection, selection_mode="single",
            label_visibility="collapsed",
            help="Every page's analyses and stats run on the active dataset. "
                 "Modeling always uses the cleaned dataset regardless.")
        if choice is None:
            choice = state.selection

        # Guard: can't select Cleaned until cleaning has produced it.
        if choice is DatasetChoice.CLEANED and not cleaned_ready:
            st.caption("Cleaned dataset not produced yet — run cleaning on the "
                       "**Data Preprocessing** page. Holding on Raw.")
            choice = DatasetChoice.RAW
        state.selection = choice

        chip_cls = "cleaned" if choice is DatasetChoice.CLEANED else "raw"
        chip_txt = "Cleaned dataset" if choice is DatasetChoice.CLEANED else "Raw dataset"
        st.markdown(f'<div class="sb-chip {chip_cls}">● {chip_txt}</div>',
                    unsafe_allow_html=True)

        # ---- Pipeline status --------------------------------------------- #
        st.markdown('<div class="sb-label" style="margin-top:14px">Pipeline</div>',
                    unsafe_allow_html=True)
        _step("Data loaded", loaded)
        _step("Cleaning applied", cleaned_ready)

        if not loaded:
            st.caption("No data loaded — start on the **Home** page.")


def _step(label: str, done: bool) -> None:
    import streamlit as st

    dot = '<span class="dot on">✓</span>' if done else '<span class="dot off">○</span>'
    cls = "sb-step" if done else "sb-step off"
    st.markdown(f'<div class="{cls}">{dot}{label}</div>', unsafe_allow_html=True)
