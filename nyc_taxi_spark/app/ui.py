"""Shared UI helpers for the Streamlit pages.

Keeps the pages themselves short by centralizing three things:

* **Placeholder signalling** -- a loud banner and per-result badges, so a
  zeroed chart can never be mistaken for a real result.
* **Result rendering** -- turning a :class:`~services.timing.Result` into a
  chart plus its metrics and execution time.
* **Guards** -- the "load the dataset first" gate every page shares.

All Streamlit and Plotly imports are local to the functions that need them so
this module stays importable outside a Streamlit runtime.
"""
from __future__ import annotations

import calendar
from typing import Any

from pipeline.analysis import Analysis, ChartType
from services.timing import Result

PLACEHOLDER_BANNER = (
    "⚠️ **PLACEHOLDER MODE** — every number on this page is a stub set to zero "
    "and every chart is intentionally empty. Only the layout, wiring, and "
    "execution timings are real. Values appear once the Spark pipeline is "
    "implemented."
)

# Light-mode polish: a soft, corner-glow gradient over the white canvas plus a
# few tasteful touches. Colors are all theme-consistent tints — the request was
# explicitly for background gradients, so this is the one place we reach past
# config.toml into CSS. Kept subtle to preserve text contrast.
_GLOBAL_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 560px at 12% -8%, #EFF6FF 0%, rgba(239,246,255,0) 60%),
    radial-gradient(950px 500px at 100% 0%, #F5F3FF 0%, rgba(245,243,255,0) 55%),
    #FFFFFF;
}
[data-testid="stSidebarContent"] {
  background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
}
[data-testid="stMetric"] {
  background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
  transition: box-shadow .2s ease, transform .2s ease;
}
[data-testid="stMetric"]:hover {
  box-shadow: 0 8px 24px -14px rgba(30, 64, 175, .45);
  transform: translateY(-1px);
}
.taxi-hero {
  display: flex; gap: 18px; align-items: center;
  padding: 22px 26px; border-radius: 16px; margin: 2px 0 14px 0;
  background: linear-gradient(120deg, #1E3A8A 0%, #1E40AF 48%, #4338CA 100%);
  color: #FFFFFF; box-shadow: 0 14px 34px -16px rgba(30, 64, 175, .55);
}
.taxi-hero .hero-icon { font-size: 2.1rem; line-height: 1; }
.taxi-hero .hero-title { font-size: 1.55rem; font-weight: 800; line-height: 1.15; }
.taxi-hero .hero-sub { opacity: .92; font-size: .95rem; margin-top: 3px; }
@media (prefers-reduced-motion: reduce) {
  [data-testid="stMetric"] { transition: none; }
}
</style>
"""


def page_header(title: str, subtitle: str = "", icon: str = "📊") -> None:
    """Inject the global light-mode polish and render a gradient hero banner.

    Called once at the top of each page so every page shares an identical,
    cohesive header. Cheap enough to run on every rerun.
    """
    import html

    import streamlit as st

    st.markdown(_GLOBAL_STYLE, unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="taxi-hero">
  <div class="hero-icon">{html.escape(icon)}</div>
  <div>
    <div class="hero-title">{html.escape(title)}</div>
    <div class="hero-sub">{html.escape(subtitle)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def placeholder_banner() -> None:
    """Render the page-level placeholder warning."""
    import streamlit as st

    st.warning(PLACEHOLDER_BANNER)


def placeholder_badge(result: Result) -> None:
    """Small inline marker shown next to a placeholder result."""
    import streamlit as st

    if result.placeholder:
        st.caption("🔸 PLACEHOLDER — zeroed stub data")


def require_dataset(state: Any) -> bool:
    """Gate a page on the dataset being loaded.

    Returns True if it's safe to proceed, otherwise renders guidance and
    returns False so the caller can ``st.stop()``.
    """
    import streamlit as st

    if not state.is_loaded:
        st.info("No dataset loaded yet. Head to the **Home** page and load the "
                "data to get started.")
        return False
    return True


def format_bytes(n: int) -> str:
    """Human-readable file size, e.g. ``1536000`` -> ``"1.5 MB"``."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - unreachable, satisfies type checkers


def format_month(month_str: str) -> str:
    """``"2023-01"`` -> ``"Jan 2023"`` for compact, human-friendly labels."""
    year, month = month_str.split("-")
    return f"{calendar.month_abbr[int(month)]} {year}"


def show_execution_time(result: Result) -> None:
    """Display the real elapsed time for a result."""
    import streamlit as st

    st.caption(f"⏱ Execution time: {result.elapsed_str}"
               + (f"  ·  {result.n_records:,} records" if result.n_records else ""))


def render_metrics(metrics: dict) -> None:
    """Render a metrics dict as a row of Streamlit metric cards."""
    import streamlit as st

    if not metrics:
        return
    display = {k: v for k, v in metrics.items() if k != "status"}
    if not display:
        return
    cols = st.columns(len(display))
    for col, (name, value) in zip(cols, display.items()):
        label = name.replace("_", " ").title()
        col.metric(label, value, border=True)


def render_analysis_result(result: Result) -> None:
    """Render one analysis Result: chart, metrics, timing, placeholder badge."""
    import streamlit as st

    payload = result.value
    analysis: Analysis = payload["analysis"]
    frame = payload["frame"]
    metrics = payload["metrics"]

    st.subheader(analysis.title)
    if analysis.description:
        st.caption(analysis.description)
    placeholder_badge(result)

    render_metrics(metrics)
    _render_chart(analysis, frame)
    show_execution_time(result)

    if frame is not None and len(frame) and analysis.chart is not ChartType.METRIC:
        with st.expander("View data", icon=":material/table_chart:"):
            st.dataframe(frame, hide_index=True, width="stretch")


def _style_fig(fig, height: int = 380) -> None:
    """Apply consistent light-mode Plotly polish shared by every chart."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7", zeroline=False)


def _render_chart(analysis: Analysis, frame: Any) -> None:
    """Dispatch to the right Plotly chart for an analysis's chart type."""
    import plotly.express as px
    import streamlit as st

    # Metric-only analyses have no frame to plot.
    if analysis.chart is ChartType.METRIC or frame is None or len(frame) == 0:
        return

    x, y = analysis.x, analysis.y

    # Special case: the pickup-vs-dropoff comparison ships two value columns and
    # is best read as grouped bars.
    if {"pickups", "dropoffs"}.issubset(frame.columns):
        melted = frame.melt(id_vars=[x], value_vars=["pickups", "dropoffs"],
                            var_name="kind", value_name="trips")
        fig = px.bar(melted, x=x, y="trips", color="kind", barmode="group", title=None)
    elif analysis.chart is ChartType.LINE:
        fig = px.area(frame, x=x, y=y, title=None)
        fig.update_traces(line=dict(width=2), fillcolor="rgba(30,64,175,0.10)")
    else:  # BAR and HISTOGRAM (pre-binned) both render as bars.
        fig = px.bar(frame, x=x, y=y, title=None)

    _style_fig(fig)
    st.plotly_chart(fig, width="stretch", theme="streamlit")
