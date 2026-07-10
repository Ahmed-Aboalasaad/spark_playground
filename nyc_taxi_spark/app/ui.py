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
        col.metric(label, value)


def render_analysis_result(result: Result) -> None:
    """Render one analysis Result: chart, metrics, timing, placeholder badge."""
    import streamlit as st

    payload = result.value
    analysis: Analysis = payload["analysis"]
    frame = payload["frame"]
    metrics = payload["metrics"]

    st.subheader(analysis.title)
    placeholder_badge(result)

    _render_chart(analysis, frame)
    render_metrics(metrics)
    show_execution_time(result)


def _render_chart(analysis: Analysis, frame: Any) -> None:
    """Dispatch to the right Plotly chart for an analysis's chart type."""
    import plotly.express as px
    import streamlit as st

    # Metric-only analyses have no frame to plot.
    if analysis.chart is ChartType.METRIC or frame is None or len(frame) == 0:
        return

    x, y = analysis.x, analysis.y
    if analysis.chart is ChartType.BAR:
        fig = px.bar(frame, x=x, y=y, title=None)
    elif analysis.chart is ChartType.LINE:
        fig = px.line(frame, x=x, y=y, title=None)
    elif analysis.chart is ChartType.HISTOGRAM:
        # Frame is pre-binned; render as a bar chart of bin counts.
        fig = px.bar(frame, x=x, y=y, title=None)
    else:
        return

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
    st.plotly_chart(fig, use_container_width=True)
