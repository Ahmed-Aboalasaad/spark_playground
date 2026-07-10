"""Home — project overview and the dataset control panel.

This is the Streamlit entrypoint (run with ``streamlit run app/Home.py``).
Sibling pages live in ``app/pages/`` and Streamlit wires up navigation
automatically.

Unlike the other pages, the data control panel here is real, not placeholder:
downloading, deleting, and loading parquet files all do genuine work regardless
of ``config.PLACEHOLDER_MODE`` (which still gates cleaning/analysis/modeling on
the other pages).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from datetime import date

import streamlit as st

from app.sidebar import render_sidebar
from app.ui import format_bytes, format_month, page_header, show_execution_time
from config import DATASET_EARLIEST_MONTH
from pipeline import dataset_manager as dm
from services.services import DataService, DatasetManagerService
from services.state import AppState
from spark.session import get_spark, session_info

st.set_page_config(page_title="NYC Taxi Analytics", page_icon="🚕", layout="wide")

state = AppState()
render_sidebar(state)

page_header(
    "NYC Taxi Analytics with Apache Spark",
    "Distributed loading, cleaning, analytics and MLlib on the NYC Yellow Taxi dataset.",
    icon="🚕",
)

st.markdown(
    """
Use the pages in the sidebar to explore the data, inspect preprocessing, run
analyses, train MLlib models, and look under the hood at Spark execution details.
Start by downloading a month or two below, then **Load into Spark**.
"""
)

with st.expander("Project objectives & technology stack", icon=":material/info:"):
    st.markdown(
        """
**Objectives:** distributed loading from Parquet, large-scale cleaning,
analytics via the DataFrame API and Spark SQL, feature engineering, MLlib
model training and evaluation, and Spark execution monitoring.

**Stack:** Apache Spark · Python · Spark MLlib · Streamlit · Plotly · Parquet.
"""
    )

# --------------------------------------------------------------------------- #
# Data control panel
# --------------------------------------------------------------------------- #
st.header("Data control panel")

manager = DatasetManagerService(state)
data_service = DataService(state)

today = date.today()
all_months = dm.month_range(DATASET_EARLIEST_MONTH, (today.year, today.month))
month_options = [dm.month_str(m) for m in all_months]

inv = manager.inventory()
downloaded_months = inv["months"]
downloaded_set = set(downloaded_months)


@st.dialog("Confirm deletion")
def _confirm_delete(months: list[dm.Month], label: str, *, delete_all: bool) -> None:
    size = sum(dm.file_path(m).stat().st_size for m in months if dm.file_path(m).exists())
    st.write(f"This will permanently delete **{label}** "
             f"({format_bytes(size)} freed). This can't be undone.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Cancel"):
            st.rerun()
        if st.button("Delete", type="primary", icon=":material/delete_forever:"):
            result = manager.delete_all() if delete_all else manager.delete(months)
            st.toast(f"Deleted {len(result.value['deleted'])} month(s)",
                     icon=":material/delete:")
            st.rerun()


# ---- Coverage overview ---------------------------------------------------- #
with st.container(border=True):
    st.subheader("Downloaded coverage")

    if inv["count"] == 0:
        st.caption("No data downloaded yet. Use the panel below to fetch some months.")
    else:
        span = f"{format_month(dm.month_str(inv['min']))} → {format_month(dm.month_str(inv['max']))}"
        with st.container(horizontal=True):
            st.metric("Coverage span", span, border=True)
            st.metric("Months downloaded", inv["count"], border=True)
            st.metric("Gaps", len(inv["gaps"]), border=True)
            st.metric("Size on disk", format_bytes(inv["total_bytes"]), border=True)

        for year in sorted({y for y, _ in dm.month_range(inv["min"], inv["max"])}):
            st.markdown(f"**{year}**")
            with st.container(horizontal=True):
                for mon in range(1, 13):
                    m = (year, mon)
                    if m < inv["min"] or m > inv["max"]:
                        continue
                    label = date(year, mon, 1).strftime("%b")
                    if m in downloaded_set:
                        st.badge(label, icon=":material/check:", color="green")
                    else:
                        st.badge(label, icon=":material/close:", color="gray")

        if inv["gaps"]:
            gap_labels = ", ".join(format_month(dm.month_str(g)) for g in inv["gaps"])
            st.caption(f":material/warning: Missing within range: {gap_labels}")

# ---- Download control ------------------------------------------------------ #
with st.container(border=True):
    st.subheader("Download data")

    default_start, default_end = month_options[max(0, len(month_options) - 6)], month_options[-1]
    if state.download_range:
        default_start = dm.month_str(state.download_range[0])
        default_end = dm.month_str(state.download_range[1])

    start_str, end_str = st.select_slider(
        "Months to download",
        options=month_options,
        value=(default_start, default_end),
        format_func=format_month,
        help="Fetches monthly yellow-taxi trip files from the NYC TLC public dataset.",
    )
    start_m, end_m = dm.parse_month_str(start_str), dm.parse_month_str(end_str)
    selected = dm.month_range(start_m, end_m)
    to_fetch = [m for m in selected if m not in downloaded_set]
    already = len(selected) - len(to_fetch)

    if to_fetch:
        st.caption(f"{len(to_fetch)} month(s) to fetch"
                   + (f" · {already} already downloaded (skipped)" if already else ""))
    else:
        st.caption("All selected months are already downloaded.")

    if st.button("Download", type="primary", icon=":material/download:", disabled=not to_fetch):
        state.download_range = (start_m, end_m)
        progress_bar = st.progress(0.0)
        status = st.status(f"Downloading {len(to_fetch)} month(s)…", expanded=True)

        def _on_progress(ev: dict) -> None:
            idx, total, month = ev["index"], ev["total"], ev["month"]
            label = format_month(dm.month_str(month))
            if ev["stage"] == "start":
                progress_bar.progress(idx / total, text=f"Fetching {label}…")
            elif ev["stage"] == "chunk" and ev["bytes_total"]:
                frac = ev["bytes_done"] / ev["bytes_total"]
                progress_bar.progress(
                    min((idx + frac) / total, 1.0),
                    text=f"{label}: {format_bytes(ev['bytes_done'])} / {format_bytes(ev['bytes_total'])}",
                )
            elif ev["stage"] == "done":
                progress_bar.progress((idx + 1) / total)
                status.write(f":material/check_circle: {label} downloaded")
            elif ev["stage"] == "skipped":
                progress_bar.progress((idx + 1) / total)
                status.write(f":material/remove_circle: {label} already present, skipped")
            elif ev["stage"] == "failed":
                progress_bar.progress((idx + 1) / total)
                status.write(f":material/error: {label} failed — {ev['error']}")

        result = manager.download(to_fetch, on_progress=_on_progress)
        summary = result.value
        n_ok, n_fail = len(summary["downloaded"]), len(summary["failed"])

        if n_fail:
            status.update(label=f"Downloaded {n_ok} month(s), {n_fail} failed",
                          state="error", expanded=True)
        else:
            status.update(label=f"Downloaded {n_ok} month(s)",
                          state="complete", expanded=False)

        show_execution_time(result)
        if n_ok:
            st.toast(f"Downloaded {n_ok} month(s)", icon=":material/download_done:")
        st.rerun()

# ---- Delete control --------------------------------------------------------- #
with st.container(border=True):
    st.subheader(":red[Danger zone]")

    if not downloaded_months:
        st.caption("Nothing downloaded yet.")
    else:
        to_delete = st.multiselect(
            "Months to delete",
            options=downloaded_months,
            format_func=lambda m: format_month(dm.month_str(m)),
            placeholder="Choose specific months…",
        )
        with st.container(horizontal=True):
            if st.button("Delete selected", icon=":material/delete:", disabled=not to_delete):
                _confirm_delete(to_delete, f"{len(to_delete)} selected month(s)", delete_all=False)
            if st.button("Delete all downloaded data", icon=":material/delete_forever:"):
                _confirm_delete(downloaded_months, f"all {len(downloaded_months)} downloaded month(s)",
                                delete_all=True)

# ---- Load control ----------------------------------------------------------- #
with st.container(border=True):
    st.subheader("Load into session")

    if not downloaded_months:
        st.caption("No data downloaded yet — use the panel above first.")
    else:
        lo, hi = downloaded_months[0], downloaded_months[-1]
        default_start_m, default_end_m = lo, hi
        if state.load_range and state.load_range[0] in downloaded_set:
            default_start_m = state.load_range[0]
        if state.load_range and state.load_range[1] in downloaded_set:
            default_end_m = state.load_range[1]

        load_options = [dm.month_str(m) for m in downloaded_months]
        start_str, end_str = st.select_slider(
            "Months to load",
            options=load_options,
            value=(dm.month_str(default_start_m), dm.month_str(default_end_m)),
            format_func=format_month,
            help="Reads the selected months' parquet files into the active Spark session.",
        )
        start_m, end_m = dm.parse_month_str(start_str), dm.parse_month_str(end_str)
        wanted = dm.month_range(start_m, end_m)
        to_load = [m for m in wanted if m in downloaded_set]
        skipped_in_range = [m for m in wanted if m not in downloaded_set]

        if skipped_in_range:
            st.caption(f":material/warning: {len(skipped_in_range)} month(s) in range "
                       f"aren't downloaded and will be skipped.")

        if st.button("Load into Spark", type="primary", icon=":material/bolt:", disabled=not to_load):
            state.load_range = (start_m, end_m)
            try:
                spark = get_spark()
                state.spark = spark
                with st.spinner(f"Loading {len(to_load)} month(s) into Spark…"):
                    result = data_service.load(spark, months=to_load)
                st.success(f"Loaded {result.n_records:,} records from {len(to_load)} month(s).")
                show_execution_time(result)
            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.error(f"Load failed: {exc}")

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

    if info.get("missing_schema_columns"):
        st.warning("Missing expected columns: " + ", ".join(info["missing_schema_columns"]),
                   icon=":material/warning:")

    with st.expander("Schema (columns)", icon=":material/table_chart:"):
        st.write(info.get("columns", []))

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
    st.caption("Spark session starts when you first load data.")
