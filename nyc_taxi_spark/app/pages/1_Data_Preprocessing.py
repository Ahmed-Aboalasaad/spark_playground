"""Data Preprocessing — data quality, the cleaning pipeline, and feature
engineering, faithfully mirroring notebook 04 but computed live in Spark.

Three tabs:
  1. Data quality   — missing values, duplicates, and the sanity checks that
     motivate the cleaning thresholds.
  2. Cleaning       — the fixed-rule filters, what they removed, and before/after
     distributions.
  3. Feature engineering — the engineered features, the time-based split, and the
     signal each feature carries.

Every heavy step runs on demand and reports its real execution time.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from app.sidebar import render_sidebar
from app.ui import page_header, require_dataset, show_execution_time
from config import (
    FARE_MAX,
    FARE_MIN,
    TRIP_DISTANCE_MAX,
    TRIP_DURATION_MAX,
    TRIP_DURATION_MIN,
)
from pipeline.features import FEATURE_RATIONALE
from services.services import DataService, ModelingService
from services.state import AppState, DatasetChoice

st.set_page_config(page_title="Data preprocessing", page_icon=":material/cleaning_services:",
                   layout="wide")

state = AppState()
render_sidebar(state)

page_header(
    "Data preprocessing",
    "Audit, clean, and engineer features — every threshold documented, every step timed.",
    icon="🧹",
)

if not require_dataset(state):
    st.stop()

is_clean = state.selection is DatasetChoice.CLEANED and state.cleaned_df is not None
badge = ":green-badge[Cleaned dataset]" if is_clean else ":blue-badge[Raw dataset]"
st.markdown(f"Working on the {badge}.")

with st.expander("Why preprocess first?", icon=":material/lightbulb:"):
    st.markdown(
        """
TLC trip data is vendor-submitted and unverified: it contains negative fares
(refunds/voids), meter glitches in the thousands of dollars, zero-distance
"trips", missing passenger counts, and sentinel *Unknown* zones. Feeding that to
a model or a chart produces misleading results.

The workflow here is deliberate and **auditable**: first we *count the damage*
(Data quality), then we apply **fixed, hand-written rules** — no statistic is
learned from the data before the split, so nothing can leak (Cleaning), and
finally we turn raw columns into features that encode what actually drives a
fare or a trip's duration (Feature engineering).
"""
    )


def _style(fig, height: int = 320):
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=height,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7", zeroline=False)
    return fig


quality_tab, cleaning_tab, feature_tab = st.tabs(
    ["Data quality", "Cleaning pipeline", "Feature engineering"]
)

# =========================================================================== #
# 1. Data quality
# =========================================================================== #
with quality_tab:
    st.caption(f"Reporting on the **{state.selection.value}**. Runs a full scan in Spark.")
    if st.button("Run quality report", icon=":material/fact_check:", key="btn_quality"):
        with st.spinner("Auditing dataset…"):
            st.session_state["pp_quality"] = DataService(state).quality_report()

    qres = st.session_state.get("pp_quality")
    if qres is not None:
        report = qres.value
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{report['n_records']:,}", border=True)
        c2.metric("Columns", report["n_columns"], border=True)
        c3.metric("Duplicate rows", f"~{report['n_duplicates']:,}", border=True,
                  help="Approximate (HyperLogLog) — exact de-duplication over the "
                       "full dataset is too memory-heavy to run on every report.")
        total_missing = sum(report["missing_per_column"].values())
        c4.metric("Missing cells", f"{total_missing:,}", border=True)

        # Missing values per column (only those that have any).
        miss = {c: n for c, n in report["missing_per_column"].items() if n}
        st.subheader("Missing values per column")
        if miss:
            mdf = pd.DataFrame({
                "column": list(miss.keys()),
                "missing": list(miss.values()),
                "missing_%": [report["missing_pct_per_column"][c] for c in miss],
            }).sort_values("missing", ascending=False)
            left, right = st.columns([3, 2])
            with left:
                fig = px.bar(mdf, x="missing", y="column", orientation="h",
                             text="missing_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(_style(fig, 300), width="stretch", theme="streamlit")
            with right:
                st.dataframe(
                    mdf, hide_index=True, width="stretch",
                    column_config={
                        "missing": st.column_config.NumberColumn("Missing", format="%d"),
                        "missing_%": st.column_config.NumberColumn("%", format="%.2f%%"),
                    },
                )
        else:
            st.success("No missing values in any column.", icon=":material/check_circle:")

        # Sanity checks — the implausible records the cleaning thresholds target.
        st.subheader("Sanity checks")
        st.caption("These are exactly the records the cleaning rules will remove.")
        s = report["sanity"]
        n = max(report["n_records"], 1)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Implausible fares", f"{s['implausible_fare']:,}",
                  delta=f"{s['implausible_fare'] / n * 100:.2f}%",
                  delta_color="inverse", border=True,
                  help=f"fare ≤ $0 or > ${FARE_MAX:g}")
        s2.metric("Zero distance", f"{s['zero_distance']:,}",
                  delta=f"{s['zero_distance'] / n * 100:.2f}%",
                  delta_color="inverse", border=True)
        s3.metric("Bad passenger count", f"{s['bad_passenger_count']:,}",
                  delta=f"{s['bad_passenger_count'] / n * 100:.2f}%",
                  delta_color="inverse", border=True, help="null or ≤ 0")
        s4.metric("Out-of-range duration", f"{s['bad_duration']:,}",
                  delta=f"{s['bad_duration'] / n * 100:.2f}%",
                  delta_color="inverse", border=True,
                  help=f"< {TRIP_DURATION_MIN:g} or > {TRIP_DURATION_MAX:g} min")

        if report.get("numeric_summary") is not None:
            st.subheader("Numeric summary")
            st.dataframe(report["numeric_summary"], width="stretch")

        show_execution_time(qres)
    else:
        st.caption("Press **Run quality report** to audit the active dataset.")

# =========================================================================== #
# 2. Cleaning pipeline
# =========================================================================== #
with cleaning_tab:
    st.markdown(
        f"""
Each filter is a fixed rule with a documented reason — applied to the whole
dataset *before* any split, so no statistic is learned from the data and nothing
can leak.

| Filter | Why | Threshold |
|---|---|---|
| Fare range | Below the flag-drop minimum are refunds/voids; above are meter glitches | `${FARE_MIN:g} … ${FARE_MAX:g}` |
| Distance range | 0 mi is a GPS/meter error; 100+ mi is not a yellow-cab trip | `0 … {TRIP_DISTANCE_MAX:g} mi` |
| Passenger count | Missing/zero means the driver never entered it | `> 0` |
| Duration range | Under a minute is an accidental meter start; over 3 h a forgotten meter | `{TRIP_DURATION_MIN:g} … {TRIP_DURATION_MAX:g} min` |
| Known zones | 264 / 265 are the *Unknown / N/A* placeholder zones | drop |

_In Spark, `NULL` fails every comparison, so rows with null fare/distance/passenger
count are dropped implicitly here — intentional, because those nulls are data-entry
failures, not meaningful absence._
"""
    )
    if st.button("Run cleaning", type="primary", icon=":material/cleaning_services:",
                 key="btn_clean"):
        with st.spinner("Cleaning dataset in Spark…"):
            st.session_state["pp_clean"] = DataService(state).clean()
            st.session_state.pop("pp_dist", None)  # invalidate stale distributions
        st.toast("Cleaning complete", icon=":material/check:")

    cres = st.session_state.get("pp_clean")
    if cres is not None:
        summ = cres.value
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input records", f"{summ['n_input']:,}", border=True)
        c2.metric("Output records", f"{summ['n_output']:,}", border=True)
        c3.metric("Removed", f"{summ['n_removed']:,}",
                  delta=f"-{summ['pct_removed']:.2f}%", delta_color="inverse", border=True)
        c4.metric("Kept", f"{summ['pct_kept']:.2f}%", border=True)

        st.subheader("Rows failing each rule")
        st.caption("Rules overlap (a row can fail several), so these don't sum to the "
                   "total removed — it's a diagnostic breakdown.")
        bd = summ["removal_breakdown"]
        labels = {
            "fare_out_of_range": "Fare out of range",
            "distance_out_of_range": "Distance out of range",
            "passengers_missing_or_zero": "Passengers missing/zero",
            "duration_out_of_range": "Duration out of range",
            "unknown_zone": "Unknown zone",
            "invalid_pickup_date": "Invalid pickup date",
        }
        # Tolerate unknown keys so a new cleaning rule never crashes this chart.
        bdf = pd.DataFrame({
            "rule": [labels.get(k, k.replace("_", " ").title()) for k in bd],
            "rows": list(bd.values()),
        })
        fig = px.bar(bdf.sort_values("rows"), x="rows", y="rule", orientation="h")
        st.plotly_chart(_style(fig, 280), width="stretch", theme="streamlit")

        with st.expander("Operations applied", icon=":material/checklist:", expanded=True):
            for op in summ["operations"]:
                st.markdown(f"- {op}")

        st.subheader("Distributions — before vs after")
        if st.button("Compute before/after distributions", icon=":material/insights:",
                     key="btn_dist"):
            with st.spinner("Sampling distributions…"):
                st.session_state["pp_dist"] = DataService(state).cleaning_distributions()

        dres = st.session_state.get("pp_dist")
        if dres is not None and dres.value["after"] is not None:
            before, after = dres.value["before"], dres.value["after"]
            specs = [
                ("fare_amount", "Fare ($)", (-20, 300), [(FARE_MIN, "min"), (FARE_MAX, "max")]),
                ("trip_duration_min", "Duration (min)", (0, 200),
                 [(TRIP_DURATION_MAX, "cap")]),
                ("trip_distance", "Distance (mi)", (0, 60), [(TRIP_DISTANCE_MAX, None)]),
            ]
            cols = st.columns(3)
            for col, (field, label, clip, lines) in zip(cols, specs):
                combo = pd.concat([
                    before[[field]].assign(phase="Before"),
                    after[[field]].assign(phase="After"),
                ])
                combo[field] = combo[field].clip(clip[0], clip[1])
                fig = px.histogram(combo, x=field, color="phase", barmode="overlay",
                                   nbins=60, opacity=0.7,
                                   color_discrete_map={"Before": "#94A3B8", "After": "#1E40AF"})
                for val, _ in lines:
                    if clip[0] <= val <= clip[1]:
                        fig.add_vline(x=val, line_dash="dash", line_color="#DC2626")
                fig.update_layout(title=label)
                col.plotly_chart(_style(fig, 280), width="stretch", theme="streamlit")
            show_execution_time(dres)

        st.info("Switch the sidebar selector to **Cleaned dataset** to run analyses on "
                "the cleaned data.", icon=":material/swap_horiz:")
        show_execution_time(cres)
    else:
        st.caption("Press **Run cleaning** to apply the pipeline and see its effect.")

# =========================================================================== #
# 3. Feature engineering
# =========================================================================== #
with feature_tab:
    st.markdown(
        """
We give the model, as explicit numbers, the things a New Yorker knows determine a
trip's fare and duration. Two rules keep it honest: features use **only what is
knowable before the trip starts** (no `tip_amount`, `total_amount`, dropoff time…
— those would leak the answer), and history features look **strictly at past
days**.
"""
    )
    rat = pd.DataFrame(
        {"Feature": list(FEATURE_RATIONALE.keys()),
         "What it encodes": list(FEATURE_RATIONALE.values())}
    )
    st.dataframe(rat, hide_index=True, width="stretch")

    st.caption("Feature engineering builds on the cleaned dataset — run cleaning first "
               "for best results.")
    if st.button("Build train/test features", type="primary",
                 icon=":material/build:", key="btn_feat"):
        with st.spinner("Engineering features and splitting by time…"):
            st.session_state["pp_feat"] = ModelingService(state).prepare_features()

    fres = st.session_state.get("pp_feat")
    if fres is not None:
        info = fres.value["info"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Train records", f"{info['n_train']:,}", border=True)
        c2.metric("Test records", f"{info['n_test']:,}", border=True)
        c3.metric("Split date", info["split_date"] or "—", border=True,
                  help="Everything before this date trains; the rest is the held-out future.")
        st.caption("**Time-based split** (not random): training on the past and testing on "
                   "a held-out future mimics real deployment. The lag features are then "
                   "imputed with **train-only** means so no future info leaks backward — "
                   f"fill values: {info['fill_means']}.")
        fc1, fc2 = st.columns(2)
        fc1.markdown("**Numeric features**\n\n" + ", ".join(info["numeric_features"]))
        fc2.markdown("**Categorical / flag features**\n\n" + ", ".join(info["categorical_features"]))
        show_execution_time(fres)

    st.subheader("Do the features carry real signal?")
    if st.button("Show feature signal", icon=":material/query_stats:", key="btn_signal"):
        with st.spinner("Aggregating signal plots in Spark…"):
            st.session_state["pp_signals"] = ModelingService(state).feature_signals()

    sres = st.session_state.get("pp_signals")
    if sres is not None:
        sig = sres.value

        st.markdown("**Hourly profile** — justifies `pickup_hour`, `is_rush_hour`, "
                    "and the cyclical encoding.")
        h = sig["hourly"]
        h1, h2, h3 = st.columns(3)
        h1.plotly_chart(_style(px.bar(h, x="pickup_hour", y="trips").update_layout(
            title="Trips by hour"), 260), width="stretch", theme="streamlit")
        h2.plotly_chart(_style(px.bar(h, x="pickup_hour", y="avg_duration").update_layout(
            title="Avg duration by hour"), 260), width="stretch", theme="streamlit")
        h3.plotly_chart(_style(px.bar(h, x="pickup_hour", y="avg_fare").update_layout(
            title="Avg fare by hour"), 260), width="stretch", theme="streamlit")

        st.markdown("**Weekday profile** — justifies `pickup_dayofweek` and `is_weekend`.")
        w = sig["weekday"]
        w1, w2 = st.columns(2)
        w1.plotly_chart(_style(px.bar(w, x="weekday", y="trips").update_layout(
            title="Trips by day of week"), 260), width="stretch", theme="streamlit")
        w2.plotly_chart(_style(px.bar(w, x="weekday", y="avg_duration").update_layout(
            title="Avg duration by day of week"), 260), width="stretch", theme="streamlit")

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Airport flag** — one 0/1 captures a whole fare regime.")
            ap = sig["airport"]
            a1.plotly_chart(_style(px.bar(ap, x="group", y="avg_fare", color="group",
                            color_discrete_sequence=["#64748B", "#D97706"]).update_layout(
                title="Avg fare: airport vs not", showlegend=False), 300),
                width="stretch", theme="streamlit")
        with a2:
            st.markdown("**Distance vs fare** — strong but sub-linear (motivates `log_distance`).")
            sc = sig["distance_fare"]
            fig = px.scatter(sc, x="trip_distance", y="fare_amount", opacity=0.2)
            fig.update_traces(marker=dict(size=4, color="#1E40AF"))
            a2.plotly_chart(_style(fig, 300), width="stretch", theme="streamlit")

        st.markdown("**Daily trips across the loaded period** — the whole experimental "
                    "design in one series; everything right of the line is the held-out future.")
        d = sig["daily"]
        d["pickup_date"] = pd.to_datetime(d["pickup_date"])
        fig = px.area(d, x="pickup_date", y="trips")
        fig.update_traces(line=dict(width=1.5, color="#1E40AF"),
                          fillcolor="rgba(30,64,175,0.10)")
        if sig.get("split_date"):
            fig.add_vline(x=pd.Timestamp(sig["split_date"]), line_dash="dash",
                          line_color="#DC2626", annotation_text="train / test split")
        st.plotly_chart(_style(fig, 320), width="stretch", theme="streamlit")
        show_execution_time(sres)
