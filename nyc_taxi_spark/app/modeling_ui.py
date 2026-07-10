"""Presentation helpers for the Modeling page.

Pure UI: styling, labels, and the small chart/metric renderers the Modeling
page reuses across its Train / Saved Models / Inference tabs. Keeping these here
keeps ``pages/3_Modeling.py`` focused on flow rather than markup. Everything
that imports Streamlit/Plotly does so lazily, so this module stays importable
outside a Streamlit runtime.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Domain labels
# --------------------------------------------------------------------------- #

# Friendly names + units for each regression target.
TARGET_META: dict[str, dict[str, str]] = {
    "fare_amount": {"label": "Fare amount", "unit": "$", "unit_pos": "prefix",
                    "icon": "💵"},
    "trip_duration_min": {"label": "Trip duration", "unit": "min", "unit_pos": "suffix",
                          "icon": "⏱️"},
}

# TLC rate codes — shown as human labels in the inference form.
RATECODE_LABELS: dict[int, str] = {
    1: "Standard rate",
    2: "JFK",
    3: "Newark",
    4: "Nassau / Westchester",
    5: "Negotiated fare",
    6: "Group ride",
}

# Stage → progress fraction, so the indeterminate training run still shows
# forward motion through its known phases.
_STAGE_FRACTION = {
    "preparing": 0.15,
    "training": 0.55,
    "evaluating": 0.85,
    "saving": 0.95,
    "done": 1.0,
}


def target_label(target: str) -> str:
    meta = TARGET_META.get(target, {})
    return f"{meta.get('icon', '')} {meta.get('label', target)}".strip()


def format_target_value(value: float, target: str) -> str:
    """Render a predicted value with the target's unit, e.g. ``$14.20`` / ``18.3 min``."""
    meta = TARGET_META.get(target, {})
    unit = meta.get("unit", "")
    if meta.get("unit_pos") == "prefix":
        return f"{unit}{value:,.2f}"
    return f"{value:,.1f} {unit}".strip()


def stage_fraction(status: str) -> float:
    return _STAGE_FRACTION.get(status, 0.1)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

_MODELING_STYLE = """
<style>
/* Tab bar: larger, pill-like, easier to scan */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, #EEF2FF 0%, #F8FAFC 100%);
}
/* Accent chips used for family / device / target badges */
.mdl-chip {
  display: inline-block; padding: 3px 11px; border-radius: 999px;
  font-size: .78rem; font-weight: 700; letter-spacing: .01em; line-height: 1.4;
}
.mdl-chip.gpu   { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.mdl-chip.cpu   { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }
.mdl-chip.xgb   { background: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; }
.mdl-chip.mllib { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.mdl-chip.star  { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
/* The headline prediction card in the Inference tab */
.mdl-pred {
  border-radius: 16px; padding: 22px 24px; margin-top: 6px;
  background: linear-gradient(120deg, #1E3A8A 0%, #4338CA 55%, #0E7490 100%);
  color: #fff; box-shadow: 0 14px 34px -16px rgba(30,64,175,.55);
}
.mdl-pred .lbl { opacity: .9; font-size: .9rem; font-weight: 600; }
.mdl-pred .val { font-size: 2.3rem; font-weight: 800; line-height: 1.1; margin-top: 2px; }
</style>
"""


def inject_style() -> None:
    import streamlit as st

    st.markdown(_MODELING_STYLE, unsafe_allow_html=True)


def chip(text: str, kind: str) -> str:
    """Return the HTML for a colored accent chip (caller renders with markdown)."""
    import html

    return f'<span class="mdl-chip {kind}">{html.escape(text)}</span>'


# --------------------------------------------------------------------------- #
# Chart / metric renderers
# --------------------------------------------------------------------------- #

def render_metric_cards(metrics: dict, *, elapsed: float | None = None,
                        n_train: int | None = None, n_test: int | None = None) -> None:
    """RMSE / MAE / R² as a row of metric cards, with optional run facts."""
    import streamlit as st

    r2 = metrics.get("R2")
    cols = st.columns(3)
    cols[0].metric("RMSE", f"{metrics.get('RMSE', 0):.3f}", border=True,
                   help="Root mean squared error — lower is better, in target units.")
    cols[1].metric("MAE", f"{metrics.get('MAE', 0):.3f}", border=True,
                   help="Mean absolute error — typical miss, in target units.")
    cols[2].metric("R²", f"{r2:.4f}" if r2 is not None else "—", border=True,
                   help="Fraction of variance explained (1.0 is perfect).")
    if elapsed is not None or n_train is not None:
        bits = []
        if elapsed is not None:
            bits.append(f"⏱ {elapsed:.1f}s")
        if n_train is not None:
            bits.append(f"{n_train:,} train rows")
        if n_test is not None:
            bits.append(f"{n_test:,} test rows")
        st.caption("  ·  ".join(bits))


def _style(fig, height: int = 320):
    fig.update_layout(
        margin=dict(l=10, r=10, t=34, b=10), height=height,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF2F7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7", zeroline=False)
    return fig


def render_eval_plots(report: dict, target: str) -> None:
    """Actual-vs-predicted, residuals, and feature importance for one report."""
    import plotly.express as px
    import streamlit as st

    sample = report.get("prediction_sample")
    unit = TARGET_META.get(target, {}).get("label", target)

    if sample is not None and len(sample):
        c1, c2 = st.columns(2)
        with c1:
            lo = float(min(sample["actual"].min(), sample["predicted"].min()))
            hi = float(max(sample["actual"].max(), sample["predicted"].max()))
            fig = px.scatter(sample, x="actual", y="predicted",
                             opacity=0.55, title="Predicted vs actual")
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(color="#EF4444", dash="dash", width=1.5))
            fig.update_traces(marker=dict(color="#4338CA", size=7))
            fig.update_layout(xaxis_title=f"Actual {unit}", yaxis_title=f"Predicted {unit}")
            st.plotly_chart(_style(fig), use_container_width=True, theme="streamlit")
        with c2:
            fig = px.histogram(sample, x="error", nbins=40, title="Residuals (predicted − actual)")
            fig.update_traces(marker=dict(color="#0E7490"))
            fig.update_layout(xaxis_title=f"Error ({unit})", yaxis_title="Count")
            st.plotly_chart(_style(fig), use_container_width=True, theme="streamlit")

    fi = report.get("feature_importance")
    if fi is not None and len(fi):
        top = fi.head(12).iloc[::-1]
        fig = px.bar(top, x="importance", y="feature", orientation="h",
                     title="Feature importance (top 12)")
        fig.update_traces(marker=dict(color="#6D28D9"))
        st.plotly_chart(_style(fig, height=380), use_container_width=True, theme="streamlit")

    if sample is not None and len(sample):
        with st.expander("Prediction sample (actual vs predicted)",
                         icon=":material/table_chart:"):
            st.dataframe(sample, use_container_width=True, hide_index=True)


def render_saved_comparison(metas: list[dict]) -> None:
    """A compact comparison table across saved models."""
    import pandas as pd
    import streamlit as st

    rows = []
    for m in metas:
        met = m.get("metrics", {})
        rows.append({
            "Model": m.get("model_name", m.get("model_key", "—")),
            "Target": TARGET_META.get(m.get("target", ""), {}).get("label", m.get("target", "—")),
            "Device": "GPU" if m.get("device") == "cuda" else "CPU",
            "RMSE": round(met.get("RMSE", 0), 3),
            "MAE": round(met.get("MAE", 0), 3),
            "R²": round(met.get("R2", 0), 4),
            "Train rows": m.get("n_train"),
            "Trained": m.get("trained_at", "—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"R²": st.column_config.NumberColumn(format="%.4f")})
