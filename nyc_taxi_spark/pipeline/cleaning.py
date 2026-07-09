"""Data cleaning and preprocessing.

Real responsibility: remove invalid records, handle missing values, convert
data types, generate derived columns (trip duration, speed, pickup hour /
weekday / month), and produce the cleaned dataset. Output is the clean Spark
DataFrame plus a summary describing what changed.

Skeleton behaviour: returns the input unchanged and a zeroed cleaning summary,
tagged as placeholder.
"""
from __future__ import annotations

from typing import Any

from config import PLACEHOLDER_MODE
from pipeline.mock import PLACEHOLDER_LABEL, zero_metrics


def clean_dataset(df: Any) -> dict:
    """Produce the cleaned dataset and a summary of the cleaning applied.

    Returns
    -------
    dict
        ``{"df": <cleaned DataFrame>, "summary": {...}}``. The summary feeds the
        Data Preprocessing page.
    """
    if PLACEHOLDER_MODE:
        return {
            "df": df,  # unchanged passthrough while stubbed
            "summary": {
                "status": PLACEHOLDER_LABEL,
                "n_input": 0,
                "n_output": 0,
                "n_removed": 0,
                "pct_removed": 0.0,
                "n_duplicates": 0,
                "operations": [
                    f"{PLACEHOLDER_LABEL}: no cleaning applied yet",
                ],
                **zero_metrics("missing_before", "missing_after"),
            },
        }

    raise NotImplementedError("Real cleaning not yet implemented.")


def data_quality_report(df: Any) -> dict:
    """Compute data-quality metrics for the Data Preprocessing page.

    Real version reports per-column missing counts/percentages, duplicate
    counts, and outlier distributions. Stub returns zeros.
    """
    if PLACEHOLDER_MODE:
        return {
            "status": PLACEHOLDER_LABEL,
            "missing_per_column": {},
            "n_duplicates": 0,
            "n_records": 0,
        }

    raise NotImplementedError("Real quality report not yet implemented.")
