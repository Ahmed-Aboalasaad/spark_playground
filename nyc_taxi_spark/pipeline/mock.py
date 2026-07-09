"""Mock-data helpers shared by every skeletal pipeline module.

While the real Spark implementations are being written in notebooks, the
pipeline returns placeholder data so the whole application is navigable and the
UI wiring can be tested end-to-end. The rule for placeholders, agreed up front:

* Correct **shape** -- right columns, right dtypes, right number of rows for
  categorical breakdowns (24 hours, 7 weekdays, 12 months).
* Zero **substance** -- every numeric value is 0.
* Impossible to **mistake** for real output -- a ``PLACEHOLDER`` label rides
  along with the data and the UI renders a loud banner.

Everything here builds pandas frames rather than Spark DataFrames on purpose:
the mock data is tiny, it needs no cluster, and the UI collects Spark
aggregations to pandas for plotting anyway. When a real implementation lands it
swaps the pandas mock for a Spark aggregation returning the same columns.
"""
from __future__ import annotations

import calendar
from typing import Sequence

import pandas as pd

# A single, obvious marker string. Search the codebase for this to find every
# spot still backed by mock data.
PLACEHOLDER_LABEL = "PLACEHOLDER"


def zero_series_frame(
    categories: Sequence, category_col: str, value_col: str = "value"
) -> pd.DataFrame:
    """A two-column frame: the given categories against an all-zero value column.

    Used for every "X by category" breakdown (by hour, weekday, month, zone).
    """
    return pd.DataFrame(
        {category_col: list(categories), value_col: [0] * len(categories)}
    )


def hours_frame(value_col: str = "trips") -> pd.DataFrame:
    """0..23 against zeros. For 'trips by hour', 'revenue by hour', etc."""
    return zero_series_frame(range(24), "hour", value_col)


def weekdays_frame(value_col: str = "trips") -> pd.DataFrame:
    """Mon..Sun against zeros."""
    names = list(calendar.day_name)  # Monday .. Sunday
    return zero_series_frame(names, "weekday", value_col)


def months_frame(value_col: str = "trips") -> pd.DataFrame:
    """Jan..Dec against zeros."""
    names = list(calendar.month_name)[1:]  # drop empty index 0
    return zero_series_frame(names, "month", value_col)


def top_n_frame(
    n: int, category_col: str = "zone", value_col: str = "trips"
) -> pd.DataFrame:
    """N placeholder rows for 'top locations' style breakdowns.

    Category labels are explicitly named as placeholders so a zeroed bar chart
    can never be mistaken for real zone data.
    """
    labels = [f"{PLACEHOLDER_LABEL} zone {i + 1}" for i in range(n)]
    return zero_series_frame(labels, category_col, value_col)


def distribution_frame(
    bins: int = 20, category_col: str = "bin", value_col: str = "count"
) -> pd.DataFrame:
    """A histogram-shaped frame with zero counts across ``bins`` buckets."""
    labels = [f"bin {i + 1}" for i in range(bins)]
    return zero_series_frame(labels, category_col, value_col)


def timeseries_frame(periods: int = 12, value_col: str = "trips") -> pd.DataFrame:
    """A date-indexed series of zeros for 'over time' analyses."""
    idx = pd.date_range("2023-01-01", periods=periods, freq="MS")
    return pd.DataFrame({"period": idx, value_col: [0] * periods})


def zero_metrics(*names: str) -> dict[str, float]:
    """A metrics dict with every requested metric set to zero."""
    return {name: 0 for name in names}
