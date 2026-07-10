"""Data cleaning and preprocessing.

Real responsibility: remove invalid records, handle missing values, convert
data types, generate derived columns (trip duration, pickup date), and produce
the cleaned dataset. Output is the clean Spark DataFrame plus a summary
describing what changed.

The rules here mirror ``notebooks/04_Preprocessing_FeatureEngineering.ipynb``
exactly: a set of fixed, hand-written filters applied to the whole dataset. Every
threshold lives in :mod:`config` with its rationale, and the summary this module
returns feeds the "Data Preprocessing" page so each decision is documented and
defensible rather than silent.

pyspark is imported lazily inside the functions so this module stays importable
without a Spark runtime (tests, tooling, smoke checks).
"""
from __future__ import annotations

from typing import Any

from config import (
    DATASET_MIN_YEAR,
    FARE_MAX,
    FARE_MIN,
    PICKUP_COL,
    DROPOFF_COL,
    TRIP_DISTANCE_MAX,
    TRIP_DURATION_MAX,
    TRIP_DURATION_MIN,
    UNKNOWN_LOCATION_IDS,
)

# Columns we surface in the numeric summary, when present on the frame.
_SUMMARY_COLUMNS = (
    "fare_amount", "total_amount", "tip_amount", "trip_distance",
    "passenger_count", "trip_duration_min",
)


def add_derived_columns(df: Any) -> Any:
    """Add ``trip_duration_min`` and ``pickup_date`` used by cleaning/features.

    ``trip_duration_min`` doubles as the duration regression target and as the
    basis for the duration sanity filter (it also enforces dropoff > pickup).
    Idempotent: columns already present are left untouched.
    """
    from pyspark.sql import functions as F

    if "trip_duration_min" not in df.columns:
        df = df.withColumn(
            "trip_duration_min",
            (F.unix_timestamp(DROPOFF_COL) - F.unix_timestamp(PICKUP_COL)) / 60.0,
        )
    if "pickup_date" not in df.columns:
        df = df.withColumn("pickup_date", F.to_date(PICKUP_COL))
    return df


def clean_dataset(df: Any) -> dict:
    """Produce the cleaned dataset and a summary of the cleaning applied.

    Returns
    -------
    dict
        ``{"df": <cleaned DataFrame>, "summary": {...}}``. The summary feeds the
        Data Preprocessing page.
    """
    from datetime import datetime

    from pyspark.sql import functions as F

    df = add_derived_columns(df)
    n_input = df.count()

    # Upper bound on a plausible pickup year: next calendar year (guards against
    # stray far-future timestamps without hardcoding a ceiling that ages out).
    max_year = datetime.now().year + 1

    # The keep-conditions, straight from the notebook. NULLs fail every
    # comparison in Spark, so rows with null fare/distance/passenger_count are
    # dropped implicitly here -- intentional: those nulls are data-entry
    # failures, not meaningful absence.
    keep = (
        (F.col("fare_amount") >= FARE_MIN) & (F.col("fare_amount") <= FARE_MAX)
        & (F.col("trip_distance") > 0) & (F.col("trip_distance") < TRIP_DISTANCE_MAX)
        & (F.col("passenger_count") > 0)
        & (F.col("trip_duration_min") >= TRIP_DURATION_MIN)
        & (F.col("trip_duration_min") <= TRIP_DURATION_MAX)
        & (~F.col("PULocationID").isin(*UNKNOWN_LOCATION_IDS))
        & (~F.col("DOLocationID").isin(*UNKNOWN_LOCATION_IDS))
        & (F.year("pickup_date") >= DATASET_MIN_YEAR)
        & (F.year("pickup_date") <= max_year)
    )

    # Per-rule "rows failing this rule" counts, computed in one pass. Rules
    # overlap (a row can fail several), so these don't sum to n_removed -- the
    # page labels them as a diagnostic breakdown, not a partition.
    def _fails(cond) -> Any:
        return F.sum(F.when(~cond | cond.isNull(), 1).otherwise(0))

    breakdown_row = df.select(
        _fails((F.col("fare_amount") >= FARE_MIN) & (F.col("fare_amount") <= FARE_MAX))
        .alias("fare_out_of_range"),
        _fails((F.col("trip_distance") > 0) & (F.col("trip_distance") < TRIP_DISTANCE_MAX))
        .alias("distance_out_of_range"),
        _fails(F.col("passenger_count") > 0).alias("passengers_missing_or_zero"),
        _fails((F.col("trip_duration_min") >= TRIP_DURATION_MIN)
               & (F.col("trip_duration_min") <= TRIP_DURATION_MAX))
        .alias("duration_out_of_range"),
        _fails((~F.col("PULocationID").isin(*UNKNOWN_LOCATION_IDS))
               & (~F.col("DOLocationID").isin(*UNKNOWN_LOCATION_IDS)))
        .alias("unknown_zone"),
        _fails((F.year("pickup_date") >= DATASET_MIN_YEAR)
               & (F.year("pickup_date") <= max_year))
        .alias("invalid_pickup_date"),
    ).first()
    breakdown = {k: int(v or 0) for k, v in breakdown_row.asDict().items()}

    cleaned = df.filter(keep)
    n_output = cleaned.count()
    n_removed = n_input - n_output
    pct_removed = (n_removed / n_input * 100.0) if n_input else 0.0

    summary = {
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "pct_removed": pct_removed,
        "pct_kept": 100.0 - pct_removed,
        "removal_breakdown": breakdown,
        "operations": [
            ("Derived `trip_duration_min` from pickup/dropoff timestamps "
             "(also enforces dropoff after pickup)"),
            (f"Kept fares in ${FARE_MIN:g}–${FARE_MAX:g} — below the flag-drop "
             "minimum are refunds/voids; above are meter glitches"),
            (f"Kept trip distance in 0–{TRIP_DISTANCE_MAX:g} mi — 0 is a GPS/meter "
             "error, 100+ is not a yellow-cab trip"),
            "Dropped trips with missing or zero passenger count",
            (f"Kept trip duration in {TRIP_DURATION_MIN:g}–{TRIP_DURATION_MAX:g} min "
             "— shorter is an accidental meter start, longer a forgotten meter"),
            (f"Dropped trips touching the sentinel 'Unknown' zones "
             f"{UNKNOWN_LOCATION_IDS} (not real geography)"),
            (f"Dropped rows with an implausible pickup year (outside "
             f"{DATASET_MIN_YEAR}–{max_year}) — stray TLC timestamps like 2002/2098"),
        ],
    }
    return {"df": cleaned, "summary": summary}


def sample_for_dist(df: Any, fraction: float = 0.01, cap: int = 6000):
    """A small random sample of the three key numeric columns for plotting.

    Used by the "before vs after cleaning" distribution overlays. Aggregating a
    histogram shape converges long before a full scan, so we move only a tiny
    sample into pandas.
    """
    df = add_derived_columns(df)
    return (
        df.select("fare_amount", "trip_duration_min", "trip_distance")
        .sample(fraction=fraction, seed=42)
        .limit(cap)
        .toPandas()
    )


def data_quality_report(df: Any) -> dict:
    """Compute data-quality metrics for the Data Preprocessing page.

    Reports per-column missing counts/percentages, the duplicate-row count, a
    numeric summary, and a small fare/distance sanity block (the implausible
    records that motivate the cleaning thresholds).
    """
    from pyspark.sql import functions as F

    df = add_derived_columns(df)
    n_records = df.count()

    # Missing counts for every column, one pass.
    missing_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    missing_row = df.select(*missing_exprs).first().asDict()
    missing_per_column = {c: int(missing_row.get(c, 0) or 0) for c in df.columns}
    missing_pct = {
        c: (missing_per_column[c] / n_records * 100.0 if n_records else 0.0)
        for c in df.columns
    }

    # Sanity block (the implausible records the cleaning thresholds target) plus
    # an approximate duplicate-row estimate -- all in a single pass.
    #
    # Duplicates use ``approx_count_distinct`` over a hash of every column
    # (HyperLogLog): fixed, tiny memory and no shuffle. A true ``distinct()``
    # over tens of millions of rows is a full shuffle that OOMs and kills the
    # local Spark driver, so it is deliberately avoided here.
    agg_row = df.select(
        F.sum(F.when((F.col("fare_amount") <= 0) | (F.col("fare_amount") > FARE_MAX), 1)
              .otherwise(0)).alias("implausible_fare"),
        F.sum(F.when(F.col("trip_distance") <= 0, 1).otherwise(0)).alias("zero_distance"),
        F.sum(F.when(F.col("trip_distance") >= TRIP_DISTANCE_MAX, 1).otherwise(0))
        .alias("extreme_distance"),
        F.sum(F.when((F.col("passenger_count").isNull()) | (F.col("passenger_count") <= 0), 1)
              .otherwise(0)).alias("bad_passenger_count"),
        F.sum(F.when((F.col("trip_duration_min") < TRIP_DURATION_MIN)
                     | (F.col("trip_duration_min") > TRIP_DURATION_MAX), 1)
              .otherwise(0)).alias("bad_duration"),
        F.approx_count_distinct(F.hash(*[F.col(c) for c in df.columns]))
        .alias("approx_distinct"),
    ).first().asDict()
    approx_distinct = int(agg_row.pop("approx_distinct") or 0)
    sanity = {k: int(v or 0) for k, v in agg_row.items()}
    n_duplicates = max(0, n_records - approx_distinct)

    # Numeric summary for the columns present. describe() -> small pandas frame.
    summary_cols = [c for c in _SUMMARY_COLUMNS if c in df.columns]
    numeric_summary = None
    if summary_cols:
        numeric_summary = (
            df.select(*summary_cols)
            .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
            .toPandas()
            .set_index("summary")
        )

    return {
        "n_records": n_records,
        "n_columns": len(df.columns),
        "n_duplicates": n_duplicates,
        "missing_per_column": missing_per_column,
        "missing_pct_per_column": missing_pct,
        "sanity": sanity,
        "numeric_summary": numeric_summary,
    }
