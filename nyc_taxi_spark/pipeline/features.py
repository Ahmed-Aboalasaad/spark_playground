"""Feature engineering.

Real responsibility: engineer the model features, split into train/test, and
impute the history-based features. Output is the train/test pair the modeling
stage consumes. Targets are ``fare_amount`` and the derived ``trip_duration_min``
(``config.ML_TARGETS``).

Mirrors ``notebooks/04_Preprocessing_FeatureEngineering.ipynb``:

* cyclical encodings so hour 23 sits next to hour 0,
* ``is_airport`` / ``is_weekend`` / ``is_rush_hour`` regime flags,
* ``log_distance`` for the sub-linear distance→fare relationship,
* per-zone lag features (yesterday's and the trailing-7-day average duration)
  whose window ends at ``-1`` day so today's trips never leak into their own
  features,
* a **time-based** split (most recent slice = held-out future), then imputation
  of the lag features using **train-only** means.

pyspark is imported lazily so the module imports without a Spark runtime.
"""
from __future__ import annotations

import math
from typing import Any

from config import (
    AIRPORT_ZONE_IDS,
    DEFAULT_TEST_FRACTION,
    FEATURE_COLUMNS,
    ML_TARGETS,
    RUSH_HOURS_AM,
    RUSH_HOURS_PM,
)
from pipeline.cleaning import add_derived_columns

# Feature groupings, surfaced on the Modeling page's "features used" display.
NUMERIC_FEATURES = (
    "trip_distance", "log_distance", "passenger_count",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "zone_prev_day_avg", "zone_7d_avg",
)
CATEGORICAL_FEATURES = (
    "PULocationID", "DOLocationID", "RatecodeID",
    "pickup_hour", "pickup_dayofweek", "pickup_month",
    "is_weekend", "is_rush_hour", "is_airport",
)

# One-line rationale per engineered feature, for the Preprocessing page.
FEATURE_RATIONALE: dict[str, str] = {
    "pickup_hour": "Traffic and demand follow a daily cycle.",
    "pickup_dayofweek": "Weekday vs weekend demand differs sharply.",
    "pickup_month": "Seasonality across the year.",
    "is_weekend": "A step change the model gets as a direct 0/1 flag.",
    "is_rush_hour": "Rush-hour congestion is a regime, not a smooth trend.",
    "hour_sin / hour_cos": "Time is circular — hour 23 borders hour 0.",
    "dow_sin / dow_cos": "Same circular trick for day of week.",
    "is_airport": "JFK/LGA/EWR carry flat rates + fixed fees (3 zone IDs).",
    "log_distance": "Fare and duration grow sub-linearly with distance.",
    "zone_prev_day_avg": "Yesterday's avg duration in the pickup zone.",
    "zone_7d_avg": "The zone's trailing-7-day congestion history.",
}


# The feature set the models actually train on -- the notebook's ``FE_FEATURES``.
# Deliberately excludes the zone lag features: they need per-zone history, which
# a single inference row can't supply, and the notebook's best model (R²≈0.97)
# didn't use them. Same set works for both the fare and duration targets.
MODEL_FEATURES: tuple[str, ...] = (
    "trip_distance", "passenger_count", "RatecodeID", "PULocationID", "DOLocationID",
    "pickup_hour", "pickup_dayofweek", "pickup_month",
    "is_weekend", "is_rush_hour", "is_airport", "log_distance",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)


def engineer_basic_features(df: Any) -> Any:
    """Add the per-row engineered columns (time, flags, cyclical, log_distance).

    Everything in :data:`MODEL_FEATURES` is produced here. No cross-row history,
    so this is safe on a single inference row as well as on the full dataset.
    """
    from pyspark.sql import functions as F

    df = add_derived_columns(df)

    am_lo, am_hi = RUSH_HOURS_AM
    pm_lo, pm_hi = RUSH_HOURS_PM

    df = (
        df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
    )
    df = (
        df.withColumn("is_weekend", F.col("pickup_dayofweek").isin(1, 7).cast("int"))
        .withColumn(
            "is_rush_hour",
            (
                (~F.col("pickup_dayofweek").isin(1, 7))
                & (F.col("pickup_hour").between(am_lo, am_hi)
                   | F.col("pickup_hour").between(pm_lo, pm_hi))
            ).cast("int"),
        )
        .withColumn(
            "is_airport",
            (F.col("PULocationID").isin(*AIRPORT_ZONE_IDS)
             | F.col("DOLocationID").isin(*AIRPORT_ZONE_IDS)).cast("int"),
        )
        .withColumn("log_distance", F.log1p("trip_distance"))
        .withColumn("hour_sin", F.sin(2 * math.pi * F.col("pickup_hour") / 24))
        .withColumn("hour_cos", F.cos(2 * math.pi * F.col("pickup_hour") / 24))
        .withColumn("dow_sin", F.sin(2 * math.pi * (F.col("pickup_dayofweek") - 1) / 7))
        .withColumn("dow_cos", F.cos(2 * math.pi * (F.col("pickup_dayofweek") - 1) / 7))
    )
    return df


def engineer_features(df: Any) -> Any:
    """Add every engineered column, including the per-zone lag history features.

    Used by the Preprocessing page's full feature showcase and the train/test
    split. Models train on :data:`MODEL_FEATURES` (the lag-free subset).
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    df = engineer_basic_features(df)

    # Per-zone daily history, joined back. Window ends at -1 day (strictly past)
    # so a trip's own day never leaks into its features.
    zone_daily = df.groupBy("PULocationID", "pickup_date").agg(
        F.avg("trip_duration_min").alias("zone_day_avg_duration")
    )
    wz = Window.partitionBy("PULocationID").orderBy("pickup_date")
    zone_daily = (
        zone_daily.withColumn("zone_prev_day_avg", F.lag("zone_day_avg_duration", 1).over(wz))
        .withColumn("zone_7d_avg", F.avg("zone_day_avg_duration").over(wz.rowsBetween(-7, -1)))
    )
    df = df.join(
        zone_daily.select("PULocationID", "pickup_date", "zone_prev_day_avg", "zone_7d_avg"),
        on=["PULocationID", "pickup_date"],
        how="left",
    )
    return df


def inference_frame(spark: Any, inputs: dict) -> Any:
    """Build a one-row engineered DataFrame from raw user inputs for prediction.

    ``inputs`` carries the raw, user-supplied values (distance, passenger count,
    rate code, pickup/dropoff zone IDs, and a pickup ``datetime``); this derives
    every :data:`MODEL_FEATURES` column from them.
    """
    from pyspark.sql import Row

    pickup = inputs["pickup_datetime"]
    row = Row(
        tpep_pickup_datetime=pickup,
        tpep_dropoff_datetime=pickup,  # duration isn't a model feature; unused
        passenger_count=float(inputs["passenger_count"]),
        trip_distance=float(inputs["trip_distance"]),
        RatecodeID=float(inputs["RatecodeID"]),
        PULocationID=int(inputs["PULocationID"]),
        DOLocationID=int(inputs["DOLocationID"]),
    )
    return engineer_basic_features(spark.createDataFrame([row]))


def _split_date(df: Any, test_fraction: float) -> Any:
    """The date cutoff whose right-hand slice is ~``test_fraction`` of the rows.

    Adaptive so the split works for whatever months are loaded, while keeping
    the notebook's "train on the past, test on the future" property. Uses the
    ``(1 - test_fraction)`` **percentile** of the row dates rather than
    ``min + fraction × span`` so that a handful of stray-dated rows (a known TLC
    quirk) can't drag the cutoff far outside the real data. Returns a
    ``datetime.date`` (or ``None`` when every row shares one date).
    """
    from datetime import date, timedelta

    from pyspark.sql import functions as F

    epoch = date(1970, 1, 1)
    day = F.datediff(F.col("pickup_date"), F.lit(epoch))  # integer day number
    row = df.select(
        F.min(day).alias("lo"),
        F.max(day).alias("hi"),
        F.percentile_approx(day, 1.0 - test_fraction).alias("cut"),
    ).first()
    if row["lo"] is None or row["hi"] is None or row["lo"] == row["hi"] or row["cut"] is None:
        return None
    return epoch + timedelta(days=int(row["cut"]))


def build_features(df: Any, test_fraction: float = DEFAULT_TEST_FRACTION, seed: int = 42) -> dict:
    """Assemble features and split into train/test sets.

    Returns
    -------
    dict
        ``{"train": <df>, "test": <df>, "info": {...}}``.
    """
    from pyspark.sql import functions as F

    feat = engineer_features(df)
    cutoff = _split_date(feat, test_fraction)

    if cutoff is None:
        # Degenerate single-day span: everything trains, no future holdout.
        train, test = feat, feat.limit(0)
    else:
        train = feat.filter(F.col("pickup_date") < F.lit(cutoff))
        test = feat.filter(F.col("pickup_date") >= F.lit(cutoff))

    # Impute the lag features with TRAIN-ONLY means, then reuse on test, so no
    # future information leaks backward into the fill value.
    fill_means: dict[str, float] = {}
    for col in ("zone_prev_day_avg", "zone_7d_avg"):
        val = train.selectExpr(f"avg({col})").first()[0]
        fill_means[col] = float(val) if val is not None else 0.0
    train = train.na.fill(fill_means)
    test = test.na.fill(fill_means)

    n_train = train.count()
    n_test = test.count()

    info = {
        "targets": list(ML_TARGETS),
        "target": ML_TARGETS[0],
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "test_fraction": test_fraction,
        "split_date": str(cutoff) if cutoff else None,
        "fill_means": fill_means,
        "n_train": n_train,
        "n_test": n_test,
    }
    return {"train": train, "test": test, "info": info}


# --------------------------------------------------------------------------- #
# Feature-signal aggregations (small pandas frames for the Preprocessing page).
# Each justifies one engineered feature by showing the signal it carries.
# Aggregation happens in Spark; only the tiny grouped result reaches pandas.
# --------------------------------------------------------------------------- #

def signal_by_hour(df: Any):
    """Trips, avg fare, avg duration by pickup hour → justifies the time features."""
    from pyspark.sql import functions as F

    df = add_derived_columns(df).withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    return (
        df.groupBy("pickup_hour").agg(
            F.count("*").alias("trips"),
            F.avg("fare_amount").alias("avg_fare"),
            F.avg("trip_duration_min").alias("avg_duration"),
        ).orderBy("pickup_hour").toPandas()
    )


def signal_by_weekday(df: Any):
    """Trips and avg duration by day of week → justifies weekday/weekend features."""
    from pyspark.sql import functions as F

    df = add_derived_columns(df).withColumn("dow", F.dayofweek("tpep_pickup_datetime"))
    pdf = (
        df.groupBy("dow").agg(
            F.count("*").alias("trips"),
            F.avg("trip_duration_min").alias("avg_duration"),
        ).orderBy("dow").toPandas()
    )
    names = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
    pdf["weekday"] = pdf["dow"].map(names)
    return pdf


def signal_airport(df: Any):
    """Avg fare/distance/trips split by the airport flag → justifies is_airport."""
    from pyspark.sql import functions as F

    df = add_derived_columns(df).withColumn(
        "is_airport",
        (F.col("PULocationID").isin(*AIRPORT_ZONE_IDS)
         | F.col("DOLocationID").isin(*AIRPORT_ZONE_IDS)).cast("int"),
    )
    pdf = (
        df.groupBy("is_airport").agg(
            F.avg("fare_amount").alias("avg_fare"),
            F.avg("trip_distance").alias("avg_distance"),
            F.count("*").alias("trips"),
        ).orderBy("is_airport").toPandas()
    )
    pdf["group"] = pdf["is_airport"].map({0: "Non-airport", 1: "Airport"})
    return pdf


def distance_fare_sample(df: Any, fraction: float = 0.001, seed: int = 42):
    """A small random sample of (distance, fare) → justifies log_distance."""
    df = add_derived_columns(df)
    return (
        df.select("trip_distance", "fare_amount")
        .sample(fraction=fraction, seed=seed)
        .limit(4000)
        .toPandas()
    )


def daily_trip_series(df: Any):
    """Trips per calendar day → the whole experimental design in one series."""
    from pyspark.sql import functions as F

    df = add_derived_columns(df)
    return df.groupBy("pickup_date").agg(F.count("*").alias("trips")).orderBy("pickup_date").toPandas()
