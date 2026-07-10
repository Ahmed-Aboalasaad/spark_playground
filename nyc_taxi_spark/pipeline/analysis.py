"""Analysis engine.

Executes analytical queries, computes summary metrics, and returns aggregated,
visualization-ready pandas frames. Aggregation happens in Spark; only the small
grouped result is collected to pandas for Plotly.

Built around a **registry**: each analysis is a small object describing its
family, title, chart type, and a producer that returns ``(frame, metrics)``.
Adding an analysis means registering one entry — nothing existing changes, which
satisfies the "extend by addition, not modification" requirement.

Every producer derives what it needs (pickup hour/weekday/month, duration,
speed) from the base trip columns, so analyses run on either the Raw or the
Cleaned dataset without assuming feature-engineered columns are present.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import pandas as pd

from config import AIRPORT_LOCATION_IDS, AIRPORT_ZONE_IDS, UNKNOWN_LOCATION_IDS
from pipeline import zones


class Family(str, Enum):
    DEMAND = "Demand"
    GEOGRAPHIC = "Geographic"
    TRIP = "Trip"
    REVENUE = "Revenue"
    PASSENGER = "Passenger"
    AIRPORT = "Airport"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    HISTOGRAM = "histogram"
    METRIC = "metric"  # single-number analyses (e.g. total revenue)


# A producer takes the active Spark DataFrame and returns (frame, metrics).
Producer = Callable[[Any], "tuple[pd.DataFrame, dict]"]


@dataclass(frozen=True)
class Analysis:
    """One registered analysis."""

    key: str
    family: Family
    title: str
    chart: ChartType
    producer: Producer
    x: str | None = None       # x column for the chart
    y: str | None = None       # y column for the chart
    description: str = ""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, Analysis] = {}


def register(analysis: Analysis) -> Analysis:
    """Add an analysis to the registry, guarding against duplicate keys."""
    if analysis.key in _REGISTRY:
        raise ValueError(f"Duplicate analysis key: {analysis.key}")
    _REGISTRY[analysis.key] = analysis
    return analysis


def get_analysis(key: str) -> Analysis:
    return _REGISTRY[key]


def analyses_in(family: Family) -> list[Analysis]:
    return [a for a in _REGISTRY.values() if a.family is family]


def all_families() -> list[Family]:
    return list(Family)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

_WEEKDAY_NAMES = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_AIRPORT_CODE = {v: k for k, v in AIRPORT_LOCATION_IDS.items()}  # id -> "JFK"/"LGA"/"EWR"


def _duration_col():
    from pyspark.sql import functions as F

    return (F.unix_timestamp("tpep_dropoff_datetime")
            - F.unix_timestamp("tpep_pickup_datetime")) / 60.0


def _fmt_int(n) -> str:
    return f"{int(n or 0):,}"


def _fmt_money(x) -> str:
    return f"${float(x or 0):,.2f}"


def _zone_names() -> dict[int, str]:
    zdf = zones.load_zone_lookup()
    return {int(r.LocationID): f"{r.Zone} ({r.Borough})" for r in zdf.itertuples()}


# --------------------------------------------------------------------------- #
# Demand
# --------------------------------------------------------------------------- #

def _trips_by_hour(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.hour("tpep_pickup_datetime").alias("hour"))
             .agg(F.count("*").alias("trips")).orderBy("hour").toPandas())
    peak = int(pdf.loc[pdf["trips"].idxmax(), "hour"]) if len(pdf) else 0
    return pdf, {"total_trips": _fmt_int(pdf["trips"].sum()), "peak_hour": f"{peak:02d}:00"}


def _trips_by_weekday(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.dayofweek("tpep_pickup_datetime").alias("dow"))
             .agg(F.count("*").alias("trips")).orderBy("dow").toPandas())
    pdf["weekday"] = pdf["dow"].map(_WEEKDAY_NAMES)
    busiest = pdf.loc[pdf["trips"].idxmax(), "weekday"] if len(pdf) else "—"
    return pdf[["weekday", "trips"]], {"total_trips": _fmt_int(pdf["trips"].sum()),
                                       "busiest_day": busiest}


def _trips_by_month(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.month("tpep_pickup_datetime").alias("m"))
             .agg(F.count("*").alias("trips")).orderBy("m").toPandas())
    pdf["month"] = pdf["m"].map(_MONTH_NAMES)
    return pdf[["month", "trips"]], {"total_trips": _fmt_int(pdf["trips"].sum())}


def _trips_over_time(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.to_date("tpep_pickup_datetime").alias("period"))
             .agg(F.count("*").alias("trips")).orderBy("period").toPandas())
    peak = pdf.loc[pdf["trips"].idxmax(), "period"] if len(pdf) else "—"
    return pdf, {"total_trips": _fmt_int(pdf["trips"].sum()),
                 "days_covered": _fmt_int(len(pdf)), "busiest_day": str(peak)}


# --------------------------------------------------------------------------- #
# Geographic
# --------------------------------------------------------------------------- #

def _top_by_location(df: Any, id_col: str, n: int = 12):
    from pyspark.sql import functions as F

    pdf = (df.filter(~F.col(id_col).isin(*UNKNOWN_LOCATION_IDS))
             .groupBy(id_col).agg(F.count("*").alias("trips"))
             .orderBy(F.desc("trips")).limit(n).toPandas())
    names = _zone_names()
    pdf["zone"] = pdf[id_col].map(lambda i: names.get(int(i), f"Zone {i}"))
    distinct = df.select(id_col).distinct().count()
    return pdf, distinct


def _top_pickups(df: Any):
    pdf, distinct = _top_by_location(df, "PULocationID")
    return pdf[["zone", "trips"]], {"distinct_zones": _fmt_int(distinct),
                                    "top_zone": pdf["zone"].iloc[0] if len(pdf) else "—"}


def _top_dropoffs(df: Any):
    pdf, distinct = _top_by_location(df, "DOLocationID")
    return pdf[["zone", "trips"]], {"distinct_zones": _fmt_int(distinct),
                                    "top_zone": pdf["zone"].iloc[0] if len(pdf) else "—"}


def _pickup_dropoff_compare(df: Any):
    from pyspark.sql import functions as F

    top, _ = _top_by_location(df, "PULocationID", n=10)
    ids = top["PULocationID"].tolist()
    drop = (df.filter(F.col("DOLocationID").isin(ids))
              .groupBy("DOLocationID").agg(F.count("*").alias("dropoffs")).toPandas())
    drop_map = dict(zip(drop["DOLocationID"], drop["dropoffs"]))
    top = top.rename(columns={"trips": "pickups"})
    top["dropoffs"] = top["PULocationID"].map(lambda i: int(drop_map.get(i, 0)))
    return top[["zone", "pickups", "dropoffs"]], {"zones_compared": _fmt_int(len(top))}


# --------------------------------------------------------------------------- #
# Trip
# --------------------------------------------------------------------------- #

def _distance_distribution(df: Any):
    from pyspark.sql import functions as F

    binned = df.filter((F.col("trip_distance") > 0) & (F.col("trip_distance") < 100)) \
               .withColumn("bin", F.when(F.col("trip_distance") >= 30, 30)
                           .otherwise(F.floor("trip_distance").cast("int")))
    pdf = binned.groupBy("bin").agg(F.count("*").alias("count")).orderBy("bin").toPandas()
    avg = df.select(F.avg("trip_distance")).first()[0]
    return pdf, {"avg_distance_mi": f"{float(avg or 0):.2f} mi"}


def _duration_distribution(df: Any):
    from pyspark.sql import functions as F

    d = df.withColumn("dur", _duration_col())
    binned = d.filter((F.col("dur") > 0) & (F.col("dur") <= 120)) \
              .withColumn("bin", F.when(F.col("dur") >= 60, 60)
                          .otherwise((F.floor(F.col("dur") / 2) * 2).cast("int")))
    pdf = binned.groupBy("bin").agg(F.count("*").alias("count")).orderBy("bin").toPandas()
    avg = d.select(F.avg("dur")).first()[0]
    return pdf, {"avg_duration_min": f"{float(avg or 0):.1f} min"}


def _avg_duration(df: Any):
    from pyspark.sql import functions as F

    avg = df.withColumn("dur", _duration_col()).select(F.avg("dur")).first()[0]
    return pd.DataFrame(), {"avg_duration_min": f"{float(avg or 0):.1f} min"}


def _avg_distance(df: Any):
    from pyspark.sql import functions as F

    avg = df.select(F.avg("trip_distance")).first()[0]
    return pd.DataFrame(), {"avg_distance_mi": f"{float(avg or 0):.2f} mi"}


def _avg_speed(df: Any):
    from pyspark.sql import functions as F

    d = df.withColumn("dur", _duration_col()).filter(F.col("dur") > 0)
    speed = d.select(F.avg(F.col("trip_distance") / (F.col("dur") / 60.0))).first()[0]
    return pd.DataFrame(), {"avg_speed_mph": f"{float(speed or 0):.1f} mph"}


# --------------------------------------------------------------------------- #
# Revenue (revenue = total_amount charged to riders)
# --------------------------------------------------------------------------- #

def _total_revenue(df: Any):
    from pyspark.sql import functions as F

    row = df.select(F.sum("total_amount").alias("rev"), F.count("*").alias("n")).first()
    return pd.DataFrame(), {"total_revenue": _fmt_money(row["rev"]),
                            "revenue_per_trip": _fmt_money((row["rev"] or 0) / (row["n"] or 1))}


def _revenue_by_hour(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.hour("tpep_pickup_datetime").alias("hour"))
             .agg(F.sum("total_amount").alias("revenue")).orderBy("hour").toPandas())
    return pdf, {"total_revenue": _fmt_money(pdf["revenue"].sum())}


def _revenue_by_weekday(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.dayofweek("tpep_pickup_datetime").alias("dow"))
             .agg(F.sum("total_amount").alias("revenue")).orderBy("dow").toPandas())
    pdf["weekday"] = pdf["dow"].map(_WEEKDAY_NAMES)
    return pdf[["weekday", "revenue"]], {"total_revenue": _fmt_money(pdf["revenue"].sum())}


def _revenue_by_month(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.groupBy(F.month("tpep_pickup_datetime").alias("m"))
             .agg(F.sum("total_amount").alias("revenue")).orderBy("m").toPandas())
    pdf["month"] = pdf["m"].map(_MONTH_NAMES)
    return pdf[["month", "revenue"]], {"total_revenue": _fmt_money(pdf["revenue"].sum())}


def _avg_fare(df: Any):
    from pyspark.sql import functions as F

    avg = df.select(F.avg("fare_amount")).first()[0]
    return pd.DataFrame(), {"avg_fare": _fmt_money(avg)}


def _avg_tip(df: Any):
    from pyspark.sql import functions as F

    row = df.select(F.avg("tip_amount").alias("t"),
                    F.avg(F.when(F.col("fare_amount") > 0,
                                 F.col("tip_amount") / F.col("fare_amount"))).alias("pct")).first()
    return pd.DataFrame(), {"avg_tip": _fmt_money(row["t"]),
                            "avg_tip_pct": f"{float(row['pct'] or 0) * 100:.1f}%"}


# --------------------------------------------------------------------------- #
# Passenger
# --------------------------------------------------------------------------- #

def _passenger_distribution(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.filter(F.col("passenger_count").isNotNull() & (F.col("passenger_count") <= 6))
             .groupBy(F.col("passenger_count").cast("int").alias("passenger_count"))
             .agg(F.count("*").alias("trips")).orderBy("passenger_count").toPandas())
    avg = df.select(F.avg("passenger_count")).first()[0]
    return pdf, {"avg_passengers": f"{float(avg or 0):.2f}"}


def _avg_fare_by_passengers(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.filter(F.col("passenger_count").isNotNull() & (F.col("passenger_count") <= 6))
             .groupBy(F.col("passenger_count").cast("int").alias("passenger_count"))
             .agg(F.avg("fare_amount").alias("avg_fare")).orderBy("passenger_count").toPandas())
    return pdf, {}


def _avg_distance_by_passengers(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.filter(F.col("passenger_count").isNotNull() & (F.col("passenger_count") <= 6))
             .groupBy(F.col("passenger_count").cast("int").alias("passenger_count"))
             .agg(F.avg("trip_distance").alias("avg_distance")).orderBy("passenger_count").toPandas())
    return pdf, {}


# --------------------------------------------------------------------------- #
# Airport
# --------------------------------------------------------------------------- #

def _airport_side(df: Any, id_col: str):
    from pyspark.sql import functions as F

    pdf = (df.filter(F.col(id_col).isin(*AIRPORT_ZONE_IDS))
             .groupBy(id_col).agg(F.count("*").alias("trips")).toPandas())
    pdf["airport"] = pdf[id_col].map(lambda i: _AIRPORT_CODE.get(int(i), str(i)))
    return pdf[["airport", "trips"]].sort_values("airport")


def _airport_pickups(df: Any):
    pdf = _airport_side(df, "PULocationID")
    return pdf, {"total_airport_trips": _fmt_int(pdf["trips"].sum())}


def _airport_dropoffs(df: Any):
    pdf = _airport_side(df, "DOLocationID")
    return pdf, {"total_airport_trips": _fmt_int(pdf["trips"].sum())}


def _airport_revenue(df: Any):
    from pyspark.sql import functions as F

    ap = F.when(F.col("PULocationID").isin(*AIRPORT_ZONE_IDS), F.col("PULocationID")) \
          .otherwise(F.col("DOLocationID"))
    pdf = (df.filter(F.col("PULocationID").isin(*AIRPORT_ZONE_IDS)
                     | F.col("DOLocationID").isin(*AIRPORT_ZONE_IDS))
             .withColumn("ap", ap)
             .groupBy("ap").agg(F.sum("total_amount").alias("revenue")).toPandas())
    pdf["airport"] = pdf["ap"].map(lambda i: _AIRPORT_CODE.get(int(i), str(i)))
    return pdf[["airport", "revenue"]].sort_values("airport"), \
        {"total_airport_revenue": _fmt_money(pdf["revenue"].sum())}


def _airport_trends(df: Any):
    from pyspark.sql import functions as F

    pdf = (df.filter(F.col("PULocationID").isin(*AIRPORT_ZONE_IDS)
                     | F.col("DOLocationID").isin(*AIRPORT_ZONE_IDS))
             .groupBy(F.to_date("tpep_pickup_datetime").alias("period"))
             .agg(F.count("*").alias("trips")).orderBy("period").toPandas())
    return pdf, {"total_airport_trips": _fmt_int(pdf["trips"].sum())}


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def _build_registry() -> None:
    entries = [
        # Demand
        Analysis("demand.by_hour", Family.DEMAND, "Trips by hour",
                 ChartType.BAR, _trips_by_hour, x="hour", y="trips",
                 description="Hourly trip volume reveals the daily demand cycle — "
                             "the morning and evening rush-hour peaks."),
        Analysis("demand.by_weekday", Family.DEMAND, "Trips by weekday",
                 ChartType.BAR, _trips_by_weekday, x="weekday", y="trips",
                 description="How demand shifts across the week (weekday commuting "
                             "vs weekend leisure)."),
        Analysis("demand.by_month", Family.DEMAND, "Trips by month",
                 ChartType.BAR, _trips_by_month, x="month", y="trips",
                 description="Seasonal demand across the calendar year."),
        Analysis("demand.over_time", Family.DEMAND, "Trips over time",
                 ChartType.LINE, _trips_over_time, x="period", y="trips",
                 description="Daily trip counts across the loaded period — good for "
                             "spotting holidays, weather events, and outages."),
        # Geographic
        Analysis("geo.top_pickups", Family.GEOGRAPHIC, "Most common pickup locations",
                 ChartType.BAR, _top_pickups, x="zone", y="trips",
                 description="The busiest pickup zones by trip count."),
        Analysis("geo.top_dropoffs", Family.GEOGRAPHIC, "Most common dropoff locations",
                 ChartType.BAR, _top_dropoffs, x="zone", y="trips",
                 description="The busiest dropoff zones by trip count."),
        Analysis("geo.hotspot_compare", Family.GEOGRAPHIC, "Pickup vs dropoff hotspots",
                 ChartType.BAR, _pickup_dropoff_compare, x="zone", y="pickups",
                 description="For the top pickup zones, how pickup volume compares "
                             "with dropoff volume — reveals directional imbalance."),
        # Trip
        Analysis("trip.distance_dist", Family.TRIP, "Trip distance distribution",
                 ChartType.HISTOGRAM, _distance_distribution, x="bin", y="count",
                 description="Distribution of trip distances (miles), binned. Note the "
                             "strong right skew that motivates log_distance."),
        Analysis("trip.duration_dist", Family.TRIP, "Trip duration distribution",
                 ChartType.HISTOGRAM, _duration_distribution, x="bin", y="count",
                 description="Distribution of trip durations (minutes), binned."),
        Analysis("trip.avg_duration", Family.TRIP, "Average trip duration",
                 ChartType.METRIC, _avg_duration,
                 description="Mean trip duration across the selected dataset."),
        Analysis("trip.avg_distance", Family.TRIP, "Average trip distance",
                 ChartType.METRIC, _avg_distance,
                 description="Mean trip distance across the selected dataset."),
        Analysis("trip.avg_speed", Family.TRIP, "Average trip speed",
                 ChartType.METRIC, _avg_speed,
                 description="Mean speed (distance ÷ duration) — a proxy for congestion."),
        # Revenue
        Analysis("rev.total", Family.REVENUE, "Total revenue",
                 ChartType.METRIC, _total_revenue,
                 description="Total fare revenue (total_amount) across the dataset."),
        Analysis("rev.by_hour", Family.REVENUE, "Revenue by hour",
                 ChartType.BAR, _revenue_by_hour, x="hour", y="revenue",
                 description="When the money is made across the day."),
        Analysis("rev.by_weekday", Family.REVENUE, "Revenue by weekday",
                 ChartType.BAR, _revenue_by_weekday, x="weekday", y="revenue",
                 description="Revenue distribution across the week."),
        Analysis("rev.by_month", Family.REVENUE, "Revenue by month",
                 ChartType.BAR, _revenue_by_month, x="month", y="revenue",
                 description="Revenue distribution across the year."),
        Analysis("rev.avg_fare", Family.REVENUE, "Average fare amount",
                 ChartType.METRIC, _avg_fare,
                 description="Mean base fare per trip."),
        Analysis("rev.avg_tip", Family.REVENUE, "Average tip amount",
                 ChartType.METRIC, _avg_tip,
                 description="Mean tip per trip, and tip as a share of fare."),
        # Passenger
        Analysis("pax.distribution", Family.PASSENGER, "Passenger count distribution",
                 ChartType.BAR, _passenger_distribution, x="passenger_count", y="trips",
                 description="How many riders share a cab."),
        Analysis("pax.avg_fare", Family.PASSENGER, "Average fare by passenger count",
                 ChartType.BAR, _avg_fare_by_passengers, x="passenger_count", y="avg_fare",
                 description="Does group size change the fare?"),
        Analysis("pax.avg_distance", Family.PASSENGER, "Average trip distance by passenger count",
                 ChartType.BAR, _avg_distance_by_passengers, x="passenger_count", y="avg_distance",
                 description="Does group size change how far people travel?"),
        # Airport
        Analysis("air.pickups", Family.AIRPORT, "Trips originating from airports",
                 ChartType.BAR, _airport_pickups, x="airport", y="trips",
                 description="Pickup volume at JFK, LaGuardia, and Newark."),
        Analysis("air.dropoffs", Family.AIRPORT, "Trips ending at airports",
                 ChartType.BAR, _airport_dropoffs, x="airport", y="trips",
                 description="Dropoff volume at the three airports."),
        Analysis("air.revenue", Family.AIRPORT, "Airport revenue",
                 ChartType.BAR, _airport_revenue, x="airport", y="revenue",
                 description="Revenue from trips touching each airport."),
        Analysis("air.trends", Family.AIRPORT, "Airport traffic trends",
                 ChartType.LINE, _airport_trends, x="period", y="trips",
                 description="Airport-related trip volume over time."),
    ]
    for entry in entries:
        register(entry)


_build_registry()
