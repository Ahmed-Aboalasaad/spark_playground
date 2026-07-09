"""Analysis engine.

Real responsibility: execute analytical queries, compute summary metrics,
generate aggregated DataFrames, and return data suitable for visualization.

This module is built around a **registry**: each analysis is a small object
describing its family, title, chart type, and a function that produces
``(dataframe, metrics)``. New analyses are added by registering another entry;
nothing existing has to change. That directly serves the "add new analytical
modules without modifying existing components" requirement.

Every producer here currently returns zeroed, correctly-shaped mock data via
``pipeline.mock``. Swapping in a real implementation means replacing one
producer function with a Spark aggregation that returns the same columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd

from pipeline import mock


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
# Mock producers (grouped by family). Each returns (frame, metrics).
# --------------------------------------------------------------------------- #

# -- Demand ----------------------------------------------------------------- #
def _trips_by_hour(df: Any):
    return mock.hours_frame("trips"), mock.zero_metrics("total_trips", "peak_hour")


def _trips_by_weekday(df: Any):
    return mock.weekdays_frame("trips"), mock.zero_metrics("total_trips")


def _trips_by_month(df: Any):
    return mock.months_frame("trips"), mock.zero_metrics("total_trips")


def _trips_over_time(df: Any):
    return mock.timeseries_frame(12, "trips"), mock.zero_metrics("total_trips")


# -- Geographic ------------------------------------------------------------- #
def _top_pickups(df: Any):
    return mock.top_n_frame(10, "zone", "trips"), mock.zero_metrics("distinct_zones")


def _top_dropoffs(df: Any):
    return mock.top_n_frame(10, "zone", "trips"), mock.zero_metrics("distinct_zones")


def _pickup_dropoff_compare(df: Any):
    frame = mock.top_n_frame(10, "zone", "pickups")
    frame["dropoffs"] = 0
    return frame, mock.zero_metrics("distinct_zones")


# -- Trip ------------------------------------------------------------------- #
def _distance_distribution(df: Any):
    return mock.distribution_frame(20, "bin", "count"), mock.zero_metrics("avg_distance")


def _duration_distribution(df: Any):
    return mock.distribution_frame(20, "bin", "count"), mock.zero_metrics("avg_duration")


def _avg_duration(df: Any):
    return pd.DataFrame(), mock.zero_metrics("avg_duration_min")


def _avg_distance(df: Any):
    return pd.DataFrame(), mock.zero_metrics("avg_distance_mi")


def _avg_speed(df: Any):
    return pd.DataFrame(), mock.zero_metrics("avg_speed_mph")


# -- Revenue ---------------------------------------------------------------- #
def _total_revenue(df: Any):
    return pd.DataFrame(), mock.zero_metrics("total_revenue")


def _revenue_by_hour(df: Any):
    return mock.hours_frame("revenue"), mock.zero_metrics("total_revenue")


def _revenue_by_weekday(df: Any):
    return mock.weekdays_frame("revenue"), mock.zero_metrics("total_revenue")


def _revenue_by_month(df: Any):
    return mock.months_frame("revenue"), mock.zero_metrics("total_revenue")


def _avg_fare(df: Any):
    return pd.DataFrame(), mock.zero_metrics("avg_fare")


def _avg_tip(df: Any):
    return pd.DataFrame(), mock.zero_metrics("avg_tip")


# -- Passenger -------------------------------------------------------------- #
def _passenger_distribution(df: Any):
    return mock.zero_series_frame(range(7), "passenger_count", "trips"), \
        mock.zero_metrics("avg_passengers")


def _avg_fare_by_passengers(df: Any):
    return mock.zero_series_frame(range(7), "passenger_count", "avg_fare"), {}


def _avg_distance_by_passengers(df: Any):
    return mock.zero_series_frame(range(7), "passenger_count", "avg_distance"), {}


# -- Airport ---------------------------------------------------------------- #
def _airport_pickups(df: Any):
    frame = mock.zero_series_frame(["EWR", "JFK", "LGA"], "airport", "trips")
    return frame, mock.zero_metrics("total_airport_trips")


def _airport_dropoffs(df: Any):
    frame = mock.zero_series_frame(["EWR", "JFK", "LGA"], "airport", "trips")
    return frame, mock.zero_metrics("total_airport_trips")


def _airport_revenue(df: Any):
    frame = mock.zero_series_frame(["EWR", "JFK", "LGA"], "airport", "revenue")
    return frame, mock.zero_metrics("total_airport_revenue")


def _airport_trends(df: Any):
    return mock.timeseries_frame(12, "trips"), mock.zero_metrics("total_airport_trips")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def _build_registry() -> None:
    entries = [
        # Demand
        Analysis("demand.by_hour", Family.DEMAND, "Trips by Hour",
                 ChartType.BAR, _trips_by_hour, x="hour", y="trips"),
        Analysis("demand.by_weekday", Family.DEMAND, "Trips by Weekday",
                 ChartType.BAR, _trips_by_weekday, x="weekday", y="trips"),
        Analysis("demand.by_month", Family.DEMAND, "Trips by Month",
                 ChartType.BAR, _trips_by_month, x="month", y="trips"),
        Analysis("demand.over_time", Family.DEMAND, "Trips over Time",
                 ChartType.LINE, _trips_over_time, x="period", y="trips"),
        # Geographic
        Analysis("geo.top_pickups", Family.GEOGRAPHIC, "Most Common Pickup Locations",
                 ChartType.BAR, _top_pickups, x="zone", y="trips"),
        Analysis("geo.top_dropoffs", Family.GEOGRAPHIC, "Most Common Dropoff Locations",
                 ChartType.BAR, _top_dropoffs, x="zone", y="trips"),
        Analysis("geo.hotspot_compare", Family.GEOGRAPHIC, "Pickup vs Dropoff Hotspots",
                 ChartType.BAR, _pickup_dropoff_compare, x="zone", y="pickups"),
        # Trip
        Analysis("trip.distance_dist", Family.TRIP, "Trip Distance Distribution",
                 ChartType.HISTOGRAM, _distance_distribution, x="bin", y="count"),
        Analysis("trip.duration_dist", Family.TRIP, "Trip Duration Distribution",
                 ChartType.HISTOGRAM, _duration_distribution, x="bin", y="count"),
        Analysis("trip.avg_duration", Family.TRIP, "Average Trip Duration",
                 ChartType.METRIC, _avg_duration),
        Analysis("trip.avg_distance", Family.TRIP, "Average Trip Distance",
                 ChartType.METRIC, _avg_distance),
        Analysis("trip.avg_speed", Family.TRIP, "Average Trip Speed",
                 ChartType.METRIC, _avg_speed),
        # Revenue
        Analysis("rev.total", Family.REVENUE, "Total Revenue",
                 ChartType.METRIC, _total_revenue),
        Analysis("rev.by_hour", Family.REVENUE, "Revenue by Hour",
                 ChartType.BAR, _revenue_by_hour, x="hour", y="revenue"),
        Analysis("rev.by_weekday", Family.REVENUE, "Revenue by Weekday",
                 ChartType.BAR, _revenue_by_weekday, x="weekday", y="revenue"),
        Analysis("rev.by_month", Family.REVENUE, "Revenue by Month",
                 ChartType.BAR, _revenue_by_month, x="month", y="revenue"),
        Analysis("rev.avg_fare", Family.REVENUE, "Average Fare Amount",
                 ChartType.METRIC, _avg_fare),
        Analysis("rev.avg_tip", Family.REVENUE, "Average Tip Amount",
                 ChartType.METRIC, _avg_tip),
        # Passenger
        Analysis("pax.distribution", Family.PASSENGER, "Passenger Count Distribution",
                 ChartType.BAR, _passenger_distribution, x="passenger_count", y="trips"),
        Analysis("pax.avg_fare", Family.PASSENGER, "Average Fare by Passenger Count",
                 ChartType.BAR, _avg_fare_by_passengers, x="passenger_count", y="avg_fare"),
        Analysis("pax.avg_distance", Family.PASSENGER, "Average Trip Distance by Passenger Count",
                 ChartType.BAR, _avg_distance_by_passengers, x="passenger_count", y="avg_distance"),
        # Airport
        Analysis("air.pickups", Family.AIRPORT, "Trips Originating from Airports",
                 ChartType.BAR, _airport_pickups, x="airport", y="trips"),
        Analysis("air.dropoffs", Family.AIRPORT, "Trips Ending at Airports",
                 ChartType.BAR, _airport_dropoffs, x="airport", y="trips"),
        Analysis("air.revenue", Family.AIRPORT, "Airport Revenue",
                 ChartType.BAR, _airport_revenue, x="airport", y="revenue"),
        Analysis("air.trends", Family.AIRPORT, "Airport Traffic Trends",
                 ChartType.LINE, _airport_trends, x="period", y="trips"),
    ]
    for entry in entries:
        register(entry)


_build_registry()
