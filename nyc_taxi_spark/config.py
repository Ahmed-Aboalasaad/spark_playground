"""Central configuration for the NYC Taxi Analytics application.

Everything that a deployment might need to change lives here: filesystem paths,
Spark session defaults, the expected dataset schema, and the small set of
domain constants (airport zones, sentinel location IDs). Nothing in this module
imports Spark or Streamlit, so it is safe to import from anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Root of the repository (this file's directory).
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory holding the monthly `yellow_tripdata_YYYY-MM.parquet` files.
# Override with the NYC_TAXI_DATA_DIR environment variable.
DATA_DIR = Path(os.environ.get("NYC_TAXI_DATA_DIR", PROJECT_ROOT / "data"))

# The TLC zone lookup table. Ships with the repo under reference/.
ZONE_LOOKUP_PATH = Path(
    os.environ.get("NYC_TAXI_ZONE_LOOKUP", PROJECT_ROOT / "reference" / "taxi_zone_lookup.csv")
)

# Glob used to discover monthly parquet files inside DATA_DIR.
PARQUET_GLOB = "yellow_tripdata_*.parquet"

# Filename template for one month of data, e.g. "yellow_tripdata_2023-01.parquet".
PARQUET_FILENAME_TEMPLATE = "yellow_tripdata_{year:04d}-{month:02d}.parquet"


# --------------------------------------------------------------------------- #
# Dataset acquisition (download control panel)
# --------------------------------------------------------------------------- #

# NYC TLC's public trip-data bucket. Monthly files live at
# f"{TLC_BASE_URL}/{PARQUET_FILENAME_TEMPLATE.format(year=y, month=m)}".
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# TLC's yellow taxi parquet publication starts January 2009.
DATASET_EARLIEST_MONTH: tuple[int, int] = (2009, 1)

# Streamed in chunks so large monthly files don't have to fit in memory at once.
DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB

# Per-request network timeout (seconds): (connect timeout, read timeout).
DOWNLOAD_TIMEOUT_SECONDS: tuple[int, int] = (10, 60)


# --------------------------------------------------------------------------- #
# Spark configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SparkConfig:
    """Spark session settings tuned for local, single-machine analytics.

    The defaults here are deliberately sized for exploring a handful of months
    of taxi data on one machine, not for a cluster. The most important override
    is ``shuffle_partitions``: Spark's default of 200 produces hundreds of tiny
    tasks on local data and makes every aggregation feel sluggish.
    """

    app_name: str = "NYCTaxiAnalytics"
    master: str = "local[*]"

    # Driver memory. In local mode the driver is also the executor, so this is
    # the single JVM heap that does all the work. The 1 GB default OOMs on the
    # zone-lag window/join once you load several months (tens of millions of
    # rows); the source notebook used 6 GB for 90M rows. 4 GB is a safe local
    # default (lower it on an <8 GB machine via NYC_TAXI_DRIVER_MEMORY).
    driver_memory: str = os.environ.get("NYC_TAXI_DRIVER_MEMORY", "4g")

    # 64 keeps shuffle partitions meaningfully sized across a wide range of
    # loads: with Adaptive Query Execution on (below), Spark coalesces them down
    # for a single small month and keeps them small enough to fit in heap for a
    # multi-month load. The old value of 8 crammed ~4M rows into each partition
    # on a large load and blew the heap during the window sort.
    shuffle_partitions: int = 64

    # Arrow makes `toPandas()` on small aggregated results fast, which is
    # exactly the access pattern the UI uses (aggregate in Spark, collect a
    # small frame, hand it to Plotly).
    arrow_enabled: bool = True

    # Extra raw Spark configs, applied last so they can override anything above.
    extra: dict = field(default_factory=dict)

    def as_spark_conf(self) -> dict:
        """Flatten into the ``spark.*`` key/value pairs the builder expects."""
        conf = {
            "spark.driver.memory": self.driver_memory,
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.sql.execution.arrow.pyspark.enabled": str(self.arrow_enabled).lower(),
            # Adaptive Query Execution: coalesce/split shuffle partitions to the
            # actual data size and handle skew — the key to one config working
            # for both a single month and a dozen.
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
        }
        conf.update(self.extra)
        return conf


# Default instance the application uses unless a caller supplies its own.
SPARK_CONFIG = SparkConfig()


# --------------------------------------------------------------------------- #
# Dataset schema (standard 19-column TLC yellow taxi schema)
# --------------------------------------------------------------------------- #

# Column names as they appear in the source parquet files. Kept as a constant
# so the loader can validate incoming data against what the pipeline expects.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
)

# The two datetime columns, called out for the cleaning / feature stages.
PICKUP_COL = "tpep_pickup_datetime"
DROPOFF_COL = "tpep_dropoff_datetime"


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

# In the zone lookup, airport zones are flagged by the service_zone column --
# but with a quirk: JFK and LaGuardia use service_zone == "Airports", while
# Newark uses its own service_zone == "EWR". Matching only "Airports" silently
# drops Newark, so airport detection matches either of these values. We keep the
# IDs below for reference and for any code path that needs them without a join.
AIRPORT_SERVICE_ZONES = ("Airports", "EWR")
AIRPORT_LOCATION_IDS: dict[str, int] = {
    "EWR": 1,     # Newark Airport
    "JFK": 132,   # JFK Airport
    "LGA": 138,   # LaGuardia Airport
}

# Sentinel location IDs that are not real geography. Cleaning should treat trips
# referencing only these as non-geographic / droppable depending on the rule.
UNKNOWN_LOCATION_IDS: tuple[int, ...] = (264, 265)


# --------------------------------------------------------------------------- #
# Cleaning thresholds (mirrors notebooks/04_Preprocessing_FeatureEngineering)
# --------------------------------------------------------------------------- #
# Each threshold below is a fixed, hand-written rule applied to the whole dataset
# before any train/test split -- no statistic is learned from the data here, so
# nothing can leak. The rationale for every value is documented alongside the
# cleaning step in the notebook and surfaced in the Data Preprocessing page.

FARE_MIN = 2.5            # $2.50 was NYC's classic flag-drop minimum; below it = artifact
FARE_MAX = 250.0          # meter glitches produce $1,000+ records; 99.99th pct sits far below
TRIP_DISTANCE_MIN = 0.0   # 0 miles = meter/GPS error (strict >)
TRIP_DISTANCE_MAX = 100.0 # NYC metro is ~30 miles across; 100 is generous (strict <)
TRIP_DURATION_MIN = 1.0   # < 1 min = accidental meter start (minutes)
TRIP_DURATION_MAX = 180.0 # > 3 h = forgotten meter (minutes)

# TLC yellow-taxi data starts in 2009. Monthly files contain a handful of stray
# rows with impossible pickup years (2002, 2098, ...) -- a known TLC quirk.
# Cleaning drops any row whose pickup year is before this or in the far future;
# leaving them in poisons the time-based train/test split.
DATASET_MIN_YEAR = 2009


# --------------------------------------------------------------------------- #
# Feature engineering (mirrors notebook 04)
# --------------------------------------------------------------------------- #
# The three airport zones, by LocationID. JFK/LGA/EWR carry flat rates and fixed
# fees, so a single 0/1 flag captures a whole fare regime. Values match
# AIRPORT_LOCATION_IDS above; kept as a plain tuple for fast isin() checks.
AIRPORT_ZONE_IDS: tuple[int, ...] = (1, 132, 138)  # EWR, JFK, LGA

# Rush-hour windows (inclusive, local time), weekdays only.
RUSH_HOURS_AM: tuple[int, int] = (7, 9)
RUSH_HOURS_PM: tuple[int, int] = (16, 19)

# The engineered feature columns produced by the feature pipeline, in the order
# the modeling stage assembles them. Mirrors the notebook's FEATURES list.
FEATURE_COLUMNS: tuple[str, ...] = (
    "trip_distance", "log_distance", "passenger_count", "RatecodeID",
    "PULocationID", "DOLocationID",
    "pickup_hour", "pickup_dayofweek", "pickup_month",
    "is_weekend", "is_rush_hour", "is_airport",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "zone_prev_day_avg", "zone_7d_avg",
)

# The regression targets. ``fare_amount`` is the default; ``trip_duration_min``
# is derived during cleaning. Kept as a tuple so the Modeling UI can offer a
# selector without a hunt through the codebase.
ML_TARGETS: tuple[str, ...] = ("fare_amount", "trip_duration_min")
ML_TARGET_COLUMN = "fare_amount"

# Fraction of the (time-ordered) span held out as the test set. The split is by
# time, not random -- see pipeline.features -- so the most recent slice becomes
# the untouched future holdout, mimicking real deployment.
DEFAULT_TEST_FRACTION = 0.2


# --------------------------------------------------------------------------- #
# Placeholder flag
# --------------------------------------------------------------------------- #

# While the pipeline is skeletal, every module returns mock data and advertises
# it loudly. Real implementations flip this to False (or remove the checks) as
# they land. The UI reads this to decide whether to show placeholder banners.
PLACEHOLDER_MODE = True
