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

    # 200 (Spark's default) is far too many for local work; 8 keeps tasks
    # meaningfully sized on a few months of data.
    shuffle_partitions: int = 8

    # Arrow makes `toPandas()` on small aggregated results fast, which is
    # exactly the access pattern the UI uses (aggregate in Spark, collect a
    # small frame, hand it to Plotly).
    arrow_enabled: bool = True

    # Extra raw Spark configs, applied last so they can override anything above.
    extra: dict = field(default_factory=dict)

    def as_spark_conf(self) -> dict:
        """Flatten into the ``spark.*`` key/value pairs the builder expects."""
        conf = {
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.sql.execution.arrow.pyspark.enabled": str(self.arrow_enabled).lower(),
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

# The regression target for the modeling pipeline. Kept as a constant so adding
# a second target later (e.g. trip duration) is a one-line change plus a UI
# selector, not a hunt through the codebase.
ML_TARGET_COLUMN = "fare_amount"


# --------------------------------------------------------------------------- #
# Placeholder flag
# --------------------------------------------------------------------------- #

# While the pipeline is skeletal, every module returns mock data and advertises
# it loudly. Real implementations flip this to False (or remove the checks) as
# they land. The UI reads this to decide whether to show placeholder banners.
PLACEHOLDER_MODE = True
