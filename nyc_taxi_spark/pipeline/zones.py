"""Zone lookup reference data.

The TLC zone lookup maps every ``LocationID`` to a borough, a zone name, and a
``service_zone``. Two things in the application depend on it:

* **Airport analysis** -- airport zones are exactly the rows where
  ``service_zone == "Airports"`` (JFK, LaGuardia, Newark). We treat that column
  as the source of truth instead of hardcoding IDs.
* **Geographic analysis** -- joining trips to this table turns raw integer IDs
  into readable zone names ("JFK Airport", "Upper East Side North").

Unlike the trip data, the lookup is tiny (265 rows) and static, so it is read
with pandas here. The real Spark pipeline can broadcast-join a Spark version of
the same table; a helper is provided to hand it over as a Spark DataFrame when
a session is available.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from config import AIRPORT_SERVICE_ZONES, UNKNOWN_LOCATION_IDS, ZONE_LOOKUP_PATH


@lru_cache(maxsize=1)
def load_zone_lookup() -> pd.DataFrame:
    """Read the zone lookup CSV once and cache it.

    Returns a frame with columns: LocationID, Borough, Zone, service_zone.
    """
    return pd.read_csv(ZONE_LOOKUP_PATH)


def airport_location_ids() -> list[int]:
    """LocationIDs for airport zones.

    Matches JFK/LaGuardia (service_zone == "Airports") *and* Newark
    (service_zone == "EWR"), since Newark is flagged differently in the lookup.
    """
    df = load_zone_lookup()
    mask = df["service_zone"].isin(AIRPORT_SERVICE_ZONES)
    return df.loc[mask, "LocationID"].tolist()


def zone_name(location_id: int) -> str:
    """Human-readable 'Zone (Borough)' label for a LocationID."""
    df = load_zone_lookup()
    row = df.loc[df["LocationID"] == location_id]
    if row.empty:
        return f"Unknown ({location_id})"
    return f"{row.iloc[0]['Zone']} ({row.iloc[0]['Borough']})"


def is_unknown(location_id: int) -> bool:
    """True for the sentinel 'Unknown' / 'Outside of NYC' location IDs."""
    return location_id in UNKNOWN_LOCATION_IDS


def spark_zone_lookup(spark: Any) -> Any:
    """Return the lookup as a Spark DataFrame for broadcast joins.

    Convenience for the real geographic/airport implementations. Reads the CSV
    directly with Spark so the join stays inside the Spark pipeline.
    """
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(ZONE_LOOKUP_PATH))
    )
