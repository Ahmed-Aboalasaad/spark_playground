"""Dataset loading.

Real responsibility (per ARCHITECTURE.md): create DataFrames from the monthly
parquet files, validate availability and schema, and report loading statistics.
Output is the raw Spark DataFrame.

Fully implemented -- this module no longer depends on ``config.PLACEHOLDER_MODE``.
It reads exactly the requested months (or every month currently on disk, if
none are specified) via ``spark.read.parquet`` and reports real statistics. The
files it reads are whatever :mod:`pipeline.dataset_manager` has already
downloaded; this module never fetches anything itself. Downstream stages
(cleaning, analysis, modeling) remain placeholder-gated independently -- their
producers ignore whatever DataFrame they're handed while stubbed, so a real
raw DataFrame flowing through them is safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from functools import reduce

from config import DATA_DIR, EXPECTED_COLUMNS, PICKUP_COL
from pipeline import dataset_manager

# Currency columns TLC has, at various points, written as a different physical
# Parquet type across monthly vintages (e.g. `airport_fee` is INT32 in months
# before the fee existed, DOUBLE once real fractional amounts appear). Reading
# such files together with a single `spark.read.parquet(*paths)` call picks one
# file's schema for all of them and crashes the vectorized reader the moment it
# hits a file encoded differently. Casting each file to a fixed DoubleType
# before unioning sidesteps the mismatch regardless of which vintage wrote it.
MONEY_COLUMNS: tuple[str, ...] = (
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge", "airport_fee",
)


def _read_one_month(spark: Any, path: Path) -> Any:
    """Read a single month's parquet file, normalizing money columns to double."""
    from pyspark.sql import functions as F

    d = spark.read.parquet(str(path))
    for c in MONEY_COLUMNS:
        if c in d.columns:
            d = d.withColumn(c, F.col(c).cast("double"))
    return d


def load_raw_dataset(
    spark: Any,
    data_dir: str | Path | None = None,
    months: list[tuple[int, int]] | None = None,
) -> dict:
    """Load the raw taxi dataset from parquet files.

    Parameters
    ----------
    spark:
        The active SparkSession.
    data_dir:
        Directory holding the monthly parquet files. Defaults to ``config.DATA_DIR``.
    months:
        Which ``(year, month)`` files to read. ``None`` means "every month
        currently downloaded". Months that aren't actually on disk are dropped
        and reported back under ``info["missing_months"]`` rather than failing
        the whole load.

    Returns
    -------
    dict
        ``{"df": <Spark DataFrame or None>, "info": {...}}`` where ``info``
        carries the statistics the Home page displays.
    """
    directory = Path(data_dir or DATA_DIR)
    target_months = months if months is not None else dataset_manager.list_downloaded(directory)
    target_months = sorted(set(target_months))

    paths = [dataset_manager.file_path(m, directory) for m in target_months]
    present = [(m, p) for m, p in zip(target_months, paths) if p.exists()]
    missing_months = [m for m, p in zip(target_months, paths) if not p.exists()]

    if not present:
        return {
            "df": None,
            "info": {
                "status": "empty",
                "n_records": 0,
                "n_columns": len(EXPECTED_COLUMNS),
                "n_files": 0,
                "size_bytes": 0,
                "date_range": (None, None),
                "columns": list(EXPECTED_COLUMNS),
                "months": [],
                "missing_months": [dataset_manager.month_str(m) for m in missing_months],
            },
        }

    present_months = [m for m, _ in present]
    present_paths = [p for _, p in present]

    df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True),
                (_read_one_month(spark, p) for p in present_paths))
    missing_columns = validate_schema(df)
    size_bytes = sum(p.stat().st_size for p in present_paths)

    start, end = None, None
    if PICKUP_COL in df.columns:
        from pyspark.sql import functions as F

        row = df.select(F.min(PICKUP_COL).alias("mn"), F.max(PICKUP_COL).alias("mx")).first()
        if row is not None:
            start = row["mn"].strftime("%Y-%m-%d") if row["mn"] else None
            end = row["mx"].strftime("%Y-%m-%d") if row["mx"] else None

    info = {
        "status": "schema_mismatch" if missing_columns else "ok",
        "n_records": df.count(),
        "n_columns": len(df.columns),
        "n_files": len(present_paths),
        "size_bytes": size_bytes,
        "date_range": (start, end),
        "columns": df.columns,
        "months": [dataset_manager.month_str(m) for m in present_months],
        "missing_months": [dataset_manager.month_str(m) for m in missing_months],
        "missing_schema_columns": missing_columns,
    }
    return {"df": df, "info": info}


def validate_schema(df: Any) -> list[str]:
    """Return the list of expected columns missing from ``df``.

    Empty list means the schema matches.
    """
    present = set(df.columns)
    return [c for c in EXPECTED_COLUMNS if c not in present]
