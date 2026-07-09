"""Dataset loading.

Real responsibility (per ARCHITECTURE.md): create DataFrames from the monthly
parquet files, validate availability and schema, and report loading statistics.
Output is the raw Spark DataFrame.

Skeleton behaviour: returns ``None`` for the DataFrame and a placeholder info
dict with zeroed statistics. The real implementation will read
``config.DATA_DIR / config.PARQUET_GLOB`` via ``spark.read.parquet`` and
validate columns against ``config.EXPECTED_COLUMNS``.
"""
from __future__ import annotations

from typing import Any

from config import EXPECTED_COLUMNS, PLACEHOLDER_MODE
from pipeline.mock import PLACEHOLDER_LABEL


def load_raw_dataset(spark: Any, data_dir: str | None = None) -> dict:
    """Load the raw taxi dataset from parquet files.

    Returns
    -------
    dict
        ``{"df": <Spark DataFrame or None>, "info": {...}}`` where ``info``
        carries the statistics the Home page displays.

    Notes
    -----
    Placeholder implementation. When real:

    >>> paths = str(Path(data_dir) / PARQUET_GLOB)
    >>> df = spark.read.parquet(paths)
    >>> _validate_schema(df)
    >>> return {"df": df, "info": _describe(df, paths)}
    """
    if PLACEHOLDER_MODE:
        return {
            "df": None,
            "info": {
                "status": PLACEHOLDER_LABEL,
                "n_records": 0,
                "n_columns": len(EXPECTED_COLUMNS),
                "n_files": 0,
                "size_bytes": 0,
                "date_range": (None, None),
                "columns": list(EXPECTED_COLUMNS),
            },
        }

    raise NotImplementedError("Real loader not yet implemented.")


def validate_schema(df: Any) -> list[str]:
    """Return the list of expected columns missing from ``df``.

    Empty list means the schema matches. Stubbed to accept anything for now.
    """
    if PLACEHOLDER_MODE:
        return []
    present = set(df.columns)
    return [c for c in EXPECTED_COLUMNS if c not in present]
