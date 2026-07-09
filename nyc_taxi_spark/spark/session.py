"""Single shared SparkSession for the whole application.

The architecture mandates exactly one SparkSession, reused by every processing
module. In a Streamlit app the natural way to guarantee a single instance
across reruns and pages is ``st.cache_resource``: the factory runs once and
every subsequent call receives the same object.

``get_spark`` is Streamlit-aware but degrades gracefully: if Streamlit is not
available (tests, scripts) it falls back to a plain module-level singleton so
the pipeline can still obtain a session.
"""
from __future__ import annotations

from typing import Any

from config import SPARK_CONFIG, SparkConfig


def _build_session(cfg: SparkConfig) -> Any:
    """Construct a SparkSession from a :class:`SparkConfig`.

    Imports pyspark lazily so importing this module doesn't require Spark to be
    installed until a session is actually requested.
    """
    import os
    import sys

    from pyspark.sql import SparkSession

    # Pin Spark's Python workers to the *same* interpreter running this process.
    # Otherwise Spark launches workers via whatever ``python`` is first on PATH,
    # which on this machine is a different minor version (3.13) than the venv
    # driver (3.12), and Spark aborts with PYTHON_VERSION_MISMATCH. Forcing both
    # to ``sys.executable`` makes the session reproducible regardless of PATH.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    builder = SparkSession.builder.appName(cfg.app_name).master(cfg.master)
    for key, value in cfg.as_spark_conf().items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


# Fallback singleton for non-Streamlit contexts.
_session_singleton: Any | None = None


def get_spark(cfg: SparkConfig | None = None) -> Any:
    """Return the shared SparkSession, creating it on first use.

    Parameters
    ----------
    cfg:
        Optional configuration override. Defaults to the application-wide
        :data:`config.SPARK_CONFIG`.
    """
    cfg = cfg or SPARK_CONFIG

    try:
        import streamlit as st
    except ImportError:
        st = None

    if st is not None:
        # Cache the session as a Streamlit resource so it survives reruns and
        # is shared across all pages. The inner function is cached by Streamlit.
        @st.cache_resource(show_spinner="Starting Spark session...")
        def _cached() -> Any:
            return _build_session(cfg)

        return _cached()

    # No Streamlit: use a plain module-level singleton.
    global _session_singleton
    if _session_singleton is None:
        _session_singleton = _build_session(cfg)
    return _session_singleton


def session_info(spark: Any) -> dict:
    """Collect display-friendly facts about an active session.

    Used by the Home page's "Spark Session Information" section.
    """
    conf = spark.sparkContext.getConf()
    return {
        "Spark version": spark.version,
        "Application name": conf.get("spark.app.name", "n/a"),
        "Master": conf.get("spark.master", "n/a"),
        "Default parallelism": spark.sparkContext.defaultParallelism,
        "Shuffle partitions": conf.get("spark.sql.shuffle.partitions", "n/a"),
    }
