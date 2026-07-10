"""Typed wrapper over Streamlit's ``st.session_state``.

The architecture calls for a fixed set of shared state (the Spark session, the
raw and cleaned datasets, the current selection, cleaning summary, trained
models, cached analysis results, and execution metrics). Addressing that state
through bare string keys is typo-prone and scatters the key names across every
page. ``AppState`` centralizes the keys and gives each one a named accessor.

The wrapper is intentionally thin: it stores everything in the real
``st.session_state`` so Streamlit's native multipage app shares it across
pages. It imports Streamlit lazily so this module can be imported (and unit
tested) outside a Streamlit runtime.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class DatasetChoice(str, Enum):
    """Which version of the dataset the user is currently viewing."""

    RAW = "Raw Dataset"
    CLEANED = "Cleaned Dataset"


# Canonical session-state keys, defined once.
class _Keys:
    SPARK = "spark_session"
    RAW_DF = "raw_df"
    CLEANED_DF = "cleaned_df"
    SELECTION = "dataset_selection"
    CLEANING_SUMMARY = "cleaning_summary"
    MODELS = "trained_models"
    ANALYSIS_CACHE = "analysis_results"
    METRICS = "execution_metrics"
    LOAD_INFO = "load_info"
    DOWNLOAD_RANGE = "download_range"
    LOAD_RANGE = "load_range"


def _session() -> Any:
    """Return the live ``st.session_state``.

    Imported lazily so importing this module never requires a running
    Streamlit server (useful for tests and for tooling).
    """
    import streamlit as st

    return st.session_state


class AppState:
    """Named, typed access to the application's shared session state."""

    # -- Spark session ---------------------------------------------------- #
    @property
    def spark(self) -> Any | None:
        return _session().get(_Keys.SPARK)

    @spark.setter
    def spark(self, value: Any) -> None:
        _session()[_Keys.SPARK] = value

    # -- Datasets --------------------------------------------------------- #
    @property
    def raw_df(self) -> Any | None:
        return _session().get(_Keys.RAW_DF)

    @raw_df.setter
    def raw_df(self, value: Any) -> None:
        _session()[_Keys.RAW_DF] = value

    @property
    def cleaned_df(self) -> Any | None:
        return _session().get(_Keys.CLEANED_DF)

    @cleaned_df.setter
    def cleaned_df(self, value: Any) -> None:
        _session()[_Keys.CLEANED_DF] = value

    @property
    def is_loaded(self) -> bool:
        """True once a dataset load has completed.

        Keyed on ``load_info`` rather than ``raw_df`` so the app is navigable in
        placeholder mode: the skeleton loader returns ``df=None`` but still
        produces a (zeroed) load-info dict, and every downstream pipeline
        function accepts ``None`` for the DataFrame. The real loader sets both
        ``raw_df`` and ``load_info``, so this stays correct once Spark lands.
        """
        return self.load_info is not None

    # -- Current selection ------------------------------------------------ #
    @property
    def selection(self) -> DatasetChoice:
        return _session().get(_Keys.SELECTION, DatasetChoice.RAW)

    @selection.setter
    def selection(self, value: DatasetChoice) -> None:
        _session()[_Keys.SELECTION] = value

    def active_df(self) -> Any | None:
        """Return whichever dataset the selector currently points at.

        Falls back to the raw dataset if the cleaned one has not been produced
        yet, so pages never break just because cleaning hasn't run.
        """
        if self.selection is DatasetChoice.CLEANED and self.cleaned_df is not None:
            return self.cleaned_df
        return self.raw_df

    # -- Cleaning summary ------------------------------------------------- #
    @property
    def cleaning_summary(self) -> dict | None:
        return _session().get(_Keys.CLEANING_SUMMARY)

    @cleaning_summary.setter
    def cleaning_summary(self, value: dict) -> None:
        _session()[_Keys.CLEANING_SUMMARY] = value

    # -- Trained models --------------------------------------------------- #
    @property
    def models(self) -> dict:
        return _session().setdefault(_Keys.MODELS, {})

    def add_model(self, name: str, model: Any) -> None:
        self.models[name] = model

    # -- Analysis result cache ------------------------------------------- #
    @property
    def analysis_cache(self) -> dict:
        return _session().setdefault(_Keys.ANALYSIS_CACHE, {})

    def cache_analysis(self, key: str, result: Any) -> None:
        self.analysis_cache[key] = result

    # -- Load info (timing, file count, date range) ---------------------- #
    @property
    def load_info(self) -> dict | None:
        return _session().get(_Keys.LOAD_INFO)

    @load_info.setter
    def load_info(self, value: dict) -> None:
        _session()[_Keys.LOAD_INFO] = value

    # -- Remembered slider selections (control panel UX) ------------------ #
    @property
    def download_range(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        return _session().get(_Keys.DOWNLOAD_RANGE)

    @download_range.setter
    def download_range(self, value: tuple[tuple[int, int], tuple[int, int]]) -> None:
        _session()[_Keys.DOWNLOAD_RANGE] = value

    @property
    def load_range(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        return _session().get(_Keys.LOAD_RANGE)

    @load_range.setter
    def load_range(self, value: tuple[tuple[int, int], tuple[int, int]]) -> None:
        _session()[_Keys.LOAD_RANGE] = value

    # -- Execution metrics log ------------------------------------------- #
    @property
    def metrics(self) -> list:
        return _session().setdefault(_Keys.METRICS, [])

    def record_metric(self, label: str, elapsed: float, n_records: int | None = None) -> None:
        """Append one timing entry to the running metrics log."""
        self.metrics.append(
            {"label": label, "elapsed": elapsed, "n_records": n_records}
        )


# A module-level instance is convenient, but because all state lives in
# st.session_state, constructing new AppState() objects is equally fine.
app_state = AppState()
