"""Dataset acquisition: inventory, download, and deletion of monthly parquet files.

Real responsibility: this module owns everything that happens to the raw
parquet files *on disk*, before Spark ever touches them -- discovering which
months are already present, fetching missing months from the NYC TLC public
bucket, and deleting files the user no longer wants to keep. It sits upstream
of :mod:`pipeline.loader` in the data flow: loader turns files that already
exist on disk into a Spark DataFrame; this module is what puts them there (or
removes them).

Deliberately Spark-free and Streamlit-free (like :mod:`pipeline.zones`), so it
is plain, testable Python: file-system + HTTP only. The download/delete
functions never raise on a per-file failure -- they collect successes and
failures into a summary dict so one bad month (e.g. a 404 for an unpublished
month) doesn't abort a whole batch. Callers decide how to surface failures.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from config import (
    DATASET_EARLIEST_MONTH,
    DATA_DIR,
    DOWNLOAD_CHUNK_BYTES,
    DOWNLOAD_TIMEOUT_SECONDS,
    PARQUET_FILENAME_TEMPLATE,
    TLC_BASE_URL,
)

Month = tuple[int, int]  # (year, month)
ProgressCallback = Callable[[dict[str, Any]], None]

_FILENAME_RE = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")


# --------------------------------------------------------------------------- #
# Filename / path helpers
# --------------------------------------------------------------------------- #

def month_str(month: Month) -> str:
    """``(2023, 1)`` -> ``"2023-01"``."""
    year, mon = month
    return f"{year:04d}-{mon:02d}"


def parse_month_str(text: str) -> Month:
    """``"2023-01"`` -> ``(2023, 1)``. Inverse of :func:`month_str`."""
    year, mon = text.split("-")
    return int(year), int(mon)


def file_name(month: Month) -> str:
    year, mon = month
    return PARQUET_FILENAME_TEMPLATE.format(year=year, month=mon)


def file_path(month: Month, data_dir: str | Path | None = None) -> Path:
    return Path(data_dir or DATA_DIR) / file_name(month)


def download_url(month: Month) -> str:
    return f"{TLC_BASE_URL}/{file_name(month)}"


def parse_filename(name: str) -> Month | None:
    """Extract ``(year, month)`` from a filename, or ``None`` if it doesn't match."""
    match = _FILENAME_RE.search(name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------- #
# Month range arithmetic
# --------------------------------------------------------------------------- #

def month_range(start: Month, end: Month) -> list[Month]:
    """Every month from ``start`` to ``end``, inclusive, ascending."""
    if start > end:
        raise ValueError(f"start {month_str(start)} must not be after end {month_str(end)}")
    months: list[Month] = []
    year, mon = start
    while (year, mon) <= end:
        months.append((year, mon))
        mon += 1
        if mon > 12:
            mon = 1
            year += 1
    return months


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #

def list_downloaded(data_dir: str | Path | None = None) -> list[Month]:
    """Months with a parquet file already present on disk, sorted ascending."""
    directory = Path(data_dir or DATA_DIR)
    if not directory.exists():
        return []
    months = [m for f in directory.glob("yellow_tripdata_*.parquet")
              if (m := parse_filename(f.name)) is not None]
    return sorted(set(months))


def coverage_summary(data_dir: str | Path | None = None) -> dict:
    """Describe what's on disk: span, gaps, file count, total size.

    Returns
    -------
    dict
        ``{"months": [...], "min": Month|None, "max": Month|None, "count": int,
        "total_bytes": int, "gaps": [...]}``. ``gaps`` lists months strictly
        between ``min`` and ``max`` that have no file -- i.e. holes in an
        otherwise-contiguous download, not months outside the span.
    """
    directory = Path(data_dir or DATA_DIR)
    months = list_downloaded(directory)
    if not months:
        return {"months": [], "min": None, "max": None, "count": 0,
                "total_bytes": 0, "gaps": []}

    lo, hi = months[0], months[-1]
    full_span = set(month_range(lo, hi))
    gaps = sorted(full_span - set(months))
    total_bytes = sum(file_path(m, directory).stat().st_size for m in months)

    return {
        "months": months,
        "min": lo,
        "max": hi,
        "count": len(months),
        "total_bytes": total_bytes,
        "gaps": gaps,
    }


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_months(
    months: Iterable[Month],
    data_dir: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    skip_existing: bool = True,
) -> dict:
    """Fetch each requested month from the TLC bucket into ``data_dir``.

    Streams to a ``.part`` file and atomically renames on success, so an
    interrupted download never leaves a corrupt file masquerading as complete.
    Never raises for an individual month's failure (bad network, 404 for an
    unpublished month, etc.) -- it's recorded in ``failed`` instead.

    ``on_progress``, if given, is called with a dict on each meaningful step:
    ``{"stage": "start"|"chunk"|"done"|"skipped"|"failed", "index": int,
    "total": int, "month": Month, "bytes_done": int, "bytes_total": int|None,
    "error": str|None}``.

    Returns
    -------
    dict
        ``{"downloaded": [...], "skipped": [...], "failed": [{"month":, "error":}]}``
    """
    directory = Path(data_dir or DATA_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    ordered = sorted(set(months))
    total = len(ordered)
    downloaded: list[Month] = []
    skipped: list[Month] = []
    failed: list[dict] = []

    def _emit(**event: Any) -> None:
        if on_progress is not None:
            on_progress(event)

    for index, month in enumerate(ordered):
        dest = file_path(month, directory)

        if skip_existing and dest.exists():
            skipped.append(month)
            _emit(stage="skipped", index=index, total=total, month=month,
                  bytes_done=0, bytes_total=None, error=None)
            continue

        _emit(stage="start", index=index, total=total, month=month,
              bytes_done=0, bytes_total=None, error=None)

        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(download_url(month), stream=True,
                               timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                total_bytes = response.headers.get("Content-Length")
                total_bytes = int(total_bytes) if total_bytes else None

                bytes_done = 0
                with open(tmp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        bytes_done += len(chunk)
                        _emit(stage="chunk", index=index, total=total, month=month,
                              bytes_done=bytes_done, bytes_total=total_bytes, error=None)

            tmp.replace(dest)
            downloaded.append(month)
            _emit(stage="done", index=index, total=total, month=month,
                  bytes_done=bytes_done, bytes_total=total_bytes, error=None)

        except requests.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            status = exc.response.status_code if exc.response is not None else None
            # The TLC CDN returns 403 (not 404) for a month that doesn't exist --
            # an S3/CloudFront quirk (no ListBucket permission means GetObject on a
            # missing key answers "Forbidden" rather than "Not Found").
            message = ("not yet published or doesn't exist" if status in (403, 404)
                       else f"HTTP {status or 'error'}")
            failed.append({"month": month, "error": message})
            _emit(stage="failed", index=index, total=total, month=month,
                  bytes_done=0, bytes_total=None, error=message)

        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            failed.append({"month": month, "error": str(exc)})
            _emit(stage="failed", index=index, total=total, month=month,
                  bytes_done=0, bytes_total=None, error=str(exc))

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #

def delete_months(months: Iterable[Month], data_dir: str | Path | None = None) -> dict:
    """Remove the parquet files for the given months.

    Returns
    -------
    dict
        ``{"deleted": [...], "freed_bytes": int, "not_found": [...]}``
    """
    directory = Path(data_dir or DATA_DIR)
    deleted: list[Month] = []
    not_found: list[Month] = []
    freed_bytes = 0

    for month in sorted(set(months)):
        path = file_path(month, directory)
        if not path.exists():
            not_found.append(month)
            continue
        freed_bytes += path.stat().st_size
        path.unlink()
        deleted.append(month)

    return {"deleted": deleted, "freed_bytes": freed_bytes, "not_found": not_found}


def delete_all(data_dir: str | Path | None = None) -> dict:
    """Remove every downloaded month. Convenience wrapper over :func:`delete_months`."""
    directory = Path(data_dir or DATA_DIR)
    return delete_months(list_downloaded(directory), directory)


__all__ = [
    "Month",
    "month_str",
    "parse_month_str",
    "file_name",
    "file_path",
    "download_url",
    "parse_filename",
    "month_range",
    "list_downloaded",
    "coverage_summary",
    "download_months",
    "delete_months",
    "delete_all",
    "DATASET_EARLIEST_MONTH",
]
