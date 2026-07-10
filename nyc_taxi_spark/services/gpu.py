"""GPU detection and live VRAM monitoring.

Thin wrapper over ``nvidia-smi`` (no extra Python dependency). Everything here
degrades gracefully to "no GPU" when the tool or a card is absent, so the app
runs identically on a machine without CUDA -- it just shows CPU-only options.

Used by the Modeling page to show the user their GPU, warn about likely
out-of-memory before a run, and display live VRAM occupancy during training.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import Any

_SMI = "nvidia-smi"
# Rough bytes-per-row estimate for the assembled training matrix (≈16 float
# features + label + Spark/XGBoost overhead). Used only for OOM *warnings*, not
# hard limits -- deliberately conservative.
BYTES_PER_ROW = 220


def _run_smi(query: str) -> list[list[str]] | None:
    """Run an ``nvidia-smi --query-gpu`` call, returning parsed CSV rows."""
    if shutil.which(_SMI) is None:
        return None
    try:
        out = subprocess.run(
            [_SMI, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None
    if not out:
        return None
    return [[c.strip() for c in line.split(",")] for line in out.splitlines()]


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True if at least one NVIDIA GPU is visible to ``nvidia-smi``."""
    rows = _run_smi("name")
    return bool(rows)


@lru_cache(maxsize=1)
def gpu_info() -> dict[str, Any] | None:
    """Static facts about GPU 0: name, total VRAM (MB), driver version.

    Cached -- these don't change during a session. Returns ``None`` with no GPU.
    """
    rows = _run_smi("name,memory.total,driver_version")
    if not rows:
        return None
    name, mem_total, driver = rows[0][0], rows[0][1], rows[0][2]
    return {
        "name": name,
        "memory_total_mb": int(float(mem_total)),
        "driver_version": driver,
    }


def vram_usage() -> dict[str, Any] | None:
    """Live VRAM occupancy for GPU 0. Not cached -- called on a polling loop.

    Returns ``{"used_mb", "total_mb", "pct"}`` or ``None`` when no GPU.
    """
    rows = _run_smi("memory.used,memory.total")
    if not rows:
        return None
    used, total = int(float(rows[0][0])), int(float(rows[0][1]))
    pct = (used / total * 100.0) if total else 0.0
    return {"used_mb": used, "total_mb": total, "pct": pct}


def estimate_oom_risk(n_rows: int) -> dict[str, Any] | None:
    """Heuristic: will ``n_rows`` likely overflow this GPU's VRAM when training?

    Returns ``{"est_mb", "budget_mb", "risky"}`` or ``None`` with no GPU. The
    estimate is intentionally rough and conservative -- it drives a *warning*,
    while the real safety net is the OOM-safe training handler.
    """
    info = gpu_info()
    if info is None:
        return None
    est_mb = n_rows * BYTES_PER_ROW / (1024 * 1024)
    # Leave headroom: only ~55% of VRAM is realistically usable for the training
    # matrix + histograms on a small laptop card.
    budget_mb = info["memory_total_mb"] * 0.55
    return {"est_mb": est_mb, "budget_mb": budget_mb, "risky": est_mb > budget_mb}
