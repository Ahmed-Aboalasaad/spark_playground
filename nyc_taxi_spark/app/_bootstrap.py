"""Make the project package importable from Streamlit pages.

Streamlit executes each page file as a script, so the project root is not
automatically on ``sys.path``. Importing this module first puts the repository
root on the path, after which ``config``, ``services``, ``pipeline``, and
``spark`` import cleanly. Every page starts with ``import _bootstrap``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # repo root (contains config.py)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
