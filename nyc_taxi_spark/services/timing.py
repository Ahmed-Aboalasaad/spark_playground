"""Execution-time reporting used across the whole application.

Every long-running operation in this project has to report how long it took.
Rather than sprinkle ``time.perf_counter()`` calls through the pipeline, all
timing goes through the two helpers here:

* ``timer()``   -- a context manager for timing an inline block.
* ``@timed``    -- a decorator that wraps a function and returns a ``Result``.

Both measure real wall-clock time via ``time.perf_counter()``. Even while the
pipeline returns mock data, the timings are real: they exercise the machinery
end-to-end and stay honest about how trivial the stub work is.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """A processing result bundled with its execution metadata.

    Attributes
    ----------
    value:
        Whatever the operation produced (a DataFrame, a metrics dict, a model).
    elapsed:
        Wall-clock seconds the operation took.
    n_records:
        Number of records processed, when known. ``None`` when not meaningful
        or not yet computed.
    placeholder:
        True when ``value`` is mock/stub data rather than a real computation.
        The UI uses this to decide whether to show a placeholder badge.
    """

    value: T
    elapsed: float
    n_records: int | None = None
    placeholder: bool = False

    @property
    def elapsed_str(self) -> str:
        """Human-friendly elapsed time, e.g. ``'0.003 s'``."""
        return f"{self.elapsed:.3f} s"


class Timer:
    """Mutable stopwatch handed out by the :func:`timer` context manager."""

    __slots__ = ("_start", "_elapsed")

    def __init__(self) -> None:
        self._start = 0.0
        self._elapsed = 0.0

    @property
    def elapsed(self) -> float:
        """Seconds elapsed. Valid once the ``with`` block has exited."""
        return self._elapsed

    @property
    def elapsed_str(self) -> str:
        return f"{self._elapsed:.3f} s"


@contextmanager
def timer() -> Iterator[Timer]:
    """Time an inline block of work.

    Example
    -------
    >>> with timer() as t:
    ...     result = do_some_work()
    >>> print(t.elapsed_str)
    """
    handle = Timer()
    handle._start = time.perf_counter()
    try:
        yield handle
    finally:
        handle._elapsed = time.perf_counter() - handle._start


def timed(func: Callable[..., T]) -> Callable[..., Result[T]]:
    """Wrap ``func`` so calling it returns a :class:`Result`.

    The wrapped function's return value becomes ``Result.value`` and the real
    elapsed time is recorded. If the function returns a ``(value, n_records)``
    tuple *and* declares that convention via the ``_reports_count`` attribute,
    the count is lifted into ``Result.n_records``; otherwise ``n_records`` is
    left as ``None``.

    Keeping this thin means services can also construct ``Result`` objects by
    hand (e.g. when they compute the record count separately).
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Result[T]:
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        n_records: int | None = None
        if getattr(func, "_reports_count", False) and isinstance(value, tuple):
            value, n_records = value  # type: ignore[misc]

        placeholder = getattr(func, "_placeholder", False)
        return Result(value=value, elapsed=elapsed, n_records=n_records,
                      placeholder=placeholder)

    return wrapper
