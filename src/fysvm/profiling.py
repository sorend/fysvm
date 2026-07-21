"""Small profiling helpers for benchmark scripts."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from time import perf_counter
from typing import TypeVar


T = TypeVar("T")


def timed_peak_memory(call: Callable[[], T]) -> tuple[T, float, float]:
    """Run ``call`` and return ``(result, seconds, peak_memory_mb)``."""

    tracemalloc.start()
    start = perf_counter()
    try:
        result = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, perf_counter() - start, peak / (1024.0 * 1024.0)
