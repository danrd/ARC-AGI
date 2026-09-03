"""Lightweight resource-usage guard for tests.

Wraps a block of code, measures wall-clock time and peak Python memory
growth (via stdlib tracemalloc - this does track numpy array data, not
just plain Python objects, since numpy registers its allocator with it),
and asserts both stay within given bounds. Meant to be dropped into
individual tests where a bug would plausibly show up as unexpected
slowdown or memory growth (repeated env stepping/MCTS search, repeated
grid-object construction) - not a blanket fixture applied to every test.

    from tests.resource_utils import resource_budget

    def test_something():
        with resource_budget(max_seconds=2.0, max_memory_mb=50.0):
            do_the_expensive_thing()

Either bound can be omitted to skip that check. If the wrapped block
raises, the budget checks are skipped and the original exception
propagates - a real bug shouldn't get masked by a budget assertion.
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from typing import Optional


@contextmanager
def resource_budget(max_seconds: Optional[float] = None, max_memory_mb: Optional[float] = None):
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()  # ignore whatever was already allocated before this block
    baseline, _ = tracemalloc.get_traced_memory()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()

    if max_seconds is not None:
        assert duration <= max_seconds, \
            f"Exceeded time budget: {duration:.2f}s > {max_seconds:.2f}s"
    if max_memory_mb is not None:
        peak_growth_mb = (peak - baseline) / (1024 * 1024)
        assert peak_growth_mb <= max_memory_mb, \
            f"Exceeded memory budget: {peak_growth_mb:.2f}MB > {max_memory_mb:.2f}MB peak allocation growth"
