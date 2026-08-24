"""Chunked aggregation: compute group-by aggregations over a CSV too large to
load into one DataFrame, by processing fixed-size chunks and combining partial
results — verified (in ``backend/tests/test_large_data.py``) to match the same
aggregation computed the naive full-load way on a small dataset, byte-for-byte
on the numeric result.

Supported aggregations and how partials combine:

- ``sum``:   per-chunk groupby-sum, folded into a running total via ``Series.add``.
- ``count``: per-chunk groupby-size, folded the same way.
- ``mean``:  computed as running ``sum`` / running ``count`` (a mean cannot be
             averaged across chunks directly — the classic bug is averaging
             per-chunk means, which is only correct if every chunk has equal
             size *and* equal group distribution; this computes the true mean).
- ``min``/``max``: per-chunk groupby-min/max, folded via elementwise min/max
             against the running accumulator.

This is the standard streaming/online aggregation pattern (the same idea SQL
engines use for partial aggregation before a final merge step).
"""

from __future__ import annotations

from os import PathLike

import pandas as pd

from app.large_data.chunked_reader import DEFAULT_CHUNKSIZE, iter_csv_chunks
from app.large_data.memory_guard import MemoryGuard

_SUPPORTED = {"sum", "mean", "count", "min", "max"}


class AggregationError(Exception):
    """Raised for invalid aggregation requests (bad column/func), mirroring the
    validation style of the existing ``app/tools`` (see ``ToolExecutionError``) —
    kept as a distinct type here since this package doesn't depend on
    ``app/tools`` (no existing coupling to introduce)."""


def chunked_group_aggregate(
    path: str | PathLike[str],
    group_col: str,
    value_col: str | None,
    agg_func: str = "sum",
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    memory_guard: MemoryGuard | None = None,
) -> pd.Series:
    """Compute ``df.groupby(group_col)[value_col].agg(agg_func)`` for a CSV at
    ``path`` without ever loading the full file into memory.

    Returns a ``pd.Series`` indexed by group, same shape as the naive
    ``DataFrame.groupby(...).agg(...)`` call would produce (sorted is left to the
    caller — this returns in first-seen group order across chunks).
    """
    if agg_func not in _SUPPORTED:
        raise AggregationError(f"Unsupported aggregation '{agg_func}'. Use one of {sorted(_SUPPORTED)}.")
    if agg_func != "count" and value_col is None:
        raise AggregationError(f"agg_func='{agg_func}' requires a value_col.")

    if agg_func == "sum":
        return _chunked_sum(path, group_col, value_col, chunksize, memory_guard)
    if agg_func == "count":
        return _chunked_count(path, group_col, chunksize, memory_guard)
    if agg_func == "mean":
        return _chunked_mean(path, group_col, value_col, chunksize, memory_guard)
    if agg_func == "min":
        return _chunked_extreme(path, group_col, value_col, chunksize, memory_guard, how="min")
    return _chunked_extreme(path, group_col, value_col, chunksize, memory_guard, how="max")


def _chunked_sum(path, group_col, value_col, chunksize, memory_guard) -> pd.Series:
    running: pd.Series | None = None
    for chunk in iter_csv_chunks(path, chunksize=chunksize, memory_guard=memory_guard):
        _validate_columns(chunk, group_col, value_col)
        partial = chunk.groupby(group_col, dropna=False)[value_col].sum()
        running = partial if running is None else running.add(partial, fill_value=0)
        if memory_guard is not None:
            memory_guard.check_accumulator(running)
    return running if running is not None else pd.Series(dtype="float64")


def _chunked_count(path, group_col, chunksize, memory_guard) -> pd.Series:
    running: pd.Series | None = None
    for chunk in iter_csv_chunks(path, chunksize=chunksize, memory_guard=memory_guard):
        _validate_columns(chunk, group_col, None)
        partial = chunk.groupby(group_col, dropna=False).size()
        running = partial if running is None else running.add(partial, fill_value=0)
        if memory_guard is not None:
            memory_guard.check_accumulator(running)
    return running.astype("int64") if running is not None else pd.Series(dtype="int64")


def _chunked_mean(path, group_col, value_col, chunksize, memory_guard) -> pd.Series:
    running_sum: pd.Series | None = None
    running_count: pd.Series | None = None
    for chunk in iter_csv_chunks(path, chunksize=chunksize, memory_guard=memory_guard):
        _validate_columns(chunk, group_col, value_col)
        grouped = chunk.groupby(group_col, dropna=False)[value_col]
        partial_sum = grouped.sum()
        partial_count = grouped.count()
        running_sum = partial_sum if running_sum is None else running_sum.add(partial_sum, fill_value=0)
        running_count = partial_count if running_count is None else running_count.add(partial_count, fill_value=0)
        if memory_guard is not None:
            memory_guard.check_accumulator(running_sum)
            memory_guard.check_accumulator(running_count)
    if running_sum is None or running_count is None:
        return pd.Series(dtype="float64")
    return running_sum / running_count


def _chunked_extreme(path, group_col, value_col, chunksize, memory_guard, *, how: str) -> pd.Series:
    running: pd.Series | None = None
    for chunk in iter_csv_chunks(path, chunksize=chunksize, memory_guard=memory_guard):
        _validate_columns(chunk, group_col, value_col)
        partial = chunk.groupby(group_col, dropna=False)[value_col].agg(how)
        if running is None:
            running = partial
        else:
            combined = pd.concat([running, partial], axis=1)
            running = combined.min(axis=1) if how == "min" else combined.max(axis=1)
        if memory_guard is not None:
            memory_guard.check_accumulator(running)
    return running if running is not None else pd.Series(dtype="float64")


def _validate_columns(chunk: pd.DataFrame, group_col: str, value_col: str | None) -> None:
    if group_col not in chunk.columns:
        raise AggregationError(f"Unknown column '{group_col}'.")
    if value_col is not None and value_col not in chunk.columns:
        raise AggregationError(f"Unknown column '{value_col}'.")
