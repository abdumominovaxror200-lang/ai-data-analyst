"""Chunked CSV reading: never hold the full file in memory at once.

Starts with CSV because it's the simplest streamable format (pandas'
``read_csv(..., chunksize=N)`` gives a true streaming iterator over the file).

XLSX follow-up (not implemented in Wave 1, noted for the next pass): openpyxl
supports read-only streaming via ``load_workbook(path, read_only=True)``, which
gives a row-by-row iterator without materializing the whole sheet — a chunked
XLSX reader would batch that row iterator into DataFrames the same shape as
``iter_csv_chunks`` below. Not built here because the existing upload path
(``app/datasets/storage.py``) already fully loads XLSX via ``pd.read_excel``, and
CSV is the higher-value target for the "100K/1M/10M rows" scenario this task is
about — XLSX files that large are rare and openpyxl's per-cell Python object
overhead makes them a poor fit for chunked processing regardless.
"""

from __future__ import annotations

from collections.abc import Iterator
from os import PathLike

import pandas as pd

from app.large_data.memory_guard import MemoryGuard

DEFAULT_CHUNKSIZE = 50_000


def iter_csv_chunks(
    path: str | PathLike[str],
    chunksize: int = DEFAULT_CHUNKSIZE,
    *,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    memory_guard: MemoryGuard | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream a CSV file in fixed-size chunks.

    At no point is the full file held in memory — ``pandas.read_csv(chunksize=...)``
    keeps an open file handle and parses ``chunksize`` rows at a time on each
    ``next()``. If ``memory_guard`` is given, each chunk's footprint is checked
    before it's yielded, so a caller iterating this generator can never receive a
    chunk that exceeds the configured ceiling.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
    )
    with reader:
        for chunk in reader:
            if memory_guard is not None:
                memory_guard.check_chunk(chunk)
            yield chunk


def count_rows_csv(path: str | PathLike[str], chunksize: int = DEFAULT_CHUNKSIZE) -> int:
    """Count rows in a CSV without ever materializing more than one chunk at a time."""
    total = 0
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        total += len(chunk)
    return total
