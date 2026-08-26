"""Sampling for large files: get a representative subset without a full scan,
for cases where an approximate/preview answer is acceptable and faster than
processing every row.

Two strategies:

- ``reservoir_sample_csv``: exact-size uniform random sample (algorithm R),
  streamed one chunk at a time. Guarantees exactly ``sample_size`` rows (or all
  rows, if the file has fewer), each row equally likely to be included,
  regardless of file size — the classic answer to "give me N random rows from a
  stream I can't hold in memory."
- ``bernoulli_sample_csv``: include each row independently with probability
  ``fraction``. Cheaper (single pass, no reservoir bookkeeping) but the output
  size is only approximately ``fraction * N`` (binomially distributed), not exact.
  Useful when an approximate preview fraction is fine and you don't need a
  precise row count.
"""

from __future__ import annotations

from os import PathLike

import numpy as np
import pandas as pd

from app.large_data.chunked_reader import DEFAULT_CHUNKSIZE, iter_csv_chunks


def reservoir_sample_csv(
    path: str | PathLike[str],
    sample_size: int,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """Uniform random sample of exactly ``min(sample_size, total_rows)`` rows,
    streamed via algorithm R (Vitter), vectorized per chunk so this stays fast at
    million-row scale instead of doing a Python-level loop over every row.

    Per chunk: once the reservoir is full, each of the chunk's rows independently
    draws a random rank ``j`` in ``[0, seen_so_far)``; a row replaces reservoir
    slot ``j`` iff ``j < sample_size``. Only rows that win a slot (a small
    fraction once ``seen`` is large — expected O(k * log(n/k)) total replacements
    for a stream of n rows and reservoir size k) are applied, and applied in
    stream order so a later winner correctly overwrites an earlier one that
    targeted the same slot. This is the same algorithm as the naive row-by-row
    version, just batched for speed.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    rng = np.random.default_rng(seed)

    reservoir: pd.DataFrame | None = None
    seen = 0
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        chunk = chunk.reset_index(drop=True)
        n = len(chunk)
        if n == 0:
            continue

        if reservoir is None:
            if n <= sample_size:
                reservoir = chunk.copy()
                seen = n
                continue
            idx = rng.choice(n, size=sample_size, replace=False)
            reservoir = chunk.iloc[idx].reset_index(drop=True)
            seen = n
            continue

        offset = 0
        if len(reservoir) < sample_size:
            # Reservoir not yet full: fill it uniformly from the front of this
            # chunk first, then run algorithm R on the remainder.
            room = sample_size - len(reservoir)
            take = min(room, n)
            idx = rng.choice(n, size=take, replace=False)
            reservoir = pd.concat([reservoir, chunk.iloc[idx]], ignore_index=True)
            remaining_mask = np.ones(n, dtype=bool)
            remaining_mask[idx] = False
            chunk = chunk.loc[remaining_mask].reset_index(drop=True)
            seen += take
            n = len(chunk)
            if len(reservoir) < sample_size or n == 0:
                continue

        # Vectorized algorithm R over the rest of this chunk.
        ranks = seen + 1 + np.arange(n)  # seen-count at each row, 1-indexed
        js = rng.integers(0, ranks)  # uniform in [0, ranks[i]) per row
        seen += n
        winners = np.flatnonzero(js < sample_size)
        if winners.size:
            # Batched, not a per-winner `.iloc[slot] = ...` loop (found via a real
            # 100M-row benchmark: the per-row assignment loop took ~6150s -- ~87x
            # slower than a comparable full pass with chunked_reader alone -- even
            # though the winner *count* stays small as designed; pandas' per-row
            # `.iloc` scalar assignment has high constant-factor overhead that
            # doesn't show up until this scale). `.iloc[fancy_index] = ...` applies
            # sequentially left-to-right for duplicate target indices (verified:
            # numpy/pandas fancy-index assignment is "last write wins" for repeated
            # indices), which is the exact same semantic the row-by-row loop had --
            # this is a performance fix, not a behavior change, and the existing
            # determinism/correctness tests (test_large_data.py) are unchanged.
            slots = js[winners]
            reservoir.iloc[slots] = chunk.iloc[winners].to_numpy()

    if reservoir is None:
        return pd.DataFrame()
    return reservoir.reset_index(drop=True)


def bernoulli_sample_csv(
    path: str | PathLike[str],
    fraction: float,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """Include each row independently with probability ``fraction``. Single pass,
    output size is approximately (not exactly) ``fraction * total_rows``."""
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1].")
    rng = np.random.default_rng(seed)

    parts: list[pd.DataFrame] = []
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        mask = rng.random(len(chunk)) < fraction
        if mask.any():
            parts.append(chunk.loc[mask])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
