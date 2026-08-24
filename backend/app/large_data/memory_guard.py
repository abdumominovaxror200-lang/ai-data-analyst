"""Explicit memory-ceiling enforcement for large-data operations.

The rest of this package (chunked reading, chunked aggregation, sampling) never
holds a full large file in memory at once — but two things can still blow up RAM
even in a "chunked" pipeline, and this module guards both:

1. A single chunk itself can be too big (caller passed an unreasonable
   ``chunksize``, or a CSV has extremely wide/heavy rows).
2. The *running accumulator* a chunked aggregation folds partial results into can
   grow unboundedly if the group-by key has very high cardinality (e.g.
   grouping by a near-unique ID column) — chunking the input does not chunk the
   output.

``MemoryGuard`` tracks both against caller-supplied byte ceilings and raises
``MemoryLimitExceededError`` — a clear, catchable error — instead of letting either
grow without bound. It intentionally does NOT try to answer "how much RAM does
the whole OS process have free" (that's environment-dependent and racy); it
measures the concrete object sizes actually flowing through this pipeline, which
is exact, cheap (`DataFrame.memory_usage(deep=True)` is O(n) but that cost is
already implied by reading the chunk), and deterministic enough to unit-test.

For the benchmark script (measuring real wall-clock/RSS numbers, not enforcing a
ceiling), see ``app/large_data/process_memory.py`` for a portable "how much memory
is this OS process actually using" reading — that one *does* talk to the OS
because for reporting real numbers an estimate isn't good enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class MemoryLimitExceededError(Exception):
    """Raised when a large-data operation would exceed its configured memory ceiling.

    This is a normal, expected control-flow error for oversized inputs — callers
    should catch it and surface a clean message (e.g. "reduce chunk size" or "use
    sampling instead of a full scan"), the same way ``ValidationError`` is handled
    in the existing upload path (``app/datasets/validation.py``).
    """


@dataclass
class MemoryGuard:
    """Tracks memory used by chunks and by an aggregation accumulator against
    configurable ceilings, raising ``MemoryLimitExceededError`` on breach.

    Parameters
    ----------
    max_chunk_bytes:
        Ceiling for any single chunk's estimated in-memory footprint
        (``DataFrame.memory_usage(deep=True).sum()``). Guards against an
        oversized ``chunksize`` or unexpectedly heavy rows.
    max_accumulator_bytes:
        Ceiling for the running partial-result accumulator a chunked aggregation
        folds into (e.g. a per-group running sum/count). Guards against
        high-cardinality group-by keys defeating the point of chunking the input.
    """

    max_chunk_bytes: int
    max_accumulator_bytes: int
    chunks_seen: int = field(default=0, init=False)
    rows_seen: int = field(default=0, init=False)
    peak_chunk_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive.")
        if self.max_accumulator_bytes <= 0:
            raise ValueError("max_accumulator_bytes must be positive.")

    def check_chunk(self, chunk: pd.DataFrame) -> int:
        """Validate one chunk's memory footprint. Returns its estimated byte size."""
        size = int(chunk.memory_usage(deep=True).sum())
        self.chunks_seen += 1
        self.rows_seen += len(chunk)
        self.peak_chunk_bytes = max(self.peak_chunk_bytes, size)
        if size > self.max_chunk_bytes:
            raise MemoryLimitExceededError(
                f"Chunk #{self.chunks_seen} uses ~{size:,} bytes, which exceeds the "
                f"configured max_chunk_bytes ceiling of {self.max_chunk_bytes:,}. "
                "Reduce chunksize or raise the ceiling explicitly."
            )
        return size

    def check_accumulator(self, accumulator: pd.Series | pd.DataFrame | dict) -> int:
        """Validate the running partial-result accumulator's memory footprint."""
        if isinstance(accumulator, (pd.Series, pd.DataFrame)):
            size = int(accumulator.memory_usage(deep=True).sum() if isinstance(accumulator, pd.DataFrame) else accumulator.memory_usage(deep=True))
        elif isinstance(accumulator, dict):
            # Cheap approximation: sum of key/value sizes. Good enough to catch
            # runaway cardinality growth without the cost of a deep introspection.
            size = sum(_approx_size(k) + _approx_size(v) for k, v in accumulator.items())
        else:
            raise TypeError(f"Unsupported accumulator type: {type(accumulator)!r}")

        if size > self.max_accumulator_bytes:
            raise MemoryLimitExceededError(
                f"Aggregation accumulator has grown to ~{size:,} bytes after "
                f"{self.chunks_seen} chunks ({self.rows_seen:,} rows), which exceeds "
                f"the configured max_accumulator_bytes ceiling of "
                f"{self.max_accumulator_bytes:,}. This usually means the group-by "
                "key has very high cardinality — chunking the input doesn't help "
                "if the output is nearly as large as the input. Consider a coarser "
                "grouping key, sampling, or raising the ceiling explicitly."
            )
        return size


def _approx_size(value: object) -> int:
    import sys

    return sys.getsizeof(value)


def estimate_row_bytes(df: pd.DataFrame) -> float:
    """Average bytes-per-row for a representative DataFrame sample, for sizing a
    ``chunksize`` from a target per-chunk byte budget before reading a large file."""
    if df.empty:
        return 0.0
    return float(df.memory_usage(deep=True).sum()) / len(df)


def suggest_chunksize(sample_df: pd.DataFrame, target_chunk_bytes: int, minimum: int = 1_000) -> int:
    """Pick a chunksize so that a chunk of that many rows, shaped like
    ``sample_df``, is expected to use about ``target_chunk_bytes``."""
    row_bytes = estimate_row_bytes(sample_df)
    if row_bytes <= 0:
        return minimum
    return max(minimum, int(target_chunk_bytes / row_bytes))
