"""Large-data handling: chunked reading, chunked aggregation, sampling, and an
explicit memory ceiling guard for datasets too big to load into one DataFrame.

This package is additive and self-contained. It does NOT modify
``app/datasets/storage.py`` (the existing single-DataFrame-in-memory model) — it is
an alternative code path for callers who know up front they are dealing with a
large file (or want a fast approximate answer) and would rather stream/chunk than
load everything at once.

Wave 1 status: built standalone (no ``app/data/`` contracts existed yet in this
worktree at the time of writing — see ``.agent/dependency_graph.md``, which notes
LARGE-DATA depends on DATA-ARCHITECT's contracts). Public functions here take a
plain file path rather than a ``DataSource``/``Dataset`` object so integration with
those contracts, once they land, is a thin adapter rather than a rewrite.
"""

from app.large_data.aggregation import chunked_group_aggregate
from app.large_data.chunked_reader import count_rows_csv, iter_csv_chunks
from app.large_data.memory_guard import MemoryGuard, MemoryLimitExceededError
from app.large_data.sampling import bernoulli_sample_csv, reservoir_sample_csv

__all__ = [
    "chunked_group_aggregate",
    "count_rows_csv",
    "iter_csv_chunks",
    "MemoryGuard",
    "MemoryLimitExceededError",
    "bernoulli_sample_csv",
    "reservoir_sample_csv",
]
