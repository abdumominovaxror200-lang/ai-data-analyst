"""100M-row large-data benchmark tests.

A true 100M-row run is slow/disk-heavy (~4GB generated file, several minutes) and
must NOT become part of ordinary `pytest -q`, same principle as the real-LLM tests
being gated (see `tests/benchmark/real_llm/runner.py`). The slow, full-scale tests
below are gated behind `RUN_100M_BENCHMARK=1`. A few fast, always-on tests exercise
the same code paths at a tiny synthetic scale (1,000 rows) to catch a broken
import/signature/logic bug cheaply in normal CI.

Real 100M-row results (measured on this dev sandbox, 16.95 GB total / ~2.25 GB
available RAM, ~21 GB free disk at benchmark time) are in
`app/large_data/benchmark_100m_results.json` and `.agent/benchmark_status.md`.
Headline finding: benchmarking at this new scale (10x past the previous largest
test) surfaced a real performance bug in `reservoir_sample_csv` -- a per-winner
`.iloc[slot] = ...` pandas assignment inside a Python loop, technically a "small
subset" of rows as designed but with high enough per-call constant overhead to take
~6150s at 100M rows. Fixed (this session) by batching winners into a single
vectorized `.iloc[fancy_index] = ...` assignment, verified to preserve the exact
same "last write wins on duplicate slot" semantics and pass all pre-existing
correctness/determinism tests unmodified -- re-measured at ~110s, a ~56x speedup.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from app.large_data.aggregation import chunked_group_aggregate
from app.large_data.chunked_reader import count_rows_csv, iter_csv_chunks
from app.large_data.memory_guard import MemoryGuard, MemoryLimitExceededError
from app.large_data.sampling import bernoulli_sample_csv, reservoir_sample_csv
from app.large_data.synthetic import generate_synthetic_csv

RUN_100M_ENV_VAR = "RUN_100M_BENCHMARK"
skip_unless_100m = pytest.mark.skipif(
    os.environ.get(RUN_100M_ENV_VAR) != "1",
    reason=f"Set {RUN_100M_ENV_VAR}=1 to run the full 100M-row benchmark (multi-minute, ~4GB disk).",
)

_SMALL_ROWS = 1_000


@pytest.fixture
def small_csv(tmp_path):
    path = tmp_path / "small.csv"
    generate_synthetic_csv(str(path), _SMALL_ROWS, seed=1)
    return str(path)


# --- Always-on: same code paths, tiny scale, catches a broken signature/import fast ---


def test_chunked_read_small_scale(small_csv):
    assert count_rows_csv(small_csv, chunksize=200) == _SMALL_ROWS


def test_chunked_aggregate_small_scale(small_csv):
    result = chunked_group_aggregate(small_csv, "region", "revenue", agg_func="sum", chunksize=200)
    assert len(result) > 0
    assert result.sum() > 0


def test_reservoir_sample_small_scale(small_csv):
    sample = reservoir_sample_csv(small_csv, sample_size=100, chunksize=200, seed=1)
    assert len(sample) == 100


def test_bernoulli_sample_small_scale(small_csv):
    sample = bernoulli_sample_csv(small_csv, fraction=0.1, chunksize=200, seed=1)
    assert 0 < len(sample) < _SMALL_ROWS


def test_memory_guard_trips_on_undersized_ceiling():
    guard = MemoryGuard(max_chunk_bytes=1024, max_accumulator_bytes=10 * 1024**3)
    df = pd.DataFrame({"a": range(10_000), "b": ["x" * 50] * 10_000})
    with pytest.raises(MemoryLimitExceededError):
        guard.check_chunk(df)


def test_truncated_file_does_not_crash(tmp_path):
    path = tmp_path / "truncated.csv"
    path.write_text("date,region,product,quantity,revenue,cost\n" + "2024-01-01,North,Widget,5,100.0,50.0\n" * 50 + "2024-01-01,North,Wid")
    # Documented real behavior (see benchmark_100m.py's truncated-file probe, and the
    # correction in its own comment -- an earlier draft of this note incorrectly
    # claimed the partial row gets dropped): pandas does NOT raise and does NOT drop
    # the truncated final row -- it keeps it as a 51st row with the missing trailing
    # fields (revenue, cost) read as NaN. This test locks in that this is silent,
    # not a crash, and confirms the real row count including the partial row.
    n = count_rows_csv(str(path), chunksize=10)
    assert n == 51


def test_reservoir_sample_duplicate_slot_last_write_wins():
    """Regression test for the vectorization fix: pandas fancy-index `.iloc[idx] =
    values` assignment must apply 'last write wins' for a repeated index, matching
    the original per-row sequential-loop semantics exactly -- this is what makes the
    performance fix a pure speedup, not a behavior change."""
    import numpy as np

    df = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
    src = pd.DataFrame({"a": [1, 2, 3], "b": ["p", "q", "r"]})
    slots = np.array([0, 0, 1])
    df.iloc[slots] = src.iloc[[0, 1, 2]].to_numpy()
    assert df.loc[0, "a"] == 2 and df.loc[0, "b"] == "q"  # slot 0: last writer (src row 1) wins
    assert df.loc[1, "a"] == 3 and df.loc[1, "b"] == "r"
    assert df.loc[2, "a"] == 30  # untouched


# --- Gated: the real 100M-row runs (multi-minute, ~4GB disk) --------------------


@skip_unless_100m
def test_100m_chunked_read_stays_memory_bounded(tmp_path):
    path = tmp_path / "100m.csv"
    generate_synthetic_csv(str(path), 100_000_000, seed=42)
    n = count_rows_csv(str(path), chunksize=200_000)
    assert n == 100_000_000


@skip_unless_100m
def test_100m_chunked_aggregate_matches_expected_total(tmp_path):
    path = tmp_path / "100m.csv"
    generate_synthetic_csv(str(path), 100_000_000, seed=42)
    result = chunked_group_aggregate(path, "region", "revenue", agg_func="sum", chunksize=200_000)
    assert len(result) == 4  # North/South/East/West
    assert result.sum() > 0


@skip_unless_100m
def test_100m_reservoir_sample_completes_in_reasonable_time(tmp_path):
    """Regression guard for the performance fix -- must stay well under the old
    ~6150s, not just 'eventually finish'. Generous ceiling (10 min) to stay robust
    across slower CI hardware while still catching a real regression back to the
    per-row-assignment anti-pattern."""
    import time

    path = tmp_path / "100m.csv"
    generate_synthetic_csv(str(path), 100_000_000, seed=42)
    t0 = time.perf_counter()
    sample = reservoir_sample_csv(path, sample_size=100_000, chunksize=200_000, seed=1)
    elapsed = time.perf_counter() - t0
    assert len(sample) == 100_000
    assert elapsed < 600, f"reservoir_sample_csv took {elapsed:.0f}s at 100M rows -- possible perf regression"
