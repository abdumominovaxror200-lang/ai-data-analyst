from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.large_data.aggregation import AggregationError, chunked_group_aggregate
from app.large_data.chunked_reader import count_rows_csv, iter_csv_chunks
from app.large_data.memory_guard import MemoryGuard, MemoryLimitExceededError, suggest_chunksize
from app.large_data.sampling import bernoulli_sample_csv, reservoir_sample_csv


@pytest.fixture
def large_ish_df() -> pd.DataFrame:
    """A few thousand rows — small enough to run fast in CI, big enough to span
    many chunks at a small chunksize and to exercise multiple groups per chunk."""
    rng = np.random.default_rng(7)
    n = 5_000
    return pd.DataFrame(
        {
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": rng.choice(["Widget", "Gadget", "Gizmo"], n),
            "revenue": rng.normal(500, 120, n).round(2),
            "quantity": rng.integers(1, 100, n),
        }
    )


@pytest.fixture
def csv_path(large_ish_df, tmp_path):
    path = tmp_path / "large_ish.csv"
    large_ish_df.to_csv(path, index=False)
    return path


# --- Chunked reading -------------------------------------------------------


def test_iter_csv_chunks_never_exceeds_requested_chunksize(csv_path):
    for chunk in iter_csv_chunks(csv_path, chunksize=777):
        assert len(chunk) <= 777


def test_iter_csv_chunks_covers_every_row_exactly_once(csv_path, large_ish_df):
    total = sum(len(chunk) for chunk in iter_csv_chunks(csv_path, chunksize=500))
    assert total == len(large_ish_df)


def test_count_rows_csv_matches_naive_len(csv_path, large_ish_df):
    assert count_rows_csv(csv_path, chunksize=333) == len(large_ish_df)


# --- Chunked aggregation correctness (the core deliverable) ----------------


@pytest.mark.parametrize("agg_func", ["sum", "mean", "count", "min", "max"])
def test_chunked_aggregate_matches_naive_full_load(csv_path, large_ish_df, agg_func):
    """The load-bearing correctness check: chunked aggregation over a streamed
    CSV must produce the same numbers as pandas' own full-load groupby/agg."""
    value_col = None if agg_func == "count" else "revenue"
    chunked = chunked_group_aggregate(
        csv_path, "region", value_col, agg_func=agg_func, chunksize=417
    ).sort_index()

    if agg_func == "count":
        naive = large_ish_df.groupby("region", dropna=False).size().sort_index()
    else:
        naive = large_ish_df.groupby("region", dropna=False)["revenue"].agg(agg_func).sort_index()

    pd.testing.assert_series_equal(chunked, naive, check_names=False, check_dtype=False, rtol=1e-9)


def test_chunked_group_sum_robust_to_chunk_boundaries_splitting_groups(tmp_path):
    """Deliberately tiny chunksize so almost every group is split across many
    chunks — this is exactly the scenario a naive 'average the per-chunk means'
    bug would get wrong for agg_func='mean'."""
    df = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "value": [10, 100, 20, 200, 30, 300, 40, 400],
        }
    )
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)

    chunked_mean = chunked_group_aggregate(path, "group", "value", agg_func="mean", chunksize=1).sort_index()
    naive_mean = df.groupby("group")["value"].mean().sort_index()
    pd.testing.assert_series_equal(chunked_mean, naive_mean, check_names=False, check_dtype=False)

    # Sanity: naive mean-of-per-chunk-means (the bug this guards against) would
    # give a different, wrong answer here since chunksize=1 makes every chunk a
    # single row — confirm our result is NOT what that broken approach would give
    # for an uneven split, using a second, unevenly-grouped fixture.
    uneven = pd.DataFrame({"group": ["A", "A", "A", "B"], "value": [10, 10, 100, 50]})
    uneven_path = tmp_path / "uneven.csv"
    uneven.to_csv(uneven_path, index=False)
    chunked_uneven_mean = chunked_group_aggregate(
        uneven_path, "group", "value", agg_func="mean", chunksize=2
    ).sort_index()
    correct_mean = uneven.groupby("group")["value"].mean().sort_index()
    pd.testing.assert_series_equal(chunked_uneven_mean, correct_mean, check_names=False, check_dtype=False)
    # correct mean for A is (10+10+100)/3 = 40, not a naive average of per-chunk
    # means (chunk1 mean=10, chunk2 mean=100 -> naive average would be 55).
    assert chunked_uneven_mean["A"] == pytest.approx(40.0)


def test_chunked_aggregate_rejects_unknown_column(csv_path):
    with pytest.raises(AggregationError):
        chunked_group_aggregate(csv_path, "not_a_column", "revenue", agg_func="sum")


def test_chunked_aggregate_rejects_unsupported_func(csv_path):
    with pytest.raises(AggregationError):
        chunked_group_aggregate(csv_path, "region", "revenue", agg_func="std")


def test_chunked_aggregate_mean_requires_value_col(csv_path):
    with pytest.raises(AggregationError):
        chunked_group_aggregate(csv_path, "region", None, agg_func="mean")


# --- Sampling ----------------------------------------------------------------


def test_reservoir_sample_returns_exact_size(csv_path, large_ish_df):
    sample = reservoir_sample_csv(csv_path, sample_size=250, chunksize=600, seed=1)
    assert len(sample) == 250
    assert set(sample.columns) == set(large_ish_df.columns)


def test_reservoir_sample_size_capped_at_total_rows(tmp_path):
    df = pd.DataFrame({"x": range(10)})
    path = tmp_path / "small.csv"
    df.to_csv(path, index=False)
    sample = reservoir_sample_csv(path, sample_size=1_000, chunksize=3, seed=1)
    assert len(sample) == 10  # can't sample more rows than exist


def test_reservoir_sample_is_deterministic_given_seed(csv_path):
    a = reservoir_sample_csv(csv_path, sample_size=100, chunksize=400, seed=99)
    b = reservoir_sample_csv(csv_path, sample_size=100, chunksize=400, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_reservoir_sample_rows_come_from_source_data(csv_path, large_ish_df):
    sample = reservoir_sample_csv(csv_path, sample_size=50, chunksize=250, seed=3)
    valid_regions = set(large_ish_df["region"].unique())
    assert set(sample["region"].unique()) <= valid_regions


def test_bernoulli_sample_size_roughly_matches_fraction(csv_path, large_ish_df):
    sample = bernoulli_sample_csv(csv_path, fraction=0.1, chunksize=500, seed=5)
    expected = len(large_ish_df) * 0.1
    # Binomial(n=5000, p=0.1): std dev ~= sqrt(5000*0.1*0.9) ~= 21.2 -> allow 6 sigma
    assert abs(len(sample) - expected) < 150


def test_bernoulli_sample_rejects_bad_fraction(csv_path):
    with pytest.raises(ValueError):
        bernoulli_sample_csv(csv_path, fraction=0.0)
    with pytest.raises(ValueError):
        bernoulli_sample_csv(csv_path, fraction=1.5)


# --- Memory guard ------------------------------------------------------------


def test_memory_guard_allows_reasonable_chunk(large_ish_df):
    guard = MemoryGuard(max_chunk_bytes=10 * 1024 * 1024, max_accumulator_bytes=10 * 1024 * 1024)
    guard.check_chunk(large_ish_df)  # ~5000 rows, well under 10MB -> no raise
    assert guard.chunks_seen == 1
    assert guard.rows_seen == len(large_ish_df)


def test_memory_guard_raises_on_oversized_chunk(large_ish_df):
    tiny_ceiling = 100  # bytes — no real chunk fits this
    guard = MemoryGuard(max_chunk_bytes=tiny_ceiling, max_accumulator_bytes=10 * 1024 * 1024)
    with pytest.raises(MemoryLimitExceededError):
        guard.check_chunk(large_ish_df)


def test_memory_guard_raises_on_oversized_accumulator():
    guard = MemoryGuard(max_chunk_bytes=10 * 1024 * 1024, max_accumulator_bytes=50)
    big_series = pd.Series(np.arange(10_000, dtype="float64"))
    with pytest.raises(MemoryLimitExceededError):
        guard.check_accumulator(big_series)


def test_memory_guard_rejects_nonpositive_ceilings():
    with pytest.raises(ValueError):
        MemoryGuard(max_chunk_bytes=0, max_accumulator_bytes=100)
    with pytest.raises(ValueError):
        MemoryGuard(max_chunk_bytes=100, max_accumulator_bytes=-1)


def test_chunked_read_stops_via_memory_guard_before_finishing(csv_path):
    """End-to-end: a memory_guard wired into iter_csv_chunks actually interrupts
    a chunked read partway through, rather than silently letting it complete."""
    guard = MemoryGuard(max_chunk_bytes=200, max_accumulator_bytes=10 * 1024 * 1024)
    consumed = 0
    with pytest.raises(MemoryLimitExceededError):
        for _ in iter_csv_chunks(csv_path, chunksize=500, memory_guard=guard):
            consumed += 1
    assert consumed == 0  # the very first chunk already exceeds a 200-byte ceiling


def test_suggest_chunksize_scales_inversely_with_row_weight():
    light_df = pd.DataFrame({"a": range(100)})
    heavy_df = pd.DataFrame({"a": range(100), "b": ["x" * 1000] * 100})
    light_suggestion = suggest_chunksize(light_df, target_chunk_bytes=1_000_000)
    heavy_suggestion = suggest_chunksize(heavy_df, target_chunk_bytes=1_000_000)
    assert heavy_suggestion < light_suggestion
