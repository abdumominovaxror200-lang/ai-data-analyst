"""Real wall-clock time / peak-memory benchmark: chunked group-by-and-sum vs. the
naive full-load-then-groupby approach, at configurable row scale.

Run as a script (each invocation is a fresh process, which matters — see
``process_memory.get_peak_rss_bytes``'s docstring for why peak-RSS is only
meaningful when measured in an otherwise-clean process rather than reused across
both approaches in one run):

    python -m app.large_data.benchmark generate --rows 1000000 --path X.csv
    python -m app.large_data.benchmark run --path X.csv --mode naive
    python -m app.large_data.benchmark run --path X.csv --mode chunked --chunksize 50000

Both modes compute the same result: total ``revenue`` summed by ``region``. The
correctness half of the claim (chunked == naive) is verified in
``backend/tests/test_large_data.py`` on small fixtures where it's fast enough to
run in every CI run; this script is for the large-scale wall-clock/memory numbers
themselves, which are a one-off measurement, not a permanent test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from app.large_data.aggregation import chunked_group_aggregate
from app.large_data.memory_guard import MemoryGuard
from app.large_data.process_memory import get_peak_rss_bytes
from app.large_data.synthetic import generate_synthetic_csv

GROUP_COL = "region"
VALUE_COL = "revenue"


def naive_group_sum(path: str) -> pd.Series:
    """The approach the app uses today (``app/datasets/storage.py``): the whole
    file is parsed into one in-memory DataFrame, then aggregated."""
    df = pd.read_csv(path)
    return df.groupby(GROUP_COL, dropna=False)[VALUE_COL].sum()


def run_naive(path: str) -> dict:
    peak_before = get_peak_rss_bytes()
    t0 = time.perf_counter()
    result = naive_group_sum(path)
    elapsed = time.perf_counter() - t0
    peak_after = get_peak_rss_bytes()
    return _report("naive_full_load", elapsed, peak_before, peak_after, result)


def run_chunked(path: str, chunksize: int) -> dict:
    # Ceilings sized generously for a benchmark run (not exercising the guard's
    # rejection path here — that's covered by dedicated unit tests). 2 GB per
    # chunk and 2 GB for the accumulator is far above what this workload needs;
    # it's there so a chunked run of an unexpectedly huge/wide file still fails
    # loudly instead of silently consuming all available RAM.
    guard = MemoryGuard(max_chunk_bytes=2 * 1024**3, max_accumulator_bytes=2 * 1024**3)
    peak_before = get_peak_rss_bytes()
    t0 = time.perf_counter()
    result = chunked_group_aggregate(
        path, GROUP_COL, VALUE_COL, agg_func="sum", chunksize=chunksize, memory_guard=guard
    )
    elapsed = time.perf_counter() - t0
    peak_after = get_peak_rss_bytes()
    report = _report("chunked", elapsed, peak_before, peak_after, result)
    report["chunksize"] = chunksize
    report["chunks_processed"] = guard.chunks_seen
    report["peak_single_chunk_bytes"] = guard.peak_chunk_bytes
    return report


def _report(mode: str, elapsed: float, peak_before: int, peak_after: int, result: pd.Series) -> dict:
    return {
        "mode": mode,
        "wall_time_sec": round(elapsed, 4),
        "peak_rss_bytes_before": peak_before,
        "peak_rss_bytes_after": peak_after,
        "peak_rss_mb_after": round(peak_after / 1024**2, 2),
        "attributable_delta_mb": round((peak_after - peak_before) / 1024**2, 2),
        "result_preview": {str(k): round(float(v), 2) for k, v in result.sort_index().items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Write a synthetic CSV.")
    gen.add_argument("--rows", type=int, required=True)
    gen.add_argument("--path", required=True)
    gen.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="Run one benchmark mode against an existing CSV.")
    run.add_argument("--path", required=True)
    run.add_argument("--mode", choices=["naive", "chunked"], required=True)
    run.add_argument("--chunksize", type=int, default=50_000)

    args = parser.parse_args(argv)

    if args.command == "generate":
        t0 = time.perf_counter()
        generate_synthetic_csv(args.path, args.rows, seed=args.seed)
        elapsed = time.perf_counter() - t0
        print(json.dumps({"generated_rows": args.rows, "path": args.path, "elapsed_sec": round(elapsed, 2)}))
        return

    if args.mode == "naive":
        report = run_naive(args.path)
    else:
        report = run_chunked(args.path, args.chunksize)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
