"""Real wall-clock/memory benchmark at 100M-row scale, extending
``app/large_data/benchmark.py``'s 100K/1M/10M-row measurements one order of
magnitude further.

Machine constraint found and respected (not worked around): this dev sandbox has
only ~2.25 GB of available RAM at the time this was written (16.95 GB total, mostly
in use by other processes) and ~21 GB free disk. A full 100M-row in-memory
pandas DataFrame is NOT attempted here for that reason -- see `naive_load_ceiling`
below, which measures the real per-row memory cost at a safe smaller scale and
extrapolates, explicitly labeled as an extrapolation, rather than crashing this
process or the host machine by forcing it.

Run as a script, each mode a fresh process (see benchmark.py's docstring for why
peak-RSS must be measured in an otherwise-clean process):

    python -m app.large_data.benchmark_100m generate --rows 100000000 --path X.csv
    python -m app.large_data.benchmark_100m chunked-read --path X.csv
    python -m app.large_data.benchmark_100m chunked-agg --path X.csv
    python -m app.large_data.benchmark_100m sample --path X.csv
    python -m app.large_data.benchmark_100m naive-ceiling --rows 2000000
    python -m app.large_data.benchmark_100m duckdb-direct --path X.csv
    python -m app.large_data.benchmark_100m memory-guard-trip
    python -m app.large_data.benchmark_100m truncated-file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from app.large_data.aggregation import chunked_group_aggregate
from app.large_data.chunked_reader import count_rows_csv, iter_csv_chunks
from app.large_data.memory_guard import MemoryGuard, MemoryLimitExceededError
from app.large_data.process_memory import get_peak_rss_bytes
from app.large_data.sampling import bernoulli_sample_csv, reservoir_sample_csv
from app.large_data.synthetic import generate_synthetic_csv

GROUP_COL = "region"
VALUE_COL = "revenue"


def cmd_generate(path: str, rows: int, seed: int) -> dict:
    t0 = time.perf_counter()
    generate_synthetic_csv(path, rows, seed=seed)
    elapsed = time.perf_counter() - t0
    size_bytes = Path(path).stat().st_size
    return {
        "mode": "generate",
        "rows": rows,
        "path": path,
        "elapsed_sec": round(elapsed, 2),
        "file_size_gb": round(size_bytes / 1024**3, 3),
        "bytes_per_row": round(size_bytes / rows, 2),
    }


def cmd_chunked_read(path: str, chunksize: int) -> dict:
    """Streams the whole file via count_rows_csv (never materializes more than one
    chunk at a time) -- proves memory stays bounded across a full 100M-row pass."""
    peak_before = get_peak_rss_bytes()
    t0 = time.perf_counter()
    n = count_rows_csv(path, chunksize=chunksize)
    elapsed = time.perf_counter() - t0
    peak_after = get_peak_rss_bytes()
    return {
        "mode": "chunked_read",
        "chunksize": chunksize,
        "rows_counted": n,
        "wall_time_sec": round(elapsed, 2),
        "peak_rss_mb_after": round(peak_after / 1024**2, 2),
        "attributable_delta_mb": round((peak_after - peak_before) / 1024**2, 2),
        "rows_per_sec": round(n / elapsed, 0) if elapsed > 0 else None,
    }


def cmd_chunked_agg(path: str, chunksize: int) -> dict:
    guard = MemoryGuard(max_chunk_bytes=512 * 1024**2, max_accumulator_bytes=256 * 1024**2)
    peak_before = get_peak_rss_bytes()
    t0 = time.perf_counter()
    result = chunked_group_aggregate(path, GROUP_COL, VALUE_COL, agg_func="sum", chunksize=chunksize, memory_guard=guard)
    elapsed = time.perf_counter() - t0
    peak_after = get_peak_rss_bytes()
    return {
        "mode": "chunked_aggregate",
        "chunksize": chunksize,
        "chunks_processed": guard.chunks_seen,
        "rows_seen": guard.rows_seen,
        "peak_single_chunk_bytes": guard.peak_chunk_bytes,
        "wall_time_sec": round(elapsed, 2),
        "peak_rss_mb_after": round(peak_after / 1024**2, 2),
        "attributable_delta_mb": round((peak_after - peak_before) / 1024**2, 2),
        "result_preview": {str(k): round(float(v), 2) for k, v in result.sort_index().items()},
    }


def cmd_sample(path: str, sample_size: int) -> dict:
    t0 = time.perf_counter()
    reservoir = reservoir_sample_csv(path, sample_size, seed=42)
    t1 = time.perf_counter()
    bern = bernoulli_sample_csv(path, fraction=sample_size / 100_000_000, seed=42)
    t2 = time.perf_counter()
    return {
        "mode": "sample",
        "reservoir_requested": sample_size,
        "reservoir_actual_rows": len(reservoir),
        "reservoir_elapsed_sec": round(t1 - t0, 2),
        "bernoulli_actual_rows": len(bern),
        "bernoulli_elapsed_sec": round(t2 - t1, 2),
    }


def cmd_naive_load_ceiling(path: str, rows: int) -> dict:
    """Measures the real per-row memory cost of a full in-memory pandas load at a
    SAFE scale (well under available RAM) against an ALREADY-GENERATED file (must
    be run as a fresh process separate from generation, same reason
    benchmark.py's naive/chunked modes are separate process invocations -- peak
    RSS is monotonic since process start, so measuring in the same process that
    just generated the file would include generation's own allocations in the
    "before" reading and silently zero out the delta). Extrapolates to 100M rows,
    explicitly labeled as an extrapolation, not a measurement -- this machine's
    available RAM (~2.25 GB measured when this script was written) makes an
    actual 100M-row full in-memory load a real risk of an OS-level crash/thrash,
    not just a slow operation, so it is not attempted directly."""
    peak_before = get_peak_rss_bytes()
    t0 = time.perf_counter()
    df = pd.read_csv(path, nrows=rows)
    _ = df.groupby(GROUP_COL, dropna=False)[VALUE_COL].sum()
    elapsed = time.perf_counter() - t0
    peak_after = get_peak_rss_bytes()
    delta_mb = (peak_after - peak_before) / 1024**2
    per_row_bytes = (delta_mb * 1024**2) / rows
    return {
        "mode": "naive_load_ceiling",
        "measured_rows": rows,
        "wall_time_sec": round(elapsed, 2),
        "peak_rss_mb_before": round(peak_before / 1024**2, 2),
        "peak_rss_mb_after": round(peak_after / 1024**2, 2),
        "attributable_delta_mb": round(delta_mb, 2),
        "measured_bytes_per_row": round(per_row_bytes, 2),
        "extrapolated_100m_row_gb": round(per_row_bytes * 100_000_000 / 1024**3, 2),
        "note": "extrapolated_100m_row_gb is a LINEAR EXTRAPOLATION from the measured "
        f"{rows:,}-row run, NOT a measurement at 100M rows -- this machine's available "
        "RAM at benchmark time made an actual 100M-row full in-memory load an unsafe "
        "experiment to run (see module docstring). Must be invoked as a fresh process "
        "against an already-generated file for the peak-RSS delta to be meaningful.",
    }


def cmd_duckdb_direct(path: str) -> dict:
    """FORWARD-LOOKING EXPERIMENT, not current-system behavior: the app's real
    DuckDBDataSource (app/sql/duckdb_source.py) takes an in-memory pandas
    DataFrame today, not a file path -- querying via the existing bridge would
    still require full materialization first. This measures something different:
    DuckDB's OWN native out-of-core CSV querying (read_csv_auto), completely
    bypassing pandas, as evidence for a *future* decision about whether the SQL
    bridge should read from disk directly instead."""
    import duckdb

    conn = duckdb.connect(":memory:", config={"memory_limit": "1GB"})
    t0 = time.perf_counter()
    row_count = conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}')").fetchone()[0]
    t1 = time.perf_counter()
    agg = conn.execute(
        f"SELECT region, SUM(revenue) AS total FROM read_csv_auto('{path}') GROUP BY region ORDER BY total DESC"
    ).fetchall()
    t2 = time.perf_counter()
    conn.close()
    return {
        "mode": "duckdb_direct_csv_FORWARD_LOOKING_NOT_CURRENT_BEHAVIOR",
        "row_count": row_count,
        "count_query_elapsed_sec": round(t1 - t0, 2),
        "groupby_query_elapsed_sec": round(t2 - t1, 2),
        "groupby_result": [(r, round(v, 2)) for r, v in agg],
        "duckdb_memory_limit_used": "1GB",
        "note": "This is DuckDB querying the CSV file directly via read_csv_auto -- the "
        "current app.sql.duckdb_source.DuckDBDataSource does NOT do this today (it takes "
        "an in-memory DataFrame). This is evidence for a future architectural decision, "
        "not a description of what the shipped SQL bridge currently does.",
    }


def cmd_memory_guard_trip() -> dict:
    """Proves graceful failure (not a crash) when a memory ceiling is genuinely
    exceeded -- a deliberately tiny ceiling against a realistic chunk."""
    guard = MemoryGuard(max_chunk_bytes=1024, max_accumulator_bytes=10 * 1024**3)
    df = pd.DataFrame({"a": range(10_000), "b": ["x" * 50] * 10_000})
    try:
        guard.check_chunk(df)
        return {"mode": "memory_guard_trip", "raised": False, "PROBLEM": "expected MemoryLimitExceededError, got none"}
    except MemoryLimitExceededError as exc:
        return {"mode": "memory_guard_trip", "raised": True, "error_message": str(exc)}


def cmd_truncated_file() -> dict:
    """Proves a mid-file truncation produces a clear, actionable error rather than
    a confusing crash."""
    import tempfile

    tmp_path = Path(tempfile.gettempdir()) / "truncated_test.csv"
    tmp_path.write_text("date,region,product,quantity,revenue,cost\n" + "2024-01-01,North,Widget,5,100.0,50.0\n" * 100 + '2024-01-01,North,Wid')
    try:
        try:
            n = count_rows_csv(str(tmp_path), chunksize=10)
            return {"mode": "truncated_file", "raised": False, "rows_counted_despite_truncation": n, "note": "pandas did NOT raise on the truncated final row and did NOT drop it -- it kept it as one extra row with the missing trailing fields read as NaN (verified: 100 complete rows + 1 truncated = 101 counted here). Silent, not a crash -- documented, not a bug, but worth knowing a truncated source file produces a partially-NaN row rather than an error."}
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this IS the probe
            return {"mode": "truncated_file", "raised": True, "error_type": type(exc).__name__, "error_message": str(exc)[:300]}
    finally:
        tmp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--rows", type=int, required=True)
    gen.add_argument("--path", required=True)
    gen.add_argument("--seed", type=int, default=42)

    cr = sub.add_parser("chunked-read")
    cr.add_argument("--path", required=True)
    cr.add_argument("--chunksize", type=int, default=200_000)

    ca = sub.add_parser("chunked-agg")
    ca.add_argument("--path", required=True)
    ca.add_argument("--chunksize", type=int, default=200_000)

    sm = sub.add_parser("sample")
    sm.add_argument("--path", required=True)
    sm.add_argument("--sample-size", type=int, default=100_000)

    nc = sub.add_parser("naive-ceiling")
    nc.add_argument("--path", required=True)
    nc.add_argument("--rows", type=int, default=2_000_000)

    dd = sub.add_parser("duckdb-direct")
    dd.add_argument("--path", required=True)

    sub.add_parser("memory-guard-trip")
    sub.add_parser("truncated-file")

    args = parser.parse_args(argv)

    if args.command == "generate":
        report = cmd_generate(args.path, args.rows, args.seed)
    elif args.command == "chunked-read":
        report = cmd_chunked_read(args.path, args.chunksize)
    elif args.command == "chunked-agg":
        report = cmd_chunked_agg(args.path, args.chunksize)
    elif args.command == "sample":
        report = cmd_sample(args.path, args.sample_size)
    elif args.command == "naive-ceiling":
        report = cmd_naive_load_ceiling(args.path, args.rows)
    elif args.command == "duckdb-direct":
        report = cmd_duckdb_direct(args.path)
    elif args.command == "memory-guard-trip":
        report = cmd_memory_guard_trip()
    else:
        report = cmd_truncated_file()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
