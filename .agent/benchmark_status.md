# Benchmark Status

**This file is now stale as a "current state" summary** — see `completed_tasks.md`'s
Phase 3C/4 sections for the full, current picture (39 tools, scripted + real-LLM
benchmarks, security controls). Kept below for historical record of the original
manual 7-question run, plus two new real, measured results appended this session.

## 100M-row large-data benchmark (this session)

Full real numbers: `backend/app/large_data/benchmark_100m_results.json`. Headline:
- Chunked reading/aggregation stay memory-bounded (~126 MB peak RSS) at 100M rows —
  confirms the existing chunking design holds one order of magnitude past its
  previous largest test (10M rows).
- **Found and fixed a real performance bug**: `reservoir_sample_csv` took ~6,150s
  (87x slower than a comparable full pass) due to a per-row pandas `.iloc[]`
  assignment anti-pattern inside a loop. Fixed by batching into one vectorized
  assignment — verified behavior-identical (same "last write wins" semantics, all
  pre-existing tests pass unmodified), re-measured at ~110s (**55.8x speedup**).
- The current SQL bridge (`DuckDBDataSource`) and `profile_dataset` both require a
  full in-memory pandas DataFrame — genuinely infeasible at 100M rows on this
  machine (~2.25 GB available RAM; a 100M-row DataFrame needs ~9.2 GB, extrapolated
  from a real 2M-row measurement). Documented as a real, current architectural
  boundary, not fixed this pass.
- Forward-looking experiment: DuckDB querying the 100M-row CSV **directly** (no
  pandas materialization) counted all rows in 5.6s and did a full groupby in 6.7s
  using only a 1GB memory limit — strong evidence for a future SQL-bridge redesign
  to stream from disk instead of requiring an in-memory DataFrame first.

## Real-LLM benchmark (live Groq, this session — see `.agent/decisions.md`)

37-case professional set: 18/37 completed with a real model response before hitting
a sustained provider-capacity wall (55 min, zero recovery across the remaining 19
cases — a hard block, not transient congestion). **Of the 18 real attempts, 100%
passed every applicable structural check.** The 5 categories that got zero real
attempts (ambiguous_questions, business_recommendations, causal_traps, data_quality,
insufficient_data) are **unmeasured**, not "0% failed" — no case in them ever got a
real model response this run. 15-case adversarial set: only 3/15 got a real
response for the same reason (all 3 passed); one additional case (misleading-outlier
average) was separately, manually re-verified live after a targeted fix (see
`decisions.md`) and confirmed corrected. Full detail in the chat transcript's Phase
5 report — not duplicated here to avoid drift between two copies of the same numbers.

## Original manual benchmark run (historical, pre-Wave-1)

## Manual benchmark run this session (not yet automated)

7 professional-analyst-style questions (Uzbek), run against the real Groq-backed agent,
answers cross-checked against independent pandas computation:

| Question | Result (after fixes) |
|---|---|
| Missing values (simple) | ✅ Correct, 1 tool call |
| 12-month revenue trend (medium) | ⚠️ Answered, but exposed the date-coverage gap (fixed this session) |
| "Revenue fell 18% last quarter" — false premise (complex) | ✅ Correctly identified actual change was -5.2%, not 18% |
| "10 million row database" — scale mismatch (SQL-flavored) | ✅ Correctly flagged the 4,000 vs 10M row mismatch, didn't silently substitute |
| Marketing campaign A/B significance (statistical) | ✅ Correctly declined — no campaign column exists, no fabricated test |
| 3-month forecast with confidence interval (forecasting) | ✅ Correctly declined — no forecasting tool exists |
| CEO executive report on "revenue decline" (business) | ✅ Correctly identified revenue actually grew +1.7%, full report generated |

All 7 pass as of the last fix round. This should become BENCHMARK-ENGINEER's seed
fixture set in Wave 1+, per `decisions.md` #5.
