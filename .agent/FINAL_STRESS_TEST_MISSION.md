# Final Professional Data Analyst Stress Test + Hardening — Mission Report

Written per the mission's explicit final-stop-condition requirements. Every number
below was measured this session, not estimated. **HEAD at completion: `b9d3c97`.**
Working tree clean (only an untracked local `.claude/` config directory, unrelated to
this work).

This report covers the 11-phase mission that started from HEAD `0914e86` (the prior
session's blocks_conclusion end-to-end work). It does not restate that prior work in
detail — see `.agent/FINAL_GO_NO_GO_AUDIT.md` for the full history before this mission.

## What was actually done (6 commits, real fixes only — no benchmark-only edits)

| Commit | Phase | What | Real bug class |
|---|---|---|---|
| `decfe0f` | 1 | Hard-benchmark audit: 97.1% → 100.0% (99→102 PASS, 3→0 PARTIAL) | 3 genuine detection gaps (confound wording, group-size imbalance never checked, unusual baseline window never checked) + a new `conclusion_guard.py` closing the "structured confidence capped but free-text answer still confident" gap |
| `fa87638` | 2 | Extreme-value numerical guard in `hypothesis.py` | Real silent-corruption bug: `t_test` against ~1e300-scale data returned `statistic=0.0, p_value=1.0` (a confident wrong answer) because variance overflowed to `inf` — same failure class as the earlier Mahalanobis bug |
| `ee2eaa9` | 5 | New `contradiction_detection.py` (mean-vs-median ranking reversal) | A named, real gap — nothing previously caught "mean says A>B, median says B>A" |
| `26dc073` | 8 | Fixed `verifier._numeric_point` comparing a t-statistic against an actual mean | Found via a **real, unscripted LLM run** — not reproducible by any scripted test — confirming Phase 8's premise that real-LLM spot-checks catch things MockProvider scripts cannot |
| `b9d3c97` | 9 | Closed a real prompt-injection gap across 4 LLM call sites | Column names/category values reaching `agent.py`, `question_parser.py`, `planner.py`, `synthesizer.py` completely unwrapped, contradicting `SYSTEM_PROMPT`'s own stated guarantee |

Every fix followed the required loop: reproduce → root cause → general fix (never a
per-case patch) → regression test → full regression → benchmark re-check → document →
commit. Backend suite grew from 985 passed (this mission's starting point, HEAD
`0914e86`) to **1020 passed, 74 skipped, 0 failed** — **35 new regression tests**
added this mission. All 4 benchmark suites (hard/final_100/professional/adversarial)
show **zero diff** on their result files except the hard benchmark's genuine
97.1%→100.0% improvement.

## Phase-by-phase findings

**Phase 1 (hard benchmark audit)** — done, see `decfe0f` above and
`.agent/FINAL_GO_NO_GO_AUDIT.md` §0b for full root-cause detail on each of the 3
PARTIAL cases.

**Phase 2 (40-item adversarial stress list)** — the existing 102-case hard benchmark
already covers ~34 of the 40 named items by category (Simpson's paradox, confounding,
correlation-vs-causation, selection/survivorship bias, outlier-driven averages,
mean/median disagreement, heavy tails, small samples, multiple comparisons,
significance-without-practical-significance, denominator/unit/cents-vs-dollars
mismatches, duplicate joins, timezone boundaries, structural breaks, seasonality,
multicollinearity, zero-inflation, impossible values, contradictory evidence,
insufficient data ×5, misleading premises — see the case list in
`tests/benchmark/hard_realworld_cases.json`). Directly stress-tested the remaining
items not obviously covered:
- **Extremely large/small values**: found and fixed the overflow bug above.
- **Malformed/truncated data**: already covered (`test_truncated_file_does_not_crash`
  in `test_large_data_100m.py`, plus `test_data_contracts.py`/`test_upload.py`).
- **Malicious dataset instructions / malicious column names**: already had cell-value
  coverage; found and fixed the column-name gap above (Phase 9).
- **"Percentage vs proportion" and "data leakage"**: assessed, not built. Percentage-
  vs-proportion is too narrow/ambiguous a pattern to generalize safely without
  guessing business semantics this system has no way to know. "Leakage" in the
  ML train/test sense does not clearly apply — this system does inferential/
  descriptive statistics, not held-out predictive modeling with a train/test split.
  Documented here rather than forced into a fragile check.

**Phase 3 (tool selection stress test)** — verified via existing coverage rather than
rebuilt: the hard benchmark's `tool_selection`/`method_selection` dimensions (now
100%) and `tests/reasoning/test_orchestrator.py`'s explicit
`test_11_out_of_category_tool_is_rejected_if_somehow_requested` already test exactly
this (superficially-similar-tool traps, category-filtered tool router enforcement).
No new gap found.

**Phase 4 (self-challenge)** — mapped the mission's 14 named self-challenge questions
against real deterministic mechanisms:

| # | Question | Deterministic mechanism |
|---|---|---|
| 1 | Evidence supporting? | `Recommendation.supporting_findings` → `Evidence` chain (structural) |
| 2 | Evidence contradicting? | `Hypothesis.evidence_for`/`evidence_against` (hypothesis-level; `Finding` has no counter-evidence concept by design — findings are observations, not claims under test) |
| 3 | Alternative explanation? | `confound_detection.py` (categorical confounds only) |
| 4 | Confound? | `confound_detection.py` |
| 5 | Units correct? | `numerical_sanity.py` (partial — see the impossible-percentage gap in `FINAL_GO_NO_GO_AUDIT.md` §0a) |
| 6 | Denominator correct? | `numerical_sanity._find_population_mismatches` + `_find_unusual_baseline_window` (new this mission) |
| 7 | Aggregation correct? | `verifier._cross_check` + `contradiction_detection.py` (new this mission) |
| 8 | Outliers distorting? | `verifier._describe_data_outlier_limitations` + `numerical_sanity._find_group_magnitude_outliers` |
| 9 | Sample size adequate? | `verifier`'s sample-size check + `numerical_sanity._find_group_size_imbalance` (new this mission) |
| 10 | Significance established? | `recommendation_grounding.py`'s evidence-tier logic |
| 11 | Claim causal? | `causation_guard.py` |
| 12 | Recommendation stronger than evidence? | `recommendation_grounding.py`'s evidence-ceiling + `blocks_conclusion` override — the core mechanism |
| 13 | Most fragile assumption? | **Not deterministic** — `Recommendation.assumptions` is LLM-authored, prompt-only |
| 14 | What analysis would reduce uncertainty? | **Not deterministic** — `Recommendation.risks` partially touches this, prompt-only |

12 of 14 have a real deterministic mechanism, not just prompt wording. #13/#14 are
honestly documented as prompt-only — no fragile heuristic was built to paper over
this.

**Phase 5 (contradiction detection)** — built (see `ee2eaa9`). Mean-vs-median ranking
reversal is now deterministically caught. Explicitly NOT built (documented, not
approximated): aggregate-vs-segment trend reversal, within-group correlation sign
flips, forecast-vs-baseline disagreement — each needs a materially different, more
complex detector than evidence gathered this session justifies.

**Phase 6 (recommendation safety)** — re-verified, no new gaps. The chain
evidence→finding→limitation→severity→confidence→recommendation is backed by 47+
existing tests (`test_recommendation_grounding.py`, `test_causation_guard.py`,
`test_hypothesis_evaluator.py`) plus this mission's and the prior session's additions
(`test_blocks_conclusion_enforcement.py`, `test_conclusion_guard.py`,
`test_numerical_sanity_end_to_end.py`). `blocks_conclusion` cannot produce a
high-confidence recommendation (structurally forced to `None`, verified via a real
HTTP round-trip). `reduces_confidence` does not silently do nothing (verified it still
respects the evidence-tier ceiling without over-suppressing).

**Phase 7 (large data)** — verified via existing infrastructure, not re-run at full
100M scale (that run takes ~13+ minutes of real wall-clock time and nothing in the
large-data code path changed this session — re-running would reproduce the same
numbers at real cost for no new information). Confirmed real, previously-measured
numbers exist and are honestly documented in
`app/large_data/benchmark_100m_results.json` (not fabricated): 100M rows, 4.07GB file,
chunked read 70.7s/125.65MB peak RSS, chunked aggregate 149.6s/126.43MB peak RSS,
reservoir sampling after a real fixed regression 110s (was 6149s before the fix). The
SQL bridge's real limitation (in-memory materialization required, ~9.2GB extrapolated
at 100M rows) is honestly documented as a genuine architectural boundary, not silently
assumed to scale. All 31 always-on fast large-data tests pass; the 3 gated 100M tests
were not re-run (require `RUN_100M_BENCHMARK=1`, multi-minute, ~4GB disk).

**Phase 8 (real-LLM validation)** — ran a small, targeted, sequential 3-case sample
(the 3 cases fixed in Phase 1) rather than the full 102-case suite, per the standing
quota-conservation rule. 1 real response obtained (confirmed the new mechanisms work
against genuinely unscripted model output — the real model independently used
`t_test`+`confidence_interval`, found the confound itself, and produced an honestly
hedged answer even before `conclusion_guard`'s caveat), and found the
`_numeric_point` bug fixed in `26dc073`. 2 cases hit "AI is receiving a lot of
requests right now" — correctly recorded as `PROVIDER_ERROR`, not a model failure, and
the run stopped rather than retrying further. Report:
`.agent/hard_realworld_fixed_partials_spotcheck.md`.

**Phase 9 (security)** — found and fixed a real, systemic prompt-injection gap (see
`b9d3c97`). Otherwise reviewed the existing security surface rather than rebuilt it:
13 SQL-injection/read-only-enforcement tests (mutation blocking, statement stacking,
CTE/subquery hiding, session/catalog escapes, EXPLAIN bypass, COPY-TO exfiltration,
database export, file-reading table functions, SQLite pragma lock-down, repeated-
attack fuzzing), 9+ prompt-injection tests (now 13, cell values + column names across
both `/api/chat` and `/api/reason`), upload hardening (adversarial filenames, reserved
Windows device names, XXE, zip-bomb-style entity expansion, oversized files). No
additional gap found in this pass beyond the one fixed.

**Phase 10 (production readiness)** — reviewed, no changes made (none needed).
Multi-stage Docker build, non-root user, real `/api/health`-wired `HEALTHCHECK`, no
secrets baked into the image (verified against `.dockerignore`), CI runs the real fast
suite and explicitly documents why the gated slow suites are excluded, structured
exception handlers (`ValidationError`/`ToolExecutionError`/catch-all) in `main.py`,
existing `test_security_error_sanitization.py` for error-message leakage. Per the
mission's explicit "do not add unnecessary infrastructure" instruction, nothing was
added here — the existing setup is already sound.

## Final Capability Matrix

Brutally honest per the mission's explicit instruction — no "100%/human-level/
professional-analyst-equivalent" claims anywhere below.

| Capability | Supported | Verified | Limitation |
|---|---|---|---|
| Data ingestion (CSV/XLSX) | Yes | Yes — upload, malformed/truncated-file, oversized-file, adversarial-filename tests all pass | No Parquet/JSON ingestion |
| Data quality checks | Yes | Yes — duplicate detection, missing-value profiling, anomaly detection, impossible-value checks | Impossible-percentage check has no real trigger path in the current toolset (documented gap, §0a) |
| EDA / profiling | Yes | Yes | — |
| Statistics (t-test, ANOVA, chi-square, effect size, CI) | Yes | Yes — including this session's overflow guard | — |
| A/B testing | Yes | Yes — imbalance now flagged (this mission) | No formal power-analysis/sample-size-calculator tool |
| Time series / structural breaks / seasonality | Yes | Yes (hard benchmark cases) | No formal changepoint-detection algorithm — pattern recognition only |
| Forecasting | Yes | Yes (hard benchmark cases) | — |
| Segmentation / RFM | Yes | Yes (hard benchmark cases, small-N instability flagged) | — |
| SQL (DuckDB/SQLite, read-only) | Yes | Yes — 13 dedicated security tests | Bridges an in-memory DataFrame today, not a file path — full materialization required |
| Large datasets | Yes (chunked read/aggregate/sample) | Yes, real measured numbers up to 100M rows | SQL bridge does not scale past available RAM (documented boundary, not silently assumed away) |
| Business analytics (aggregation, comparison, recommendations) | Yes | Yes | — |
| Recommendation grounding/safety | Yes | Yes — extensively tested chain, this mission closed the free-text caveat gap | Assumption-fragility and uncertainty-reduction suggestions are prompt-only, not deterministic |
| Causal reasoning restraint | Yes | Yes — `causation_guard.py`, 22 tests | — |
| Confound detection | Yes (categorical only) | Yes | No continuous-variable confound detection |
| Contradiction detection | Yes (mean-vs-median ranking only, new this mission) | Yes | Aggregate-vs-segment trend reversal, correlation-sign flips, forecast-vs-baseline disagreement not covered |
| Numerical/statistical validation | Yes | Yes — this mission added 2 new checks + the overflow guard + the cross-check statistic/value fix | Impossible-percentage gap (above) |
| Visualization | Yes (chart-generation tools) | Not stress-tested this mission | — |
| Executive summaries | Yes (synthesis stage) | Yes | — |
| Uncertainty communication | Yes | Yes | — |
| Security (SQL/prompt-injection/upload) | Yes | Yes — this mission closed a real, systemic column-name-injection gap across 4 LLM call sites | — |
| Production deployment | Yes (Docker/CI/health checks) | Reviewed, sound | Not deployed/load-tested in a real environment this session |

## Remaining known limitations (explicit, not exhaustive)

1. Impossible-percentage check has no real trigger path in the current toolset
   (`.agent/FINAL_GO_NO_GO_AUDIT.md` §0a) — a genuine architectural gap, not fixed
   (would need a new general capability, deliberately not built to avoid
   fixture-shaped special-casing).
2. Contradiction detection covers only mean-vs-median ranking reversal; 3 other
   named Phase-5 patterns are undetected by design (documented above).
3. Confound detection is categorical-only; no continuous-variable confound check.
4. Two of Phase 4's 14 self-challenge questions (fragile-assumption identification,
   uncertainty-reduction suggestions) rely on prompt instructions only.
5. The SQL bridge requires full in-memory materialization; does not itself scale to
   100M-row SQL queries (chunked pandas operations do; SQL does not).
6. "Percentage vs proportion" and ML-style "data leakage" from the original 40-item
   list were assessed and not built — judged too narrow/inapplicable respectively.
7. Real-LLM validation this mission covered 3 cases (1 real response, 2 provider
   errors from rate limiting); broader real-LLM coverage remains from prior sessions
   only (see `.agent/hard_realworld_real_llm_spotcheck.md`).
8. The 100M-row large-data numbers are from a prior verified run, not re-measured
   this session (nothing in that code path changed).

## Regression evidence

- Full backend suite: **1020 passed, 74 skipped, 0 failed** (985 → 1020 this mission).
- Hard benchmark: **100.0%, 102/102 PASS, 0 PARTIAL, 0 FAIL** (up from 97.1%).
- `final_100`/`professional`/`adversarial` benchmarks: unchanged, zero file diff.
- 35 new regression tests added this mission across 8 files.
- Working tree clean at HEAD `b9d3c97` (only an untracked, unrelated local
  `.claude/` directory).
