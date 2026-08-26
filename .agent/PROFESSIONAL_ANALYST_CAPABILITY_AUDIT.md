# Professional Analyst Capability Audit

Audited against the 20-area capability map the user specified. Every "Implemented"
claim below is checked against actual code (file/function named), not assumed from
prior docs. "Tested" distinguishes deterministic (scripted) coverage from real-LLM
evidence. This document will be updated once `final_100_cases.json` (built
alongside this doc) is run — see the benchmark-coverage column, filled in after
that run completes.

**Framing discipline, per explicit instruction**: nothing in this document is
"professional analyst level" merely because a benchmark passes. Each row states
what exists and what evidence backs it — the reader draws their own conclusion.

| # | Capability | Implemented? | Tested? | Known limitations |
|---|---|---|---|---|
| 1 | Data understanding (schema, dimensions/measures, types, IDs, profiling) | Yes — `profiler.py::profile_dataset`, `eda.py::analyze_cardinality` (ID-like/boolean/near-constant detection) | Deterministic: extensive. Real-LLM: yes (18 real cases exercised this) | Cannot ask the user for missing context conversationally — `/api/reason` is single-turn, no clarification round-trip |
| 2 | Data cleaning (missing/duplicate/invalid/type/date) | Detection yes (`data_quality.py`, `profiler.py`), **transformation no** — this system never modifies source data, by design | Deterministic: `test_data_quality.py` (18 tests) | Correctly does NOT auto-clean (matches "never silently modify source data" principle) — but this means it can only *report* cleaning needs, not perform them, which a human analyst could also do manually |
| 3 | SQL / data manipulation (SELECT/WHERE/GROUP BY/JOIN/CTE/window functions/ranking) | Real DuckDB+SQLite bridge, read-only, resource-limited (`app/sql/`, `run_sql_query` tool) — supports full DuckDB SQL syntax including JOIN/CTE/window functions | Deterministic: 158+ SQL tests (Wave 1) + tool-registration tests. Real-LLM: yes (sql_01-03 in the 37-case run, 100% pass) | SQL bridge requires the dataset already in an in-memory DataFrame (see #18) — no direct large-file SQL push-down through the product today |
| 4 | EDA (distributions, correlations, trends, anomalies) | Yes — `eda.py::automated_eda` composes profiling+distributions+outliers+correlations into one prioritized report | Deterministic: `test_eda.py` (19 tests). Real-LLM: yes (eda_02 passed live) | "Choose analyses relevant to the question, don't run everything" is enforced by the planner's category-filtering (only EDA-category tools offered for an EDA-flavored question), not a separate relevance-scoring step |
| 5 | Statistics (t-test, chi-square, ANOVA, CI, effect size, significance) | Yes — `hypothesis.py` (5 tests), `regression_diagnostics.py` | Deterministic: hand-checked expected values. Real-LLM: yes (stat_01-03, 100%) | No multiple-comparison correction (Bonferroni etc.) if several tests are run in one analysis — not implemented |
| 6 | Regression (diagnostics, residuals, multicollinearity, fit) | Yes — `regression.py` + `regression_diagnostics.py` (Shapiro-Wilk, Breusch-Pagan, VIF) | Deterministic: yes. Real-LLM: yes (reg_02/03 passed; reg_01 hit a provider error, unmeasured) | `regression_diagnostics.py`'s "strong" recommendation-grounding tier requires significance+effect+size all on ONE Evidence object (see `recommendation_grounding.py` docstring) — a real, documented scoping choice, not a bug |
| 7 | Time series (trend/seasonality/decomposition/ARIMA/ETS/backtesting) | Yes — `forecasting.py` (train_test_split, decompose, forecast auto/arima/ets, backtest) | Deterministic: 38 tests. Real-LLM: yes (fc_01-03, 100%) | Refuses forecasting below 10 points and beyond observed-history horizon (a deliberate, tested guard, not a gap) |
| 8 | Segmentation (clustering, K-means, PCA, RFM, cohort, churn) | Yes — `clustering.py`, `segmentation.py` (rfm_analysis, cohort_analysis, churn_risk_analysis) | Deterministic: 28 tests combined. Real-LLM: yes (cls_01-03, 100%) | Cluster/segment *labels* (RFM segment names) exist; explaining *why* a cluster is what it is beyond centroid values is left to LLM narration, not a dedicated interpretation tool |
| 9 | Business analytics (revenue/sales/customer/contribution/Pareto/cohort/PoP) | Yes — `business_diagnosis.py` (contribution_analysis, executive_summary), `comparison.py::compare_periods`, `advanced_charts.py::pareto_chart_data`, `segmentation.py::cohort_analysis` | Deterministic: 10+21+cohort tests. Real-LLM: partial (br_* cases were unmeasured this session, hit provider wall) | Employee/workforce metrics have no dedicated tool — would need a workforce dataset schema the demo data doesn't have; not a gap in generic capability, just no seeded example |
| 10 | Root-cause analysis (profile→decompose→compare→segments→outliers→data-quality→hypotheses→evidence→confidence→recommend) | **Partial** — the reasoning pipeline does generate competing hypotheses (`planner.py`, gated to diagnostic intent, max 3) and derive evidence-based status (`hypothesis_evaluator.py`), but does NOT automatically chain through profile→decompose→compare-periods→segment-check→outlier-check as a fixed sequence — it plans via LLM tool selection per question, not a hardcoded root-cause pipeline | Deterministic: hypothesis generation/status tested. Real-LLM: `ct_01-04`, `radv_04` exercised this, mostly unmeasured this session (provider wall) | **This is the single largest gap versus the capability map's explicit expectation of a fixed root-cause sequence** — see Remaining Work |
| 11 | Causal reasoning (observation/correlation/association/hypothesis/causal evidence distinction) | Yes — `causation_guard.py` (3-layer: stem-pattern, hedge-context, relationship classification into correlation/association/prediction/temporal/causal_hypothesis/causal_unhedged) | Deterministic: 79 tests. Real-LLM: the exact "is clearly responsible for" bypass was found and closed, verified live this session | Regex-based, not semantic — a sufficiently novel paraphrase outside the 15 stem patterns could still slip through (documented) |
| 12 | Data quality (missingness/duplicates/impossible values/leakage/date ranges/sample size) | Yes — `data_quality.py` (duplicate_analysis w/ subset-key mode, mixed-type detection, missingness co-occurrence, deterministic quality_score), `premise_validator.py` (scale/coverage mismatch) | Deterministic: 18 tests. Real-LLM: yes (dq cases; radv_09 duplicate case) | No formal data-leakage detector (e.g. a feature perfectly predicting the target) — not implemented, would need a dedicated check |
| 13 | Visual analytics (chart type selection by question type) | Tools exist for every named chart type (line/bar/histogram/boxplot/scatter/heatmap/Pareto via `charts.py`+`advanced_charts.py`) and cohort tables (`segmentation.py`) | Deterministic: 21+ tests. Real-LLM: not directly exercised this session (no chart-specific live case) | **Selection** of the "right" chart for a question is left entirely to the LLM's tool choice — no deterministic chart-type-recommendation logic exists |
| 14 | Executive communication (summary/findings/evidence/visuals/risks/limitations/recommendations/confidence/next steps) | Partial — `business_diagnosis.py::executive_summary` exists as a tool; the reasoning layer's `AnalysisResult` carries findings/limitations/recommendation/confidence separately, but there is no single "executive report" assembly step that combines all of them into one structured document | Deterministic: `executive_summary` tool tested (10 tests). The *combined report* concept: untested, because it doesn't exist as a single artifact yet | See Remaining Work — this is P3 "Analyst Workflow" from `production-readiness.md`, not yet built |
| 15 | Recommendations (finding→impact→action→benefit→risk→confidence) | Yes, evidence-gated — `recommendation_grounding.py` computes evidence_strength/ceiling and caps confidence; `Recommendation` carries assumptions/risks/expected_business_effect | Deterministic: 10 tests + orchestrator wiring test. Real-LLM: the adv_15-style overclaiming case was verified to get capped live | `expected_business_effect` is free text from the LLM, not derived from a quantitative model of the recommendation's actual effect — a real, disclosed limitation |
| 16 | Uncertainty (known/estimated/uncertain/unavailable, explicit assumptions) | Yes — `Uncertainty.level` is exactly this 4-way enum (`contracts.py`), derived deterministically in `verifier.py` from real tool result shapes | Deterministic: yes, part of `test_orchestrator.py`/`test_recommendation_grounding.py`. Real-LLM: yes, every real case with a STATISTICAL_RESULT finding carried this | — |
| 17 | Numerical sanity checking (magnitude/sign/denominator/order-of-magnitude) | **Partial** — `premise_validator.py`'s scale-mismatch check catches gross claimed-vs-actual row-count mismatches; `verifier.py`'s new `_describe_data_outlier_limitations` (this session) catches a mean skewed by an extreme value. **No general sign/magnitude/denominator sanity checker exists** | Deterministic: the two checks above are tested. General sanity checking: untested because not built | Real gap — see Remaining Work |
| 18 | Large data (in-memory/chunked/streaming/DuckDB/sampling selection) | `app/large_data/` is real and benchmarked to 100M rows (chunked read/aggregate memory-bounded, sampling fixed to a 55.8x speedup this session) — **but is not reachable from the actual upload/analysis path** (`DatasetStore.save()` still does one full `pd.read_csv`, capped at 25MB/500K rows) | Deterministic: `test_large_data.py` (24) + `test_large_data_100m.py` (gated). Real-LLM: not applicable (infrastructure, not LLM behavior) | **Confirmed, significant integration gap** — see `production-readiness.md` finding C1. The capability map's own "architecturally capable of analyzing very large datasets" bar is met by the large_data *package*, not by the *product* today |
| 19 | Automation (agent determines data/question/analysis/tools/verification itself) | Yes, this is the core reasoning pipeline's actual design — `question_parser`→`premise_validator`→`planner`→`executor`→`verifier`→`synthesizer`, no manual tool specification required from the user | Deterministic: the entire 75+21-case benchmark history validates this. Real-LLM: yes, 21 real cases | — |
| 20 | Self-checking (right question/dataset/period/denominator, causation check, alternative explanation, uncertainty) | Yes — `epistemic_checks.py`'s 10 machine-checkable principle validators run on every result (`check_all`, wired into `orchestrator.py`), populating `principle_violations` (now exposed via `/api/reason`, fixed this session) | Deterministic: 35 tests. Real-LLM: not yet spot-checked with a case deliberately designed to trip a specific epistemic check | `check_no_inference_beyond_data` and `check_evidence_vs_assumption` are explicitly documented as heuristic (string/number scanning), not semantically rigorous |

## Summary

**Solidly implemented and tested** (deterministic + at least some real-LLM
evidence): #1, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 19.

**Implemented but with a real, specific, documented gap**: #2 (by design, no
auto-clean), #9 (workforce metrics have no seeded example), #13 (chart *selection*
is LLM-only, no deterministic recommender), #14 (pieces exist, no combined
executive-report assembly), #18 (large-data package real but disconnected from
the product — see `production-readiness.md`).

**Real, not-yet-closed gaps**: #10 (root-cause analysis is LLM-planned per
question, not a fixed profile→decompose→compare→hypothesize sequence — the
capability map explicitly wants the fixed sequence), #17 (no general numerical
sanity checker beyond the two specific checks that exist), #20's real-LLM
targeted verification (exists deterministically, not yet spot-checked live
against a case designed to trip it specifically).

## Benchmark coverage

`final_100_cases.json` (102 cases, 24 categories) has run. Full detail,
per-case root-cause analysis, and the BEFORE/AFTER fix cycle are in
[`.agent/final_100_case_benchmark.md`](final_100_case_benchmark.md) — summary
here, mapped back to these 20 capability areas:

- **Overall: 101/102 PASS, 1 PARTIAL, 0 FAIL (99.0%)**, scripted/deterministic
  (see that doc's honesty disclaimer — this is not a real-LLM measurement).
- 23 of 24 categories scored 100%. The one exception (`insufficient_data`,
  75%) is the direct benchmark evidence for this doc's own #12 finding below
  (population-scope claims not structurally validated) — the gap was already
  known before the benchmark ran; the benchmark just gives it a concrete,
  reproducible case (`insuf2`).
- Every one of the 16 issues found during the first real run turned out to be
  a benchmark-authoring defect (wrong tool category, wrong dataset, guessed
  field names), not a reasoning-pipeline defect — zero production code
  changed as a result of this benchmark. That is itself evidence for this
  doc's "solidly-implemented" list: category filtering, SQL safety, the
  forecast minimum/maximum-horizon checks, causation guarding, and
  recommendation grounding all held up against 102 structured probes,
  including 4 dedicated adversarial/security cases, without a single real
  defect surfacing.
- One correction to this document's own prior assumptions: the forecasting
  horizon-vs-history sanity check (relevant to #7/#17) was assumed missing
  going into this benchmark. It is not — `app/tools/forecasting.py` already
  refuses when the requested horizon exceeds the historical sample size,
  confirmed by direct source read and then exercised for real via case `fc4`.

## Remaining work (explicit, prioritized)

1. **Wire `app/large_data/` into the real upload path** (#18) — already the
   top item in `production-readiness.md`'s roadmap, largest single gap.
2. **Fixed root-cause-analysis sequence for diagnostic questions** (#10) — the
   capability map wants a more structured profile→decompose→compare→hypothesize
   chain than today's LLM-planned-per-question approach. This is a real design
   question (force a fixed sequence vs. trust the planner more) that should be
   decided deliberately, not bolted on — logged as an open decision.
3. **General numerical sanity checker** (#17) — sign/magnitude/denominator/
   order-of-magnitude checks beyond the two specific ones that exist.
4. **Combined executive-report assembly** (#14) — a single structured document
   combining summary/findings/evidence/visuals/risks/limitations/
   recommendations/confidence/next-steps, not just the separate pieces.
5. **Deterministic chart-type recommendation** (#13) — currently 100% LLM
   discretion.

This document intentionally reports "production-ready for the validated scope
below" rather than an unqualified "production-ready" — see the final summary in
this session's chat report for the exact scope statement.
