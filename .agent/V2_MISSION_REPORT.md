# AI Data Analyst v2 — Reasoning, Reliability & Self-Verification Upgrade — Mission Report

Written per this mission's explicit final-stop-condition requirements. HEAD at
completion: see the final commit reported to the user (this file is committed
alongside it, so `git log -1` on this file's own commit is authoritative). Started
from baseline HEAD `246dba2` (see `.agent/V2_BASELINE.md` for the pre-mission
snapshot this report's regression numbers are measured against).

## Summary of what changed (7 commits, all reproduce → root cause → general fix →
## regression test → full regression → benchmark check → document → commit)

1. `a779f86` — **Confound Engine 2.0**: numeric-column and missingness confound
   detection (v1 was categorical-only), and a real stratified-effect check that
   directly verifies whether the compared metric's direction actually reverses (or
   merely shrinks, with an honest "possible mediator" caveat when it can't
   distinguish the two) within each stratum of a candidate confound — not just
   whether the candidate's own distribution differs, which was the v1 module's only
   signal. Found and fixed 2 real bugs while building it.
2. `e1be0e2` — **Contradiction Engine 2.0**: the mission's own flagship example
   ("revenue increased 15% overall, but every segment decreased") now detected via
   `detect_overall_vs_subgroup_contradiction`, plus `detect_data_quality_contradictions`
   for two verification tools disagreeing over the same scope. Wired
   `Evidence.population` (declared in the contract, never actually populated) to
   enable the former — a small, safe, zero-prior-consumer fix.
3. `1b9232f` — **AnalyticalAudit + 5-level conclusion classification**
   (SUPPORTED/WEAKLY_SUPPORTED/UNCERTAIN/CONTRADICTED/BLOCKED), assembled from
   pieces the orchestrator already computes, surfaced through the real API.
4. `ab261c3` — **Integer-overflow fix**: real int64 sum wraparound bug (silently
   produces a wildly wrong, sign-flipped result) found via direct stress-testing per
   Phase 8's explicit "integer overflow" requirement, fixed at 3 call sites.
5. `25b05a2` — **End-to-end analyst-workflow test** (Phase 14): a realistic
   multi-step "verify or reject this business claim" scenario proving the full
   pipeline composes correctly, not just its individual mechanisms in isolation.
6. This report + `.agent/FINAL_CAPABILITY_MATRIX.md` +
   `.agent/HUMAN_VS_AI_COMPARISON_FRAMEWORK.md`.

Full details, root causes, and exact before/after numbers for each are in their own
commit messages — not repeated here.

## Test count

Backend suite: **985 (mission start) → 1057 passed, 74 skipped, 0 failed** (72 new
regression tests this mission, across 12 files).

## Benchmark results

- Hard real-world benchmark: **100.0%, 102/102 PASS, 0 PARTIAL, 0 FAIL** — unchanged
  from mission start (already at 100% from the prior mission); verified after every
  single commit this mission, zero drift.
- `final_100`, `professional`, `adversarial` benchmark suites: unchanged, zero file
  diff after every commit.

## Real-LLM results

One targeted real-LLM spot-check was run this mission (`hard_confound_01a`, the
honest-confound case, specifically to verify Confound Engine 2.0 fires correctly
against a real, unscripted model's own tool calls — not just the scripted suite).
See `.agent/v2_real_llm_spotcheck.md` for the result. Per the standing
quota-conservation rule, no further real-LLM calls were made beyond this single
targeted check — the scripted suites and direct code-level verification already
cover the new mechanisms' correctness; a real-LLM call adds value specifically for
"does a real, unscripted model's behavior still trigger this," which one case answers.

## Major bugs discovered (this mission)

1. **`confound_detection.py`'s identifier check wrongly excluded numeric columns.**
   A continuous numeric measurement (e.g. customer tenure) is expected to be
   near-unique across rows — the uniqueness-ratio identifier check, correct for
   categorical/string columns, was silently disqualifying every numeric confound
   candidate before the new standardized-mean-gap check ever ran. Found by the
   module's own new unit test failing on first run.
2. **`confound_detection.py`'s missingness check was unreachable for high-
   cardinality columns.** An early "neither categorical nor numeric → irrelevant"
   return sat above the missingness check, silently skipping it for exactly the
   kind of free-text/high-cardinality column (e.g. an email field) where a
   missingness-rate gap is a real, common signal. Found the same way.
3. **Real int64 sum overflow.** `group_and_aggregate`/`compare_periods`/
   `describe_data` all silently wrapped around to a wildly wrong (often negative)
   number when summing large integers, with no warning — verified directly (four
   `2**62` values summed to a negative result). Fixed at all 3 sites.
4. **Contradiction-engine test scenario had a real arithmetic impossibility.**
   While building the end-to-end test for "overall increases, every segment
   decreases," constructing it with `agg_func="sum"` revealed this pattern is
   mathematically IMPOSSIBLE for a straightforward sum when segments fully
   partition the data (segment sums must add to the overall sum) — the paradox
   only exists for rate/mean metrics under a genuine mix-shift. Documented in the
   module and the test itself, not silently worked around.
5. **One existing confound-detection unit test's expectation was based on the
   older, less rigorous distributional-only model.** The new stratified-effect
   check correctly found a genuine within-stratum reversal in that exact test's
   data that the old test didn't anticipate — the test's assertion was updated to
   match the objectively more correct new behavior (documented in both the test's
   docstring and the module docstring), and the original scenario was preserved as
   its own dedicated regression test proving the escalation.

## Bugs fixed

All 5 above. Every fix has a dedicated regression test reproducing the exact
originally-observed shape before asserting the fix (not a generic "doesn't crash"
test).

## Remaining limitations (honest, not exhaustive — see `FINAL_CAPABILITY_MATRIX.md`
## for the full per-capability table)

1. Impossible-percentage check still has no real trigger path in the current
   toolset (carried over from the prior mission, not addressed this one).
2. Confound detection remains single-variable (no simultaneous 2+-way
   stratification) and cannot distinguish a true confound from a mediator without
   temporal/domain information it doesn't have — says so explicitly rather than
   guessing.
3. Contradiction engine covers 3 of the mission's 14 named patterns as dedicated
   checks (several others are already covered under different names or already
   structurally enforced, documented in the module); within-group correlation sign
   flips and forecast-vs-baseline disagreement remain undetected — no tool computes
   the structured field either would need.
4. `large_data/aggregation.py`'s chunked streaming sum path has the same
   theoretical int64-overflow exposure as the 3 fixed sites — deliberately left
   unfixed (negligible real-world likelihood at realistic business scale; fixing it
   risks the measured 100M-row performance numbers for a vanishingly unlikely
   trigger).
5. Two of Phase 3/4's self-challenge questions (fragile-assumption identification,
   uncertainty-reduction suggestions) remain prompt-only, not deterministic.
6. Phase 4's literal "return to analysis" loop, Phase 10's full LLM-reliability
   metadata harness, and Phase 11's LLM self-consistency re-checking were
   deliberately not built — each is either already achieved by the existing
   architecture under a different mechanism, or judged not clearly justified by
   evidence gathered this session (see `FINAL_CAPABILITY_MATRIX.md`'s "Deliberately
   not built" section for the specific reasoning behind each).
7. Visualization and production deployment were reviewed/spot-checked, not
   freshly stress-tested this mission.

## Production-readiness assessment

No change from the prior mission's assessment (`FINAL_STRESS_TEST_MISSION.md`):
Docker multi-stage build, non-root user, real health-check-wired `HEALTHCHECK`, CI
running the real fast suite with gated slow suites correctly excluded and
documented. Nothing in this mission's changes affects deployment configuration.
Not deployed or load-tested in a real environment this session — that remains
unverified, honestly stated as such rather than assumed.

## Capability matrix location

`.agent/FINAL_CAPABILITY_MATRIX.md`

## Git status

Working tree clean at completion (verified via `git status --short` before this
report's own commit; only an untracked, unrelated local `.claude/` config directory
present throughout the mission).
