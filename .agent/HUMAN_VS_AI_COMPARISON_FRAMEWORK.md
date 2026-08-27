# Human vs. AI Data Analyst Comparison Framework (Phase 15)

Per explicit instruction: this is the framework only. **No comparison has been run.**
No human analyst was available this session, and none of the numbers below are
filled in or estimated — doing so without a real human analyst would be exactly the
kind of fabricated result the standing rules prohibit.

## Purpose

A structured, repeatable protocol for later comparing this system's output against a
human data analyst given the *same* dataset and the *same* question, so the
comparison is fair (same inputs, same time budget disclosed, same scoring rubric)
and reproducible (any two people scoring the same pair of outputs should reach
materially the same conclusion).

## Protocol

1. Pick a question from the existing hard/professional/adversarial benchmark suites
   (`backend/tests/benchmark/*.json`) rather than inventing a new one — these already
   have documented ground truth and known traps, so a human's answer can be checked
   against the same facts the AI is checked against, without any post-hoc rationalization.
2. Give the human analyst the raw dataset and the question text only — no access to
   this system's own answer, no coaching.
3. Time-box the human analyst's session and record the actual wall-clock time taken
   (not an estimate).
4. Run the same question through `/api/reason` (real LLM, not a scripted case) and
   record wall-clock time the same way.
5. Score both outputs independently, blind to which is which where practical, using
   the rubric below.

## Metrics (mission's own list, each with a concrete measurement method)

| Metric | How it's measured |
|---|---|
| Time to completion | Real wall-clock time, both sides, as timed in the protocol above |
| Numerical correctness | Compare every stated number against the benchmark case's documented ground truth (`hard_realworld_cases.json` entries already carry this for many cases) |
| Statistical correctness | Was the right test/method used, and was its result interpreted correctly (significance, effect size, confidence) |
| Tool selection | Did the approach taken (whichever tools/methods, human or AI) fit the question, judged against this project's own `tool_selection`/`method_selection` benchmark dimensions (`hard_scoring.py`) |
| Insight quality | Independent human rating (1-5) on a fixed rubric: relevance, depth, business-actionability |
| Missed insights | Cross-reference against the benchmark case's documented hidden traps (`hidden_traps` field) -- how many did each side catch |
| False conclusions | Any claim that contradicts the documented ground truth |
| Unsupported causal claims | Any claim of causation not backed by a randomized/quasi-experimental design or explicit hedging |
| Recommendation quality | Independent human rating (1-5): is it actionable, appropriately hedged, proportionate to evidence strength |
| Reproducibility | Given the same inputs again, does the same method/reasoning reproduce the same answer (deterministic for the AI by construction; for a human, whether their stated method is well-specified enough to redo) |
| Explanation quality | Independent human rating (1-5): clarity, whether the reasoning steps are visible/auditable, not just the final number |

## Why this system's own internal machinery helps here

This system already exposes, per analysis, real artifacts a human analyst's own
notebook usually doesn't produce automatically -- these should be captured
alongside its answer whenever this framework is actually run:

- `AnalyticalAudit.conclusion_status` (SUPPORTED/WEAKLY_SUPPORTED/UNCERTAIN/
  CONTRADICTED/BLOCKED) and its component lists (confounds, contradictions,
  numerical issues) -- a structured self-assessment a human analyst would have to
  be separately interviewed to produce.
- `reasoning_trace` -- the actual sequence of checks run, for the "explanation
  quality"/"reproducibility" metrics.
- `principle_violations` -- the epistemic-principle canary list.

## Explicitly NOT part of this framework

- No claim that the AI is "as good as" or "better than" a human analyst — that
  claim can only ever be made after running this protocol with a real result.
- No synthetic/simulated "human baseline" — a fabricated comparison is worse than
  no comparison.
- No scoring dimension without a concrete, described measurement method above.

## Status

Framework prepared, not executed. To run it: schedule time with a real data
analyst, pick 3-5 cases spanning different hard-benchmark categories (at minimum
one honesty-pair, one multi-step, one "insufficient data" case), and follow the
protocol above.
