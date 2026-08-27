"""General-purpose numerical sanity checking (Phase 5).

Motivated by evidence gathered independently across three separate benchmark waves
this project has run (`.agent/PROFESSIONAL_ANALYST_CAPABILITY_AUDIT.md` #17,
`.agent/final_100_case_benchmark.md`'s documented gaps, and the hard real-world
benchmark's own trap categories -- `funnel_denominator`, `finance_units_mismatch`,
`hr_impossible_values`): this project had no deterministic mechanism catching an
impossible or badly-scaled numeric value in a tool's own output before it reaches the
user, relying entirely on whether the model's own prose happened to notice one.

This module runs purely against `Evidence` objects the executor already gathered --
no new tool call, no LLM call -- exactly the same "zero additional cost" discipline
`verifier.py`'s existing `_describe_data_outlier_limitations` and `_cross_check`
already follow. It checks three concrete, mechanically well-defined things:

1. **Impossible percentage/rate values** -- any `_pct`-suffixed field in a tool's own
   result is negative or exceeds 100 (with a documented, narrow exception for metrics
   that are legitimately allowed to exceed 100%, e.g. MAPE). Directly targets the
   "denominator correctness" and "impossible values" requirements.
2. **Population/denominator mismatch across evidence for the same metric** -- two
   `Evidence` items about the same metric with wildly different `sample_size` values
   signal that a conclusion comparing them may be comparing different populations
   (a filtered subset against the whole, for instance).
3. **Within-evidence magnitude outliers** -- a `group_and_aggregate`-shaped result
   (a "groups" list of `{"group": ..., "value": ...}` dicts) where one group's value is
   wildly larger than the rest is often a units mismatch (cents vs. dollars) rather
   than a genuine business signal, and is at least worth a flag either way.

Deliberately conservative: every rule here is a real, mechanically-checkable
consistency property of the tool's own numbers, not a guess about business semantics
this module has no way to know.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence, Limitation

# MAPE (mean absolute percentage error) and any other genuinely unbounded percentage
# metric can legitimately exceed 100% -- never flagged by the impossible-percentage rule.
_UNBOUNDED_PCT_FIELDS = {"mape_pct"}

_POPULATION_MISMATCH_RATIO = 5.0  # one sample_size at least 5x another -> worth flagging
_GROUP_MAGNITUDE_OUTLIER_RATIO = 50.0  # one group value >=50x the median of the rest


def _find_impossible_percentages(evidence: list[Evidence]) -> list[Limitation]:
    limitations: list[Limitation] = []
    for i, ev in enumerate(evidence):
        r = ev.result_summary
        if not isinstance(r, dict):
            continue
        for key, value in r.items():
            if not key.endswith("_pct") or key in _UNBOUNDED_PCT_FIELDS:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if value < 0 or value > 100:
                limitations.append(
                    Limitation(
                        category="other",
                        text=(
                            f"{ev.source_tool}'s '{key}' is {value}, which is outside the "
                            "possible 0-100% range for a percentage -- this points to a "
                            "denominator or units problem upstream, not a real business figure."
                        ),
                        severity="blocks_conclusion",
                        affected_findings=[f"finding_{i}"],
                    )
                )
    return limitations


def _find_population_mismatches(evidence: list[Evidence]) -> list[Limitation]:
    limitations: list[Limitation] = []
    by_metric: dict[str, list[Evidence]] = {}
    for e in evidence:
        if e.metric and e.sample_size:
            by_metric.setdefault(e.metric, []).append(e)

    for metric, items in by_metric.items():
        distinct_tools = list({e.source_tool: e for e in items}.values())
        if len(distinct_tools) < 2:
            continue
        sizes = [(e, e.sample_size) for e in distinct_tools]
        sizes.sort(key=lambda pair: pair[1])
        smallest_e, smallest_n = sizes[0]
        largest_e, largest_n = sizes[-1]
        if smallest_n > 0 and largest_n / smallest_n >= _POPULATION_MISMATCH_RATIO:
            limitations.append(
                Limitation(
                    category="methodological",
                    text=(
                        f"'{metric}' is examined by both {smallest_e.source_tool} (n={smallest_n}) "
                        f"and {largest_e.source_tool} (n={largest_n}) -- these cover very different "
                        "population sizes, so comparing or combining their results directly may not "
                        "be apples-to-apples."
                    ),
                    severity="reduces_confidence",
                )
            )
    return limitations


def _find_group_magnitude_outliers(evidence: list[Evidence]) -> list[Limitation]:
    limitations: list[Limitation] = []
    for i, ev in enumerate(evidence):
        r = ev.result_summary
        if not isinstance(r, dict):
            continue
        groups = r.get("groups")
        if not isinstance(groups, list) or len(groups) < 3:
            continue
        values = [g["value"] for g in groups if isinstance(g, dict) and isinstance(g.get("value"), (int, float)) and not isinstance(g.get("value"), bool)]
        if len(values) < 3:
            continue
        abs_values = sorted(abs(v) for v in values)
        median = abs_values[len(abs_values) // 2]
        if median <= 0:
            continue
        largest = abs_values[-1]
        if largest / median >= _GROUP_MAGNITUDE_OUTLIER_RATIO:
            limitations.append(
                Limitation(
                    category="methodological",
                    text=(
                        f"{ev.source_tool}'s breakdown for '{ev.metric or 'this metric'}' has one group "
                        f"roughly {round(largest / median)}x the typical group value -- worth checking "
                        "for a units or data-entry inconsistency before treating this as a real outlier "
                        "group."
                    ),
                    severity="reduces_confidence",
                    affected_findings=[f"finding_{i}"],
                )
            )
    return limitations


def check_numerical_sanity(evidence: list[Evidence]) -> list[Limitation]:
    """Runs all numerical-sanity rules against already-gathered evidence. Additive to
    (never a replacement for) verifier.py's existing outlier/sample-size/cross-check
    limitations -- called from build_findings alongside them."""
    return (
        _find_impossible_percentages(evidence)
        + _find_population_mismatches(evidence)
        + _find_group_magnitude_outliers(evidence)
    )
