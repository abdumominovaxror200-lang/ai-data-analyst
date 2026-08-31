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
from app.datasets.metric_registry import MetricRegistry

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


_GROUP_IMBALANCE_RATIO = 5.0  # largest compared group at least 5x the smallest -> worth flagging


def _group_sizes_from_result(r: dict) -> dict[str, int] | None:
    """Reads each compared group's real size directly off a tool's own result shape
    -- never re-derived or guessed. Three shapes recognized (confirmed against
    app/tools/hypothesis.py, not guessed):

    - `t_test`/`effect_size` (two-sample): "group_a"/"group_b" dicts each with "n".
    - `anova_test`: "groups" DICT keyed by group name, each value a dict with "n".
    - `chi_square_test`: "contingency_table" -- {row_value: {col_value: count}} --
      each row's real group size is the sum of its own counts across columns."""
    group_a, group_b = r.get("group_a"), r.get("group_b")
    if isinstance(group_a, dict) and isinstance(group_b, dict) and isinstance(group_a.get("n"), int) and isinstance(group_b.get("n"), int):
        return {str(group_a.get("label", "group_a")): group_a["n"], str(group_b.get("label", "group_b")): group_b["n"]}

    groups = r.get("groups")
    if isinstance(groups, dict) and all(isinstance(v, dict) and isinstance(v.get("n"), int) for v in groups.values()) and len(groups) >= 2:
        return {name: v["n"] for name, v in groups.items()}

    table = r.get("contingency_table")
    if isinstance(table, dict) and len(table) >= 2:
        sizes: dict[str, int] = {}
        for row, cols in table.items():
            if not isinstance(cols, dict):
                return None
            sizes[str(row)] = sum(v for v in cols.values() if isinstance(v, int) and not isinstance(v, bool))
        return sizes

    return None


def _find_group_size_imbalance(evidence: list[Evidence]) -> list[Limitation]:
    """Flags a real, mechanically-detectable statistical-power concern: when the
    compared groups in a two-or-more-group statistical test are very unevenly
    sized, the comparison has less power to detect a real difference than its total
    N suggests, and the smaller group's own estimate is especially noisy -- a
    distinct problem from `verifier.py`'s existing tiny-*total*-sample check, which
    only looks at overall evidence sample size and says nothing about a lopsided
    split (e.g. 480 vs 40 has a perfectly adequate total N of 520, but the 40-row
    group individually is far too small to trust).

    Found as a real, general gap via the hard real-world benchmark (a scripted
    480-vs-40 imbalanced two-group comparison produced zero limitations at all,
    despite `recommendation_grounding.py` separately (and correctly) capping
    confidence from the test's own non-significance -- the imbalance itself, and
    the reason a real effect could easily be missed here, was never surfaced to the
    user)."""
    limitations: list[Limitation] = []
    for i, ev in enumerate(evidence):
        r = ev.result_summary
        if not isinstance(r, dict):
            continue
        sizes = _group_sizes_from_result(r)
        if not sizes or len(sizes) < 2:
            continue
        smallest_label = min(sizes, key=lambda k: sizes[k])
        largest_label = max(sizes, key=lambda k: sizes[k])
        smallest_n, largest_n = sizes[smallest_label], sizes[largest_label]
        if smallest_n <= 0 or largest_n / smallest_n < _GROUP_IMBALANCE_RATIO:
            continue
        limitations.append(
            Limitation(
                category="sample_size",
                text=(
                    f"{ev.source_tool}'s compared samples are very unevenly sized "
                    f"('{smallest_label}' n={smallest_n} vs '{largest_label}' n={largest_n}, a "
                    f"{round(largest_n / smallest_n)}x imbalance) -- this reduces statistical power "
                    "to detect a real difference and makes the smaller sample's own estimate "
                    "especially noisy; treat this comparison with caution."
                ),
                severity="reduces_confidence",
                affected_findings=[f"finding_{i}"],
            )
        )
    return limitations


_PERIOD_WINDOW_RATIO = 3.0  # comparison window at least 3x the baseline window -> worth flagging


def _find_unusual_baseline_window(evidence: list[Evidence]) -> list[Limitation]:
    """Flags `compare_periods`-shaped evidence whose two windows cover very
    different amounts of data (read directly off its own real result shape --
    "current_period"/"previous_period", each a dict with "n", confirmed against
    app/tools/comparison.py, not guessed). Comparing a normal-length period against
    an unusually short/narrow baseline (e.g. "the 5 days right before a change" vs
    "the following full month") is a real regression-to-the-mean / cherry-picked-
    baseline trap distinct from every other check in this module: nothing else here
    (or in verifier.py) looks at the RELATIVE size of the two windows a period
    comparison itself chose.

    Found as a real, general gap via the hard real-world benchmark: a scripted
    pre/post pricing comparison against a 5-day pre-period (vs. a 30-day post-period)
    produced zero limitations, even though an unusually short/narrow baseline window
    is exactly the kind of thing that makes an ordinary fluctuation look like a large
    structural jump."""
    limitations: list[Limitation] = []
    for i, ev in enumerate(evidence):
        r = ev.result_summary
        if not isinstance(r, dict):
            continue
        current, previous = r.get("current_period"), r.get("previous_period")
        if not (isinstance(current, dict) and isinstance(previous, dict)):
            continue
        n_current, n_previous = current.get("n"), previous.get("n")
        if not (isinstance(n_current, int) and isinstance(n_previous, int) and not isinstance(n_current, bool) and not isinstance(n_previous, bool)):
            continue
        if n_current <= 0 or n_previous <= 0:
            continue
        smaller, larger = min(n_current, n_previous), max(n_current, n_previous)
        if larger / smaller < _PERIOD_WINDOW_RATIO:
            continue
        shorter_label = "previous" if n_previous < n_current else "current"
        limitations.append(
            Limitation(
                category="methodological",
                text=(
                    f"{ev.source_tool}'s two compared periods cover very different amounts of data "
                    f"(previous: {n_previous}, current: {n_current}) -- the {shorter_label} period is an "
                    "unusually short or narrow baseline, which can make an ordinary fluctuation look "
                    "like a larger structural change than it really is (a regression-to-the-mean risk); "
                    "treat the size of this comparison with caution."
                ),
                severity="reduces_confidence",
                affected_findings=[f"finding_{i}"],
            )
        )
    return limitations


def check_metric_denominators(evidence: list[Evidence], registry: MetricRegistry | None) -> list[Limitation]:
    if registry is None:
        return []
    limitations: list[Limitation] = []
    seen: set[str] = set()
    for index, item in enumerate(evidence):
        if not item.metric or item.metric in seen:
            continue
        definition = registry.definition_for(item.metric)
        if definition is None or definition.kind not in ("rate", "ratio") or definition.status == "resolved":
            continue
        seen.add(item.metric)
        limitations.append(Limitation(
            category="methodological", severity="blocks_conclusion",
            text=(f"Metric '{item.metric}' is a derived rate/ratio without a verified denominator. "
                  "Define its numerator, denominator, aggregation, and eligible population before drawing a conclusion."),
            affected_findings=[f"finding_{index}"],
        ))
    return limitations


def check_numerical_sanity(evidence: list[Evidence], metric_registry: MetricRegistry | None = None) -> list[Limitation]:
    """Runs all numerical-sanity rules against already-gathered evidence. Additive to
    (never a replacement for) verifier.py's existing outlier/sample-size/cross-check
    limitations -- called from build_findings alongside them."""
    return (
        _find_impossible_percentages(evidence)
        + _find_population_mismatches(evidence)
        + _find_group_magnitude_outliers(evidence)
        + _find_group_size_imbalance(evidence)
        + _find_unusual_baseline_window(evidence)
        + check_metric_denominators(evidence, metric_registry)
    )
