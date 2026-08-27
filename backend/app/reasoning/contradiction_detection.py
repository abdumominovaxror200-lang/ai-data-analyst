"""Deterministic detection of a ranking contradiction between two different
aggregations of the same group comparison (final stress-test mission, Phase 5:
"mean says A > B, median says B > A" -- the system must not blindly pick one).

Runs against already-gathered evidence, exactly like `confound_detection.py` --
zero new tool call, zero new LLM call. Scoped deliberately narrow and mechanically
well-defined: two `group_and_aggregate` calls over the SAME `group_by`/`agg_column`
but a DIFFERENT `agg_func` (e.g. "mean" vs "median") whose top-ranked group differs.
This is a real, common, and easy-to-miss trap -- a skewed or outlier-heavy
distribution can make the mean-based ranking and the median-based ranking of the
same groups disagree about which group is actually "best", and nothing previously
checked for this (`verifier._cross_check` only compares a directly-shared flat
scalar field like "mean"/"value"/"coefficient"/"statistic" across DIFFERENT tools,
which `group_and_aggregate`'s own per-group "value" field never surfaces at that
top level -- a structurally different shape this module reads directly instead).

Other Phase-5-named contradiction patterns (aggregate-vs-segment trend reversal,
correlation sign flipping within groups, forecast-vs-baseline disagreement) are
deliberately NOT covered here: each would need a materially different, more complex
detector (time-direction reasoning, per-group correlation comparison, or a second
forecast run) that isn't justified by evidence gathered so far -- documented as a
known gap rather than approximated with a fragile heuristic.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence, Limitation


def _top_group(groups: object) -> tuple[str, float] | None:
    if not isinstance(groups, list):
        return None
    valid = [
        (g["group"], g["value"]) for g in groups
        if isinstance(g, dict) and "group" in g and isinstance(g.get("value"), (int, float)) and not isinstance(g.get("value"), bool)
    ]
    if len(valid) < 2:
        return None
    return max(valid, key=lambda pair: pair[1])


def detect_ranking_contradictions(evidence: list[Evidence]) -> list[Limitation]:
    limitations: list[Limitation] = []
    seen: set[tuple] = set()
    by_key: dict[tuple, dict[str, Evidence]] = {}

    for ev in evidence:
        r = ev.result_summary
        if not isinstance(r, dict) or ev.source_tool != "group_and_aggregate":
            continue
        group_by, agg_column, agg_func = r.get("group_by"), r.get("agg_column"), r.get("agg_func")
        if not (isinstance(group_by, str) and isinstance(agg_column, str) and isinstance(agg_func, str)):
            continue
        by_key.setdefault((group_by, agg_column), {})[agg_func] = ev

    for (group_by, agg_column), by_func in by_key.items():
        funcs = sorted(by_func)
        for i in range(len(funcs)):
            for j in range(i + 1, len(funcs)):
                f1, f2 = funcs[i], funcs[j]
                top1 = _top_group(by_func[f1].result_summary.get("groups"))
                top2 = _top_group(by_func[f2].result_summary.get("groups"))
                if not top1 or not top2 or top1[0] == top2[0]:
                    continue
                signature = (group_by, agg_column, f1, f2)
                if signature in seen:
                    continue
                seen.add(signature)
                limitations.append(
                    Limitation(
                        category="methodological",
                        text=(
                            f"'{agg_column}' ranks '{group_by}' groups differently depending on the "
                            f"aggregation used: {f1} ranks '{top1[0]}' highest, but {f2} ranks "
                            f"'{top2[0]}' highest -- this is a genuine contradiction, likely because "
                            "one or more groups has a skewed or outlier-influenced distribution. Do "
                            "not present either ranking as the single correct answer without noting "
                            "this disagreement."
                        ),
                        severity="reduces_confidence",
                    )
                )
    return limitations
