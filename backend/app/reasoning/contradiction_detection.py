"""Deterministic Evidence Contradiction Engine (Phase 5; upgraded to "Contradiction
Engine 2.0" for the v2 reliability mission).

Runs against already-gathered evidence, exactly like `confound_detection.py` -- zero
new tool call, zero new LLM call. Three independent, mechanically well-defined checks:

1. `detect_ranking_contradictions` -- "mean says A > B, median says B > A": two
   `group_and_aggregate` calls over the SAME `group_by`/`agg_column` but a DIFFERENT
   `agg_func` (e.g. "mean" vs "median") whose top-ranked group differs. A skewed or
   outlier-heavy distribution can make the mean-based ranking and the median-based
   ranking of the same groups disagree about which group is actually "best", and
   nothing else checks for this (`verifier._cross_check` only compares a directly-
   shared flat scalar field like "mean"/"value"/"coefficient" across DIFFERENT
   tools, which `group_and_aggregate`'s own per-group "value" field never surfaces
   at that top level).
2. `detect_overall_vs_subgroup_contradiction` -- the mission's own flagship example:
   "revenue increased 15% overall, but every major segment decreased". Reads
   `compare_periods` evidence's own real `pct_change` field, using
   `Evidence.population` (wired into `executor.py` for exactly this -- see that
   module's `_guess_population` docstring) to tell an unfiltered ("overall") call
   apart from a segment-filtered ("subgroup") one for the same metric. Flags when
   the overall direction and EVERY examined subgroup's direction disagree -- a
   classic Simpson's-paradox-via-mix-shift pattern.
3. `detect_data_quality_contradictions` -- two independent verification tools
   (`detect_anomalies`/`duplicate_analysis`/`data_quality_report`) examining the
   SAME population disagreeing about whether the data is clean. Reuses
   `verifier._VERIFICATION_TOOLS_CLEAN_CHECK`'s exact clean-check rules rather than
   reimplementing them, so this module and `verifier.py`'s own corroboration logic
   never disagree about what "clean" means for a given tool.

Other Phase-2-named contradiction patterns from the v2 mission's 14-item list
(correlation sign flipping within groups, forecast-vs-baseline disagreement,
numerator/denominator inconsistency, time-window contradiction as a distinct check
from #2 above) are deliberately NOT covered here: each would need either a materially
different detector (per-group correlation comparison, a second forecast run) or a
tool this project's toolset does not have (nothing computes a cross-aggregation
ratio as a structured field -- the same gap documented for `numerical_sanity.py`'s
impossible-percentage check in `.agent/FINAL_GO_NO_GO_AUDIT.md` §0a). Documented as
a known gap rather than approximated with a fragile heuristic. "Two independent tools
producing incompatible numerical results" and "unit contradiction" are already
covered, under different names, by `verifier._cross_check` and
`numerical_sanity._find_group_magnitude_outliers`/`_find_impossible_percentages`
respectively -- not duplicated here. "Recommendation vs evidence contradiction" and
"conclusion vs limitation contradiction" are already structurally enforced by
`recommendation_grounding.py`'s `blocks_conclusion` override and
`conclusion_guard.py`'s caveat injection, not merely detected-and-reported -- a
stronger guarantee than this module's advisory `reduces_confidence` limitations, so
also not duplicated here.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence, Limitation
from app.reasoning.verifier import _VERIFICATION_TOOLS_CLEAN_CHECK


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


def detect_overall_vs_subgroup_contradiction(evidence: list[Evidence]) -> list[Limitation]:
    """The mission's own flagship example: "revenue increased 15% overall, but
    every major segment decreased." Groups `compare_periods` evidence by metric,
    splits into "overall" (Evidence.population is None -- no filters were applied)
    vs "subgroup" (population is set) calls, and flags when the overall's
    `pct_change` direction disagrees with EVERY examined subgroup's -- not just one
    (a single dissenting subgroup is normal and not a contradiction; a unanimous
    reversal across every subgroup actually checked is the real, surprising,
    worth-investigating pattern, and usually signals a shift in the MIX of
    subgroups rather than a genuine change within any one of them)."""
    by_metric: dict[str, list[Evidence]] = {}
    for ev in evidence:
        r = ev.result_summary
        if not isinstance(r, dict) or ev.source_tool != "compare_periods" or not ev.metric:
            continue
        pct_change = r.get("pct_change")
        if isinstance(pct_change, (int, float)) and not isinstance(pct_change, bool):
            by_metric.setdefault(ev.metric, []).append(ev)

    limitations: list[Limitation] = []
    for metric, items in by_metric.items():
        overall = next((e for e in items if e.population is None), None)
        subgroups = [e for e in items if e.population is not None]
        if overall is None or len(subgroups) < 2:
            continue
        overall_pct = overall.result_summary["pct_change"]
        if overall_pct == 0:
            continue

        opposing = [
            e for e in subgroups
            if e.result_summary["pct_change"] != 0 and (e.result_summary["pct_change"] > 0) != (overall_pct > 0)
        ]
        if len(opposing) != len(subgroups):
            continue

        direction = "increased" if overall_pct > 0 else "decreased"
        opposite = "decreased" if overall_pct > 0 else "increased"
        subgroup_desc = "; ".join(f"{e.population} ({round(e.result_summary['pct_change'], 1)}%)" for e in subgroups)
        limitations.append(
            Limitation(
                category="methodological",
                text=(
                    f"'{metric}' {direction} overall ({round(overall_pct, 1)}%), but every examined "
                    f"subgroup {opposite} instead ({subgroup_desc}) -- this is a genuine contradiction "
                    "(a Simpson's-paradox-style pattern, often caused by a shift in the MIX of "
                    "subgroups rather than a real change within any of them). Do not present the "
                    "overall figure as representative of what happened within each subgroup without "
                    "investigating this further."
                ),
                severity="blocks_conclusion",
            )
        )
    return limitations


def detect_data_quality_contradictions(evidence: list[Evidence]) -> list[Limitation]:
    """Two independent verification tools examining the SAME population (matched
    via `Evidence.population` -- both `None`, i.e. both examined the whole dataset
    with no filters, is the common and most valuable case) disagreeing about
    whether the data is clean. Reuses `verifier._VERIFICATION_TOOLS_CLEAN_CHECK`'s
    exact clean-check rules -- this module and `verifier.py`'s own corroboration
    logic must never disagree about what "clean" means for a given tool."""
    candidates = [e for e in evidence if e.source_tool in _VERIFICATION_TOOLS_CLEAN_CHECK]
    limitations: list[Limitation] = []
    seen: set[tuple] = set()

    for i, e1 in enumerate(candidates):
        for e2 in candidates[i + 1:]:
            if e1.source_tool == e2.source_tool or e1.population != e2.population:
                continue
            clean1 = _VERIFICATION_TOOLS_CLEAN_CHECK[e1.source_tool](e1.result_summary)
            clean2 = _VERIFICATION_TOOLS_CLEAN_CHECK[e2.source_tool](e2.result_summary)
            if clean1 == clean2:
                continue

            signature = tuple(sorted([e1.source_tool, e2.source_tool])) + (e1.population,)
            if signature in seen:
                continue
            seen.add(signature)

            clean_ev, dirty_ev = (e1, e2) if clean1 else (e2, e1)
            scope = f"the '{e1.population}' subset" if e1.population else "the whole dataset"
            limitations.append(
                Limitation(
                    category="methodological",
                    text=(
                        f"{clean_ev.source_tool} found no data-quality problem over {scope}, but "
                        f"{dirty_ev.source_tool} found one over the same scope -- these are conflicting "
                        "data-quality signals; investigate before treating the data as either fully "
                        "clean or meaningfully compromised."
                    ),
                    severity="reduces_confidence",
                )
            )
    return limitations
