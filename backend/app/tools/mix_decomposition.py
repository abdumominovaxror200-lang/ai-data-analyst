"""Deterministic Mix Decomposition Engine (shift-share / Blinder-Oaxaca-style
3-term decomposition).

Answers a specific, common, and commonly-misjudged question: when a per-unit
metric (e.g. average order value, churn rate, average revenue per customer)
changes between two periods, how much of that change is because the MIX of
segments shifted (composition effect) versus each segment's own average
actually changing (within-segment effect)? This is exactly the mechanism
behind Simpson's-paradox-style reversals, where an aggregate trend and every
segment's own trend can point in different directions.

THE FORMULA (exact algebraic identity, not an approximation):

For segment s present in both periods with share w_s (fraction of that
period's rows in segment s) and mean value m_s:

    overall = sum_s( w_s * m_s )

Let A = previous period, B = current period. Then:

    total_change      = overall_B - overall_A
    mix_effect        = sum_s( (w_B_s - w_A_s) * m_A_s )   -- composition shift,
                         holding each segment's own average fixed at period A
    within_effect     = sum_s( w_A_s * (m_B_s - m_A_s) )   -- each segment's own
                         average changing, holding period-A mix fixed
    interaction       = sum_s( (w_B_s - w_A_s) * (m_B_s - m_A_s) )

    mix_effect + within_effect + interaction == total_change   (EXACT identity)

This is the standard 3-term shift-share decomposition (equivalent to a
Blinder-Oaxaca decomposition with the "before" period as the reference/
counterfactual mix and rates). The interaction term is not optional in the
sense of being droppable -- omitting it (a common but sloppy 2-term
presentation) would only approximately reconcile, and by how much depends on
an arbitrary allocation convention for splitting it into the other two. Always
reporting it, labeled as its own transparent term, is what makes the
reconciliation exact and auditable rather than approximate.

WHAT THIS DOES NOT DO: identify WHY the mix shifted or WHY a segment's average
changed, and does not establish that either effect CAUSED the outcome to
change -- this is a purely descriptive, arithmetic decomposition of an
observed difference, always returned with `"causal_interpretation":
"not_supported"` and never phrased as one factor "driving"/"causing" the
other (see the module-level `_ensure_no_causal_language` check, defense in
depth against literally including a causal phrase in this tool's own output,
even though every field here is fixed, dataset-agnostic, arithmetic text).
"""

from __future__ import annotations

import pandas as pd

from app.reasoning.causation_guard import find_causal_phrases
from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

# Mirrors confound_detection.py's own established floor ("_MIN_GROUP_SIZE = 5
# # each compared group needs at least this many rows to trust a proportion")
# -- the same threshold below which a segment's own average is too noisy to
# treat as a real per-segment rate rather than sampling noise.
_MIN_SEGMENT_GROUP_SIZE = 5

# Below this many valid (non-missing-segment) rows in EITHER period, a
# per-segment breakdown is not attempted at all -- there is not enough data to
# support splitting further than the two period totals already computed by
# compare_periods.
_MIN_TOTAL_OBSERVATIONS = 10

# The 3-term identity is exact algebra -- this only guards against a coding
# bug corrupting it, not against the math failing (it cannot, absent a bug).
_RECONCILIATION_TOLERANCE = 1e-6


def mix_decomposition(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    segment_column: str,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
    filters: list[dict] | None = None,
) -> dict:
    """Decomposes the change in the PER-ROW MEAN of `value_column` between the
    previous window (`previous_start`..`previous_end`) and the current window
    (`current_start`..`current_end`) into composition (mix) and within-segment
    components, by category of `segment_column`.

    ASSUMPTION, stated explicitly: the metric this decomposes is a per-unit/
    per-order average (`value_column` per row, e.g. one row per order/customer/
    period-record) -- the mean, not the sum. A sum-based "which segment
    contributed most to the total dollar change" question is a different,
    already-covered question (see `contribution_analysis`); conflating the two
    would silently misrepresent whether an aggregate change reflects real
    per-unit performance change or just more/fewer units in a segment.

    Segments are restricted to those with at least `_MIN_SEGMENT_GROUP_SIZE`
    valid rows in BOTH periods ("common support") -- a segment present in only
    one period, or too small in either, is EXCLUDED and reported explicitly in
    `excluded_segments`, never silently folded into another segment or given a
    fabricated rate. If no segment has common support at all, this raises
    (nothing to decompose)."""
    working = apply_filters(df, filters) if filters else df

    for col in (date_column, value_column, segment_column):
        if col not in working.columns:
            raise ToolExecutionError(f"Unknown column '{col}'.")
    if not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' must be numeric.")

    current_start_ts, current_end_ts = _parse_boundary_dates(current_start, current_end, "current")
    previous_start_ts, previous_end_ts = _parse_boundary_dates(previous_start, previous_end, "previous")

    dates = pd.to_datetime(working[date_column], errors="coerce")
    available_min, available_max = dates.min(), dates.max()

    current_mask = (dates >= current_start_ts) & (dates <= current_end_ts)
    previous_mask = (dates >= previous_start_ts) & (dates <= previous_end_ts)
    current_all = working.loc[current_mask]
    previous_all = working.loc[previous_mask]

    # Missing segment values: excluded from the decomposition (never treated as
    # their own category, never dropped silently -- counted and reported).
    current_missing_segment = int(current_all[segment_column].isna().sum())
    previous_missing_segment = int(previous_all[segment_column].isna().sum())
    current_valid = current_all.dropna(subset=[segment_column])
    previous_valid = previous_all.dropna(subset=[segment_column])

    if len(current_valid) < _MIN_TOTAL_OBSERVATIONS or len(previous_valid) < _MIN_TOTAL_OBSERVATIONS:
        raise ToolExecutionError(
            f"Insufficient observations for a mix decomposition: current period has "
            f"{len(current_valid)} valid row(s), previous period has {len(previous_valid)} "
            f"(need at least {_MIN_TOTAL_OBSERVATIONS} in each, after excluding rows with a "
            f"missing '{segment_column}' value)."
        )

    current_stats = current_valid.groupby(segment_column)[value_column].agg(["count", "mean"])
    previous_stats = previous_valid.groupby(segment_column)[value_column].agg(["count", "mean"])

    all_segments = sorted(set(current_stats.index) | set(previous_stats.index), key=str)
    common_segments = [
        s for s in all_segments
        if s in current_stats.index and s in previous_stats.index
        and current_stats.loc[s, "count"] >= _MIN_SEGMENT_GROUP_SIZE
        and previous_stats.loc[s, "count"] >= _MIN_SEGMENT_GROUP_SIZE
    ]
    excluded_segments = []
    for s in all_segments:
        if s in common_segments:
            continue
        in_current = s in current_stats.index
        in_previous = s in previous_stats.index
        reason = (
            "absent in previous period" if in_current and not in_previous else
            "absent in current period" if in_previous and not in_current else
            f"below the minimum group size ({_MIN_SEGMENT_GROUP_SIZE}) in at least one period"
        )
        excluded_segments.append({
            "segment": str(s),
            "reason": reason,
            "n_current": int(current_stats.loc[s, "count"]) if in_current else 0,
            "n_previous": int(previous_stats.loc[s, "count"]) if in_previous else 0,
        })

    if not common_segments:
        raise ToolExecutionError(
            f"No common segment support between the two periods: no category of "
            f"'{segment_column}' has at least {_MIN_SEGMENT_GROUP_SIZE} valid rows in BOTH "
            f"the current and previous period. A mix decomposition requires segments that "
            f"exist, with enough data, in both periods being compared."
        )

    # The decomposition can only cover the common-support subset -- restricting
    # to it here (rather than the full period) is what the decomposition
    # reconciles against; `full_sample_*` below separately reports the
    # unrestricted headline figures so coverage loss is never hidden.
    n_current_common = int(sum(current_stats.loc[s, "count"] for s in common_segments))
    n_previous_common = int(sum(previous_stats.loc[s, "count"] for s in common_segments))

    segment_rows = []
    mix_effect = 0.0
    within_effect = 0.0
    interaction = 0.0
    overall_current_common = 0.0
    overall_previous_common = 0.0
    for s in common_segments:
        n_cur = int(current_stats.loc[s, "count"])
        n_prev = int(previous_stats.loc[s, "count"])
        m_cur = float(current_stats.loc[s, "mean"])
        m_prev = float(previous_stats.loc[s, "mean"])
        w_cur = n_cur / n_current_common
        w_prev = n_prev / n_previous_common

        seg_mix = (w_cur - w_prev) * m_prev
        seg_within = w_prev * (m_cur - m_prev)
        seg_interaction = (w_cur - w_prev) * (m_cur - m_prev)

        mix_effect += seg_mix
        within_effect += seg_within
        interaction += seg_interaction
        overall_current_common += w_cur * m_cur
        overall_previous_common += w_prev * m_prev

        segment_rows.append({
            "segment": str(s),
            "n_previous": n_prev,
            "n_current": n_cur,
            "share_previous": round(w_prev, 6),
            "share_current": round(w_cur, 6),
            "mean_previous": round(m_prev, 6),
            "mean_current": round(m_cur, 6),
            "mix_contribution": round(seg_mix, 6),
            "within_contribution": round(seg_within, 6),
            "interaction_contribution": round(seg_interaction, 6),
        })

    total_change_common = overall_current_common - overall_previous_common
    reconciled_sum = mix_effect + within_effect + interaction
    reconciliation_diff = total_change_common - reconciled_sum
    reconciles = abs(reconciliation_diff) <= _RECONCILIATION_TOLERANCE

    full_sample_current_mean = float(current_valid[value_column].mean())
    full_sample_previous_mean = float(previous_valid[value_column].mean())

    coverage_pct_current = round(n_current_common / len(current_valid) * 100, 2) if len(current_valid) else 0.0
    coverage_pct_previous = round(n_previous_common / len(previous_valid) * 100, 2) if len(previous_valid) else 0.0

    result = {
        "metric": value_column,
        "segment_dimension": segment_column,
        "current_period": {"start": current_start, "end": current_end},
        "previous_period": {"start": previous_start, "end": previous_end},
        "dataset_date_coverage": {
            "min": available_min.strftime("%Y-%m-%d") if pd.notna(available_min) else None,
            "max": available_max.strftime("%Y-%m-%d") if pd.notna(available_max) else None,
        },
        "sample_counts": {
            "n_current_total_valid": int(len(current_valid)),
            "n_previous_total_valid": int(len(previous_valid)),
            "n_current_common_support": n_current_common,
            "n_previous_common_support": n_previous_common,
            "n_current_missing_segment_excluded": current_missing_segment,
            "n_previous_missing_segment_excluded": previous_missing_segment,
            "coverage_pct_current": coverage_pct_current,
            "coverage_pct_previous": coverage_pct_previous,
        },
        "full_sample_overall_change": {
            "previous_mean": round(full_sample_previous_mean, 6),
            "current_mean": round(full_sample_current_mean, 6),
            "delta": round(full_sample_current_mean - full_sample_previous_mean, 6),
            "note": (
                "Computed over ALL valid rows in each period (not just common-support "
                "segments) -- this is the real headline change. It may differ slightly "
                "from decomposition_overall_change below whenever coverage_pct is below "
                "100%, since the decomposition itself can only cover segments with common "
                "support."
            ),
        },
        "decomposition_overall_change": {
            "previous_mean": round(overall_previous_common, 6),
            "current_mean": round(overall_current_common, 6),
            "delta": round(total_change_common, 6),
        },
        "components": {
            "mix_effect": round(mix_effect, 6),
            "within_segment_effect": round(within_effect, 6),
            "interaction_or_residual": round(interaction, 6),
        },
        "reconciliation": {
            "reconciled_sum": round(reconciled_sum, 6),
            "observed_change": round(total_change_common, 6),
            "difference": round(reconciliation_diff, 10),
            "tolerance": _RECONCILIATION_TOLERANCE,
            "reconciles": reconciles,
        },
        "segments": segment_rows,
        "excluded_segments": excluded_segments,
        "assumptions": [
            f"Metric is the per-row mean of '{value_column}' (a per-unit/per-order average), not a sum.",
            f"Only segments with at least {_MIN_SEGMENT_GROUP_SIZE} valid rows in BOTH periods are included in the decomposition.",
            "The previous period's mix and within-segment rates are used as the reference point for the mix_effect and within_segment_effect terms respectively (a standard shift-share convention) -- the interaction_or_residual term captures the remainder exactly, so the three terms always sum to the observed change over the common-support subset.",
            "Rows with a missing segment value are excluded entirely, not treated as their own category.",
        ],
        "causal_interpretation": "not_supported",
        "disclaimer": (
            "This is a descriptive, arithmetic decomposition of an observed change into "
            "composition (mix) and within-segment components. It does not identify why "
            "the mix shifted or why any segment's own average changed, and it does not "
            "establish causation for the overall change."
        ),
    }

    if not reconciles:
        # Should be structurally unreachable (the identity is exact algebra) --
        # fail loudly rather than silently return an untrustworthy breakdown if
        # a future edit ever breaks it.
        raise ToolExecutionError(
            f"Internal error: mix decomposition components did not reconcile within "
            f"tolerance (observed={total_change_common!r}, reconciled_sum={reconciled_sum!r}, "
            f"diff={reconciliation_diff!r}). This indicates a bug, not a data problem -- "
            f"no result is returned rather than an unverified one."
        )

    _ensure_no_causal_language(result)
    return result


def _parse_boundary_dates(start: str, end: str, label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
    except Exception as exc:  # pandas/dateutil raise different exception types across versions
        raise ToolExecutionError(f"Invalid {label} period date(s): {start!r} / {end!r}.") from exc
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ToolExecutionError(f"Invalid {label} period date(s): {start!r} / {end!r}.")
    if start_ts > end_ts:
        raise ToolExecutionError(f"The {label} period's start ({start}) is after its end ({end}).")
    return start_ts, end_ts


def _ensure_no_causal_language(result: dict) -> None:
    """Defense in depth: every string this tool ever emits is fixed,
    dataset-agnostic, arithmetic/descriptive text (never LLM-generated, never
    built from a dataset value in a way that could smuggle in phrasing) -- this
    asserts that invariant holds rather than merely intending it, so a future
    edit that carelessly adds a causal-sounding word here is caught immediately
    rather than silently leaking past the causation guard downstream (exactly
    the real regression this project already hit once with an objective
    description -- see investigation_objectives.py's concentration_analysis
    description fix)."""
    texts = [result["disclaimer"], *result["assumptions"]]
    for excluded in result["excluded_segments"]:
        texts.append(excluded["reason"])
    for text in texts:
        found = find_causal_phrases(text)
        if found:
            raise ToolExecutionError(
                f"Internal error: mix decomposition output unexpectedly contains causal "
                f"language {found!r} in {text!r}. Refusing to return a result that could "
                f"be misread as a causal claim."
            )
