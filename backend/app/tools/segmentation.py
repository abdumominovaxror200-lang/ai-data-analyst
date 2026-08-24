from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_MAX_COHORTS_RETURNED = 24
_MAX_PERIODS_SINCE_RETURNED = 24

# Segment labels, keyed by (R, F, M) score tier ("high" = score 4-5, "mid" = 3,
# "low" = 1-2). This is the exact, documented rule `rfm_analysis` uses to turn
# three 1-5 quintile scores into one human label — see `_label_segment`.
_SEGMENT_RULES = [
    # (recency_tier, frequency_tier, monetary_tier) -> label. Checked in order;
    # first match wins. "high" R = bought recently, "high" F/M = buys often /
    # spends a lot.
    (("high", "high", "high"), "Champions"),
    (("high", "high", "mid"), "Loyal Customers"),
    (("high", "mid", "high"), "Loyal Customers"),
    (("high", "high", "low"), "Loyal Customers"),
    (("high", "mid", "mid"), "Potential Loyalists"),
    (("high", "low", "low"), "New Customers"),
    (("high", "mid", "low"), "Potential Loyalists"),
    (("high", "low", "high"), "Potential Loyalists"),
    (("high", "low", "mid"), "New Customers"),
    (("mid", "high", "high"), "Loyal Customers"),
    (("mid", "high", "mid"), "Loyal Customers"),
    (("mid", "high", "low"), "Potential Loyalists"),
    (("mid", "mid", "high"), "Potential Loyalists"),
    (("mid", "mid", "mid"), "Needs Attention"),
    (("mid", "mid", "low"), "Needs Attention"),
    (("mid", "low", "high"), "Needs Attention"),
    (("mid", "low", "mid"), "Needs Attention"),
    (("mid", "low", "low"), "About to Sleep"),
    (("low", "high", "high"), "At Risk"),
    (("low", "high", "mid"), "At Risk"),
    (("low", "mid", "high"), "At Risk"),
    (("low", "high", "low"), "At Risk"),
    (("low", "mid", "mid"), "At Risk"),
    (("low", "low", "high"), "Cannot Lose Them"),
    (("low", "mid", "low"), "Hibernating"),
    (("low", "low", "mid"), "Hibernating"),
    (("low", "low", "low"), "Lost"),
]
_SEGMENT_LOOKUP = dict(_SEGMENT_RULES)


def _tier(score: int) -> str:
    if score >= 4:
        return "high"
    if score == 3:
        return "mid"
    return "low"


def _label_segment(r_score: int, f_score: int, m_score: int) -> str:
    key = (_tier(r_score), _tier(f_score), _tier(m_score))
    return _SEGMENT_LOOKUP.get(key, "Needs Attention")


def _quintile_score(series: pd.Series, ascending: bool) -> pd.Series:
    """1-5 quintile score. `ascending=True` means a higher raw value earns a
    higher score (Frequency, Monetary); `ascending=False` means a *lower* raw
    value earns a higher score (Recency — fewer days since last purchase is
    better). Falls back to a rank-based split when there are too few distinct
    values for `pd.qcut` to form 5 real bins (common with small/degenerate
    inputs), so the tool never crashes on data with lots of ties."""
    ranks = series.rank(method="first", ascending=ascending)
    try:
        scores = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        return scores.astype(int)
    except ValueError:
        # Not enough distinct rank values for 5 quantile bins - fall back to an
        # even split of the sorted rank order into 5 groups.
        n = len(series)
        bin_size = max(1, -(-n // 5))  # ceil division
        positions = ranks.rank(method="first").astype(int) - 1
        return (positions // bin_size + 1).clip(upper=5).astype(int)


def rfm_analysis(
    df: pd.DataFrame,
    customer_column: str,
    date_column: str,
    value_column: str,
    reference_date: str | None = None,
    filters: list[dict] | None = None,
) -> dict:
    """Recency/Frequency/Monetary segmentation, one row per customer collapsed
    into per-segment summaries (never the full per-customer table — see the
    module docstring in `insights.py` on why unbounded per-row payloads have
    caused real 413 errors on this project).

    Scoring rule (documented, not a black box): for each of R, F, M
    independently, customers are split into 5 quintiles and scored 1
    (worst) to 5 (best) — for Recency, "best" means *fewest* days since their
    last transaction, so the ascending/descending direction is flipped
    relative to Frequency/Monetary. Each 1-5 score is collapsed to a tier
    ("low" = 1-2, "mid" = 3, "high" = 4-5), and the (R-tier, F-tier, M-tier)
    triple maps to one of 10 standard RFM segment labels (Champions, Loyal
    Customers, Potential Loyalists, New Customers, Needs Attention, About to
    Sleep, At Risk, Cannot Lose Them, Hibernating, Lost) via the fixed lookup
    table in `_SEGMENT_RULES` — e.g. (high, high, high) = "Champions" (bought
    recently, often, and for a lot), (low, low, low) = "Lost" (haven't bought
    in a long time, rarely bought, spent little).
    """
    working = apply_filters(df, filters) if filters else df
    for col in (customer_column, date_column, value_column):
        if col not in working.columns:
            raise ToolExecutionError(f"Unknown column '{col}'.")
    if not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' must be numeric.")

    subset = working[[customer_column, date_column, value_column]].dropna()
    subset = subset.copy()
    subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
    subset = subset.dropna(subset=[date_column])
    if subset.empty:
        raise ToolExecutionError("No usable rows (need non-null customer, date, and value).")

    if reference_date is not None:
        ref = pd.to_datetime(reference_date)
    else:
        ref = subset[date_column].max()

    n_customers = subset[customer_column].nunique()
    if n_customers < 5:
        raise ToolExecutionError(
            f"At least 5 distinct customers are required for quintile-based RFM scoring (found {n_customers})."
        )

    grouped = subset.groupby(customer_column)
    rfm = grouped.agg(
        recency_days=(date_column, lambda s: (ref - s.max()).days),
        frequency=(customer_column, "count"),
        monetary=(value_column, "sum"),
    )
    rfm = rfm[rfm["recency_days"] >= 0]  # guard against a reference_date before some customer's last purchase
    if rfm.empty:
        raise ToolExecutionError("No customers have transactions on or before the reference date.")

    rfm["r_score"] = _quintile_score(rfm["recency_days"], ascending=False)
    rfm["f_score"] = _quintile_score(rfm["frequency"], ascending=True)
    rfm["m_score"] = _quintile_score(rfm["monetary"], ascending=True)
    rfm["segment"] = [
        _label_segment(r, f, m) for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"])
    ]

    summary = (
        rfm.groupby("segment")
        .agg(
            customer_count=("segment", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
    )
    segments = [
        {
            "segment": row["segment"],
            "customer_count": int(row["customer_count"]),
            "pct_of_customers": round(float(row["customer_count"]) / len(rfm) * 100, 2),
            "avg_recency_days": round(float(row["avg_recency_days"]), 2),
            "avg_frequency": round(float(row["avg_frequency"]), 2),
            "avg_monetary": round(float(row["avg_monetary"]), 2),
        }
        for _, row in summary.iterrows()
    ]
    segments.sort(key=lambda s: s["customer_count"], reverse=True)

    return {
        "customer_column": customer_column,
        "date_column": date_column,
        "value_column": value_column,
        "reference_date": ref.strftime("%Y-%m-%d"),
        "n_customers": int(len(rfm)),
        "segments": segments,
    }


def cohort_analysis(
    df: pd.DataFrame,
    customer_column: str,
    date_column: str,
    value_column: str | None = None,
    period: str = "M",
    filters: list[dict] | None = None,
) -> dict:
    """Cohort retention table: each customer's cohort is the calendar period of
    their *first* transaction; for every subsequent period, the table reports
    what fraction of that cohort is still active (transacted at all), or, if
    `value_column` is given, the cohort's total value in that period instead
    of a retention fraction.

    `period` follows pandas offset-alias convention — "M" (month, default),
    "W" (week), "D" (day), "Q" (quarter), "Y" (year) — via
    `Series.dt.to_period(period)`.

    Bounded output: at most `_MAX_COHORTS_RETURNED` cohorts (most recent
    first) and `_MAX_PERIODS_SINCE_RETURNED` periods-since-first-purchase per
    cohort, to keep the payload size-safe regardless of how long the dataset's
    history is.
    """
    working = apply_filters(df, filters) if filters else df
    required = [customer_column, date_column] + ([value_column] if value_column else [])
    for col in required:
        if col not in working.columns:
            raise ToolExecutionError(f"Unknown column '{col}'.")
    if value_column and not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' must be numeric.")

    valid_periods = {"D", "W", "M", "Q", "Y"}
    if period not in valid_periods:
        raise ToolExecutionError(f"period must be one of {sorted(valid_periods)}.")

    subset = working[required].dropna().copy()
    subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
    subset = subset.dropna(subset=[date_column])
    if subset.empty:
        raise ToolExecutionError("No usable rows (need non-null customer and date).")

    subset["_period"] = subset[date_column].dt.to_period(period)
    first_period = subset.groupby(customer_column)["_period"].transform("min")
    subset["_cohort"] = first_period
    subset["_periods_since"] = (subset["_period"] - subset["_cohort"]).apply(lambda x: x.n)

    if value_column:
        pivot = subset.pivot_table(
            index="_cohort", columns="_periods_since", values=value_column, aggfunc="sum", fill_value=0.0
        )
        cohort_sizes = subset.groupby("_cohort")[customer_column].nunique()
    else:
        active = subset.groupby(["_cohort", "_periods_since"])[customer_column].nunique()
        pivot = active.unstack(fill_value=0)
        cohort_sizes = subset.groupby("_cohort")[customer_column].nunique()

    cohorts_sorted = sorted(pivot.index)[-_MAX_COHORTS_RETURNED:]
    periods_sorted = sorted(pivot.columns)[:_MAX_PERIODS_SINCE_RETURNED]

    rows = []
    for cohort in cohorts_sorted:
        cohort_size = int(cohort_sizes.get(cohort, 0))
        row_values = {}
        for p in periods_sorted:
            if p not in pivot.columns:
                continue
            raw = pivot.loc[cohort, p] if cohort in pivot.index else 0
            if value_column:
                row_values[str(int(p))] = round(float(raw), 4)
            else:
                retention_rate = round(float(raw) / cohort_size, 4) if cohort_size else 0.0
                row_values[str(int(p))] = retention_rate
        rows.append(
            {
                "cohort": str(cohort),
                "cohort_size": cohort_size,
                "periods_since_first": row_values,
            }
        )

    return {
        "customer_column": customer_column,
        "date_column": date_column,
        "value_column": value_column,
        "period": period,
        "metric": "total_value" if value_column else "retention_rate",
        "n_cohorts": len(rows),
        "cohorts": rows,
    }


def churn_risk_analysis(
    df: pd.DataFrame,
    customer_column: str,
    date_column: str,
    reference_date: str | None = None,
    churn_threshold_days: float | None = None,
    filters: list[dict] | None = None,
) -> dict:
    """Flags each customer as churned / at_risk / active based on days since
    their last transaction (recency) relative to a threshold.

    Threshold inference (used only when `churn_threshold_days` is not given):
    for every customer with at least 2 transactions, compute their median
    inter-purchase gap (days between consecutive transactions), then take the
    *median across customers* of that per-customer median gap, and multiply
    by 3. Rationale: 3x a customer's own typical repurchase cadence is a
    standard, explainable "they're overdue" heuristic in retention analysis —
    a customer who reliably buys every ~10 days but hasn't been seen in 30 is
    a much stronger churn signal than an arbitrary fixed day count (e.g. "90
    days") that ignores how often this business's customers normally buy at
    all. Using the *median* (not mean) inter-purchase gap, both within a
    customer and across customers, avoids a few customers with erratic gaps
    or a few one-off bulk buyers distorting the threshold.

    If fewer than 2 customers have 2+ transactions (so no inter-purchase gap
    can be computed at all), inference is impossible and the caller must
    supply `churn_threshold_days` explicitly.

    `at_risk` is defined as recency between 1x and the full threshold (i.e.
    over their typical gap but not yet past the churn line) — active means
    recency at or below their typical gap, churned means recency at or past
    the full threshold.
    """
    working = apply_filters(df, filters) if filters else df
    for col in (customer_column, date_column):
        if col not in working.columns:
            raise ToolExecutionError(f"Unknown column '{col}'.")

    subset = working[[customer_column, date_column]].dropna().copy()
    subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
    subset = subset.dropna(subset=[date_column])
    if subset.empty:
        raise ToolExecutionError("No usable rows (need non-null customer and date).")

    if reference_date is not None:
        ref = pd.to_datetime(reference_date)
    else:
        ref = subset[date_column].max()

    grouped = subset.groupby(customer_column)[date_column]
    last_seen = grouped.max()
    recency_days = (ref - last_seen).dt.days
    recency_days = recency_days[recency_days >= 0]
    if recency_days.empty:
        raise ToolExecutionError("No customers have transactions on or before the reference date.")

    typical_gap_used = None
    inference_note = None
    if churn_threshold_days is None:
        per_customer_median_gap = []
        for _, dates in grouped:
            dates_sorted = dates.sort_values()
            if len(dates_sorted) < 2:
                continue
            gaps = dates_sorted.diff().dropna().dt.days
            if len(gaps) > 0:
                per_customer_median_gap.append(float(gaps.median()))

        if len(per_customer_median_gap) < 2:
            raise ToolExecutionError(
                "Cannot infer a churn threshold: fewer than 2 customers have 2+ transactions to measure a "
                "purchase cadence from. Pass churn_threshold_days explicitly."
            )
        typical_gap = pd.Series(per_customer_median_gap).median()
        if typical_gap <= 0:
            typical_gap = 1.0
        churn_threshold_days = round(float(typical_gap) * 3, 2)
        typical_gap_used = round(float(typical_gap), 2)
        inference_note = (
            f"Inferred as 3x the median customer's typical inter-purchase gap ({typical_gap_used} days) "
            f"across {len(per_customer_median_gap)} customers with 2+ transactions."
        )
    else:
        if churn_threshold_days <= 0:
            raise ToolExecutionError("churn_threshold_days must be positive.")
        typical_gap_used = round(float(churn_threshold_days) / 3, 2)

    at_risk_floor = typical_gap_used if typical_gap_used else churn_threshold_days / 3

    def _classify(days: float) -> str:
        if days >= churn_threshold_days:
            return "churned"
        if days > at_risk_floor:
            return "at_risk"
        return "active"

    risk = recency_days.apply(_classify)
    counts = risk.value_counts()
    total = len(risk)

    return {
        "customer_column": customer_column,
        "date_column": date_column,
        "reference_date": ref.strftime("%Y-%m-%d"),
        "churn_threshold_days": round(float(churn_threshold_days), 2),
        "threshold_inferred": inference_note is not None,
        "inference_note": inference_note,
        "n_customers": int(total),
        "counts": {
            "active": int(counts.get("active", 0)),
            "at_risk": int(counts.get("at_risk", 0)),
            "churned": int(counts.get("churned", 0)),
        },
        "pct": {
            "active": round(float(counts.get("active", 0)) / total * 100, 2),
            "at_risk": round(float(counts.get("at_risk", 0)) / total * 100, 2),
            "churned": round(float(counts.get("churned", 0)) / total * 100, 2),
        },
    }
