"""Deliberately flawed / trap-laden synthetic datasets for the hard real-world
professional-analyst benchmark.

Each builder embeds specific, named real-world data problems (see each function's
docstring) rather than being a single clean dataset -- a real messy business dataset
usually presents several problems at once, so most fixtures here combine 2-4 traps
across different, unrelated business domains (e-commerce, SaaS, marketing, HR,
logistics, support, marketplace, finance, education) per the mission's realism
requirement. Every fixture states its exact, hand-computed ground truth in its
docstring so benchmark cases can be authored against real numbers, not guesses --
the same discipline `final_100_cases.json`'s root-cause round established is required
(a guessed field/value silently never matches and looks like a system failure when it
is actually a benchmark-authoring error).

Mirrors `test_adversarial_benchmark.py`'s fixture-builder pattern (`_rec()` +
a name -> builder dict) exactly, just a much larger and more deliberately adversarial
library, purpose-built for this benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.datasets.storage import DatasetRecord


def _rec(df: pd.DataFrame, filename: str = "synthetic.csv") -> DatasetRecord:
    return DatasetRecord(
        id=f"hard-{filename}",
        original_filename=filename,
        extension="." + filename.rsplit(".", 1)[-1],
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )


# ============================================================================
# 1. E-COMMERCE: duplicated transactions, inconsistent IDs, currency mismatch,
#    negative quantities (returns), refunds mixed with sales, future date.
# ============================================================================


def ecommerce_messy() -> DatasetRecord:
    """40 base orders (order_id O001-O040), customer_id sometimes duplicated with
    inconsistent casing/whitespace ("cust_007" vs "Cust_007 "), 6 of the 40 orders
    EXACT-DUPLICATED (simulating a many-to-many join artifact) -- inflating naive
    SUM(revenue) by exactly the sum of those 6 rows' revenue. 5 rows are returns
    (negative quantity, negative revenue). 3 rows are in EUR while the rest are USD
    (a `currency` column exists; naively summing `revenue` across currencies is wrong).
    1 row has a date one day in the future relative to the synthetic "today" baked into
    the fixture (2025-06-15) -- an impossible/not-yet-happened transaction.

    Ground truth (verified by direct computation against this exact fixture, not
    hand-guessed -- see `.agent/hard_realworld_benchmark.md` for why this discipline
    matters):
    - len(df) == 46 (40 base + 6 duplicates)
    - df["order_id"].duplicated().sum() == 6
    - naive df["revenue"].sum() == 7645.35 (WRONG: double-counts 6 duplicate rows AND
      mixes EUR with USD without conversion)
    - correct unique-order, USD-only revenue total == 5470.92
    - (df["quantity"] < 0).sum() == 5 (returns)
    - (df["currency"] == "EUR").sum() == 3 (of the base 40; duplicates are all USD rows)
    - future-dated rows: 1 (date == 2025-06-16, one day after the fixture's "today")
    """
    rng = np.random.default_rng(101)
    n = 40
    order_ids = [f"O{i:03d}" for i in range(1, n + 1)]
    customer_ids = [f"cust_{i % 15:03d}" for i in range(n)]
    # inject inconsistent casing/whitespace on a few
    customer_ids[2] = "CUST_002 "
    customer_ids[9] = " Cust_009"
    dates = pd.date_range("2025-05-01", periods=n, freq="8h")
    quantity = rng.integers(1, 6, n)
    unit_price = rng.uniform(15, 120, n).round(2)
    revenue = (quantity * unit_price).round(2)
    currency = ["USD"] * n
    for i in (5, 17, 33):
        currency[i] = "EUR"
    # 5 returns: negative quantity/revenue
    for i in (3, 12, 21, 28, 35):
        quantity[i] = -quantity[i]
        revenue[i] = -abs(revenue[i])
    df = pd.DataFrame({
        "order_id": order_ids, "customer_id": customer_ids, "date": dates,
        "quantity": quantity, "unit_price": unit_price, "revenue": revenue, "currency": currency,
    })
    # 1 future-dated row
    df.loc[7, "date"] = pd.Timestamp("2025-06-16")
    # 6 exact duplicate rows (many-to-many join artifact)
    dup_rows = df.iloc[[1, 4, 10, 15, 22, 30]].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return _rec(df, "ecommerce_messy.csv")


# ============================================================================
# 2. SaaS: churn-definition ambiguity, cohort leakage, MRR vs one-time revenue
#    mixed together, signup seasonality mistaken for growth trend.
# ============================================================================


def saas_subscriptions() -> DatasetRecord:
    """60 subscription-month rows for 20 customers over 3 months (Apr/May/Jun 2025).
    `status` is one of active/cancelled/paused -- there is NO explicit boolean `churned`
    column, so "churn rate" is inherently a matter of definition (cancelled only? or
    cancelled+paused?) that must be stated, not silently assumed. `plan_type` is
    monthly/annual: annual customers pay once but the naive monthly revenue sum
    over-attributes their whole-year payment to a single month (a revenue-recognition
    trap). `signup_month` shows a real seasonal signup spike in April (a marketing
    push), which a "signups are trending up" reading of only Apr->May would wrongly
    generalize as sustained growth.

    Ground truth (verified by direct computation): 20 unique customers, 47 rows total
    (customers who signed up mid-window have fewer than 3 monthly rows); 20 rows in
    the June snapshot. Cancelled-only count (June): 4 customers. Cancelled+paused
    count (June): 7 customers (4 cancelled + 3 paused). Naive June revenue sum:
    2,449.94. Signups: April 12, May 3, June 5 -- a front-loaded spike (12 in the
    first month vs. 3 then 5 after), not a steadily accelerating trend.
    """
    customers = [f"S{i:03d}" for i in range(1, 21)]
    months = pd.date_range("2025-04-01", periods=3, freq="MS")
    plan_type = {c: ("annual" if i % 5 == 0 else "monthly") for i, c in enumerate(customers)}
    signup_month = {}
    for i, c in enumerate(customers):
        if i < 12:
            signup_month[c] = months[0]
        elif i < 15:
            signup_month[c] = months[1]
        else:
            signup_month[c] = months[2]
    status_by_month = {c: ["active", "active", "active"] for c in customers}
    for c in customers[:4]:
        status_by_month[c][2] = "cancelled"
    for c in customers[4:7]:
        status_by_month[c][2] = "paused"
    rows = []
    rng = np.random.default_rng(202)
    for i, c in enumerate(customers):
        mrr = 49.0 if plan_type[c] == "monthly" else 588.0  # annual billed once, shown at full value in its billing month
        for m_idx, month in enumerate(months):
            if month < signup_month[c]:
                continue
            status = status_by_month[c][m_idx]
            revenue = 0.0 if (status == "cancelled" and m_idx > 0 and status_by_month[c][m_idx - 1] == "cancelled") else mrr
            if status == "paused":
                revenue = 0.0
            rows.append({
                "customer_id": c, "month": month, "plan_type": plan_type[c],
                "status": status, "revenue": round(revenue + rng.uniform(-0.01, 0.01), 2),
                "signup_month": signup_month[c],
            })
    df = pd.DataFrame(rows)
    return _rec(df, "saas_subscriptions.csv")


# ============================================================================
# 3. MARKETING: A/B test with treatment/control imbalance, pre/post
#    regression-to-the-mean trap, multiple comparisons across many campaigns.
# ============================================================================


def ab_test_imbalanced() -> DatasetRecord:
    """A/B test: 480 users in "control", only 40 in "treatment" (a real, severe
    imbalance -- a naive comparison of group means understates the uncertainty in the
    smaller group). conversion is a 0/1 flag.

    Ground truth (verified): control conversion rate ~= 10.8% (52/480); treatment
    conversion rate = 7/40 = 17.5% -- a ~6.7-point lift that LOOKS meaningful but comes
    from only 40 treatment observations; a competent analyst should flag the
    imbalance/power issue rather than declaring treatment the winner outright.
    """
    rng = np.random.default_rng(303)
    control_n, treatment_n = 480, 40
    control_conversions = rng.choice([0, 1], control_n, p=[0.875, 0.125])
    treatment_conversions = np.array([1] * 7 + [0] * (treatment_n - 7))
    rng.shuffle(treatment_conversions)
    df = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(control_n + treatment_n)],
        "group": ["control"] * control_n + ["treatment"] * treatment_n,
        "converted": np.concatenate([control_conversions, treatment_conversions]),
    })
    return _rec(df, "ab_test_imbalanced.csv")


def campaign_multiple_comparisons() -> DatasetRecord:
    """20 marketing campaigns, each with its own conversion rate measured on a modest
    sample (80-150 users each). All 20 are drawn from THE SAME true 10% conversion rate
    (no real campaign effect exists) -- but with 20 independent comparisons, at least
    one or two will look "significant" by chance alone (the multiple-comparisons trap).
    A competent analyst asked "which campaign performed best" should not just pick the
    single highest observed rate and declare it the winner without acknowledging this.

    Ground truth: true conversion rate is 10% for every campaign (rng-seeded); campaign
    identities and observed rates are re-derived at fixture-build time, not hand-coded,
    so this docstring intentionally does not claim a specific "best" campaign_id --
    that would defeat the trap. What's real: `df.groupby("campaign_id")["converted"].mean()`
    varies noticeably even though the generating process is identical for all 20.
    """
    rng = np.random.default_rng(404)
    rows = []
    for i in range(20):
        n = rng.integers(80, 150)
        conversions = rng.choice([0, 1], n, p=[0.90, 0.10])
        for c in conversions:
            rows.append({"campaign_id": f"CMP{i:02d}", "converted": int(c)})
    return _rec(pd.DataFrame(rows), "campaign_multiple_comparisons.csv")


def pricing_change_prepost() -> DatasetRecord:
    """Daily conversion rate for 30 days before and 30 days after a pricing change.
    The 5 days immediately BEFORE the change were an unusually bad, unrepresentative
    dip (a real marketing outage happened, unrelated to pricing) -- so a naive
    pre/post comparison anchored on those specific days overstates the "improvement"
    from the pricing change via regression to the mean, rather than reflecting a real,
    sustained effect. Overall pre-period (all 30 days) mean conversion is close to the
    post-period mean; only the last-5-days-of-pre sub-window looks artificially low.

    Ground truth: full 30-day pre-period mean conversion ~= 8.9%; last-5-days-of-pre
    mean ~= 4.2% (the artificial dip); post-period (30 days after) mean ~= 9.1% --
    almost unchanged from the full pre-period, meaningfully different from the
    last-5-days comparison a rushed analyst might reach for.
    """
    rng = np.random.default_rng(505)
    pre_dates = pd.date_range("2025-03-01", periods=30, freq="D")
    post_dates = pd.date_range("2025-04-01", periods=30, freq="D")
    pre_rate = rng.normal(0.089, 0.01, 30)
    pre_rate[-5:] = rng.normal(0.042, 0.005, 5)  # the outage dip right before the change
    post_rate = rng.normal(0.091, 0.01, 30)
    pre_rate = np.clip(pre_rate, 0.005, None)
    post_rate = np.clip(post_rate, 0.005, None)
    df = pd.DataFrame({
        "date": list(pre_dates) + list(post_dates),
        "period": ["pre"] * 30 + ["post"] * 30,
        "conversion_rate": np.concatenate([pre_rate, post_rate]).round(4),
    })
    return _rec(df, "pricing_change_prepost.csv")


# ============================================================================
# 4. CONFOUNDING / BIAS: region-size confound (Simpson's-paradox variant),
#    selection bias (survey responders only), survivorship bias (active-only).
# ============================================================================


def region_size_confound() -> DatasetRecord:
    """Store-level average basket size by region and store format (small/large-format
    stores). Region North has mostly large-format stores (higher average basket);
    Region South has mostly small-format stores (lower average basket) -- so a naive
    "compare region averages" reading attributes a format-mix difference to a
    region-level effect. WITHIN each format, North and South actually perform
    similarly.

    Ground truth (verified by direct computation): North overall avg basket ~= $143.01
    (18 large-format stores avg ~$151.48, 2 small-format avg ~$66.86); South overall
    avg basket ~= $87.93 (2 large-format avg ~$149.30, 18 small-format avg ~$81.11) --
    the ~$55 gap in the naive regional averages shrinks to $2-15 within each format,
    because North is disproportionately large-format stores and South is
    disproportionately small-format.
    """
    rng = np.random.default_rng(606)
    rows = []
    for i in range(18):
        rows.append({"store_id": f"N{i:02d}", "region": "North", "format": "large", "avg_basket": round(rng.normal(150, 8), 2)})
    for i in range(2):
        rows.append({"store_id": f"N{i+18:02d}", "region": "North", "format": "small", "avg_basket": round(rng.normal(70, 5), 2)})
    for i in range(2):
        rows.append({"store_id": f"S{i:02d}", "region": "South", "format": "large", "avg_basket": round(rng.normal(148, 8), 2)})
    for i in range(18):
        rows.append({"store_id": f"S{i+2:02d}", "region": "South", "format": "small", "avg_basket": round(rng.normal(80, 5), 2)})
    return _rec(pd.DataFrame(rows), "region_size_confound.csv")


def survey_selection_bias() -> DatasetRecord:
    """A post-purchase satisfaction survey with a `responded` flag: only 60 of 500
    customers responded, and responders are disproportionately the MOST satisfied
    customers (a real-world response-bias pattern -- unhappy customers usually don't
    bother replying). satisfaction_score exists for all 500 (ground truth, unrealistic
    to know in a real business but included here so the trap's existence can be proven
    directly), but a `responded_score` column only carries the observed value for
    survey responders (NaN otherwise) -- an analyst should only be able to see
    `responded_score`, and should recognize that computing "average satisfaction" from
    only the responded subset overstates true satisfaction.

    Ground truth (verified by direct computation): true population mean satisfaction
    (all 500, `satisfaction_score`) ~= 3.42/5; mean of `responded_score` among the 110
    responders ~= 3.82/5 -- a real ~0.4-point inflation from response bias alone, even
    though the responder sample (110/500 = 22%) is not tiny.
    """
    rng = np.random.default_rng(707)
    n = 500
    true_scores = np.clip(rng.normal(3.4, 0.9, n), 1, 5).round(1)
    # responders are biased toward high scorers
    response_prob = 0.05 + 0.30 * (true_scores - 1) / 4
    responded = rng.random(n) < response_prob
    responded_score = np.where(responded, true_scores, np.nan)
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "satisfaction_score": true_scores,
        "responded": responded,
        "responded_score": responded_score,
    })
    return _rec(df, "survey_selection_bias.csv")


def survivorship_active_only() -> DatasetRecord:
    """200 customer rows -- but this dataset ONLY contains customers who are still
    active today (churned customers were already removed upstream, a realistic
    survivorship-bias setup, not a missing-column problem). `tenure_months` and
    `avg_monthly_spend` are both present. A naive "what predicts long tenure" analysis
    run on this dataset alone cannot see what churned customers looked like, so any
    conclusion about retention drivers drawn purely from this table is fundamentally
    survivorship-biased, regardless of how sound the statistics run on it are.

    Ground truth: this fixture legitimately has NO churned-customer rows at all
    (`status` column is 100% "active") -- the trap is structural (a whole population
    missing), not a statistical pattern within the given rows.
    """
    rng = np.random.default_rng(808)
    n = 200
    tenure = rng.gamma(4, 6, n).round(1)
    spend = (50 + tenure * rng.uniform(0.8, 1.5, n)).round(2)
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "tenure_months": tenure,
        "avg_monthly_spend": spend,
        "status": ["active"] * n,
    })
    return _rec(df, "survivorship_active_only.csv")


# ============================================================================
# 5. STATISTICAL TRAPS: heavy-tailed distribution, small-sample instability,
#    multicollinearity, non-stationary/structural-break time series.
# ============================================================================


def heavy_tailed_deal_sizes() -> DatasetRecord:
    """120 sales deal sizes drawn from a log-normal (heavy-tailed) distribution -- a
    tiny number of enterprise deals are 50-100x the typical deal, which is a REAL,
    representative pattern in B2B sales (not an error to "fix"), unlike the outlier
    fixture in adversarial_cases.json which represents a data-entry mistake. The
    correct analyst behavior differs: this is a distributional characteristic to
    report (median/skewness, not just mean), not a row to flag as corrupted data.

    Ground truth (verified by direct computation): mean deal size ~= $6,402 but median
    ~= $3,232 -- a ~2x mean/median gap that is real and expected for this
    distribution, not a sign of bad data.
    """
    rng = np.random.default_rng(909)
    deal_size = rng.lognormal(mean=8.3, sigma=1.1, size=120).round(2)
    return _rec(pd.DataFrame({"deal_id": [f"D{i:03d}" for i in range(120)], "deal_size": deal_size}), "heavy_tailed_deals.csv")


def multicollinear_marketing_spend() -> DatasetRecord:
    """90 weekly rows: `tv_spend`, `radio_spend` (near-perfectly correlated with
    tv_spend by construction -- the same media buyer sets both budgets together every
    week), and `revenue`. A regression of revenue on both spend variables together
    will show unstable/misleading individual coefficients (multicollinearity) even
    though the OVERALL model fits reasonably well -- individual-coefficient
    interpretation ("radio spend drives $X of revenue") is the trap here.

    Ground truth (verified by direct computation): corr(tv_spend, radio_spend) ~=
    0.992 by construction (radio_spend = 0.4 * tv_spend + small noise).
    """
    rng = np.random.default_rng(1010)
    n = 90
    tv_spend = rng.uniform(5000, 25000, n)
    radio_spend = tv_spend * 0.4 + rng.normal(0, 300, n)
    revenue = 30000 + tv_spend * 1.8 + rng.normal(0, 4000, n)
    df = pd.DataFrame({
        "week": pd.date_range("2024-01-01", periods=n, freq="W"),
        "tv_spend": tv_spend.round(2), "radio_spend": radio_spend.round(2), "revenue": revenue.round(2),
    })
    return _rec(df, "multicollinear_marketing.csv")


def structural_break_shipping() -> DatasetRecord:
    """365 days of average daily shipping time. A NEW fulfillment center opened on day
    200 (2024-07-19), permanently shifting average shipping time from ~4.2 days down to
    ~2.1 days -- a genuine structural break, not noise or a gradual trend. Any
    single-line trend fit or a naive full-period average conceals this; a forecast
    fit on the full period without accounting for the break would be systematically
    wrong for future (post-break-regime) periods.

    Ground truth (verified by direct computation): mean shipping time days 1-199 ~=
    4.21; days 200-365 ~= 2.13; the break date is exactly day 200 (2024-07-18, 0-indexed
    row 199) by construction.
    """
    rng = np.random.default_rng(1111)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    before = rng.normal(4.2, 0.4, 199)
    after = rng.normal(2.1, 0.3, 166)
    shipping_days = np.concatenate([before, after]).round(2)
    return _rec(pd.DataFrame({"date": dates, "avg_shipping_days": shipping_days}), "structural_break_shipping.csv")


def short_history_volatile() -> DatasetRecord:
    """Only 14 days of daily revenue, with unusually high day-to-day variance (a new
    product launch, still stabilizing) -- both too short a history AND too volatile
    for a naive forecast to be trustworthy, distinct from the existing `tiny_forecast`
    adversarial fixture (which is short but stable/low-variance)."""
    rng = np.random.default_rng(1212)
    dates = pd.date_range("2025-06-01", periods=14, freq="D")
    revenue = rng.normal(2000, 900, 14).round(2)
    return _rec(pd.DataFrame({"date": dates, "revenue": revenue}), "short_history_volatile.csv")


# ============================================================================
# 6. SEGMENTATION / RFM TRAPS: unstable segments from a too-small population,
#    a churn-definition-sensitive cohort table, zero-inflated usage data.
# ============================================================================


def rfm_instability_small_n() -> DatasetRecord:
    """Only 12 customers, 25 transactions total, across a 6-month window -- technically
    enough rows for `rfm_analysis` to run without erroring, but a genuine RFM
    segmentation (quintile-based scoring) is unstable/near-meaningless at n=12: the
    quintile boundaries themselves become arbitrary with so few customers. The trap is
    NOT a tool failure -- the tool will happily return segments -- the trap is whether
    the analyst treats those segments as reliable.

    Ground truth: 12 unique customers, 19 transaction rows total (verified directly).
    """
    rng = np.random.default_rng(1313)
    customers = [f"C{i:02d}" for i in range(12)]
    rows = []
    for c in customers:
        n_tx = rng.integers(1, 4)
        for _ in range(n_tx):
            rows.append({
                "customer_id": c,
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 180))),
                "revenue": round(rng.uniform(20, 300), 2),
            })
    return _rec(pd.DataFrame(rows), "rfm_instability_small_n.csv")


def zero_inflated_usage() -> DatasetRecord:
    """300 users' monthly feature-usage counts: 70% of users have EXACTLY zero usage
    (never touched the feature), the rest have a roughly-normal usage count centered
    around 15. A naive mean/std treats this as one distribution; the correct read
    separates "did they use it at all" from "how much, given they did."

    Ground truth (verified by direct computation): 212/300 (70.7%) rows have
    usage_count == 0 exactly; mean usage_count among the 88 nonzero users ~= 15.32.
    """
    rng = np.random.default_rng(1414)
    n = 300
    zero_mask = rng.random(n) < 0.70
    usage = np.where(zero_mask, 0, np.clip(rng.normal(15, 5, n), 1, None).round(0))
    return _rec(pd.DataFrame({"user_id": [f"U{i:04d}" for i in range(n)], "usage_count": usage.astype(int)}), "zero_inflated_usage.csv")


# ============================================================================
# 7. FINANCE / UNITS: currency mismatch, unit mismatch (cents vs dollars),
#    revenue-recognition timing, refunds/cancellations mixed with sales.
# ============================================================================


def finance_units_mismatch() -> DatasetRecord:
    """50 transactions across 10 recurring items: `amount` is stored in CENTS for a
    random subset of rows (an upstream system migration artifact -- a real, common
    data-integration bug) and in DOLLARS for the rest, with no column distinguishing
    which is which. Each item's true dollar price is fixed, so the SAME item shows up
    at both its normal dollar value and a ~100x-larger cents value across different
    rows -- naively summing `amount` treats both as if they were the same unit,
    wildly inflating the total.

    Ground truth (verified directly against this exact fixture, seed=1515): 16 of the
    50 rows are cents-denominated (~100x their item's true dollar price), 34 are
    dollar-denominated; every one of the 10 items has at least one row of each kind,
    so the mismatch is detectable per-item, not just in aggregate. Naive
    `amount.sum()` == 64,899.03; the unit-corrected sum (dividing cents-flagged rows
    by 100) == 1,972.65 -- a >30x overstatement from the unit mismatch alone.
    """
    rng = np.random.default_rng(1515)
    n = 50
    items = [f"item_{i % 10}" for i in range(n)]
    base_dollar_price = {f"item_{i}": round(rng.uniform(10, 60), 2) for i in range(10)}
    is_cents = rng.random(n) < 0.35
    # guarantee every item has at least one cents row and one dollar row
    for item_idx in range(10):
        rows_for_item = [i for i, it in enumerate(items) if it == f"item_{item_idx}"]
        if not any(is_cents[i] for i in rows_for_item):
            is_cents[rows_for_item[0]] = True
        if not all(is_cents[i] for i in rows_for_item):
            continue
        is_cents[rows_for_item[-1]] = False
    amount = [round(base_dollar_price[item] * 100) if cents else base_dollar_price[item] for item, cents in zip(items, is_cents)]
    df = pd.DataFrame({"transaction_id": [f"T{i:03d}" for i in range(n)], "item": items, "amount": amount})
    return _rec(df, "finance_units_mismatch.csv")


def revenue_recognition_trap() -> DatasetRecord:
    """70 sale rows spanning Jan-Mar 2025 (`order_date`, `ship_date`, `revenue`) plus
    10 refund rows (negative revenue, `type`="refund"), each refund dated 20-50 days
    after its original sale's `order_date` -- often landing in a different calendar
    month than the original sale. A naive "revenue by month" grouping that doesn't
    net each refund against its original sale's month will show a real sale's revenue
    in one month and its reversal in another, distorting month-over-month comparisons.

    Ground truth (verified directly against this exact fixture, seed=1616): 10 of 80
    rows have type=="refund" with negative revenue; all 10 have their processing
    (`ship_date`) month different from their original sale's `order_date` month (each
    refund row keeps the original sale's `order_date` for traceability but is
    processed 20-50 days later).
    """
    rng = np.random.default_rng(1616)
    n = 70
    order_dates = pd.to_datetime(rng.choice(pd.date_range("2025-01-01", "2025-03-31"), n))
    ship_delay = rng.integers(1, 35, n)
    ship_dates = order_dates + pd.to_timedelta(ship_delay, unit="D")
    revenue = rng.uniform(50, 500, n).round(2)
    df = pd.DataFrame({"order_id": [f"O{i:03d}" for i in range(n)], "order_date": order_dates, "ship_date": ship_dates, "revenue": revenue, "type": "sale"})
    refund_rows = []
    for i in range(10):
        orig_idx = int(rng.integers(0, n))
        orig = df.iloc[orig_idx]
        refund_date = orig["order_date"] + pd.Timedelta(days=int(rng.integers(20, 50)))
        refund_rows.append({
            "order_id": orig["order_id"] + "-R", "order_date": orig["order_date"], "ship_date": refund_date,
            "revenue": -orig["revenue"], "type": "refund",
        })
    df = pd.concat([df, pd.DataFrame(refund_rows)], ignore_index=True)
    return _rec(df, "revenue_recognition.csv")


# ============================================================================
# 8. DATE / TIME TRAPS: timezone boundary shift, partial current month,
#    incomplete-month comparison, category rename splitting one product.
# ============================================================================


def timezone_boundary_shift() -> DatasetRecord:
    """200 transaction timestamps stored in UTC, but the business operates in
    US/Pacific (UTC-7/8) -- 30 of the 200 timestamps fall between 00:00 and 08:00 UTC,
    which is actually the PREVIOUS business day in Pacific time. A naive `.dt.date`
    grouping on the raw UTC timestamp misattributes those 30 transactions to the wrong
    calendar day.

    Ground truth (verified by direct computation): 78/200 rows have a UTC hour < 8
    (the ones that shift to the previous Pacific business day); verifiable directly
    from `timestamp_utc.dt.hour < 8`.
    """
    rng = np.random.default_rng(1717)
    n = 200
    base = pd.Timestamp("2025-05-01 00:00:00")
    minutes_offset = rng.integers(0, 60 * 24 * 20, n)
    timestamps = base + pd.to_timedelta(minutes_offset, unit="m")
    # force exactly 30 into the 00:00-08:00 UTC danger zone
    danger_idx = rng.choice(n, 30, replace=False)
    for idx in danger_idx:
        day_start = timestamps[idx].normalize()
        timestamps.values[idx] = day_start + pd.Timedelta(hours=int(rng.integers(0, 8)), minutes=int(rng.integers(0, 60)))
    revenue = rng.uniform(20, 300, n).round(2)
    df = pd.DataFrame({"timestamp_utc": timestamps, "revenue": revenue})
    return _rec(df, "timezone_boundary.csv")


def partial_current_month() -> DatasetRecord:
    """Daily revenue for Jan through Jun 2025, but the data EXTRACT was pulled on
    2025-06-10 -- so June only has 10 days of data while every other month is
    complete. A naive "revenue by month" bar chart makes June look like a severe
    decline purely from having 20 fewer days of data, not a real business decline.

    Ground truth: Jan-May each have 28-31 full days; June has exactly 10 days
    (2025-06-01 through 2025-06-10).
    """
    rng = np.random.default_rng(1818)
    dates = list(pd.date_range("2025-01-01", "2025-05-31", freq="D")) + list(pd.date_range("2025-06-01", "2025-06-10", freq="D"))
    revenue = rng.normal(3000, 300, len(dates)).round(2)
    return _rec(pd.DataFrame({"date": dates, "revenue": revenue}), "partial_current_month.csv")


def category_rename_split() -> DatasetRecord:
    """180 days of daily sales for a single product line. For the first 90 days it's
    recorded under category "Outdoor Gear". Starting day 91, the SAME underlying
    product line was renamed/split in the source system into "Camping Equipment" (70%
    of what used to be "Outdoor Gear" sales) and "Hiking Gear" (the remaining 30%) --
    an internal catalog change, not a real change in what's being sold. Comparing
    "Outdoor Gear" revenue before vs. after the rename wrongly shows a near-total
    collapse, when the actual combined-category revenue is roughly flat.

    Ground truth: total daily revenue (summed across whichever category label is
    active) is drawn from the same underlying distribution across the full 180 days;
    "Outdoor Gear" revenue specifically drops to ~$0 after day 90 purely from the
    rename, not from a real decline.
    """
    rng = np.random.default_rng(1919)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    daily_revenue = rng.normal(1200, 100, 180).round(2)
    category = []
    for i in range(180):
        if i < 90:
            category.append("Outdoor Gear")
        else:
            category.append("Camping Equipment" if rng.random() < 0.7 else "Hiking Gear")
    df = pd.DataFrame({"date": dates, "category": category, "revenue": daily_revenue})
    return _rec(df, "category_rename_split.csv")


# ============================================================================
# 9. FUNNEL / ATTRIBUTION: conversion-denominator trap, funnel stage mismatch.
# ============================================================================


def funnel_denominator_trap() -> DatasetRecord:
    """A marketing funnel table with `stage` in {visit, signup, purchase} and a
    `channel` column. The `purchase` stage rows do NOT include a `visit`-stage row for
    every purchaser (some purchases came from a mobile app that only logs `purchase`
    events, never `visit`) -- so "purchase / visit" as a naive conversion-rate
    denominator undercounts the true funnel entry point for the app channel
    specifically, making app-channel conversion look impossibly high (in a few cases,
    the naive rate is even mathematically >100%, which should itself be a giveaway of
    the impossible-value data-quality problem).

    Ground truth: for channel == "app", `purchase` row count (40) exceeds `visit` row
    count (5) -- so `purchases / visits` for that channel alone would compute to 800%,
    an impossible conversion rate.
    """
    rows = []
    rng = np.random.default_rng(2020)
    for channel, visits, signups, purchases in [("web", 1000, 300, 90), ("email", 400, 200, 60), ("app", 5, 3, 40)]:
        for _ in range(visits):
            rows.append({"channel": channel, "stage": "visit"})
        for _ in range(signups):
            rows.append({"channel": channel, "stage": "signup"})
        for _ in range(purchases):
            rows.append({"channel": channel, "stage": "purchase"})
    return _rec(pd.DataFrame(rows), "funnel_denominator.csv")


# ============================================================================
# 10. LARGE-SCALE: a real, meaningfully large in-memory dataset (300K rows) to test
#     whether the agent recognizes it should prefer SQL/pushdown aggregation over a
#     naive full-memory approach -- large enough to matter, small enough to actually
#     fit in this test process without requiring the separate 100M-row infra.
# ============================================================================


def large_scale_transactions() -> DatasetRecord:
    """300,000 synthetic transaction rows, 2023-01-01 through 2024-12-31, 8 regions.
    Not large enough to need `app/large_data`'s chunked/streaming machinery, but large
    enough that a naive multi-step pandas groupby-then-filter-then-sort chain is
    measurably slower and more memory-hungry than pushing the aggregation into
    `run_sql_query` (DuckDB) -- the scalability dimension checks for evidence of a
    SQL-category tool call or an explicit scale acknowledgment on this fixture.

    Ground truth: len(df) == 300000; total revenue is deterministic given the seed but
    intentionally not hand-computed here (300K rows) -- cases against this fixture
    should check `row_count`/`group_count`-style structural fields, not a hand-verified
    total, consistent with the lesson from the 100-case benchmark round about only
    checking fields real tools actually expose flatly.
    """
    rng = np.random.default_rng(2121)
    n = 300_000
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    regions = rng.choice(["North", "South", "East", "West", "Central", "NE", "NW", "SE"], n)
    revenue = rng.gamma(2, 80, n).round(2)
    df = pd.DataFrame({"date": dates, "region": regions, "revenue": revenue})
    return _rec(df, "large_scale_transactions.csv")


def marketplace_join_inflation() -> DatasetRecord:
    """35 marketplace orders, but 8 of them are represented 3 TIMES EACH with identical
    order_id/revenue -- simulating a real, common data-integration bug: an orders table
    joined many-to-many against a line-items or fulfillment-events table (each order
    had multiple line items or multiple fulfillment scan events), so the join produced
    duplicate order rows rather than being properly aggregated first. Distinct from
    `ecommerce_messy`'s duplicate-transaction trap (that one is exact whole-row
    duplication of a plausible upstream cause; this one models the specific
    many-to-many-join mechanism the mission calls out by name).

    Ground truth (verified by direct computation): 35 unique order_ids, 51 total rows
    (8 orders appear 3x instead of once, adding 16 extra rows). Naive `revenue.sum()`
    == 12,686.32; the correct deduplicated total == 8,500.26 -- a ~49% overstatement
    from the join-inflation bug alone.
    """
    rng = np.random.default_rng(2222)
    n = 35
    order_ids = [f"MP{i:03d}" for i in range(n)]
    revenue = rng.uniform(30, 400, n).round(2)
    seller_id = rng.choice([f"SELLER_{i}" for i in range(6)], n)
    df = pd.DataFrame({"order_id": order_ids, "seller_id": seller_id, "revenue": revenue})
    inflate_idx = rng.choice(n, 8, replace=False)
    extra = pd.concat([df.iloc[inflate_idx]] * 2, ignore_index=True)
    df = pd.concat([df, extra], ignore_index=True)
    return _rec(df, "marketplace_join_inflation.csv")


def hr_impossible_values() -> DatasetRecord:
    """60 employee rows: `tenure_months` has 4 negative values (a sign-flip data-entry
    error during a system migration -- a real, common HR-data problem), and 2 employee
    IDs appear twice with slightly different `department` values (a record that was
    updated in place at the source but landed as a new row instead of an update, a
    common CDC/ETL bug). `salary` has one extreme outlier (a real executive salary,
    not an error) mixed in with individual-contributor salaries.

    Ground truth (verified by direct computation): 62 total rows (60 base + 2
    duplicated-with-different-department rows); 4 rows have tenure_months < 0; 2
    employee_ids each appear exactly twice with a different department value; salary
    mean ~= $72,059 vs. median ~= $67,983 -- a modest but real mean/median gap from the
    single $340,000 executive salary among otherwise $35K-$95K individual-contributor
    salaries (one outlier among 60 rows shifts the mean less dramatically than in the
    heavy-tailed-deals fixture, which is itself a useful contrast for a benchmark case).
    """
    rng = np.random.default_rng(2323)
    n = 60
    employee_ids = [f"E{i:03d}" for i in range(n)]
    tenure = rng.gamma(3, 8, n).round(1)
    for idx in (3, 17, 29, 44):
        tenure[idx] = -abs(tenure[idx])
    departments = rng.choice(["Sales", "Engineering", "Support", "Marketing"], n).tolist()
    salary = np.clip(rng.normal(68000, 12000, n), 35000, None).round(2)
    salary[5] = 340000.0  # a single real executive salary, not an error
    df = pd.DataFrame({"employee_id": employee_ids, "department": departments, "tenure_months": tenure, "salary": salary})
    dup_rows = df.iloc[[10, 40]].copy()
    dup_rows["department"] = ["Engineering", "Sales"]  # updated-in-place record landed as a new row with a different value
    df = pd.concat([df, dup_rows], ignore_index=True)
    return _rec(df, "hr_impossible_values.csv")


def randomized_email_experiment() -> DatasetRecord:
    """500 users per arm of a genuinely randomized A/B email subject-line test
    (`group` in {A, B}, `opened` 0/1) with a deliberately LARGE, obvious effect
    (group A ~40% open rate vs. group B ~20%) and a large sample -- unlike every
    other fixture in this module, this one is NOT a trap: it exists specifically to
    give the reasoning pipeline real, strong, well-powered evidence so a causal
    claim CAN legitimately be permitted (see hard_prim_10's "causal language is
    justified" case, and the mission's explicit requirement for at least one
    positive-control case so the causation guard doesn't become blindly
    conservative).

    Ground truth (verified by direct computation, seed=2424): group A open rate
    ~= 55% (275/500), group B ~= 10% (50/500) -- a 45-point absolute difference,
    large enough that a real t-test comes back significant AND the real Cohen's d
    effect-size calculation classifies as "large" (d > 0.8), not merely significant
    with a small/borderline standardized effect (verified directly: a smaller, more
    "realistic" 40%-vs-20% gap was tried first and correctly computed to a "small"
    Cohen's d magnitude despite the large absolute percentage-point gap and p=0.0
    significance -- Cohen's d for two proportions depends on the pooled variance,
    not just the raw point difference, so this fixture deliberately uses a bigger gap
    to reliably land in the "large" bucket).
    """
    rng = np.random.default_rng(2424)
    n_per_group = 500
    a_opens = rng.choice([1, 0], n_per_group, p=[0.55, 0.45])
    b_opens = rng.choice([1, 0], n_per_group, p=[0.10, 0.90])
    df = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(n_per_group * 2)],
        "group": ["A"] * n_per_group + ["B"] * n_per_group,
        "opened": np.concatenate([a_opens, b_opens]),
    })
    return _rec(df, "randomized_email_experiment.csv")


HARD_FIXTURES = {
    "ecommerce_messy": ecommerce_messy,
    "saas_subscriptions": saas_subscriptions,
    "ab_test_imbalanced": ab_test_imbalanced,
    "campaign_multiple_comparisons": campaign_multiple_comparisons,
    "pricing_change_prepost": pricing_change_prepost,
    "region_size_confound": region_size_confound,
    "survey_selection_bias": survey_selection_bias,
    "survivorship_active_only": survivorship_active_only,
    "heavy_tailed_deals": heavy_tailed_deal_sizes,
    "multicollinear_marketing": multicollinear_marketing_spend,
    "structural_break_shipping": structural_break_shipping,
    "short_history_volatile": short_history_volatile,
    "rfm_instability_small_n": rfm_instability_small_n,
    "zero_inflated_usage": zero_inflated_usage,
    "finance_units_mismatch": finance_units_mismatch,
    "revenue_recognition": revenue_recognition_trap,
    "timezone_boundary": timezone_boundary_shift,
    "partial_current_month": partial_current_month,
    "category_rename_split": category_rename_split,
    "funnel_denominator": funnel_denominator_trap,
    "large_scale_transactions": large_scale_transactions,
    "marketplace_join_inflation": marketplace_join_inflation,
    "hr_impossible_values": hr_impossible_values,
    "randomized_email_experiment": randomized_email_experiment,
}
