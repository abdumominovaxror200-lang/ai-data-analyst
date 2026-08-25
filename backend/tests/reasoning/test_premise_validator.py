from __future__ import annotations

from app.reasoning.contracts import AnalyticalQuestion
from app.reasoning.premise_validator import validate_question


def test_existing_metric_and_dimension_are_verified_true(sales_record):
    question = AnalyticalQuestion(
        original_question="q", intent="descriptive", requested_metrics=["revenue"], requested_dimensions=["region"]
    )
    claims, limitations, _profile = validate_question(question, sales_record.df)
    metric_claim = next(c for c in claims if "revenue" in c.text)
    dim_claim = next(c for c in claims if "region" in c.text)
    assert metric_claim.status == "verified_true"
    assert dim_claim.status == "verified_true"
    assert limitations == []


def test_nonexistent_metric_is_flagged_missing_data(sales_record):
    question = AnalyticalQuestion(original_question="q", intent="descriptive", requested_metrics=["conversion_rate"])
    claims, limitations, _profile = validate_question(question, sales_record.df)
    metric_claim = next(c for c in claims if "conversion_rate" in c.text)
    assert metric_claim.status == "verified_false"
    assert any(l.category == "missing_data" and l.severity == "blocks_conclusion" for l in limitations)


def test_nonexistent_dimension_is_flagged_missing_data(sales_record):
    question = AnalyticalQuestion(original_question="q", intent="descriptive", requested_dimensions=["customer_segment"])
    claims, limitations, _profile = validate_question(question, sales_record.df)
    assert any(l.category == "missing_data" for l in limitations)


def test_time_range_matching_actual_coverage_is_verified_true(sales_record):
    # sales_record spans ~8 months; asking for "last 3 months" fits comfortably.
    question = AnalyticalQuestion(original_question="q", intent="descriptive", requested_time_range="last 3 months")
    claims, limitations, _profile = validate_question(question, sales_record.df)
    range_claim = next(c for c in claims if "covers the requested" in c.text)
    assert range_claim.status == "verified_true"
    assert not any(l.category == "insufficient_coverage" for l in limitations)


def test_time_range_exceeding_actual_coverage_is_flagged_never_silently_substituted(sales_record):
    """The exact scenario from the Phase 3B spec: dataset has ~8 months, user asks for
    the last 12 -- must be flagged, never silently answered as if it were 12 months."""
    question = AnalyticalQuestion(original_question="q", intent="descriptive", requested_time_range="last 12 months")
    claims, limitations, _profile = validate_question(question, sales_record.df)
    range_claim = next(c for c in claims if "covers the requested" in c.text)
    assert range_claim.status == "verified_false"
    assert range_claim.note is not None and "8" in range_claim.note or "months" in range_claim.note
    coverage_limitations = [l for l in limitations if l.category == "insufficient_coverage"]
    assert len(coverage_limitations) == 1
    assert coverage_limitations[0].severity == "blocks_conclusion"
    assert "12" in coverage_limitations[0].text


def test_grossly_mismatched_scale_claim_is_flagged(sales_record):
    """Recreates this project's original benchmark finding: a claim describing a
    dataset two-to-three orders of magnitude larger than what's actually loaded."""
    question = AnalyticalQuestion(
        original_question="q", intent="descriptive", explicit_constraints=["a database of 10 million rows"]
    )
    claims, limitations, profile = validate_question(question, sales_record.df)
    scale_claim = next(c for c in claims if "10 million" in c.text)
    assert scale_claim.status == "verified_false"
    assert str(profile["rows"]) in scale_claim.note
    assert any(l.category == "insufficient_coverage" for l in limitations)


def test_scale_claim_close_to_actual_row_count_is_verified_true(sales_record):
    n = len(sales_record.df)
    question = AnalyticalQuestion(original_question="q", intent="descriptive", explicit_constraints=[f"about {n} rows"])
    claims, limitations, _profile = validate_question(question, sales_record.df)
    scale_claim = next(c for c in claims if "rows" in c.text)
    assert scale_claim.status == "verified_true"


def test_population_is_marked_unverifiable_not_silently_assumed_true(sales_record):
    question = AnalyticalQuestion(original_question="q", intent="descriptive", requested_population="VIP customers")
    claims, _limitations, _profile = validate_question(question, sales_record.df)
    pop_claim = next(c for c in claims if "VIP customers" in c.text)
    assert pop_claim.status == "unverifiable"
