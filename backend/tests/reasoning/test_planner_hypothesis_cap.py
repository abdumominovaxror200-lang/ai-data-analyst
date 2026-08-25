"""Phase 4 P2: `planner.py`'s hypothesis cap is now 3 (was 4).

Follows the same scripted `MockProvider` pattern as `test_orchestrator.py` /
`conftest.py` -- no network call, no dependency on Groq.
"""

from __future__ import annotations

from app.agent.providers import MockProvider
from app.reasoning import planner
from app.reasoning.contracts import AnalyticalQuestion
from tests.reasoning.conftest import json_response, plan_payload


def test_planner_truncates_more_than_3_hypotheses_to_3():
    assert planner._MAX_HYPOTHESES == 3

    five_hypotheses = [
        {"description": "Seasonality drove the decline", "is_causal": False},
        {"description": "A pricing change drove the decline", "is_causal": False},
        {"description": "Demand shifted away from this product", "is_causal": False},
        {"description": "A tracking/reporting change caused an apparent decline", "is_causal": False},
        {"description": "A key customer churned", "is_causal": False},
    ]
    script = [
        json_response(
            plan_payload(capability_categories=["EDA"], hypotheses=five_hypotheses)
        )
    ]
    provider = MockProvider(script)
    question = AnalyticalQuestion(original_question="Why did revenue decline?", intent="diagnostic")

    plan = planner.plan_analysis(provider, question, [], [])

    assert len(plan.hypotheses) == 3
    # the first 3 in the raw script order survive, truncation is deterministic
    assert [h.description for h in plan.hypotheses] == [
        "Seasonality drove the decline",
        "A pricing change drove the decline",
        "Demand shifted away from this product",
    ]


def test_planner_allows_fewer_than_3_hypotheses_unchanged():
    two_hypotheses = [
        {"description": "Seasonality drove the decline", "is_causal": False},
        {"description": "A pricing change drove the decline", "is_causal": False},
    ]
    script = [
        json_response(
            plan_payload(capability_categories=["EDA"], hypotheses=two_hypotheses)
        )
    ]
    provider = MockProvider(script)
    question = AnalyticalQuestion(original_question="Why did revenue decline?", intent="diagnostic")

    plan = planner.plan_analysis(provider, question, [], [])

    assert len(plan.hypotheses) == 2
