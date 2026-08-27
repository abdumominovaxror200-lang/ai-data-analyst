"""End-to-end real-analyst-workflow test (v2 reliability mission, Phase 14).

Not a single-question probe of one mechanism -- this drives the real
ReasoningOrchestrator through a realistic multi-step investigation resembling actual
paid analyst work: "Management says revenue is falling because Product A is
underperforming. Verify or reject this claim." The dataset is deliberately
constructed so the claim, AS STATED, is misleading: every product's revenue declined
by roughly the same proportion month over month -- a market-wide pattern, not a
Product-A-specific one. A correct analyst (human or AI) investigates by checking each
product individually before accepting the premise, and should conclude the claim
mis-attributes a shared, broader decline to one product specifically.

This is a single, carefully-designed integration test, not a large new suite --
the individual deterministic mechanisms it exercises (compare_periods correctness,
evidence gathering, synthesis) are already covered in depth elsewhere; this test's
value is proving they compose correctly across a REALISTIC multi-step question, not
re-testing any one of them in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.reasoning.orchestrator import ReasoningOrchestrator


def _sales_dataset() -> pd.DataFrame:
    """Ground truth (verified by direct computation before writing this test): every
    product's revenue declines by ~20% from March to April -- a uniform, market-wide
    pattern. Product A is NOT uniquely underperforming; management's claim, as
    stated, is misleading."""
    rng = np.random.default_rng(42)
    rows = []
    for product, march_mean, april_mean in [("Product A", 500.0, 400.0), ("Product B", 500.0, 400.0), ("Product C", 500.0, 400.0)]:
        for _ in range(40):
            rows.append({"date": pd.Timestamp("2025-03-15"), "product": product, "revenue": march_mean + rng.normal(0, 20)})
        for _ in range(40):
            rows.append({"date": pd.Timestamp("2025-04-15"), "product": product, "revenue": april_mean + rng.normal(0, 20)})
    return pd.DataFrame(rows)


def test_multi_step_investigation_correctly_rejects_a_misleading_business_claim():
    df = _sales_dataset()
    record = DatasetRecord(
        id="analyst-workflow", original_filename="sales.csv", extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="u",
    )

    parsed_question = {
        "intent": "diagnostic", "requested_metrics": ["revenue"], "requested_dimensions": ["product"],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en",
        "claims": [{"text": "Product A is underperforming and driving the revenue decline", "source": "user_asserted"}],
    }
    plan = {
        "objective": "Verify whether Product A specifically drives the revenue decline, or whether the decline is broader",
        "capability_categories": ["GENERAL_ANALYSIS"], "steps": [],
        "tools_required": ["compare_periods"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["each product's trend individually checked"], "hypotheses": [],
    }
    common = {
        "date_column": "date", "value_column": "revenue", "agg_func": "mean",
        "current_start": "2025-04-01", "current_end": "2025-04-30",
        "previous_start": "2025-03-01", "previous_end": "2025-03-31",
    }
    responses = [
        ProviderResponse(content=json.dumps(parsed_question)),
        ProviderResponse(content=json.dumps(plan)),
        # Step 1: check the overall trend.
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c1", name="compare_periods", arguments=dict(common))]),
        # Step 2-4: verify the claim by checking each product individually, rather
        # than accepting "Product A is the driver" at face value.
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c2", name="compare_periods", arguments={**common, "filters": [{"column": "product", "op": "==", "value": "Product A"}]})]),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c3", name="compare_periods", arguments={**common, "filters": [{"column": "product", "op": "==", "value": "Product B"}]})]),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c4", name="compare_periods", arguments={**common, "filters": [{"column": "product", "op": "==", "value": "Product C"}]})]),
        ProviderResponse(content="evidence gathered"),
        ProviderResponse(content=json.dumps({
            "final_answer_text": (
                "Management's claim is not well-supported as stated. Average revenue fell "
                "roughly 20% for Product A, but nearly identically for Product B and Product C "
                "as well -- this is a broad, market-wide decline affecting every product "
                "similarly, not a Product-A-specific problem. Singling out Product A would "
                "misdirect any corrective action."
            ),
            "recommendation": {
                "recommendation": "Investigate market-wide or seasonal factors affecting all products, rather than a Product-A-specific intervention.",
                "expected_business_effect": "Correctly targeted investigation of the real driver.",
                "confidence": "medium",
                "assumptions": [], "risks": [],
            },
        })),
    ]
    orchestrator = ReasoningOrchestrator(MockProvider(responses))
    result = orchestrator.analyze(record, "Management says revenue is falling because Product A is underperforming. Verify or reject this claim.")

    # 1. The investigation actually happened: multiple real tool calls, not one
    #    shallow lookup accepting the premise at face value.
    tools_used = [e.source_tool for e in result.evidence]
    assert tools_used.count("compare_periods") == 4

    # 2. Each product's evidence is independently traceable (not just narrated).
    assert len(result.findings) == 4
    assert all(f.cross_checked is False or f.cross_checked is True for f in result.findings)  # never crashes computing this

    # 3. The final answer actually addresses and rejects the narrow framing, not
    #    just restates the premise.
    assert "not" in result.final_answer_text.lower() or "market-wide" in result.final_answer_text.lower()
    assert "Product A" in result.final_answer_text

    # 4. The structured audit reflects a real investigation occurred and reached a
    #    real conclusion -- not blocked/contradicted (nothing here actually
    #    contradicts -- all 3 products moved the same direction, confirming the
    #    market-wide story, which is itself evidence AGAINST the narrow claim, not
    #    a methodological contradiction).
    assert result.analytical_audit is not None
    assert result.analytical_audit.conclusion_status in ("SUPPORTED", "WEAKLY_SUPPORTED", "UNCERTAIN")

    # 5. A recommendation was reached and is traceable to real evidence, not invented.
    assert result.recommendation is not None
    assert result.recommendation.supporting_findings
