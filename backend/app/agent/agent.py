from __future__ import annotations

import json
import logging

from app.agent.providers import LLMProvider
from app.agent.tool_router import ToolRouter
from app.datasets.storage import DatasetRecord
from app.schemas import ToolCallRecord
from app.tools.errors import ToolExecutionError
from app.tools.profiler import profile_dataset

logger = logging.getLogger(__name__)

# CRITICAL DATA ANALYSIS RULE (see project instructions): the LLM must never invent
# numbers. It only sees dataset *metadata* (shape, column names/types) here — every
# concrete figure it uses has to come from a tool-call result appended to the
# conversation below. tests/test_agent_tools.py asserts this boundary holds.
SYSTEM_PROMPT = (
    "You are an AI data analyst. You help users understand a dataset that has already "
    "been uploaded. You NEVER invent numbers. Every numeric claim you make must come "
    "from a tool call result provided to you in this conversation — you only interpret "
    "and explain those results in clear, business-friendly language. If you need a "
    "number, call a tool for it instead of guessing. When you have enough information, "
    "give a concise, direct answer and mention concrete numbers from tool results to "
    "support it. If the data cannot answer the question, say so plainly."
)

MAX_TOOL_ITERATIONS = 6


class DataAnalystAgent:
    def __init__(self, provider: LLMProvider, tool_router: ToolRouter | None = None) -> None:
        self._provider = provider
        self._router = tool_router or ToolRouter()

    def ask(self, record: DatasetRecord, question: str, history: list[dict[str, str]] | None = None) -> dict:
        profile = profile_dataset(record.df)
        dataset_context = (
            f"Dataset '{record.original_filename}': {profile['rows']} rows, {profile['columns']} columns. "
            f"Numeric columns: {profile['numeric_columns']}. "
            f"Categorical columns: {profile['categorical_columns']}. "
            f"Date columns: {profile['date_columns']}. "
            "This is metadata only — you do not have raw row values. Call tools to get real numbers."
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": dataset_context},
        ]
        for turn in history or []:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        tool_call_records: list[ToolCallRecord] = []
        charts: list[dict] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._provider.complete(messages, self._router.available_tools())

            if not response.tool_calls:
                return {
                    "answer": response.content or "I couldn't generate an answer.",
                    "tool_calls": tool_call_records,
                    "charts": charts,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                try:
                    result = self._router.execute(call.name, record, call.arguments)
                    tool_call_records.append(ToolCallRecord(tool=call.name, params=call.arguments, result=result))
                    if call.name == "generate_chart":
                        charts.append(result)
                    # default=str is defense-in-depth: tools are expected to already
                    # return JSON-native values (see app/tools/serialization.py), but a
                    # future tool returning e.g. a stray Timestamp must never crash the
                    # whole chat request — degrade to its string form instead.
                    payload = json.dumps(result, default=str)
                except ToolExecutionError as exc:
                    payload = json.dumps({"error": str(exc)})
                    logger.warning("tool_error tool=%s error=%s", call.name, exc)

                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": payload})

        return {
            "answer": (
                "I gathered several results but reached the tool-call limit before finishing. "
                "Try asking a more specific follow-up question."
            ),
            "tool_calls": tool_call_records,
            "charts": charts,
        }
