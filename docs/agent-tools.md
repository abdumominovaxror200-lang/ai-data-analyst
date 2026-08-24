# Agent tools

Every tool is a pure function over a `pandas.DataFrame` (`backend/app/tools/*.py`), independently
unit-tested, and exposed to the LLM as an OpenAI-style function-calling schema in
`backend/app/agent/tool_router.py`. The agent can only obtain numbers by calling these — see
[`architecture.md`](architecture.md#llm-provider-abstraction) for how that boundary is enforced.

All tools except `profile_dataset` and `generate_report` accept an optional `filters` array —
`[{"column": ..., "op": "==|!=|>|>=|<|<=|in|contains|between", "value": ...}]` — applied before
the tool's own computation, so the LLM can combine "filter to region=North" with any other tool
in a single call.

| Tool | Purpose | Key parameters |
|---|---|---|
| `profile_dataset` | Shape, column types/roles, missing values, duplicate rows. | — |
| `describe_data` | Mean/median/std/min/max/sum for numeric columns, top values for categorical. | `columns?`, `filters?` |
| `filter_data` | Row count and preview for a set of filter conditions. | `filters` |
| `group_and_aggregate` | Group by a column, aggregate another (`sum/mean/median/count/min/max`). | `group_by`, `agg_column`, `agg_func`, `top_n?`, `filters?` |
| `compare_periods` | Aggregate a value over two date ranges and compute delta/%-change. | `date_column`, `value_column`, `current_start/end`, `previous_start/end`, `agg_func?`, `filters?` |
| `correlation_analysis` | Pearson/Spearman/Kendall correlation matrix + ranked strongest pairs. | `columns?`, `method?`, `filters?` |
| `detect_anomalies` | IQR or z-score outlier detection on one numeric column. | `column`, `method?`, `threshold?`, `filters?` |
| `generate_chart` | Chart-ready `{labels, series}` or `{points}` data for line/bar/histogram/scatter/pie. | `chart_type`, `x`, `y?`, `agg_func?`, `bins?`, `top_n?`, `filters?` |
| `generate_business_insights` | Bundles profiling + stats + anomalies + correlations + data-quality flags into one payload for narration. | `filters?` |
| `generate_report` | Full structured report (overview, statistics, anomalies, correlations, key findings). | — |

## Agent loop

`DataAnalystAgent.ask()` (`backend/app/agent/agent.py`):

1. Builds a system prompt stating the no-hallucination rule, plus a dataset-metadata message
   (shape, column names/types — not raw values).
2. Sends the conversation + `ToolRouter.available_tools()` to the configured `LLMProvider`.
3. If the model returns tool calls, each is executed via `ToolRouter.execute(name, record,
   params)`; results (or a clean `ToolExecutionError` message) are appended as `tool` messages.
4. Repeats until the model returns plain content or `MAX_TOOL_ITERATIONS` (6) is reached, at
   which point the agent returns a "reached the tool-call limit" message rather than looping
   forever.
5. Returns `{answer, tool_calls, charts}` — `tool_calls` is the full audit trail (tool name,
   params, result) shown in the UI as badges under each answer, and any `generate_chart` result
   is collected into `charts` for inline rendering.

## Adding a new tool

1. Write the pure function in `backend/app/tools/<name>.py`, raising `ToolExecutionError` for
   invalid input — it should take a `DataFrame` (plus a `filters` param if it makes sense) and
   return a JSON-serializable `dict`.
2. Add its OpenAI function-calling schema to `TOOL_SCHEMAS` and a lambda to `_HANDLERS` in
   `backend/app/agent/tool_router.py`.
3. Write unit tests in `backend/tests/test_<name>.py` covering the happy path and at least one
   validation error.
