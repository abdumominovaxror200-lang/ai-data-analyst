from __future__ import annotations

import pandas as pd

from app.tools.filtering import apply_filters


def interpretation_echo(df: pd.DataFrame, params: dict | None = None, metric_hint: str | None = None) -> str:
    params = params or {}
    metric = metric_hint or next((params.get(key) for key in ("metric_name", "value_column", "agg_column", "column", "target_column") if isinstance(params.get(key), str)), "unspecified")
    period_parts = []
    for prefix in ("previous", "current"):
        start, end = params.get(f"{prefix}_start"), params.get(f"{prefix}_end")
        if start is not None and end is not None:
            period_parts.append(f"{prefix} {start}..{end}")
    period = "; ".join(period_parts) or "all available dates"
    segment = next((params.get(key) for key in ("segment_column", "group_by", "group_column", "dimension_column") if isinstance(params.get(key), str)), "all rows")
    filters = params.get("filters") if isinstance(params.get("filters"), list) else []
    try:
        used = len(apply_filters(df, filters)) if filters else len(df)
    except Exception:
        used = 0
    excluded = max(0, len(df) - used)
    filter_text = str(filters) if filters else "none"
    return f"Understood as: metric={metric}; period={period}; segment={segment}; filters={filter_text}. Computed from {used} rows ({excluded} excluded)."


def prepend_interpretation(answer: str, df: pd.DataFrame, params: dict | None = None, metric_hint: str | None = None) -> str:
    if answer.startswith("Understood as:"):
        return answer
    return interpretation_echo(df, params, metric_hint) + "\n\n" + answer
