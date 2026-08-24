from __future__ import annotations

import pandas as pd


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame slice into JSON-safe records.

    `.astype(object)` alone unboxes numeric numpy scalars to native Python
    int/float, but leaves datetime64 values as `pandas.Timestamp`, which
    `json.dumps` cannot serialize — that combination is what this fixes.
    """
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%dT%H:%M:%S").where(safe[col].notna(), None)
    safe = safe.astype(object).where(pd.notnull(safe), None)
    return safe.to_dict(orient="records")
