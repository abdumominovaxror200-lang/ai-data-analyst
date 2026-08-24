from __future__ import annotations

import pandas as pd

from .duckdb_source import DuckDBDataSource
from .models import DataSource, QueryResult
from .sqlite_source import SQLiteDataSource

__all__ = [
    "DataSource",
    "QueryResult",
    "DuckDBDataSource",
    "SQLiteDataSource",
    "create_data_source",
]


def create_data_source(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    engine: str = "duckdb",
    **kwargs,
) -> DataSource:
    """Build a read-only SQL data source over a DataFrame.

    `data` is typically `DatasetRecord.df` from `app.datasets.storage`
    (imported by the caller, not by this module — this package stays
    decoupled from the dataset-storage layer per the task's constraint not
    to modify `storage.py`). Pass a `dict[str, pd.DataFrame]` instead of a
    bare DataFrame to register more than one table (for future joins across
    datasets) under their given names.

    engine: "duckdb" (default, primary backend — JOIN/CTE/window functions,
            EXPLAIN plan, best performance) or "sqlite" (lighter, no extra
            dependency).
    """
    if engine == "duckdb":
        return DuckDBDataSource(data, **kwargs)
    if engine == "sqlite":
        return SQLiteDataSource(data, **kwargs)
    raise ValueError(f"Unknown SQL engine '{engine}'. Use 'duckdb' or 'sqlite'.")
