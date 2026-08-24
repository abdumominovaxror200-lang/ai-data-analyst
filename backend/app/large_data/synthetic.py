"""Generate synthetic CSV files for large-data benchmarking, similar in shape to
the existing demo dataset (see ``backend/tests/conftest.py::sample_df``): a date
column, two low-cardinality category columns, and numeric quantity/revenue/cost
columns.

Generation itself is streamed in batches (not one giant in-memory DataFrame) so
this can produce a 10M-row file without needing 10M rows resident in RAM at once
— consistent with the rest of this package's "never hold it all in memory"
approach.
"""

from __future__ import annotations

from os import PathLike

import numpy as np
import pandas as pd

_REGIONS = ["North", "South", "East", "West"]
_PRODUCTS = ["Widget", "Gadget", "Gizmo", "Doohickey", "Thingamajig"]


def generate_synthetic_csv(
    path: str | PathLike[str],
    n_rows: int,
    *,
    batch_size: int = 500_000,
    seed: int = 42,
) -> None:
    """Write an n_rows-row synthetic CSV to ``path``, streamed in batches of
    ``batch_size`` rows so peak memory during generation stays bounded regardless
    of ``n_rows``. Columns: date, region, product, quantity, revenue, cost —
    same shape/dtypes as the app's real demo dataset.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    rng = np.random.default_rng(seed)
    base_date = np.datetime64("2020-01-01")

    written = 0
    first = True
    while written < n_rows:
        this_batch = min(batch_size, n_rows - written)
        day_offsets = rng.integers(0, 1800, size=this_batch)
        dates = base_date + day_offsets.astype("timedelta64[D]")
        region = rng.choice(_REGIONS, size=this_batch)
        product = rng.choice(_PRODUCTS, size=this_batch)
        quantity = rng.integers(1, 200, size=this_batch)
        revenue = rng.normal(1000, 250, size=this_batch).clip(min=1).round(2)
        cost = (revenue * rng.uniform(0.4, 0.7, size=this_batch)).round(2)

        batch_df = pd.DataFrame(
            {
                "date": dates,
                "region": region,
                "product": product,
                "quantity": quantity,
                "revenue": revenue,
                "cost": cost,
            }
        )
        batch_df.to_csv(path, mode="w" if first else "a", header=first, index=False)
        first = False
        written += this_batch
