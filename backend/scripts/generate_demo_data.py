"""Generates the demo dataset used to showcase the AI Data Analyst.

Produces data/demo/sales_data.xlsx: ~2 years of synthetic B2C retail transactions
with realistic seasonality plus a handful of intentional anomalies (revenue spikes,
a cost-data-entry error producing negative profit, and some missing customer ids)
so the anomaly-detection and insights tools have something real to find.

Run from the backend/ directory: venv/Scripts/python scripts/generate_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"

PRODUCTS = {
    "Aurora Desk Lamp": ("Home & Office", 39.0),
    "Zenith Office Chair": ("Home & Office", 189.0),
    "Pulse Wireless Mouse": ("Electronics", 24.0),
    "Nimbus Bluetooth Speaker": ("Electronics", 59.0),
    "Verve Running Shoes": ("Apparel", 79.0),
    "Solstice Backpack": ("Apparel", 65.0),
    "Ember Coffee Maker": ("Home & Office", 84.0),
    "Glide Yoga Mat": ("Fitness", 29.0),
    "Torque Dumbbell Set": ("Fitness", 119.0),
    "Halo Desk Monitor": ("Electronics", 219.0),
}
REGIONS = ["North", "South", "East", "West", "Central"]
SALESPEOPLE = ["A. Karimova", "B. Yusupov", "D. Rashidov", "E. Tosheva", "F. Normatov", "G. Islomova"]


def generate(seed: int = 20260824, n_rows: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    day_weights = 1 + 0.35 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))  # seasonality
    day_weights += np.where(pd.Series(dates).dt.month.isin([11, 12]), 0.6, 0.0)  # holiday bump
    day_probs = day_weights / day_weights.sum()
    row_dates = rng.choice(dates, size=n_rows, p=day_probs)

    product_names = list(PRODUCTS.keys())
    products = rng.choice(product_names, size=n_rows)
    categories = [PRODUCTS[p][0] for p in products]
    base_prices = np.array([PRODUCTS[p][1] for p in products])

    quantity = rng.integers(1, 8, size=n_rows)
    price_noise = rng.normal(1.0, 0.06, size=n_rows)
    unit_price = np.round(base_prices * price_noise, 2)
    revenue = np.round(unit_price * quantity, 2)

    margin = rng.uniform(0.35, 0.55, size=n_rows)
    cost = np.round(revenue * (1 - margin), 2)
    profit = np.round(revenue - cost, 2)

    df = pd.DataFrame(
        {
            "date": row_dates,
            "product": products,
            "category": categories,
            "region": rng.choice(REGIONS, size=n_rows),
            "salesperson": rng.choice(SALESPEOPLE, size=n_rows),
            "quantity": quantity,
            "unit_price": unit_price,
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
            "customer_id": [f"CUST-{i:05d}" for i in rng.integers(1, 1200, size=n_rows)],
        }
    )
    df = df.sort_values("date").reset_index(drop=True)

    _inject_anomalies(df, rng)
    return df


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> None:
    n = len(df)

    # Revenue spikes: a few bulk-order days with unrealistically high revenue.
    spike_idx = rng.choice(n, size=6, replace=False)
    df.loc[spike_idx, "quantity"] = rng.integers(80, 150, size=6)
    df.loc[spike_idx, "revenue"] = np.round(df.loc[spike_idx, "unit_price"] * df.loc[spike_idx, "quantity"], 2)
    df.loc[spike_idx, "cost"] = np.round(df.loc[spike_idx, "revenue"] * 0.5, 2)
    df.loc[spike_idx, "profit"] = np.round(df.loc[spike_idx, "revenue"] - df.loc[spike_idx, "cost"], 2)

    # Data-entry error: cost accidentally entered higher than revenue -> negative profit.
    error_idx = rng.choice(n, size=8, replace=False)
    df.loc[error_idx, "cost"] = np.round(df.loc[error_idx, "revenue"] * rng.uniform(1.2, 1.8, size=8), 2)
    df.loc[error_idx, "profit"] = np.round(df.loc[error_idx, "revenue"] - df.loc[error_idx, "cost"], 2)

    # Missing customer ids: incomplete records, a realistic data-quality issue.
    missing_idx = rng.choice(n, size=25, replace=False)
    df.loc[missing_idx, "customer_id"] = None


def main() -> None:
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
