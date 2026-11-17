"""Load the online retail transactions and build a product-week panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_WEEKS = 30


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and keep completed sales lines.

    Cancellation invoices start with ``C``. Lines with non-positive quantity
    or unit price are returns, adjustments, or bad rows and are dropped.
    Rows without a CustomerID are kept because they are still sales.
    """
    raw = pd.read_csv(path, encoding="latin1", dtype={"InvoiceNo": str, "StockCode": str})
    keep = (
        ~raw["InvoiceNo"].str.startswith("C")
        & (raw["Quantity"] > 0)
        & (raw["UnitPrice"] > 0)
    )
    out = raw.loc[keep].copy()
    out["ts"] = pd.to_datetime(out["InvoiceDate"], format="%m/%d/%Y %H:%M")
    return out.reset_index(drop=True)


def weekly_panel(transactions: pd.DataFrame, min_weeks: int = MIN_WEEKS) -> pd.DataFrame:
    """Aggregate to product-week rows and keep products sold in enough weeks.

    ``price`` is revenue divided by units, so it is the average price paid
    that week. Log quantity and log price are added for the elasticity model.
    """
    tx = transactions.assign(
        week=transactions["ts"].dt.to_period("W-SUN").dt.start_time,
        revenue=transactions["Quantity"] * transactions["UnitPrice"],
    )
    panel = tx.groupby(["StockCode", "week"], as_index=False).agg(
        quantity=("Quantity", "sum"), revenue=("revenue", "sum")
    )
    weeks_per_product = panel.groupby("StockCode")["week"].transform("nunique")
    panel = panel.loc[weeks_per_product >= min_weeks].copy()
    panel["price"] = panel["revenue"] / panel["quantity"]
    panel["log_q"] = np.log(panel["quantity"])
    panel["log_p"] = np.log(panel["price"])
    return panel.reset_index(drop=True)
