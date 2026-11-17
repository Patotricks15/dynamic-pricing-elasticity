import numpy as np
import pandas as pd

from dp_elasticity.data import load_transactions, weekly_panel
from dp_elasticity.model import estimate_elasticity


def _simulated_panel(true_elasticity: float, n_products: int = 80, n_weeks: int = 40, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    product_fe = rng.normal(2.0, 0.5, n_products)
    week_fe = rng.normal(0.0, 0.3, n_weeks)
    for i in range(n_products):
        log_p = rng.normal(1.0, 0.3, n_weeks)
        log_q = product_fe[i] + true_elasticity * log_p + week_fe + rng.normal(0, 0.2, n_weeks)
        rows.append(
            pd.DataFrame(
                {
                    "StockCode": f"P{i}",
                    "week": pd.Timestamp("2011-01-03") + pd.to_timedelta(np.arange(n_weeks) * 7, unit="D"),
                    "log_p": log_p,
                    "log_q": log_q,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_estimator_recovers_known_elasticity_inside_ci():
    panel = _simulated_panel(true_elasticity=-1.8)
    res = estimate_elasticity(panel)
    assert res.ci_low <= -1.8 <= res.ci_high
    assert abs(res.elasticity + 1.8) < 0.05


def test_loader_drops_cancellations_and_non_positive_lines(tmp_path):
    path = tmp_path / "tx.csv"
    path.write_text(
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "1,A,x,2,12/1/2010 8:26,3.0,1,UK\n"
        "C2,A,x,-1,12/1/2010 9:00,3.0,1,UK\n"
        "3,A,x,0,12/1/2010 9:10,3.0,1,UK\n"
        "4,A,x,1,12/1/2010 9:20,0.0,,UK\n",
        encoding="latin1",
    )
    tx = load_transactions(path)
    assert tx["InvoiceNo"].tolist() == ["1"]


def test_weekly_panel_keeps_only_products_with_enough_weeks():
    tx = pd.DataFrame(
        {
            "InvoiceNo": [str(i) for i in range(4)],
            "StockCode": ["A", "A", "B", "B"],
            "Quantity": [1, 1, 1, 1],
            "UnitPrice": [2.0, 2.0, 5.0, 5.0],
            "ts": pd.to_datetime(["2011-01-03", "2011-01-10", "2011-01-03", "2011-01-03"]),
        }
    )
    panel = weekly_panel(tx, min_weeks=2)
    assert set(panel["StockCode"]) == {"A"}
