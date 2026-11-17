"""Log-log price elasticity with product fixed effects and week controls."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

ALPHA = 0.05
MIN_PRODUCTS_PER_SEGMENT = 50


@dataclass(frozen=True)
class ElasticityResult:
    elasticity: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_products: int

    @property
    def ci_includes_minus_one(self) -> bool:
        return self.ci_low <= -1.0 <= self.ci_high

    @property
    def ci_includes_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high


def _demean_by_product(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols] - frame.groupby("StockCode")[cols].transform("mean")


def fit_elasticity(panel: pd.DataFrame):
    """Fit ``log q = α_product + ε·log p + γ_week + u`` by within-product OLS.

    Product effects are removed by within-product demeaning. Week effects
    enter as dummies, also demeaned within product. Standard errors are HC3.
    Returns the statsmodels results object so callers can inspect residuals.
    """
    week_dummies = pd.get_dummies(panel["week"], prefix="w", drop_first=True, dtype=float)
    design = pd.concat([panel[["StockCode", "log_p"]], week_dummies], axis=1)
    value_cols = ["log_p"] + list(week_dummies.columns)
    within = _demean_by_product(design, value_cols)
    y = panel["log_q"] - panel.groupby("StockCode")["log_q"].transform("mean")
    return sm.OLS(y, within).fit(cov_type="HC3")


def estimate_elasticity(panel: pd.DataFrame) -> ElasticityResult:
    """Point estimate and 95% CI for ε from ``fit_elasticity``."""
    fit = fit_elasticity(panel)
    low, high = fit.conf_int(ALPHA).loc["log_p"]
    return ElasticityResult(
        elasticity=float(fit.params["log_p"]),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=int(fit.nobs),
        n_products=int(panel["StockCode"].nunique()),
    )



def naive_elasticity(panel: pd.DataFrame) -> ElasticityResult:
    """Pooled OLS of log q on log p with no product or week effects."""
    x = sm.add_constant(panel[["log_p"]])
    fit = sm.OLS(panel["log_q"], x).fit(cov_type="HC3")
    low, high = fit.conf_int(ALPHA).loc["log_p"]
    return ElasticityResult(
        elasticity=float(fit.params["log_p"]),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=int(fit.nobs),
        n_products=int(panel["StockCode"].nunique()),
    )


def segment_elasticities(
    panel: pd.DataFrame, n_bins: int = 4, min_products: int = MIN_PRODUCTS_PER_SEGMENT
) -> pd.DataFrame:
    """Estimate elasticity per product price tier.

    Tiers are quantiles of each product's mean log price. A tier is reported
    only if it has at least ``min_products`` products.
    """
    product_price = panel.groupby("StockCode")["log_p"].mean()
    tiers = pd.qcut(product_price, n_bins, labels=False, duplicates="drop")
    panel = panel.assign(tier=panel["StockCode"].map(tiers))
    rows = []
    for tier, sub in panel.groupby("tier"):
        if sub["StockCode"].nunique() < min_products:
            continue
        res = estimate_elasticity(sub)
        rows.append(
            {
                "tier": int(tier),
                "elasticity": res.elasticity,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "n_products": res.n_products,
                "n_obs": res.n_obs,
            }
        )
    return pd.DataFrame(rows)
