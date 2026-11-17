# %% [markdown]
# # Price elasticity of online retail demand
#
# **Question.** How responsive is weekly demand to the average price paid for a product, and how certain are we about it?
#
# **Data.** Kaggle `thedevastator/online-retail-transaction-data` (license listed as "other"; read 2026-11-17).
# The rows are invoice-line sales from a UK-based online gift retailer, Dec 2010 to Dec 2011.
# Raw data is not committed. See `data/README.md`.
#
# **Model.** Within-product log-log regression with week effects, HC3 standard errors, and a 95% CI.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dp_elasticity.data import load_transactions, weekly_panel
from dp_elasticity.model import (
    estimate_elasticity,
    fit_elasticity,
    naive_elasticity,
    segment_elasticities,
)

RAW = Path("../data/raw/online_retail.csv")
FIG = Path("../reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load and clean

# %%
raw = pd.read_csv(RAW, encoding="latin1", dtype={"InvoiceNo": str, "StockCode": str})
tx = load_transactions(RAW)

quality = pd.Series(
    {
        "raw lines": len(raw),
        "cancellation invoices (C*)": int(raw["InvoiceNo"].str.startswith("C").sum()),
        "quantity <= 0": int((raw["Quantity"] <= 0).sum()),
        "unit price <= 0": int((raw["UnitPrice"] <= 0).sum()),
        "lines kept": len(tx),
        "lines without CustomerID (kept)": int(tx["CustomerID"].isna().sum()),
    }
)
quality

# %% [markdown]
# ## 2. Product-week panel
#
# Products sold in at least 30 distinct weeks are kept. Price is revenue divided by units in each week.

# %%
panel = weekly_panel(tx, min_weeks=30)
print(f"products: {panel['StockCode'].nunique()}, product-weeks: {len(panel)}")
print(f"weeks covered: {panel['week'].nunique()}")
panel[["StockCode", "week", "quantity", "price"]].head()

# %% [markdown]
# ## 3. Naive versus within-product estimate
#
# A pooled regression mixes cheap high-volume products with expensive low-volume ones, which pulls the slope toward zero.
# The within-product estimate removes product level differences.

# %%
naive = naive_elasticity(panel)
main = estimate_elasticity(panel)
pd.DataFrame(
    {
        "specification": ["pooled OLS (no product effects)", "product + week effects (main)"],
        "elasticity": [naive.elasticity, main.elasticity],
        "ci_low": [naive.ci_low, main.ci_low],
        "ci_high": [naive.ci_high, main.ci_high],
        "observations": [naive.n_obs, main.n_obs],
    }
)

# %%
fit = fit_elasticity(panel)
x_within = fit.model.exog[:, fit.model.exog_names.index("log_p")]
partial_y = fit.resid + main.elasticity * x_within

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x_within, partial_y, s=4, alpha=0.2, color="steelblue")
xs = np.linspace(x_within.min(), x_within.max(), 100)
ax.plot(xs, main.elasticity * xs, color="crimson", lw=2, label=f"slope = {main.elasticity:.2f}")
ax.set_xlabel("log price, demeaned within product")
ax.set_ylabel("log quantity, partial (after product and week effects)")
ax.set_title("Online retail: partial relation of log price and log quantity\n(product-week panel, products with 30+ weeks)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "within_product_scatter.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Main estimate and interval
#
# The 95% CI excludes −1, so demand is elastic on average: a 1% price rise is associated with a larger percentage drop in quantity.

# %%
print(f"elasticity = {main.elasticity:.3f}, 95% CI [{main.ci_low:.3f}, {main.ci_high:.3f}]")
print(f"CI includes -1 (unit-elastic threshold): {main.ci_includes_minus_one}")
print(f"CI includes 0: {main.ci_includes_zero}")

# %% [markdown]
# ## 5. Sensitivity
#
# Two concerns drive the sensitivity checks.
# 1. **Minimum weeks per product.** Changing the threshold changes which products enter.
# 2. **Bulk orders.** Lines with 100+ units are about 1% of lines but about 21% of revenue. Wholesale buyers often pay lower unit prices, which can make the slope more negative for reasons unrelated to consumer response.
#
# The table shows whether the conclusion depends on these choices.

# %%
rows = []
for min_weeks in [20, 30, 40]:
    res = estimate_elasticity(weekly_panel(tx, min_weeks=min_weeks))
    rows.append({"spec": f"min weeks = {min_weeks}", "elasticity": res.elasticity, "ci_low": res.ci_low, "ci_high": res.ci_high, "products": res.n_products})
for cap in [100, 24]:
    res = estimate_elasticity(weekly_panel(tx[tx["Quantity"] < cap], min_weeks=30))
    rows.append({"spec": f"lines with quantity < {cap}", "elasticity": res.elasticity, "ci_low": res.ci_low, "ci_high": res.ci_high, "products": res.n_products})
sens = pd.DataFrame(rows)
sens

# %%
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(sens))
ax.errorbar(
    sens["elasticity"], y,
    xerr=[sens["elasticity"] - sens["ci_low"], sens["ci_high"] - sens["elasticity"]],
    fmt="o", color="navy", capsize=4,
)
ax.axvline(-1, color="crimson", ls="--", label="unit elastic (-1)")
ax.set_yticks(y, sens["spec"])
ax.invert_yaxis()
ax.set_xlabel("price elasticity (95% CI)")
ax.set_title("Elasticity across specifications")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "sensitivity_forest.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Diagnostics
#
# Residuals should centre on zero without strong structure against the fitted values.

# %%
resid = fit.resid
fitted = fit.fittedvalues
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(resid, bins=80, color="slategray")
axes[0].set_title("Residual distribution (within-product model)")
axes[0].set_xlabel("residual")
axes[1].scatter(fitted, resid, s=3, alpha=0.2, color="slategray")
axes[1].axhline(0, color="crimson")
axes[1].set_title("Residuals versus fitted values")
axes[1].set_xlabel("fitted (demeaned)")
axes[1].set_ylabel("residual")
fig.tight_layout()
fig.savefig(FIG / "residual_diagnostics.png", dpi=150)
plt.show()
print(f"residual mean: {resid.mean():.2e}, residual sd: {resid.std():.3f}")

# %% [markdown]
# ## 7. Elasticity by price tier
#
# Tiers are quartiles of each product's mean log price. A tier is reported only if it has at least 50 products.

# %%
seg = segment_elasticities(panel)
seg

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    seg["elasticity"], seg["tier"],
    xerr=[seg["elasticity"] - seg["ci_low"], seg["ci_high"] - seg["elasticity"]],
    fmt="o", color="darkgreen", capsize=4,
)
ax.axvline(-1, color="crimson", ls="--")
ax.set_yticks(seg["tier"], [f"tier {t} (Q{t + 1})" for t in seg["tier"]])
ax.invert_yaxis()
ax.set_xlabel("price elasticity (95% CI)")
ax.set_title("Elasticity by product price tier (0 = cheapest)")
fig.tight_layout()
fig.savefig(FIG / "elasticity_by_tier.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Summary
#
# - Within-product elasticity is about **−2.0** on all lines, and about **−1.4** on retail-sized lines (quantity < 24).
#   Both intervals exclude −1, so demand is elastic in both specifications. The magnitude depends on the sample.
# - The pooled estimate (about −0.6) is biased toward zero by cross-product differences.
# - The estimate is not a causal price effect. Prices are set by the retailer and react to demand, and promotions coincide with demand spikes.
#   Week effects remove common seasonality only. The result should be read as an association, and it is conditional on this sample.

# %%
summary = pd.DataFrame(
    {
        "estimate": ["naive pooled", "within-product, all lines", "within-product, quantity < 24"],
        "elasticity": [naive.elasticity, main.elasticity, sens.iloc[4]["elasticity"]],
    }
)
summary
