# Data

Raw data is not committed. Only this README is tracked.

| Field | Value |
|---|---|
| Kaggle slug | `thedevastator/online-retail-transaction-data` |
| License | "other" (as listed on the Kaggle page; not confirmed) |
| Declared origin | Invoice-line sales of a UK-based online gift retailer, Dec 2010 to Dec 2011 |
| Synthetic check | Not declared synthetic. Columns and row count (541,909) match the widely used Online Retail transaction table. |
| Read date | 2026-11-17 |
| Local file | `data/raw/online_retail.csv` (git-ignored) |

## Download

```bash
export KAGGLE_API_TOKEN=<your token>   # never commit or write the token to a file
bash scripts/download_data.sh
```

## Columns used

`InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate` (format `%m/%d/%Y %H:%M`), `UnitPrice`, `CustomerID`, `Country`.

Cost is not in the data. Elasticity does not need cost, so none is assumed here.

## Cleaning

- Cancellation invoices (`InvoiceNo` starting with `C`) are dropped.
- Lines with quantity ≤ 0 or unit price ≤ 0 are dropped.
- Lines without `CustomerID` are kept because they are still sales.
- The file is read as `latin1`.
