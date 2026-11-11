"""Identify retained purchase lines that were later reversed by a cancellation (product-level eligibility rule).

The segmentation pipeline uses *gross* positive purchase lines: cancellation invoices are excluded, not subtracted.
For product, sales, basket and forecasting views this would let a cancelled bulk order (e.g. 80,995 units of one
product, cancelled 12 minutes later) dominate rankings. This script matches every cancellation line to the purchase
line it reverses and writes those purchase lines to Product_Reversed_Purchase_Lines.csv, so the dashboard can exclude them.

Matching rule (exact, one-to-one, chronological):
  same CustomerID, same StockCode, same UnitPrice, cancelled quantity == purchased quantity,
  purchase timestamp <= cancellation timestamp; each cancellation reverses at most one purchase line (the latest
  unmatched one before it) and each purchase line is reversed at most once.
Partial cancellations (different quantity) are not matched and the purchase stays eligible.
Read-only for data/raw/Online Retail.xlsx. Run:  python scripts/product_eligibility.py
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
KEY = ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "InvoiceDate", "CustomerID"]


def find_reversed(raw: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    r = raw.drop_duplicates()
    r = r[r["CustomerID"].notna()]
    canc = r[r["InvoiceNo"].astype(str).str.startswith("C") & (r["Quantity"] < 0)].copy()
    canc["CustomerID"] = canc["CustomerID"].astype("int64")
    pool = defaultdict(list)                                  # key -> purchase rows sorted by time
    for row in lines.sort_values("InvoiceDate").itertuples(index=False):
        pool[(row.CustomerID, row.StockCode, round(row.UnitPrice, 4), row.Quantity)].append(row)
    used, out = set(), []
    for c in canc.sort_values("InvoiceDate").itertuples(index=False):
        cands = pool.get((c.CustomerID, str(c.StockCode), round(c.UnitPrice, 4), -c.Quantity), [])
        for p in reversed(cands):                             # latest purchase before the cancellation
            k = (p.InvoiceNo, p.StockCode, p.Quantity, p.UnitPrice, p.InvoiceDate, p.CustomerID)
            if p.InvoiceDate <= c.InvoiceDate and k not in used:
                used.add(k)
                out.append(dict(zip(KEY, k), CancelInvoiceNo=c.InvoiceNo, CancelDate=c.InvoiceDate, LineValue=p.Quantity * p.UnitPrice))
                break
    return pd.DataFrame(out, columns=KEY + ["CancelInvoiceNo", "CancelDate", "LineValue"])


def main():
    raw = pd.read_excel(ROOT / "data" / "raw" / "Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
    lines = pd.read_csv(DATA / "Online_Retail_Cleaned_Purchase_Lines.csv", dtype={"InvoiceNo": str, "StockCode": str},
                        parse_dates=["InvoiceDate"])
    lines["CustomerID"] = lines["CustomerID"].astype("int64")
    rev = find_reversed(raw, lines)
    rev.to_csv(DATA / "Product_Reversed_Purchase_Lines.csv", index=False)
    print(f"cancellation lines considered: {int((raw.drop_duplicates().dropna(subset=['CustomerID'])['InvoiceNo'].str.startswith('C')).sum()):,}")
    print(f"purchase lines reversed in full: {len(rev):,}  (units {int(rev['Quantity'].sum()):,}, £{rev['LineValue'].sum():,.2f})")


if __name__ == "__main__":
    main()
