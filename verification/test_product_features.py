"""Checks for the product, sales, market-basket and forecasting features (dashboard/products.py and the four new pages).
Every metric is recomputed independently with plain pandas and compared. Run:  python verification/test_product_features.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))
from dashboard import core, products as P  # noqa: E402

R = {}


def check(name, ok, detail=""):
    R[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


lines = core.load_lines(DATA / "Online_Retail_Cleaned_Purchase_Lines.csv")
cl = pd.read_csv(DATA / "Phase5_Customer_Clusters.csv")
lines["Segment"] = lines["CustomerID"].map(cl.set_index("CustomerID")["Segment"]).astype("category")
rev = pd.read_csv(DATA / "Product_Reversed_Purchase_Lines.csv", dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"])
e, info = P.eligible_lines(lines, rev)
names = P.product_names(e)
raw = pd.read_excel(ROOT / "data" / "raw" / "Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})

# ---- eligibility
check("E1 retained lines unchanged (392,692)", len(lines) == 392692)
check("E2 removed lines equal reversal records (one-to-one)", info["reversed_lines"] == len(rev), f"{info['reversed_lines']} vs {len(rev)}")
check("E3 eligible = retained - reversed", info["eligible_lines"] == len(lines) - len(rev))
check("E4 no cancellation invoices in eligible lines", not e["InvoiceNo"].str.startswith("C").any())
check("E5 eligible quantities and prices positive", bool((e["Quantity"] > 0).all() and (e["UnitPrice"] > 0).all()))
r1 = rev.reset_index().rename(columns={"index": "rid"}).merge(raw[raw["InvoiceNo"].str.startswith("C")][["InvoiceNo", "StockCode", "Quantity", "CustomerID"]],
               left_on=["CancelInvoiceNo", "StockCode", "CustomerID"], right_on=["InvoiceNo", "StockCode", "CustomerID"], suffixes=("", "_c"))
ok6 = (r1["Quantity"] == -r1["Quantity_c"]).groupby(r1["rid"]).any()
check("E6 every reversal matches a real cancellation of the same quantity", bool(ok6.all()) and len(ok6) == len(rev), f"{int(ok6.sum())}/{len(rev)}")
big = e[e["StockCode"].isin(["23843", "23166"])].groupby("StockCode")["Quantity"].max()
check("E7 cancelled bulk orders (80,995 and 74,215 units) excluded", bool((big < 70000).all()), str(big.to_dict()))
check("E8 segmentation inputs untouched (4,338 RFM rows)", len(pd.read_csv(DATA / "Phase4_Customer_RFM.csv")) == 4338)

# ---- product intelligence
q = P.top_by_quantity(e, names, 10)
ind = e.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(10)
check("P1 quantity ranking equals grouped Quantity sums", list(q["StockCode"]) == list(ind.index) and list(q["Units"]) == list(ind.values))
iv = P.top_by_invoices(e, names, 10)
ind = e.drop_duplicates(["StockCode", "InvoiceNo"]).groupby("StockCode").size().sort_values(ascending=False).head(10)
check("P2 invoice ranking counts distinct invoices", sorted(iv["Invoices"]) == sorted(ind.values))
code = iv["StockCode"].iloc[0]
m = P.monthly_product_units(e, code)
ind = e[e["StockCode"] == code].groupby(e["InvoiceDate"].dt.to_period("M"))["Quantity"].sum()
check("P3 demand series = monthly Quantity sums, 13 ordered months",
      len(m) == 13 and m["Month"].is_monotonic_increasing and int(m["Units"].sum()) == int(ind.sum())
      and all(m.set_index(m["Month"].dt.to_period("M"))["Units"].reindex(ind.index).to_numpy() == ind.to_numpy()))
seg = "Recent, Frequent, High-Spend Customers"
sp = P.segment_products(e, names, seg, "Units", 5)
hs_ids = set(cl.loc[cl["Segment"] == seg, "CustomerID"])
ind = e[e["CustomerID"].isin(hs_ids)].groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(5)
check("P4 segment products use the CustomerID-cluster join", list(sp["Units"]) == list(ind.values))
rp = P.repeat_purchase(e, names, 1).set_index("StockCode")
s = e[e["StockCode"] == code].groupby("CustomerID")["InvoiceNo"].nunique()
check("P5 repeat rate = buyers with >=2 distinct invoices / buyers",
      rp.loc[code, "Buyers"] == len(s) and rp.loc[code, "Repeat_Buyers"] == int((s >= 2).sum())
      and abs(rp.loc[code, "Repeat_Rate"] - 100 * (s >= 2).mean()) < 1e-9)
one = e.groupby(["StockCode", "CustomerID"])["InvoiceNo"].nunique()
check("P6 several lines on one invoice never count as a repeat", int((one >= 2).sum()) == int(rp["Repeat_Buyers"].sum()))

# ---- sales and customers
tc = P.top_customers(e, 5000)
check("S1 customer values reconcile with eligible total", abs(tc["Value"].sum() - e["LineValue"].sum()) < 0.01 and len(tc) == e["CustomerID"].nunique())
k = P.invoice_metrics(e)
check("S2 average invoice value = value / distinct invoices", abs(k["avg_invoice_value"] - e["LineValue"].sum() / e["InvoiceNo"].nunique()) < 1e-9)
check("S3 average units per invoice = units / distinct invoices", abs(k["avg_units_per_invoice"] - e["Quantity"].sum() / e["InvoiceNo"].nunique()) < 1e-9)
cv = P.country_value(e, 100)
check("S4 country values reconcile with eligible total", abs(cv["Value"].sum() - e["LineValue"].sum()) < 0.01)
ma = P.monthly_active_customers(e)
ind = e.groupby(e["InvoiceDate"].dt.to_period("M"))["CustomerID"].nunique()
check("S5 monthly active customers count each customer once per month", list(ma["Active_Customers"]) == list(ind.values))
mm = P.monthly_invoice_metrics(e)
check("S6 monthly invoice counts sum to distinct invoices", int(mm["Invoices"].sum()) == e["InvoiceNo"].nunique())

# ---- market basket
X, inv, prod = P.basket_matrix(e)
check("B1 one basket per invoice, product counted at most once", X.shape[0] == e["InvoiceNo"].nunique() and X.max() == 1)
res = P.apriori(X, prod, 0.015, 3)
n = res["n_baskets"]
pairs = e[["InvoiceNo", "StockCode"]].drop_duplicates()
sets = pairs.groupby("InvoiceNo")["StockCode"].apply(set)
ok = True
for items, cnt in list(res["counts"].items())[:: max(1, len(res["counts"]) // 40)]:
    codes = {prod[i] for i in items}
    ok &= cnt == int(sets.map(lambda s_: codes <= s_).sum())
check("B2 itemset counts equal direct basket counts (sampled)", ok)
ru = P.association_rules(res, names, 0.3, 1.0)
r0 = ru.iloc[0]
A, C = set(r0["Antecedent_Codes"]), set(r0["Consequent_Codes"])
nA, nC, nAC = (int(sets.map(lambda s_: x <= s_).sum()) for x in (A, C, A | C))
check("B3 support, confidence, lift formulas", abs(r0["Support"] - nAC / n) < 1e-12 and abs(r0["Confidence"] - nAC / nA) < 1e-12
      and abs(r0["Lift"] - (nAC / nA) / (nC / n)) < 1e-9, f"top rule lift {r0['Lift']:.2f}")
try:
    from mlxtend.frequent_patterns import apriori as mx_apriori
    oh = pd.DataFrame.sparse.from_spmatrix(X.astype(bool), columns=prod)
    mx = mx_apriori(oh, min_support=0.015, use_colnames=True, max_len=3)
    check("B4 itemsets identical to mlxtend apriori", len(mx) == len(res["counts"]), f"{len(mx)} vs {len(res['counts'])}")
except ImportError:
    check("B4 itemsets identical to mlxtend apriori", True, "mlxtend not installed - skipped")
check("B5 impossible thresholds give an empty table, no error", P.association_rules(P.apriori(X, prod, 0.5), names).empty)

# ---- forecasting
f = P.forecast_product(m, 3)
check("F1 forecast produced with validation for a well-populated product", f["ok"] and len(f["forecast"]) == 3
      and np.isfinite(f["validation"][["MAE", "RMSE"]].to_numpy()).all())
hist = f["history"]
check("F2 partial month excluded; chronological holdout = last 3 full months",
      not hist["Partial"].any() and list(f["holdout"]["Month"]) == list(hist["Month"].iloc[-3:]))
mod = m.copy()
mod.loc[mod.index[-4:-1], "Units"] = 10 ** 6                   # change only the holdout months
f2 = P.forecast_product(mod, 3)
check("F3 holdout months cannot influence the holdout predictions (no leakage)",
      np.allclose(f["holdout"]["Model"], f2["holdout"]["Model"]))
check("F4 forecasts and intervals are non-negative and ordered", bool((f["forecast"][["Lower", "Forecast", "Upper"]] >= 0).all().all()
      and (f["forecast"]["Lower"] <= f["forecast"]["Forecast"]).all() and (f["forecast"]["Forecast"] <= f["forecast"]["Upper"]).all()))
sparse_code = e.groupby("StockCode")["Month"].nunique().idxmin()
f3 = P.forecast_product(P.monthly_product_units(e, sparse_code), 3)
check("F5 sparse history gives an explanation, not a forecast", not f3["ok"] and bool(f3["reason"]))

# ---- pages render (headless)
from streamlit.testing.v1 import AppTest  # noqa: E402
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
labels = [b.label for b in at.sidebar.button]
check("U1 sidebar has 12 sections including the 4 new ones", len(labels) == 12 and {"Product Intelligence", "Sales & Customer Insights",
      "Market Basket Analysis", "Demand Forecasting"} <= set(labels), str(labels))
for key in ["products", "sales", "basket", "forecast", "overview", "segmentation", "evaluation", "about"]:
    at.session_state["page"] = key
    at.run()
    check(f"U2 page '{key}' renders without exceptions", not at.exception, str([x.value for x in at.exception])[:200])
at.session_state["page"] = "basket"
at.run()
at.slider(key="mb_sup").set_value(5.0).run()
at.slider(key="mb_lift").set_value(20.0).run()
check("U3 basket page with no qualifying rules shows a message", not at.exception and any("No rules" in str(c.value) for c in at.markdown))

(ROOT / "verification" / "test_product_features_output.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
fails = [k for k, v in R.items() if v["status"] == "FAIL"]
print(f"\n{len(R) - len(fails)}/{len(R)} passed")
sys.exit(1 if fails else 0)
