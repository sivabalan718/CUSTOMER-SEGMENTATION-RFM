"""Product, sales, market-basket and demand-forecast calculations (pure pandas/NumPy; no Streamlit).

Single eligibility rule for every function here ("eligible purchase lines"):
  the project's retained purchase lines (duplicates, unidentified rows, cancellation invoices, non-positive quantity
  and price already removed) MINUS purchase lines that a later cancellation reversed in full
  (Product_Reversed_Purchase_Lines.csv, produced by product_eligibility.py).
The customer segmentation (RFM / K-Means) keeps using all retained lines; it is not changed by this module.
Values are gross positive purchase value (Quantity x UnitPrice), never profit or net revenue.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import sparse

KEY = ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "InvoiceDate", "CustomerID"]
PARTIAL_MONTH = pd.Timestamp("2011-12-01")      # data end on 9 Dec 2011


# ----------------------------------------------------------------------------- eligibility + labels
def eligible_lines(lines: pd.DataFrame, reversed_lines: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    d = lines.copy()
    d["StockCode"] = d["StockCode"].astype(str)
    info = {"retained_lines": len(d), "reversed_lines": 0, "reversed_units": 0, "reversed_value": 0.0, "rule_applied": False}
    if reversed_lines is not None and len(reversed_lines):
        r = reversed_lines[KEY].copy()
        r["InvoiceNo"], r["StockCode"] = r["InvoiceNo"].astype(str), r["StockCode"].astype(str)
        r["InvoiceDate"] = pd.to_datetime(r["InvoiceDate"])
        r["_rev"] = True
        # one-to-one: the k-th identical purchase line can only be matched by the k-th identical reversal record
        d["_n"], r["_n"] = d.groupby(KEY, sort=False).cumcount(), r.groupby(KEY, sort=False).cumcount()
        m = d.merge(r, on=KEY + ["_n"], how="left")
        d = d.drop(columns="_n")
        flag = m["_rev"].eq(True).to_numpy()
        info.update(reversed_lines=int(flag.sum()), reversed_units=int(d.loc[flag, "Quantity"].sum()),
                    reversed_value=float(d.loc[flag, "LineValue"].sum()), rule_applied=True)
        d = d.loc[~flag]
    d = d.reset_index(drop=True)
    d["Month"] = d["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    info.update(eligible_lines=len(d), invoices=int(d["InvoiceNo"].nunique()), customers=int(d["CustomerID"].nunique()))
    return d, info


def product_names(e: pd.DataFrame) -> pd.Series:
    """StockCode -> readable name: the most frequent description of that code; codes are never merged by name."""
    desc = e.dropna(subset=["Description"]).astype({"Description": str})
    mode = desc.groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0])
    names = pd.Series(e["StockCode"].unique()).to_frame("StockCode").set_index("StockCode").join(mode)["Description"]
    names = names.fillna("(no description)").map(lambda s: " ".join(str(s).replace(" ,", ",").split()).title())
    dup = names.duplicated(keep=False)                      # different codes, same description -> show the code
    names[dup] = names[dup] + " [" + names[dup].index + "]"
    return names


def _with_names(df, names):
    return df.assign(Product=df.index.map(names)).reset_index()


# ----------------------------------------------------------------------------- product intelligence
def top_by_quantity(e, names, n=10):
    g = e.groupby("StockCode").agg(Units=("Quantity", "sum"), Invoices=("InvoiceNo", "nunique"), Lines=("Quantity", "size"))
    return _with_names(g.sort_values(["Units", "Invoices"], ascending=False).head(n), names)


def top_by_invoices(e, names, n=10):
    g = e.groupby("StockCode").agg(Invoices=("InvoiceNo", "nunique"), Units=("Quantity", "sum"))
    return _with_names(g.sort_values(["Invoices", "Units"], ascending=False).head(n), names)


def full_months(e) -> pd.DatetimeIndex:
    return pd.date_range(e["Month"].min(), e["Month"].max(), freq="MS")


def monthly_product_units(e, code) -> pd.DataFrame:
    s = e.loc[e["StockCode"] == code].groupby("Month")["Quantity"].sum()
    s = s.reindex(full_months(e), fill_value=0)
    return pd.DataFrame({"Month": s.index, "Units": s.to_numpy().astype(int), "Partial": s.index == PARTIAL_MONTH})


def describe_trend(series: pd.DataFrame) -> str:
    """Plain description of the displayed full-month series (no extrapolation)."""
    y = series.loc[~series["Partial"], "Units"].to_numpy(dtype=float)
    active = int((y > 0).sum())
    if len(y) < 4 or active < 4:
        return f"too sparse to describe a pattern: units were recorded in only {active} of {len(y)} full months"
    share = y.max() / y.sum()
    first, last = y[: len(y) // 2].mean(), y[len(y) // 2:].mean()
    cv = y.std() / y.mean() if y.mean() else 0
    if share > 0.5:
        return f"dominated by one month that holds {share:.0%} of all units, so no stable pattern can be read"
    if last > first * 1.25:
        trend = "higher in the second half of the period than in the first"
    elif last < first * 0.8:
        trend = "lower in the second half of the period than in the first"
    else:
        trend = "broadly level between the first and second half of the period"
    return trend + ("; month-to-month values fluctuate strongly" if cv > 0.6 else "")


def segment_products(e, names, segment, measure="Units", n=10):
    d = e.loc[e["Segment"].astype(str) == segment]
    g = d.groupby("StockCode").agg(Units=("Quantity", "sum"), Invoices=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"),
                                   Value=("LineValue", "sum"))
    return _with_names(g.sort_values([measure, "Units"], ascending=False).head(n), names)


def repeat_purchase(e, names, min_buyers=50):
    """Per product: customers buying it at least once, and in >= 2 distinct invoices (same customer, same product)."""
    pc = e.groupby(["StockCode", "CustomerID"])["InvoiceNo"].nunique()
    g = pc.groupby(level="StockCode").agg(Buyers="size", Repeat_Buyers=lambda s: int((s >= 2).sum()), Median_Invoices="median")
    g["Repeat_Rate"] = 100 * g["Repeat_Buyers"] / g["Buyers"]
    g = g[g["Buyers"] >= min_buyers]
    return _with_names(g, names)


# ----------------------------------------------------------------------------- customers + sales
def top_customers(e, n=20):
    g = e.groupby("CustomerID").agg(Value=("LineValue", "sum"), Invoices=("InvoiceNo", "nunique"), Units=("Quantity", "sum"),
                                    Last_Purchase=("InvoiceDate", "max"), Segment=("Segment", "first"), Country=("Country", "first"))
    return g.sort_values("Value", ascending=False).head(n).reset_index()


def invoice_metrics(e) -> dict:
    inv = e["InvoiceNo"].nunique()
    v, u = float(e["LineValue"].sum()), int(e["Quantity"].sum())
    return {"invoices": int(inv), "value": v, "units": u, "avg_invoice_value": v / inv if inv else np.nan,
            "avg_units_per_invoice": u / inv if inv else np.nan}


def monthly_invoice_metrics(e) -> pd.DataFrame:
    first = e.groupby("InvoiceNo")["Month"].min()              # an invoice belongs to the month it was issued
    d = e.assign(InvMonth=e["InvoiceNo"].map(first))
    g = d.groupby("InvMonth").agg(Value=("LineValue", "sum"), Units=("Quantity", "sum"), Invoices=("InvoiceNo", "nunique"),
                                  Active_Customers=("CustomerID", "nunique"))
    g = g.reindex(full_months(e), fill_value=0)
    g["Avg_Invoice_Value"] = np.where(g["Invoices"] > 0, g["Value"] / g["Invoices"].replace(0, np.nan), np.nan)
    g["Avg_Units_per_Invoice"] = np.where(g["Invoices"] > 0, g["Units"] / g["Invoices"].replace(0, np.nan), np.nan)
    g.index.name = "Month"
    return g.reset_index().assign(Partial=lambda x: x["Month"] == PARTIAL_MONTH)


def country_value(e, n=10):
    g = e.groupby("Country", observed=True).agg(Value=("LineValue", "sum"), Customers=("CustomerID", "nunique"),
                                                Invoices=("InvoiceNo", "nunique"))
    g["Value_Share"] = 100 * g["Value"] / g["Value"].sum()
    return g.sort_values("Value", ascending=False).head(n).reset_index()


def monthly_active_customers(e) -> pd.DataFrame:
    s = e.groupby("Month")["CustomerID"].nunique().reindex(full_months(e), fill_value=0)
    return pd.DataFrame({"Month": s.index, "Active_Customers": s.to_numpy(), "Partial": s.index == PARTIAL_MONTH})


# ----------------------------------------------------------------------------- market basket (Apriori)
def basket_matrix(e):
    """Invoice x product presence matrix (each product counted once per invoice)."""
    pairs = e[["InvoiceNo", "StockCode"]].drop_duplicates()
    inv = pd.Index(pairs["InvoiceNo"].unique())
    prod = pd.Index(pairs["StockCode"].unique())
    X = sparse.csr_matrix((np.ones(len(pairs), dtype=np.int32), (inv.get_indexer(pairs["InvoiceNo"]), prod.get_indexer(pairs["StockCode"]))),
                          shape=(len(inv), len(prod)))
    return X.tocsc(), inv, prod


def apriori(X, products, min_support=0.02, max_len=3) -> dict:
    """Level-wise Apriori: frequent k-itemsets are built only from frequent (k-1)-itemsets (downward closure)."""
    n = X.shape[0]
    min_count = int(np.ceil(min_support * n))
    col = np.asarray(X.sum(axis=0)).ravel()
    freq1 = np.flatnonzero(col >= min_count)
    counts = {(int(i),): int(col[i]) for i in freq1}
    if max_len >= 2 and len(freq1) > 1:
        Xf = X[:, freq1]
        co = (Xf.T @ Xf).toarray()
        for a, b in zip(*np.triu_indices(len(freq1), 1)):
            if co[a, b] >= min_count:
                counts[(int(freq1[a]), int(freq1[b]))] = int(co[a, b])
    if max_len >= 3:
        pairs = {k for k in counts if len(k) == 2}
        items2 = sorted({i for p in pairs for i in p})
        cols = {i: X[:, i].toarray().ravel().astype(bool) for i in items2}
        for a, b in sorted(pairs):
            for c in items2:
                if c > b and (a, c) in pairs and (b, c) in pairs:
                    cnt = int((cols[a] & cols[b] & cols[c]).sum())
                    if cnt >= min_count:
                        counts[(a, b, c)] = cnt
    return {"n_baskets": n, "min_count": min_count, "counts": counts, "products": products}


def association_rules(res, names, min_confidence=0.3, min_lift=1.0) -> pd.DataFrame:
    n, counts, prods = res["n_baskets"], res["counts"], res["products"]
    rows = []
    for items, cnt in counts.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ante in itertools.combinations(items, r):
                cons = tuple(i for i in items if i not in ante)
                conf = cnt / counts[ante]
                lift = conf / (counts[cons] / n)
                if conf >= min_confidence and lift >= min_lift:
                    rows.append({"Antecedent": " + ".join(names[prods[i]] for i in ante),
                                 "Consequent": " + ".join(names[prods[i]] for i in cons),
                                 "Antecedent_Codes": tuple(prods[i] for i in ante), "Consequent_Codes": tuple(prods[i] for i in cons),
                                 "Support": cnt / n, "Confidence": conf, "Lift": lift, "Invoices": cnt})
    cols = ["Antecedent", "Consequent", "Antecedent_Codes", "Consequent_Codes", "Support", "Confidence", "Lift", "Invoices"]
    return pd.DataFrame(rows, columns=cols).sort_values(["Lift", "Support"], ascending=False, ignore_index=True)


# ----------------------------------------------------------------------------- demand forecasting
def _holt_damped(y, alpha, beta, phi, h):
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    fitted = []
    for t in range(1, len(y)):
        f = level + phi * trend                      # one-step-ahead forecast made with data up to t-1
        fitted.append(f)
        new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    steps = np.cumsum([phi ** k for k in range(1, h + 1)])
    return np.array(fitted), level + steps * trend


def fit_holt(y):
    """Grid-search a damped-trend Holt model on the given (training) data only, by one-step-ahead squared error."""
    best = None
    for a in np.arange(0.1, 1.0, 0.1):
        for b in np.arange(0.0, 0.6, 0.1):
            for p in (0.8, 0.9, 0.98):
                fitted, _ = _holt_damped(y, a, b, p, 1)
                sse = float(np.sum((y[1:] - fitted) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, a, b, p)
    return {"alpha": round(best[1], 2), "beta": round(best[2], 2), "phi": best[3]}


def forecast_product(monthly: pd.DataFrame, horizon=3, holdout=3) -> dict:
    """monthly: output of monthly_product_units. The partial final month is excluded from fitting and evaluation."""
    hist = monthly.loc[~monthly["Partial"]].reset_index(drop=True)
    y = hist["Units"].to_numpy(dtype=float)
    out = {"history": hist, "ok": False, "reason": None}
    active = int((y > 0).sum())
    if len(y) < 9 or active < 8:
        out["reason"] = (f"only {active} of {len(y)} full months have recorded units; at least 8 active months are needed "
                         "for a forecast and a holdout check")
        return out
    if y.max() / y.sum() > 0.5:
        out["reason"] = "one month holds more than half of all units (a bulk order), so a time-series forecast would be misleading"
        return out
    # chronological validation: fit on the earlier months only, predict the last `holdout` months
    train, test = y[:-holdout], y[-holdout:]
    p = fit_holt(train)
    _, pred = _holt_damped(train, p["alpha"], p["beta"], p["phi"], holdout)
    pred = np.clip(pred, 0, None)
    naive = np.repeat(train[-1], holdout)
    ma3 = np.repeat(train[-3:].mean(), holdout)

    def mae(f): return float(np.mean(np.abs(test - f)))
    def rmse(f): return float(np.sqrt(np.mean((test - f) ** 2)))
    out["validation"] = pd.DataFrame({
        "Method": ["Damped-trend exponential smoothing (Holt)", "Baseline: last observed month", "Baseline: 3-month moving average"],
        "MAE": [mae(pred), mae(naive), mae(ma3)], "RMSE": [rmse(pred), rmse(naive), rmse(ma3)]})
    out["holdout"] = pd.DataFrame({"Month": hist["Month"].iloc[-holdout:].to_numpy(), "Actual": test, "Model": pred,
                                   "Last value": naive, "Moving average": ma3})
    # final model on all full months; interval from one-step-ahead residual spread (approximate)
    pf = fit_holt(y)
    fitted, fc = _holt_damped(y, pf["alpha"], pf["beta"], pf["phi"], horizon)
    sd = float(np.std(y[1:] - fitted, ddof=1))
    width = 1.2816 * sd * np.sqrt(np.arange(1, horizon + 1))        # approximate 80% interval, widening with the horizon
    months = pd.date_range(hist["Month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    out["forecast"] = pd.DataFrame({"Month": months, "Forecast": np.clip(fc, 0, None), "Lower": np.clip(fc - width, 0, None),
                                    "Upper": np.clip(fc + width, 0, None)})
    out.update(ok=True, params=pf, train_months=len(train), holdout_months=holdout,
               beats_baselines=bool(mae(pred) <= min(mae(naive), mae(ma3))))
    return out
