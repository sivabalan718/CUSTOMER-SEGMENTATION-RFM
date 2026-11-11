"""SENSITIVITY ANALYSES (not the primary result).

Re-examines three documented modelling choices of the primary pipeline without changing it:
  S1  Gross vs net Monetary (cancellations subtracted at customer level)
  S2  Excluding non-product / service lines from Monetary, Frequency and Recency
  S3  log1p-transforming Recency as well as Frequency and Monetary
Each variant re-runs the same K-Means set-up (log1p, StandardScaler, K=3, random_state=42, n_init=10)
and compares it with the primary Phase 5 labels using the Adjusted Rand Index (label-permutation invariant).

Reads the project files (never writes them) and writes results to verification/sensitivity/.
Usage (from the project folder):  python verification/sensitivity_analysis.py [optional raw-pickle cache]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "data" / "processed"
OUT = HERE / "sensitivity"
OUT.mkdir(exist_ok=True)
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
COLS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
NON_PRODUCT_RULE_A = {"POST", "DOT", "C2", "M", "BANK CHARGES"}   # explicit service / adjustment stock codes
NON_PRODUCT_RULE_B = NON_PRODUCT_RULE_A | {"23444"}                # + "Next Day Carriage" (5-digit service code)
summary = {}

# ---------------------------------------------------------------- load and standardise exactly as Phase 3
raw = pd.read_pickle(CACHE) if CACHE and CACHE.exists() else pd.read_excel(ROOT.parent / "raw" / "Online Retail.xlsx")
if CACHE and not CACHE.exists():
    raw.to_pickle(CACHE)
d = raw[COLS].copy()
for c in ["InvoiceNo", "StockCode"]:
    d[c] = d[c].astype(str).str.strip()
for c in ["Description", "Country"]:
    d[c] = d[c].astype("string").str.strip()
d["CustomerID"] = pd.to_numeric(d["CustomerID"], errors="coerce").astype("Int64")
d = d.drop_duplicates(subset=COLS, keep="first")
ident = d[d["CustomerID"].notna()].copy()
ident["CustomerID"] = ident["CustomerID"].astype("int64")
ident["LineValue"] = ident["Quantity"] * ident["UnitPrice"]
is_cancel = ident["InvoiceNo"].str.upper().str.startswith("C")
purch = ident[~is_cancel & (ident["Quantity"] > 0) & (ident["UnitPrice"] > 0)].copy()
canc = ident[is_cancel].copy()
assert len(purch) == 392692, "Purchase lines differ from the verified Phase 3 output"

rfm = pd.read_csv(ROOT / "Phase4_Customer_RFM.csv")
primary = pd.read_csv(ROOT / "Phase5_Customer_Clusters.csv")[["CustomerID", "Cluster"]]
assert rfm["CustomerID"].is_unique and primary["CustomerID"].is_unique
gross = purch.groupby("CustomerID")["LineValue"].sum()
assert np.allclose(gross.reindex(rfm["CustomerID"]).to_numpy(), rfm["Monetary"].to_numpy(), atol=0.006)


def name_segments(profile, medians):
    """Same strict-threshold naming rule as Phase 7 (cluster median vs population median)."""
    def one(r):
        recent = r["Recency_Median"] < medians["Recency"]
        frequent = r["Frequency_Median"] > medians["Frequency"]
        high = r["Monetary_Median"] > medians["Monetary"]
        if recent and frequent and high:
            return "Recent, Frequent, High-Spend Customers"
        if recent and frequent:
            return "Recent, Frequent Customers"
        if recent and high:
            return "Recent, High-Spend Customers"
        if frequent and high:
            return "Frequent, High-Spend Customers"
        if r["Recency_Median"] > medians["Recency"] and r["Frequency_Median"] < medians["Frequency"]:
            return "At-Risk / Inactive, Low-Frequency Customers"
        if r["Recency_Median"] > medians["Recency"]:
            return "Less-Recent Customers"
        return "Occasional / Lower-Spend Customers"
    return profile.apply(one, axis=1)


def cluster_variant(table, label, log_recency=False):
    X = table[["Recency", "Frequency", "Monetary"]].astype(float).copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])
    if log_recency:
        X["Recency"] = np.log1p(X["Recency"])
    Xs = StandardScaler().fit_transform(X)
    sil = {k: silhouette_score(Xs, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)) for k in range(2, 9)}
    lab = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Xs)
    t = table.assign(Variant_Cluster=lab).merge(primary, on="CustomerID", how="left", validate="one_to_one")
    prof = t.groupby("Variant_Cluster").agg(
        Customers=("CustomerID", "count"), Recency_Median=("Recency", "median"),
        Frequency_Median=("Frequency", "median"), Monetary_Median=("Monetary", "median"),
        Monetary_Mean=("Monetary", "mean")).reset_index()
    prof["Segment_Name"] = name_segments(prof, table[["Recency", "Frequency", "Monetary"]].median())
    prof.insert(0, "Variant", label)
    common = t["Cluster"].notna()
    res = {
        "customers": int(len(t)),
        "best_k_by_silhouette": int(max(sil, key=sil.get)),
        "silhouette_k3": round(sil[3], 4),
        "silhouette_by_k": {k: round(v, 4) for k, v in sil.items()},
        "ari_vs_primary_k3": round(adjusted_rand_score(t.loc[common, "Cluster"], t.loc[common, "Variant_Cluster"]), 4),
        "cluster_sizes": sorted(np.bincount(lab).tolist()),
    }
    xt = pd.crosstab(t.loc[common, "Cluster"], t.loc[common, "Variant_Cluster"])
    xt.index = [f"primary_{i}" for i in xt.index]
    xt.columns = [f"variant_{c}" for c in xt.columns]
    return res, prof, xt


# ---------------------------------------------------------------- S0: evidence for CustomerID 12346
ev = raw[pd.to_numeric(raw["CustomerID"], errors="coerce") == 12346][COLS].copy()
ev["LineValue"] = ev["Quantity"] * ev["UnitPrice"]
ev.to_csv(OUT / "S0_customer_12346_raw_lines.csv", index=False, lineterminator="\n")
summary["S0_customer_12346"] = {
    "raw_lines": ev[["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "LineValue"]]
        .astype(str).to_dict(orient="records"),
    "gross_positive_value": round(float(ev.loc[ev["Quantity"] > 0, "LineValue"].sum()), 2),
    "net_value": round(float(ev["LineValue"].sum()), 2),
    "primary_rfm_row": rfm.loc[rfm["CustomerID"] == 12346, ["Recency", "Frequency", "Monetary"]].to_dict(orient="records")[0],
    "primary_cluster": int(primary.loc[primary["CustomerID"] == 12346, "Cluster"].iloc[0]),
}

# ---------------------------------------------------------------- S1: gross vs net
summary["S1_cancellations"] = {
    "rule": "Cancellation line = InvoiceNo starting with 'C' among de-duplicated identified-customer rows; "
            "net = gross positive purchase value + sum(Quantity x UnitPrice) of that customer's cancellation lines "
            "over the whole data period (customer-level netting, no line-level matching).",
    "cancellation_lines": int(len(canc)),
    "cancellation_lines_with_quantity_ge_0": int((canc["Quantity"] >= 0).sum()),
    "cancellation_value": round(float(canc["LineValue"].sum()), 2),
    "cancellation_value_by_stockcode_type": {
        "product_codes": round(float(canc.loc[canc["StockCode"].str.match(r"^\d{5}"), "LineValue"].sum()), 2),
        "non_product_codes": round(float(canc.loc[~canc["StockCode"].str.match(r"^\d{5}"), "LineValue"].sum()), 2),
    },
    "non_product_cancellation_codes": canc.loc[~canc["StockCode"].str.match(r"^\d{5}")]
        .groupby("StockCode")["LineValue"].agg(["size", "sum"]).round(2)
        .rename(columns={"size": "lines", "sum": "value"}).to_dict(orient="index"),
    "customers_with_any_cancellation": int(canc["CustomerID"].nunique()),
    "customers_with_cancellations_but_no_purchase_lines": int((~canc["CustomerID"].drop_duplicates().isin(purch["CustomerID"])).sum()),
}
cancel_by_cust = canc.groupby("CustomerID")["LineValue"].sum()
net = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()
net["Gross_Monetary"] = net["Monetary"]
net["Cancelled_Value"] = net["CustomerID"].map(cancel_by_cust).fillna(0.0).round(2)
net["Net_Monetary"] = (net["Gross_Monetary"] + net["Cancelled_Value"]).round(2)
tol = 0.005
summary["S1_net_spend"] = {
    "rfm_customers": int(len(net)),
    "customers_net_le_0": int((net["Net_Monetary"] <= tol).sum()),
    "customers_net_eq_0": int((net["Net_Monetary"].abs() <= tol).sum()),
    "customers_net_lt_0": int((net["Net_Monetary"] < -tol).sum()),
    "customers_net_below_50pct_of_gross": int((net["Net_Monetary"] < 0.5 * net["Gross_Monetary"]).sum()),
    "total_gross": round(float(net["Gross_Monetary"].sum()), 2),
    "total_net": round(float(net["Net_Monetary"].sum()), 2),
    "tolerance_used_for_zero": tol,
}
net.merge(primary, on="CustomerID").sort_values("Cancelled_Value").to_csv(
    OUT / "S1_gross_vs_net_by_customer.csv", index=False, lineterminator="\n")

# Matching diagnostic: can each cancellation line be traced to an earlier purchase of the same item by the same customer?
p = purch[["CustomerID", "StockCode", "InvoiceDate", "Quantity"]].sort_values("InvoiceDate").copy()
p["cum_bought"] = p.groupby(["CustomerID", "StockCode"])["Quantity"].cumsum()
c = canc[["CustomerID", "StockCode", "InvoiceDate", "Quantity", "LineValue"]].sort_values("InvoiceDate").copy()
c["cum_cancelled"] = (-c["Quantity"]).groupby([c["CustomerID"], c["StockCode"]]).cumsum()
m = pd.merge_asof(c, p[["CustomerID", "StockCode", "InvoiceDate", "cum_bought"]], on="InvoiceDate",
                  by=["CustomerID", "StockCode"], direction="backward")
m["match_status"] = np.select(
    [m["cum_bought"].isna(), m["cum_bought"] >= m["cum_cancelled"]],
    ["unmatched (no earlier purchase of same StockCode by same customer in data window)",
     "covered by earlier purchases"],
    default="partially covered")
summary["S1_matching_diagnostic"] = {
    "rule": "A cancellation line is 'covered' if the same customer bought at least the cumulative cancelled quantity "
            "of the same StockCode at or before the cancellation timestamp. Price, invoice and line-level pairing are "
            "not used, so this is a plausibility diagnostic, not a definitive match.",
    "lines": m["match_status"].value_counts().to_dict(),
    "value": m.groupby("match_status")["LineValue"].sum().round(2).to_dict(),
}

net_pos = net[net["Net_Monetary"] > tol][["CustomerID", "Recency", "Frequency", "Net_Monetary"]].rename(
    columns={"Net_Monetary": "Monetary"})
res, prof1, xt1 = cluster_variant(net_pos, "S1 net Monetary (customers with net > 0)")
res["excluded_customers_net_le_0"] = int(len(net) - len(net_pos))
summary["S1_clustering"] = res

# ---------------------------------------------------------------- S2: non-product lines
def rfm_from(lines):
    ref = lines["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    g = lines.groupby("CustomerID")
    return pd.DataFrame({
        "Recency": (ref - g["InvoiceDate"].max().dt.normalize()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["LineValue"].sum().round(2),
    }).reset_index(), ref


s2 = {}
profs2, xts2 = [], []
for rule_name, codes in [("rule_A", NON_PRODUCT_RULE_A), ("rule_B", NON_PRODUCT_RULE_B)]:
    np_mask = purch["StockCode"].isin(codes)
    lines = purch[~np_mask]
    t, ref = rfm_from(lines)
    base = rfm.set_index("CustomerID")
    tt = t.set_index("CustomerID")
    common_ids = tt.index
    info = {
        "stock_codes": sorted(codes),
        "lines_removed": int(np_mask.sum()),
        "value_removed": round(float(purch.loc[np_mask, "LineValue"].sum()), 2),
        "customers_affected": int(purch.loc[np_mask, "CustomerID"].nunique()),
        "customers_dropping_out": int(len(base) - len(tt)),
        "reference_date": str(ref.date()),
        "customers_with_changed_frequency": int((base.loc[common_ids, "Frequency"] != tt["Frequency"]).sum()),
        "customers_with_changed_recency": int((base.loc[common_ids, "Recency"] != tt["Recency"]).sum()),
        "removed_lines_by_code": purch.loc[np_mask].groupby("StockCode")["LineValue"].agg(["size", "sum"])
            .round(2).rename(columns={"size": "lines", "sum": "value"}).to_dict(orient="index"),
    }
    res, prof, xt = cluster_variant(t, f"S2 non-product excluded ({rule_name})")
    info.update(res)
    s2[rule_name] = info
    profs2.append(prof)
    xts2.append(xt.assign(Variant=rule_name))
summary["S2_non_product"] = s2
summary["S2_non_product"]["note"] = ("PADS ('PADS TO MATCH ALL CUSHIONS', 3 lines, value 0.003) is treated as a product. "
                                      "Rule A matches the five non-numeric service/adjustment codes; rule B adds 23444 'Next Day Carriage'.")

# ---------------------------------------------------------------- S3: log1p(Recency)
res, prof3, xt3 = cluster_variant(rfm[["CustomerID", "Recency", "Frequency", "Monetary"]],
                                  "S3 log1p(Recency) + log1p(F) + log1p(M)", log_recency=True)
summary["S3_log_recency"] = res

# ---------------------------------------------------------------- primary reference + save
pp = rfm.merge(primary, on="CustomerID").groupby("Cluster").agg(
    Customers=("CustomerID", "count"), Recency_Median=("Recency", "median"),
    Frequency_Median=("Frequency", "median"), Monetary_Median=("Monetary", "median"),
    Monetary_Mean=("Monetary", "mean")).reset_index().rename(columns={"Cluster": "Variant_Cluster"})
pp["Segment_Name"] = name_segments(pp, rfm[["Recency", "Frequency", "Monetary"]].median())
pp.insert(0, "Variant", "PRIMARY (Phase 5)")
pd.concat([pp, prof1, *profs2, prof3], ignore_index=True).to_csv(
    OUT / "sensitivity_cluster_profiles.csv", index=False, lineterminator="\n")
pd.concat([xt1.assign(Variant="S1"), *xts2, xt3.assign(Variant="S3")]).to_csv(
    OUT / "sensitivity_crosstabs_vs_primary.csv", lineterminator="\n")
summary["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                       "sklearn": __import__("sklearn").__version__}
(OUT / "sensitivity_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print(json.dumps(summary, indent=1, default=str))
