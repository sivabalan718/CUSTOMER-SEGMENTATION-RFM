"""Independent re-computation of Phase 3-5 results from the raw Excel file.

Written separately from the notebooks so that it can catch errors in them.
It reads (never writes) the project files and prints a JSON summary.
Usage (from the project folder):  python verification/independent_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
# Working layout: project folder = parent of verification/. Submission layout: ../03_Datasets_and_Results.
ROOT = _HERE.parent / "data" / "processed"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # optional pickle cache of the raw sheet
out = {}

# ---------------- Phase 3 ----------------
if CACHE and CACHE.exists():
    raw = pd.read_pickle(CACHE)
else:
    raw = pd.read_excel(ROOT.parent / "raw" / "Online Retail.xlsx")
    if CACHE:
        raw.to_pickle(CACHE)

cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
out["raw_shape"] = list(raw.shape)
out["raw_columns"] = raw.columns.tolist()
out["raw_missing_customerid"] = int(raw["CustomerID"].isna().sum())
out["raw_cancel_rows"] = int(raw["InvoiceNo"].astype(str).str.upper().str.startswith("C").sum())
out["raw_nonpos_qty"] = int((raw["Quantity"] <= 0).sum())
out["raw_nonpos_price"] = int((raw["UnitPrice"] <= 0).sum())
out["raw_invoice_prefixes"] = raw["InvoiceNo"].astype(str).str.extract(r"^([A-Za-z]*)")[0].value_counts().to_dict()
out["raw_exact_dups_untrimmed"] = int(raw.duplicated(subset=cols).sum())
_inv = raw["InvoiceNo"].astype(str).str.strip().str.upper()
_noncancel_nonpos = (raw["Quantity"] <= 0) & ~_inv.str.startswith("C")
out["raw_noncancel_nonpos_qty"] = int(_noncancel_nonpos.sum())
out["raw_noncancel_nonpos_qty_with_customerid"] = int((_noncancel_nonpos & raw["CustomerID"].notna()).sum())
out["raw_cancel_rows_with_qty_gt_0"] = int((_inv.str.startswith("C") & (raw["Quantity"] > 0)).sum())
out["raw_A_prefix_rows_with_customerid"] = int((_inv.str.startswith("A") & raw["CustomerID"].notna()).sum())
out["raw_missing_customerid_rows_that_are_duplicates"] = int((raw.duplicated(subset=cols) & raw["CustomerID"].isna()).sum())
out["raw_missing_description_with_customerid"] = int((raw["Description"].isna() & raw["CustomerID"].notna()).sum())
out["raw_date_range"] = [str(raw["InvoiceDate"].min()), str(raw["InvoiceDate"].max())]

d = raw[cols].copy()
for c in ["InvoiceNo", "StockCode"]:
    d[c] = d[c].astype(str).str.strip()
for c in ["Description", "Country"]:
    d[c] = d[c].astype("string").str.strip()
d["CustomerID"] = pd.to_numeric(d["CustomerID"], errors="coerce").astype("Int64")

# Sequential funnel: each count is measured on the rows that survived the previous step.
funnel = [("raw", len(d))]
s = d.drop_duplicates(subset=cols, keep="first")
funnel.append(("after_dedup", len(s)))
s = s[s["CustomerID"].notna()]
funnel.append(("after_customerid", len(s)))
s = s[~s["InvoiceNo"].str.upper().str.startswith("C")]
funnel.append(("after_cancel", len(s)))
q_bad = int((s["Quantity"] <= 0).sum())
p_bad = int((s["UnitPrice"] <= 0).sum())
both_bad = int(((s["Quantity"] <= 0) & (s["UnitPrice"] <= 0)).sum())
s = s[s["Quantity"] > 0]
funnel.append(("after_qty", len(s)))
s = s[s["UnitPrice"] > 0]
funnel.append(("after_price", len(s)))
req = s["InvoiceNo"].ne("") & s["InvoiceDate"].notna() & s["StockCode"].ne("")
s = s[req]
funnel.append(("after_required", len(s)))
out["funnel"] = funnel
out["removed_per_step"] = {funnel[i][0]: funnel[i - 1][1] - funnel[i][1] for i in range(1, len(funnel))}
out["nonpos_qty_after_cancel"] = q_bad
out["nonpos_price_after_cancel"] = p_bad
out["nonpos_both_after_cancel"] = both_bad
out["dups_after_trim"] = int(d.duplicated(subset=cols).sum())
out["cancel_rows_after_dedup_all_customers"] = int(
    d.drop_duplicates(subset=cols)["InvoiceNo"].str.upper().str.startswith("C").sum())

clean_ref = s.assign(LineValue=s["Quantity"] * s["UnitPrice"])

# Compare with the saved cleaned CSV
saved = pd.read_csv(ROOT / "Online_Retail_Cleaned_Purchase_Lines.csv", dtype={"InvoiceNo": str, "StockCode": str})
out["saved_clean_shape"] = list(saved.shape)
out["saved_clean_columns"] = saved.columns.tolist()
out["saved_clean_checks"] = {
    "missing_customerid": int(saved["CustomerID"].isna().sum()),
    "nonpos_qty": int((saved["Quantity"] <= 0).sum()),
    "nonpos_price": int((saved["UnitPrice"] <= 0).sum()),
    "cancel_prefix": int(saved["InvoiceNo"].str.upper().str.startswith("C").sum()),
    "full_duplicates": int(saved.duplicated().sum()),
    "linevalue_mismatch": int((~np.isclose(saved["LineValue"], saved["Quantity"] * saved["UnitPrice"])).sum()),
    "null_counts": saved.isna().sum().to_dict(),
}
a = clean_ref.reset_index(drop=True).copy()
b = saved.copy()
a["InvoiceDate"] = a["InvoiceDate"].astype(str)
a["CustomerID"] = a["CustomerID"].astype("int64")
a["Description"] = a["Description"].astype(object).where(a["Description"].notna(), np.nan)
a["Country"] = a["Country"].astype(object)
# Exact row-by-row comparison in saved order (a key-based merge would multiply rows that share a key).
text_cols = ["InvoiceNo", "StockCode", "Description", "InvoiceDate", "Country"]
out["clean_rows_identical_in_order"] = bool(
    len(a) == len(b)
    and all((a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all() for c in text_cols)
    and np.array_equal(a["Quantity"].to_numpy(), b["Quantity"].to_numpy())
    and np.array_equal(a["CustomerID"].to_numpy(), b["CustomerID"].to_numpy())
    and np.allclose(a["UnitPrice"], b["UnitPrice"], rtol=0, atol=1e-12)
    and np.allclose(a["LineValue"], b["LineValue"], rtol=1e-12, atol=1e-9))
out["clean_total_linevalue"] = round(float(clean_ref["LineValue"].sum()), 2)
out["outlier_q995"] = {c: float(clean_ref[c].quantile(0.995)) for c in ["Quantity", "UnitPrice", "LineValue"]}
out["outlier_rows_above_q995"] = {c: int((clean_ref[c] > clean_ref[c].quantile(0.995)).sum())
                                  for c in ["Quantity", "UnitPrice", "LineValue"]}

# ---------------- Phase 4 ----------------
c = clean_ref.copy()
latest = c["InvoiceDate"].max()
ref = latest.normalize() + pd.Timedelta(days=1)
g = c.groupby("CustomerID")
rfm = pd.DataFrame({
    "Recency": (ref - g["InvoiceDate"].max().dt.normalize()).dt.days,
    "Frequency": g["InvoiceNo"].nunique(),
    "Monetary": g["LineValue"].sum().round(2),
    "Lines": g.size(),
}).reset_index()
rfm["CustomerID"] = rfm["CustomerID"].astype("int64")
out["latest_invoice"] = str(latest)
out["reference_date"] = str(ref.date())
out["rfm_customers"] = len(rfm)
out["frequency_if_counting_lines_median"] = float(rfm["Lines"].median())
out["rfm_summary"] = {k: [float(rfm[k].min()), float(rfm[k].median()), float(rfm[k].max())]
                      for k in ["Recency", "Frequency", "Monetary"]}
out["rfm_total_monetary"] = round(float(rfm["Monetary"].sum()), 2)
# Invoices that map to >1 customer or >1 timestamp (would affect frequency semantics)
inv = c.groupby("InvoiceNo").agg(n_cust=("CustomerID", "nunique"), n_ts=("InvoiceDate", "nunique"))
out["invoices_with_multiple_customers"] = int((inv["n_cust"] > 1).sum())
out["invoices_with_multiple_timestamps"] = int((inv["n_ts"] > 1).sum())
out["customers_with_multiple_countries"] = int((c.groupby("CustomerID")["Country"].nunique() > 1).sum())

saved_rfm = pd.read_csv(ROOT / "Phase4_Customer_RFM.csv")
out["saved_rfm_shape"] = list(saved_rfm.shape)
out["saved_rfm_dup_ids"] = int(saved_rfm["CustomerID"].duplicated().sum())
out["saved_rfm_nulls"] = saved_rfm.isna().sum().to_dict()
cmp = rfm.merge(saved_rfm, on="CustomerID", how="outer", suffixes=("_ref", "_saved"), indicator=True)
out["rfm_id_match"] = cmp["_merge"].value_counts().to_dict()
both = cmp[cmp["_merge"] == "both"]
out["rfm_value_mismatches"] = {
    "Recency": int((both["Recency_ref"] != both["Recency_saved"]).sum()),
    "Frequency": int((both["Frequency_ref"] != both["Frequency_saved"]).sum()),
    "Monetary": int((~np.isclose(both["Monetary_ref"], both["Monetary_saved"], atol=0.006)).sum()),
}

# ---------------- Phase 5 ----------------
X = saved_rfm[["Recency", "Frequency", "Monetary"]].astype(float).copy()
X["Frequency"] = np.log1p(X["Frequency"])
X["Monetary"] = np.log1p(X["Monetary"])
Xs = StandardScaler().fit_transform(X)
evals = []
labels_by_k = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(Xs)
    labels_by_k[k] = lab
    vc = np.bincount(lab)
    evals.append({"K": k, "Inertia": km.inertia_, "Silhouette Score": silhouette_score(Xs, lab),
                  "Davies-Bouldin Index": davies_bouldin_score(Xs, lab),
                  "Calinski-Harabasz Score": calinski_harabasz_score(Xs, lab),
                  "Smallest Cluster": int(vc.min()), "Largest Cluster": int(vc.max())})
ev = pd.DataFrame(evals)
saved_ev = pd.read_csv(ROOT / "Phase5_KMeans_Evaluation.csv")
out["eval_recomputed"] = ev.round(6).to_dict(orient="records")
out["eval_max_abs_diff_vs_saved"] = {col: float((ev[col] - saved_ev[col]).abs().max()) for col in ev.columns if col != "K"}
best = int(ev.loc[ev["Silhouette Score"].idxmax(), "K"])
out["best_k_by_silhouette"] = best
out["best_k_by_db"] = int(ev.loc[ev["Davies-Bouldin Index"].idxmin(), "K"])
out["best_k_by_ch"] = int(ev.loc[ev["Calinski-Harabasz Score"].idxmax(), "K"])

saved_cl = pd.read_csv(ROOT / "Phase5_Customer_Clusters.csv")
lab = labels_by_k[best]
out["cluster_label_mismatch_vs_saved"] = int((saved_cl["Cluster"].to_numpy() != lab).sum())
out["clusters_ids_aligned"] = bool((saved_cl["CustomerID"].to_numpy() == saved_rfm["CustomerID"].to_numpy()).all())
prof = saved_rfm.assign(Cluster=lab).groupby("Cluster").agg(
    Customers=("CustomerID", "count"), Recency_Median=("Recency", "median"),
    Frequency_Median=("Frequency", "median"), Monetary_Median=("Monetary", "median"),
    Recency_Mean=("Recency", "mean"), Frequency_Mean=("Frequency", "mean"),
    Monetary_Mean=("Monetary", "mean")).reset_index()
saved_prof = pd.read_csv(ROOT / "Phase5_Cluster_Profiles.csv")
out["profile_recomputed"] = prof.round(4).to_dict(orient="records")
out["profile_max_abs_diff_vs_saved"] = float((prof - saved_prof[prof.columns]).abs().max().max())  # numeric columns only
# Stability of the K=3 solution across seeds (labels compared up to permutation via ARI)
from sklearn.metrics import adjusted_rand_score
out["k3_ari_across_seeds"] = [round(adjusted_rand_score(lab, KMeans(n_clusters=best, random_state=s, n_init=10).fit_predict(Xs)), 4)
                              for s in [0, 1, 7, 123, 2024]]
out["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                   "sklearn": __import__("sklearn").__version__}

# ---------------- explicit expected-vs-observed checks (documented figures; tolerances stated)
exp_removed = {"after_dedup": 5268, "after_customerid": 135037, "after_cancel": 8872, "after_qty": 0, "after_price": 40, "after_required": 0}
d_eval = out["eval_max_abs_diff_vs_saved"]
checks = {
    "Raw file shape is 541,909 x 8 with the 8 documented columns": out["raw_shape"] == [541909, 8] and out["raw_columns"] == cols,
    "Sequential removal counts 5,268 / 135,037 / 8,872 / 0 / 40 / 0": out["removed_per_step"] == exp_removed,
    "Saved audit CSV reconciles with recomputed funnel": bool(
        (lambda a: a.set_index("Step")["Rows"].iloc[1:-1].tolist() == list(exp_removed.values())
         and int(a["Rows"].iloc[0]) == 541909 and int(a["Rows"].iloc[-1]) == 392692)(pd.read_csv(ROOT / "Phase3_Preprocessing_Audit.csv"))),
    "Cleaned CSV has 392,692 rows x 9 expected columns": out["saved_clean_shape"] == [392692, 9]
        and out["saved_clean_columns"] == cols + ["LineValue"],
    "Cleaned CSV identical (row order and values) to recomputation": out["clean_rows_identical_in_order"],
    "Cleaned CSV: no missing IDs, cancellations, non-positive qty/price, duplicates or LineValue errors":
        all(v == 0 for k, v in out["saved_clean_checks"].items() if k != "null_counts")
        and all(v == 0 for v in out["saved_clean_checks"]["null_counts"].values()),
    "RFM: 4,338 customers, unique IDs, no missing values": out["rfm_customers"] == 4338 and out["saved_rfm_dup_ids"] == 0
        and out["rfm_id_match"].get("both") == 4338 and all(v == 0 for v in out["saved_rfm_nulls"].values()),
    "RFM: Recency, Frequency, Monetary match for every customer (Monetary tol 0.006)": all(v == 0 for v in out["rfm_value_mismatches"].values()),
    "RFM reference date is 2011-12-10": out["reference_date"] == "2011-12-10",
    "Frequency = distinct invoices (no invoice shared by two customers)": out["invoices_with_multiple_customers"] == 0,
    "Phase 5 metrics match saved evaluation (abs tol 1e-9; inertia/CH rel tol 1e-12)":
        d_eval["Silhouette Score"] <= 1e-9 and d_eval["Davies-Bouldin Index"] <= 1e-9
        and d_eval["Inertia"] <= 1e-12 * 7000 and d_eval["Calinski-Harabasz Score"] <= 1e-12 * 5000
        and d_eval["Smallest Cluster"] == 0 and d_eval["Largest Cluster"] == 0,
    "Silhouette, Davies-Bouldin and Calinski-Harabasz all select K=3": out["best_k_by_silhouette"] == out["best_k_by_db"] == out["best_k_by_ch"] == 3,
    "All 4,338 saved cluster labels equal recomputed labels": out["cluster_label_mismatch_vs_saved"] == 0 and out["clusters_ids_aligned"],
    "Saved cluster profiles match recomputation (abs tol 1e-9)": out["profile_max_abs_diff_vs_saved"] <= 1e-9,
    "K=3 solution stable across seeds (ARI >= 0.99)": min(out["k3_ari_across_seeds"]) >= 0.99,
}
out["checks"] = {k: "PASS" if v else "FAIL" for k, v in checks.items()}
out["n_pass"] = sum(bool(v) for v in checks.values())
out["n_fail"] = sum(not v for v in checks.values())
print(json.dumps(out, indent=1, default=str))
sys.exit(1 if out["n_fail"] else 0)
