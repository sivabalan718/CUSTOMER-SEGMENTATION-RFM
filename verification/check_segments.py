"""Independent checks for the K-Means evaluation (Phase 5) and the segment interpretation (Phase 7).

Recomputes K-Means for K=2..8 from Phase4_Customer_RFM.csv, compares the saved evaluation table and cluster labels,
and checks every Phase 7 table. All label files are joined by CustomerID (never by row position).
Reads project files only; prints a JSON summary with PASS/FAIL per check.
Usage (from the project folder):  python verification/check_segments.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "data" / "processed"
TOL = 1e-9  # absolute tolerance for floating-point metrics (differences seen between runs are ~1e-12)
checks = {}


def record(name, ok, detail=None):
    checks[name] = {"status": "PASS" if ok else "FAIL", **({"detail": detail} if detail is not None else {})}


rfm = pd.read_csv(ROOT / "Phase4_Customer_RFM.csv")
rfm_cols = ["Recency", "Frequency", "Monetary"]
X = rfm[rfm_cols].astype(float).copy()
X["Frequency"] = np.log1p(X["Frequency"])
X["Monetary"] = np.log1p(X["Monetary"])
Xs = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------- Phase 5 evaluation table (K = 2..8)
rows, labels = [], {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    labels[k] = km.labels_
    vc = np.bincount(km.labels_)
    rows.append({"K": k, "Inertia": km.inertia_, "Silhouette Score": silhouette_score(Xs, km.labels_),
                 "Davies-Bouldin Index": davies_bouldin_score(Xs, km.labels_),
                 "Calinski-Harabasz Score": calinski_harabasz_score(Xs, km.labels_),
                 "Smallest Cluster": int(vc.min()), "Largest Cluster": int(vc.max())})
ref = pd.DataFrame(rows).set_index("K")
saved = pd.read_csv(ROOT / "Phase5_KMeans_Evaluation.csv").set_index("K")
record("P5 evaluation table covers K = 2..8", list(saved.index) == list(range(2, 9)))
diff = {c: float(np.max(np.abs(ref[c].to_numpy() - saved[c].to_numpy())))
        for c in ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score", "Inertia"]}
record("P5 metrics match recomputation (abs tol 1e-9; CH and inertia rel tol 1e-9)",
       diff["Silhouette Score"] <= TOL and diff["Davies-Bouldin Index"] <= TOL
       and diff["Calinski-Harabasz Score"] <= 1e-9 * 5000 and diff["Inertia"] <= 1e-9 * 7000, diff)
record("P5 cluster sizes match recomputation exactly",
       ref[["Smallest Cluster", "Largest Cluster"]].equals(saved[["Smallest Cluster", "Largest Cluster"]]))
best = {"Silhouette Score": int(ref["Silhouette Score"].idxmax()), "Davies-Bouldin Index": int(ref["Davies-Bouldin Index"].idxmin()),
        "Calinski-Harabasz Score": int(ref["Calinski-Harabasz Score"].idxmax())}
record("P5 K = 3 is selected (maximum silhouette)", best["Silhouette Score"] == 3, best["Silhouette Score"])
record("P5 claim: K = 3 is best on silhouette, Davies-Bouldin and Calinski-Harabasz", set(best.values()) == {3}, best)

# ---------------------------------------------------------------- Phase 5 labels and stability
p5 = pd.read_csv(ROOT / "Phase5_Customer_Clusters.csv")
j = p5[["CustomerID", "Cluster"]].merge(rfm[["CustomerID"]], on="CustomerID", how="outer", validate="one_to_one", indicator=True)
record("P5 labels: same 4,338 customers as Phase 4 (CustomerID join)", bool((j["_merge"] == "both").all() and len(j) == 4338))
jr = p5[["CustomerID", "Cluster"]].merge(pd.DataFrame({"CustomerID": rfm["CustomerID"], "ref": labels[3]}), on="CustomerID",
                                         validate="one_to_one")
record("P5 K = 3 labels identical to recomputation", bool((jr["Cluster"] == jr["ref"]).all()))
stab = [adjusted_rand_score(labels[3], KMeans(n_clusters=3, random_state=s, n_init=10).fit_predict(Xs)) for s in [0, 1, 7, 21, 99]]
record("P5 stability: mean ARI across five other random seeds >= 0.99", float(np.mean(stab)) >= 0.99, round(float(np.mean(stab)), 4))
j3 = p5.merge(rfm, on="CustomerID", suffixes=("", "_rfm"), validate="one_to_one")
record("P5 RFM values in label file match Phase 4 by CustomerID", all(np.allclose(j3[c], j3[c + "_rfm"]) for c in rfm_cols))


def profile(df, col):
    return df.groupby(col).agg(Customers=("CustomerID", "count"), Recency_Median=("Recency", "median"),
                               Frequency_Median=("Frequency", "median"), Monetary_Median=("Monetary", "median"),
                               Recency_Mean=("Recency", "mean"), Frequency_Mean=("Frequency", "mean"),
                               Monetary_Mean=("Monetary", "mean")).reset_index()


# ---------------------------------------------------------------- Phase 7
p7 = pd.read_csv(ROOT / "Phase7_Customer_Segment_Labels.csv")
j7 = p7[["CustomerID", "KMeans_Cluster", "Segment"] + rfm_cols].merge(
    p5[["CustomerID", "Cluster", "Segment"]], on="CustomerID", how="outer", suffixes=("", "_p5"), validate="one_to_one", indicator=True)
record("P7 labels: same 4,338 customers as Phase 5 (CustomerID join)", bool((j7["_merge"] == "both").all() and len(j7) == 4338))
record("P7 cluster labels identical to Phase 5 by CustomerID", bool((j7["KMeans_Cluster"] == j7["Cluster"]).all()))
record("P7 segment names identical to Phase 5 by CustomerID", bool((j7["Segment"] == j7["Segment_p5"]).all()))
j7r = j7.merge(rfm, on="CustomerID", suffixes=("", "_rfm"))
record("P7 RFM values match Phase 4 by CustomerID", all(np.allclose(j7r[c], j7r[c + "_rfm"]) for c in rfm_cols))

prof7 = pd.read_csv(ROOT / "Phase7_KMeans_Segment_Profiles.csv")
j7r = j7r.rename(columns={"Cluster": "C"})  # C = Phase 5 label, independent of the Phase 7 file's own label column
pr = profile(j7r, "C")
record("P7 profile counts/medians/means match recomputation",
       bool(np.allclose(pr.drop(columns="C").to_numpy(float), prof7[pr.columns.drop("C")].to_numpy(float), rtol=1e-12, atol=1e-9)))
q = j7r.groupby("C")[rfm_cols].quantile([0.25, 0.75]).unstack()
qok = all(np.allclose(q[(m, p)].to_numpy(), prof7[f"{m}_P{int(p * 100)}"].to_numpy()) for m in rfm_cols for p in (0.25, 0.75))
record("P7 percentiles (P25/P75) match recomputation", bool(qok))
record("P7 customer percentages sum to 100 and counts to 4,338",
       bool(abs(prof7["Customer_Percent"].sum() - 100) < 1e-9 and prof7["Customers"].sum() == 4338))

med = rfm[rfm_cols].median()
justify = []
for _, r in prof7.iterrows():
    name = r["Suggested_Segment_Name"]
    facts = {"recent": r["Recency_Median"] < med["Recency"], "frequent": r["Frequency_Median"] > med["Frequency"],
             "high_spend": r["Monetary_Median"] > med["Monetary"], "less_recent": r["Recency_Median"] > med["Recency"],
             "low_frequency": r["Frequency_Median"] < med["Frequency"]}
    needs = {"Recent, Frequent, High-Spend Customers": ["recent", "frequent", "high_spend"],
             "At-Risk / Inactive, Low-Frequency Customers": ["less_recent", "low_frequency"],
             "Occasional / Lower-Spend Customers": []}.get(name)
    ok = needs is not None and all(facts[n] for n in needs)
    if name == "Occasional / Lower-Spend Customers":
        ok = ok and not facts["frequent"] and not facts["high_spend"] and r["Monetary_Median"] < med["Monetary"]
    justify.append({"cluster": int(r["KMeans_Cluster"]), "name": name, "medians": {
        "R": r["Recency_Median"], "F": r["Frequency_Median"], "M": r["Monetary_Median"]}, "justified": bool(ok)})
record("P7 every segment name is supported by its medians vs population medians (strict)",
       all(x["justified"] for x in justify), {"population_medians": med.round(2).to_dict(), "segments": justify})
record("P7 names are unique", bool(prof7["Suggested_Segment_Name"].is_unique))

# ---------------------------------------------------------------- figures saved by the notebooks
figs = {"phase5_outputs": ["elbow_method.png", "silhouette_scores.png", "cluster_sizes.png", "cluster_scatter_pca.png",
                           "cluster_scatter_recency_monetary.png"],
        "phase7_outputs": ["recency_median_by_segment.png", "frequency_median_by_segment.png", "monetary_median_by_segment.png", "customers_by_segment.png"]}
missing = [f"{d}/{f}" for d, fs in figs.items() for f in fs if not (HERE.parent / "artifacts" / "phase_outputs" / d / f).exists()]
record("Figures saved by Phase 5 and Phase 7 notebooks exist", not missing, missing or None)

summary = {"root": ROOT.name, "checks": checks,
           "n_pass": sum(c["status"] == "PASS" for c in checks.values()),
           "n_fail": sum(c["status"] == "FAIL" for c in checks.values()),
           "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                        "sklearn": __import__("sklearn").__version__}}
print(json.dumps(summary, indent=1, default=str))
sys.exit(1 if summary["n_fail"] else 0)
