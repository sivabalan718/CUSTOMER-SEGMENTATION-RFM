"""Data loading, validation, joins and business calculations (no Streamlit code, so it can be unit-tested).

Rules followed throughout:
* customer labels are joined by CustomerID (validated one-to-one), never by row position;
* nothing is retrained: PCA and scaling only re-express the documented model inputs;
* Monetary / LineValue are gross positive purchase-line values (cancellations excluded, not netted).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXPECTED, FILES, NON_PRODUCT_CODES, SEGMENT_ORDER

CORE = ["Recency", "Frequency", "Monetary"]


class DataError(Exception):
    """Raised when a required input is missing or fails validation."""


@dataclass
class FileStatus:
    key: str
    name: str
    path: Path
    required: bool
    exists: bool = False
    rows: int | None = None
    columns: int | None = None
    ok: bool = False
    message: str = ""
    size_mb: float | None = None
    modified: str | None = None


@dataclass
class Validation:
    checks: list = field(default_factory=list)  # (name, status, detail) with status PASS / FAIL / WARN / N/A

    def add(self, name, ok, detail="", warn=False):
        self.checks.append((name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail))

    def na(self, name, detail):
        self.checks.append((name, "N/A", detail))

    @property
    def failed(self):
        return [c for c in self.checks if c[1] == "FAIL"]


# ----------------------------------------------------------------------------- loading
def file_status(data_dir: Path) -> dict:
    out = {}
    for key, (name, cols, required) in FILES.items():
        p = data_dir / name
        st = FileStatus(key, name, p, required)
        if p.exists():
            st.exists = True
            st.size_mb = round(p.stat().st_size / 1e6, 2)
            st.modified = pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            try:
                head = pd.read_csv(p, nrows=5)
                missing = [c for c in cols if c not in head.columns]
                st.columns = len(head.columns)
                if missing:
                    st.message = f"missing columns: {', '.join(missing)}"
                else:
                    st.ok = True
                    st.message = "columns OK"
            except Exception as e:  # unreadable file
                st.message = f"unreadable: {type(e).__name__}"
        else:
            st.message = "file not found"
        out[key] = st
    return out


def read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def load_lines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str, "Description": "string", "Country": "string"},
                     parse_dates=["InvoiceDate"])
    for c in ["StockCode", "Description", "Country"]:
        df[c] = df[c].astype("category")
    df["CustomerID"] = df["CustomerID"].astype("int64")
    return df


# ----------------------------------------------------------------------------- customers + labels
def build_customers(rfm: pd.DataFrame, labels: pd.DataFrame, v: Validation) -> pd.DataFrame:
    """Join RFM rows with cluster labels by CustomerID. Unmatched customers stay 'Unlabelled' (never guessed)."""
    for name, df in [("RFM", rfm), ("labels", labels)]:
        if df["CustomerID"].isna().any():
            raise DataError(f"{name} file has missing CustomerID values")
        dups = int(df["CustomerID"].duplicated().sum())
        v.add(f"{name}: CustomerID unique", dups == 0, f"{dups} duplicate IDs")
        if dups:
            raise DataError(f"{name} file has {dups} duplicate CustomerID values; refusing to join")
    nulls = int(rfm[CORE].isna().sum().sum())
    v.add("RFM: no missing Recency/Frequency/Monetary", nulls == 0, f"{nulls} missing values")
    bad = int(((rfm["Recency"] < 0) | (rfm["Frequency"] < 1) | (rfm["Monetary"] <= 0)).sum())
    v.add("RFM: values in valid range (R ≥ 0, F ≥ 1, M > 0)", bad == 0, f"{bad} invalid rows")
    lab = labels[["CustomerID", "Cluster", "Segment"]].copy()
    m = rfm.merge(lab, on="CustomerID", how="left", validate="one_to_one", indicator=True)
    unmatched = int((m["_merge"] == "left_only").sum())
    extra = int((~lab["CustomerID"].isin(rfm["CustomerID"])).sum())
    v.add("Labels joined by CustomerID: every RFM customer has a label", unmatched == 0, f"{unmatched} unmatched customers")
    v.add("Labels file has no customers outside the RFM table", extra == 0, f"{extra} extra label rows", warn=True)
    m = m.drop(columns="_merge")
    m["Cluster"] = m["Cluster"].astype("Int64")
    m["Segment"] = m["Segment"].fillna("Unlabelled").astype(str)
    per_cluster = m.dropna(subset=["Cluster"]).groupby("Cluster")["Segment"].nunique()
    v.add("Each cluster maps to exactly one segment name", bool((per_cluster == 1).all()), per_cluster.to_dict())
    return m


def segment_order(customers: pd.DataFrame) -> list:
    present = list(dict.fromkeys(customers["Segment"]))
    return [s for s in SEGMENT_ORDER if s in present] + sorted(s for s in present if s not in SEGMENT_ORDER)


def naming_rule(profile_row, medians) -> str:
    """Same strict-threshold rule as Phase 5/7 (cluster median vs population median)."""
    recent = profile_row["Recency_Median"] < medians["Recency"]
    frequent = profile_row["Frequency_Median"] > medians["Frequency"]
    high = profile_row["Monetary_Median"] > medians["Monetary"]
    if recent and frequent and high:
        return "Recent, Frequent, High-Spend Customers"
    if recent and frequent:
        return "Recent, Frequent Customers"
    if recent and high:
        return "Recent, High-Spend Customers"
    if frequent and high:
        return "Frequent, High-Spend Customers"
    if profile_row["Recency_Median"] > medians["Recency"] and profile_row["Frequency_Median"] < medians["Frequency"]:
        return "At-Risk / Inactive, Low-Frequency Customers"
    if profile_row["Recency_Median"] > medians["Recency"]:
        return "Less-Recent Customers"
    return "Occasional / Lower-Spend Customers"


def segment_profiles(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers[customers["Segment"] != "Unlabelled"]
    g = c.groupby(["Cluster", "Segment"], observed=True)
    prof = g.agg(Customers=("CustomerID", "count"),
                 Recency_Mean=("Recency", "mean"), Recency_Median=("Recency", "median"),
                 Recency_P25=("Recency", lambda s: s.quantile(.25)), Recency_P75=("Recency", lambda s: s.quantile(.75)),
                 Frequency_Mean=("Frequency", "mean"), Frequency_Median=("Frequency", "median"),
                 Frequency_P25=("Frequency", lambda s: s.quantile(.25)), Frequency_P75=("Frequency", lambda s: s.quantile(.75)),
                 Monetary_Mean=("Monetary", "mean"), Monetary_Median=("Monetary", "median"),
                 Monetary_P25=("Monetary", lambda s: s.quantile(.25)), Monetary_P75=("Monetary", lambda s: s.quantile(.75)),
                 Monetary_Total=("Monetary", "sum")).reset_index()
    prof["Customer_Share_%"] = 100 * prof["Customers"] / len(customers)
    prof["Value_Share_%"] = 100 * prof["Monetary_Total"] / customers["Monetary"].sum()
    order = {s: i for i, s in enumerate(segment_order(customers))}
    return prof.sort_values("Segment", key=lambda s: s.map(order)).reset_index(drop=True)


def validate_profiles(customers, prof, prof7, v: Validation):
    med = customers[CORE].median()
    names_ok = all(naming_rule(r, med) == r["Segment"] for _, r in prof.iterrows())
    v.add("Segment names reproduce the documented naming rule", names_ok,
          f"population medians R={med['Recency']:.0f}, F={med['Frequency']:.0f}, M={med['Monetary']:.2f}")
    if prof7 is None:
        v.na("Profiles match Phase7_KMeans_Segment_Profiles.csv", "file not available")
        return
    j = prof.merge(prof7, left_on="Cluster", right_on="KMeans_Cluster", how="outer", suffixes=("", "_p7"))
    ok = (len(j) == len(prof) and (j["Customers"] == j["Customers_p7"]).all()
          and (j["Segment"] == j["Suggested_Segment_Name"]).all()
          and np.allclose(j["Monetary_Median"], j["Monetary_Median_p7"]) and np.allclose(j["Recency_Median"], j["Recency_Median_p7"]))
    v.add("Profiles match Phase7_KMeans_Segment_Profiles.csv (counts, names, medians)", bool(ok))


def validate_against_expected(customers, lines, audit, v: Validation):
    v.add(f"Customer count = {EXPECTED['customers']:,}", len(customers) == EXPECTED["customers"], f"{len(customers):,} loaded")
    if lines is not None:
        v.add(f"Purchase lines = {EXPECTED['clean_lines']:,}", len(lines) == EXPECTED["clean_lines"], f"{len(lines):,} loaded")
        ref = (lines["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)).date().isoformat()
        v.add(f"Reference date = {EXPECTED['reference_date']}", ref == EXPECTED["reference_date"], ref)
        v.add("Sum of Monetary = sum of LineValue", abs(customers["Monetary"].sum() - lines["LineValue"].sum()) < 0.01 * len(customers),
              f"{customers['Monetary'].sum():,.2f} vs {lines['LineValue'].sum():,.2f}")
        v.add("Sum of Frequency = distinct invoices", int(customers["Frequency"].sum()) == lines["InvoiceNo"].nunique())
    else:
        v.na("Transaction-level checks", "cleaned purchase-line file not available")
    if audit is not None:
        rows = audit.set_index("Step")["Rows"]
        removed = rows.iloc[1:-1]
        chain_ok = int(rows.iloc[0]) - int(removed.sum()) == int(rows.iloc[-1])
        v.add("Audit: raw − removals = retained", chain_ok, f"{int(rows.iloc[0]):,} − {int(removed.sum()):,} = {int(rows.iloc[-1]):,}")
        exp = [EXPECTED[k] for k in ["duplicates", "missing_customerid", "cancellations", "nonpos_quantity", "nonpos_price"]]
        v.add("Audit counts match verified figures", removed.tolist()[:5] == exp and int(rows.iloc[0]) == EXPECTED["raw_rows"]
              and int(rows.iloc[-1]) == EXPECTED["clean_lines"])
        if lines is not None:
            v.add("Audit retained count = rows in cleaned file", int(rows.iloc[-1]) == len(lines))
    else:
        v.na("Audit checks", "audit file not available")


def load_verification(vdir: Path) -> dict:
    out = {}
    for key, rel in [("core", "independent_check_output.json"), ("segments", "check_segments_output.json"),
                     ("dashboard", "test_dashboard_output.json"), ("sens", "sensitivity/sensitivity_summary.json"),
                     ("runlog", "run_all_log.json")]:
        p = vdir / rel
        try:
            out[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (json.JSONDecodeError, OSError):
            out[key] = None
    return out


# ----------------------------------------------------------------------------- filters + exports
DEFAULT_FILTERS = {"search": "", "segments": None, "countries": None, "recency": None, "frequency": None, "monetary": None}


def filter_customers(customers: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = customers
    if f.get("search"):
        d = d[d["CustomerID"].astype(str).str.contains(str(f["search"]).strip(), regex=False)]
    if f.get("segments") is not None:
        d = d[d["Segment"].isin(f["segments"])]
    if f.get("countries"):
        d = d[d["Country"].isin(f["countries"])]
    for col in CORE:
        rng = f.get(col.lower())
        if rng is not None:
            d = d[d[col].between(rng[0], rng[1])]
    return d


def sort_and_page(d: pd.DataFrame, sort_col: str, ascending: bool, page: int, page_size: int):
    d = d.sort_values([sort_col, "CustomerID"], ascending=[ascending, True], kind="mergesort")
    pages = max(1, int(np.ceil(len(d) / page_size)))
    page = min(max(1, page), pages)
    return d.iloc[(page - 1) * page_size: page * page_size], pages, page


EXPORT_COLUMNS = {"CustomerID": "CustomerID", "Cluster": "Cluster", "Segment": "Segment_Name", "Recency": "Recency_days",
                  "Frequency": "Frequency_invoices", "Monetary": "Monetary_GBP_gross_positive", "TotalItems": "Total_Items",
                  "DistinctProducts": "Distinct_Products", "Country": "Country", "LastPurchaseDate": "Last_Purchase_Date"}


def export_customers(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPORT_COLUMNS if c in d.columns]
    return d[cols].rename(columns=EXPORT_COLUMNS)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


# ----------------------------------------------------------------------------- analytics helpers
def model_features(customers: pd.DataFrame):
    """Documented Phase 5 inputs: log1p(F), log1p(M), raw R, then StandardScaler (population mean / std, ddof=0)."""
    X = customers[CORE].astype(float).copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])
    mean, std = X.mean(), X.std(ddof=0)
    return (X - mean) / std, X, mean, std


def pca_projection(customers: pd.DataFrame):
    Z, _, _, _ = model_features(customers)
    Zc = Z.to_numpy() - Z.to_numpy().mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2 / (len(Zc) - 1)
    ratio = var / var.sum()
    comps = Vt[:2]
    # deterministic sign: make the largest absolute loading of each component positive
    for i in range(2):
        if comps[i, np.argmax(np.abs(comps[i]))] < 0:
            comps[i] *= -1
    coords = Zc @ comps.T
    loadings = pd.DataFrame(comps.T, index=CORE, columns=["PC1", "PC2"])
    return coords, ratio[:2], loadings, ratio


def concentration(customers: pd.DataFrame) -> dict:
    m = np.sort(customers["Monetary"].to_numpy())[::-1]
    cum = np.cumsum(m) / m.sum()
    n = len(m)
    top = {p: float(cum[max(0, int(np.ceil(n * p / 100)) - 1)]) for p in (1, 5, 10, 20, 50)}
    asc = np.sort(m)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ asc / (n * asc.sum()))
    lorenz = pd.DataFrame({"customer_share": np.r_[0, np.arange(1, n + 1) / n],
                           "value_share": np.r_[0, np.cumsum(asc) / asc.sum()]})
    step = max(1, n // 400)
    return {"top_share": top, "gini": gini, "lorenz": lorenz.iloc[::step].reset_index(drop=True)}


def invoices_table(lines: pd.DataFrame) -> pd.DataFrame:
    g = lines.groupby("InvoiceNo", observed=True)
    return pd.DataFrame({"CustomerID": g["CustomerID"].first(), "InvoiceDate": g["InvoiceDate"].min(),
                         "Value": g["LineValue"].sum(), "Items": g["Quantity"].sum(), "Lines": g.size(),
                         "Segment": g["Segment"].first()}).reset_index()


def percentile_rank(series: pd.Series, value: float) -> float:
    return float(100 * (series < value).mean() + 50 * (series == value).mean())


def key_findings(customers, prof, lines) -> list:
    med = customers[CORE].median()
    top = prof.sort_values("Value_Share_%", ascending=False).iloc[0]
    low = prof.sort_values("Recency_Median", ascending=False).iloc[0]
    conc = concentration(customers)
    f = [
        f"{top['Segment']} (cluster {int(top['Cluster'])}) contains {top['Customer_Share_%']:.1f}% of customers but "
        f"{top['Value_Share_%']:.1f}% of gross positive purchase value.",
        f"{low['Segment']} (cluster {int(low['Cluster'])}) has a median of {low['Recency_Median']:.0f} days since the last purchase, "
        f"versus {med['Recency']:.0f} days for all customers.",
        f"The top 10% of customers by Monetary account for {100 * conc['top_share'][10]:.1f}% of gross positive purchase value "
        f"(Gini coefficient {conc['gini']:.2f}).",
        f"The median customer placed {med['Frequency']:.0f} invoices and spent £{med['Monetary']:,.2f} (gross).",
    ]
    if lines is not None:
        uk = 100 * lines.loc[lines["Country"] == "United Kingdom", "CustomerID"].nunique() / lines["CustomerID"].nunique()
        f.append(f"{uk:.1f}% of customers are UK-based, so international patterns rest on small samples.")
    return f


def interpret_segment(row, med) -> list:
    """Plain-language observations, each traceable to a measured statistic."""
    out = []
    r, fq, m = row["Recency_Median"], row["Frequency_Median"], row["Monetary_Median"]
    out.append(f"Median recency {r:.0f} days ({'more' if r < med['Recency'] else 'less'} recent than the population median of {med['Recency']:.0f}).")
    rel = "above" if fq > med["Frequency"] else ("equal to" if fq == med["Frequency"] else "below")
    out.append(f"Median frequency {fq:.0f} invoices ({rel} the population median of {med['Frequency']:.0f}).")
    rel = "above" if m > med["Monetary"] else "below"
    out.append(f"Median gross spend £{m:,.2f} ({rel} the population median of £{med['Monetary']:,.2f}).")
    out.append(f"Holds {row['Customer_Share_%']:.1f}% of customers and {row['Value_Share_%']:.1f}% of gross positive purchase value.")
    return out


def non_product_summary(lines: pd.DataFrame) -> dict:
    mask = lines["StockCode"].isin(NON_PRODUCT_CODES)
    return {"lines": int(mask.sum()), "value": float(lines.loc[mask, "LineValue"].sum()),
            "share_value_%": float(100 * lines.loc[mask, "LineValue"].sum() / lines["LineValue"].sum()),
            "by_code": lines.loc[mask].groupby("StockCode", observed=True)["LineValue"].agg(["size", "sum"]).rename(
                columns={"size": "Lines", "sum": "Value_GBP"}).reset_index()}
