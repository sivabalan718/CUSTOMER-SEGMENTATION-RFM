"""Acceptance tests for the SegmentIQ Streamlit dashboard (app.py + dashboard/ package), white redesign.

Headless (Streamlit AppTest) plus a real server start-up check. Altered inputs go to temporary folders; project files are
never modified. Prints a JSON summary; exit code 1 if any test fails.
Usage (from the project folder):  python verification/test_dashboard.py
"""
import io
import json
import os
import py_compile
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.pop("GEMINI_API_KEY", None)
from streamlit.testing.v1 import AppTest  # noqa: E402

HERE = Path(__file__).resolve().parent
APP = next(p for p in [HERE.parent / "app.py", HERE.parent / "05_Dashboard_Optional" / "app.py"] if p.exists())
sys.path.insert(0, str(APP.parent))
from dashboard import core  # noqa: E402
from dashboard.config import FILES, SEGMENT_SHORT  # noqa: E402

DATA = next(p for p in [APP.parent / "data" / "processed", APP.parent, APP.parent.parent / "03_Datasets_and_Results"] if (p / "Phase4_Customer_RFM.csv").exists())
PAGES = ["overview", "dataset", "preprocessing", "rfm", "segmentation", "evaluation", "results", "products", "sales", "basket", "forecast", "about"]
LABELS = ["Overview", "Dataset & EDA", "Preprocessing", "RFM Analysis", "Customer Segmentation", "Model Evaluation",
          "Results & Insights", "Product Intelligence", "Sales & Customer Insights",
          "Market Basket Analysis", "Demand Forecasting", "About the Project"]
FORBIDDEN = [r"\bPhase\s*\d", r"Phase\d", r"\.csv\b", r"\.json\b", r"verification", r"re-?verified", r"load-time",
             r"\bPASS\b", r"\bFAIL\b", r"Data folder", r"independent(ly)? (re-)?check", r"\bdebug\b", r"Module \d"]
results, timings = {}, {}


def record(cid, name, ok, detail=None):
    results[f"{cid} {name}"] = {"status": "PASS" if ok else "FAIL", **({"detail": detail} if detail is not None else {})}


def new_app(data_dir=None):
    if data_dir is None:
        os.environ.pop("SEGMENTATION_DATA_DIR", None)
    else:
        os.environ["SEGMENTATION_DATA_DIR"] = str(data_dir)
    return AppTest.from_file(str(APP), default_timeout=240)


def md(at):
    return "\n".join(m.value for m in at.markdown)


def plain(html):
    return unescape(re.sub(r"<[^>]+>", " ", html))


def goto(at, page):
    at.button(key=f"nav_{page}").click().run()
    return at


def match_count(at):
    m = re.search(r"\*\*([\d,]+)\*\* of ([\d,]+) customers match", md(at))
    return int(m.group(1).replace(",", "")) if m else None


def visible_text(at):
    """All user-visible strings of the current page: markdown/captions, widget labels, tabs, expanders, chart titles."""
    parts = [plain(m.value) for m in at.markdown] + [c.value for c in at.caption]
    for kind in ["button", "selectbox", "multiselect", "slider", "number_input", "text_input", "toggle", "radio", "checkbox"]:
        for w in getattr(at, kind):
            parts.append(str(getattr(w, "label", "")))
    parts += [t.label for t in at.tabs] + [e.label for e in at.expander]
    for el in at.get("vega_lite_chart") + at.get("arrow_vega_lite_chart"):
        spec = json.loads(el.proto.spec)
        stack = [spec]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "title" and isinstance(v, str):
                        parts.append(v)
                    elif k == "title" and isinstance(v, dict) and isinstance(v.get("text"), str):
                        parts.append(v["text"])
                    else:
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
    for df in at.dataframe:
        parts += [str(c) for c in df.value.columns if "Phase" in str(c)]
    return "\n".join(parts)


rfm = pd.read_csv(DATA / "Phase4_Customer_RFM.csv")
lab = pd.read_csv(DATA / "Phase5_Customer_Clusters.csv")
p7 = pd.read_csv(DATA / "Phase7_KMeans_Segment_Profiles.csv")
e5 = pd.read_csv(DATA / "Phase5_KMeans_Evaluation.csv")
audit = pd.read_csv(DATA / "Phase3_Preprocessing_Audit.csv")
insp = pd.read_csv(DATA / "Phase3_Raw_Data_Inspection.csv").set_index("Attribute")
lines = pd.read_csv(DATA / "Online_Retail_Cleaned_Purchase_Lines.csv", usecols=["InvoiceNo", "LineValue"], dtype={"InvoiceNo": str})
ref = rfm.merge(lab[["CustomerID", "Cluster", "Segment"]], on="CustomerID", validate="one_to_one")
k_best = e5.loc[e5["Silhouette Score"].idxmax()]
risk = "At-Risk / Inactive, Low-Frequency Customers"

# ------------------------------------------------------------------ code health
bad = []
for f in [APP, *sorted((APP.parent / "dashboard").rglob("*.py"))]:
    try:
        py_compile.compile(str(f), doraise=True)
    except py_compile.PyCompileError as e:
        bad.append(f"{f.name}: {e.msg[:120]}")
record("T01", "all app modules compile", not bad, bad or None)

# ------------------------------------------------------------------ start + navigation + visible-text scan
at = new_app()
t = time.time()
at.run()
timings["cold_start_s"] = round(time.time() - t, 1)
record("T02", "app starts without exceptions", len(at.exception) == 0, [e.value[:200] for e in at.exception] or None)
side = list(at.sidebar.button)
record("T03", "sidebar shows exactly the 12 specified sections, in order",
       [b.label for b in side] == LABELS and [b.key for b in side] == [f"nav_{p}" for p in PAGES], [b.label for b in side])
sb = "\n".join(m.value for m in at.sidebar.markdown)
record("T03", "sidebar shows the SegmentIQ title and a dataset label", "Segment" in sb and "IQ" in sb and "UCI Online Retail" in sb)
errors, leaks = {}, {}
for p in PAGES:
    t = time.time()
    goto(at, p)
    timings[f"page_{p}_s"] = round(time.time() - t, 2)
    if at.exception or at.error or at.session_state["page"] != p:
        errors[p] = [e.value[:200] for e in at.exception] + [e.value[:200] for e in at.error]
    active = [b.key for b in at.sidebar.button if b.proto.type == "primary"]
    if active != [f"nav_{p}"]:
        errors.setdefault(p, []).append(f"highlighted: {active}")
    txt = visible_text(at)
    hits = sorted({m.group(0) for pat in FORBIDDEN for m in re.finditer(pat, txt, re.I)})
    if hits:
        leaks[p] = hits
record("T04", "every section opens without errors and is highlighted when active", not errors, errors or None)
record("T05", "no phase references, file names, PASS/FAIL tags or verification wording in any visible text", not leaks, leaks or None)
record("T06", "responsive enough for the full dataset (cold start < 60 s, each page < 10 s)",
       timings["cold_start_s"] < 60 and all(v < 10 for k, v in timings.items() if k.startswith("page_")), timings)

# ------------------------------------------------------------------ overview
goto(at, "overview")
txt = md(at)
kv = re.findall(r'<div class="v">([^<]+)</div>', txt)
exp = [f"{len(rfm):,}", str(lab["Cluster"].nunique()), "392,692", f"{k_best['Silhouette Score']:.4f}"]
record("T07", "overview shows four primary KPIs computed from the files (4,338 · 3 · 392,692 · 0.4157)", kv == exp, kv)
need = [f"{lines['InvoiceNo'].nunique():,}", f"£{lines['LineValue'].sum():,.0f}", "01 Dec 2010 – 09 Dec 2011", "541,909"]
record("T07", "secondary figures (invoices, gross value, coverage, raw rows) match the files", all(n in txt for n in need),
       {n: n in txt for n in need})
record("T07", "required gross-value disclaimer is shown",
       "Gross positive purchase value reflects retained positive purchase lines. It is not net revenue or profit and may not "
       "account for the full effect of returns and cancellations." in txt)

# ------------------------------------------------------------------ dataset & preprocessing
goto(at, "dataset")
kv = re.findall(r'<div class="v">([^<]+)</div>', md(at))
exp = ["541,909", "8", f"{int(insp.loc['CustomerID', 'missing']):,}", f"{int(insp.loc['CustomerID', 'unique_values']):,}"]
record("T08", "dataset KPIs: 541,909 rows · 8 attributes · 135,080 missing IDs · 4,372 raw customers", kv == exp, kv)
for m in ["Invoices", "Active customers", "Gross positive purchase value (£)"]:
    at.button_group(key="ds_metric").set_value(m).run() if hasattr(at, "button_group") else None
at.slider(key="ds_topn").set_value(15).run()
record("T08", "dataset controls (trend measure, top-N) work", len(at.exception) == 0)
goto(at, "preprocessing")
txt = md(at)
after = audit["Rows_After"].tolist()
need = [f"{after[0]:,}", f"{after[1]:,}", f"{after[2]:,}", f"{after[3]:,}", f"{after[-1]:,}"]
record("T09", "preprocessing pipeline shows the audited row counts (541,909 → 536,641 → 401,604 → 392,732 → 392,692)",
       all(n in txt for n in need), need)
tab = at.dataframe[0].value
record("T09", "cleaning audit table equals the audit file", tab["Rows"].tolist() == audit["Rows"].iloc[1:-1].tolist())

# ------------------------------------------------------------------ RFM
goto(at, "rfm")
at.selectbox(key="rfm_x").set_value("Frequency").run()
at.selectbox(key="rfm_y").set_value("Monetary").run()
at.selectbox(key="rfm_cid").set_value(12347).run()
cust = core.build_customers(rfm, lab, core.Validation())
Z = core.model_features(cust)[0]
i = cust.index[cust.CustomerID == 12347][0]
insp_tab = at.dataframe[-1].value
record("T10", "RFM model-input inspection shows the correct standardised values",
       len(at.exception) == 0 and np.isclose(insp_tab.loc[insp_tab.Feature == "Monetary", "Standardised value"].iloc[0], Z.loc[i, "Monetary"]))

# ------------------------------------------------------------------ segmentation (+ customer lookup)
goto(at, "segmentation")
txt = md(at)
record("T11", "segment cards show the verified counts and shares",
       all(f"{int(r.Customers):,} · {r.Customer_Percent:.1f}%" in txt for r in p7.itertuples()))
record("T11", "headline states 3 segments from 4,338 customers", "K-Means identified 3 customer segments from 4,338 customer records." in txt)
coords, ratio2, _, _ = core.pca_projection(cust)
from sklearn.decomposition import PCA  # noqa: E402
record("T12", "PCA explained variance matches scikit-learn and is displayed",
       np.allclose(ratio2, PCA(n_components=2).fit(Z).explained_variance_ratio_) and f"{100 * ratio2.sum():.1f}%" in txt)
spec = json.dumps([json.loads(e.proto.spec) for e in at.get("vega_lite_chart")])
record("T12", "cluster chart legend uses the verified names and segment colours",
       all(n in spec for n in p7["Suggested_Segment_Name"]) and "#4C6FD6" in spec and "#2F9E77" in spec)
names = {int(k): v for k, v in cust.groupby("Cluster")["Segment"].first().items()}
record("T12", "cluster 0 is 'Occasional / Lower-Spend Customers' in labels and profiles",
       names.get(0) == "Occasional / Lower-Spend Customers" == p7.set_index("KMeans_Cluster").loc[0, "Suggested_Segment_Name"])
ok = True
for r in p7.itertuples():
    at.button_group(key="seg_pick").set_value(r.Suggested_Segment_Name).run()
    stats = next(d.value for d in at.dataframe if "Statistic" in d.value.columns)
    ok &= abs(stats.loc[stats.Statistic == "Median", "Monetary (£ gross)"].iloc[0] - r.Monetary_Median) < 1e-6
record("T13", "segment detail medians equal the verified profiles for every segment", bool(ok))
record("T14", "customer lookup lists all customers", match_count(at) == len(rfm), match_count(at))
at.text_input(key="ex_search").set_value("1234").run()
exp_n = int(rfm["CustomerID"].astype(str).str.contains("1234").sum())
record("T14", "CustomerID search filters correctly", match_count(at) == exp_n, {"expected": exp_n, "shown": match_count(at)})
at.button(key="ex_reset").click().run()
record("T14", "reset filters restores all customers", match_count(at) == len(rfm))
at.multiselect(key="ex_segments").set_value([risk]).run()
record("T14", "segment filter returns exactly that segment", match_count(at) == int((ref.Segment == risk).sum()))
at.button(key="ex_reset").click().run()
at.slider(key="ex_r").set_value((1, 30)).run()
at.slider(key="ex_f").set_value((5, 209)).run()
at.number_input(key="ex_m_min").set_value(1000.0).run()
exp_n = int((rfm.Recency.between(1, 30) & rfm.Frequency.between(5, 209) & (rfm.Monetary >= 1000)).sum())
record("T14", "Recency, Frequency and Monetary range filters combine correctly", match_count(at) == exp_n, {"expected": exp_n, "shown": match_count(at)})
at.button(key="ex_reset").click().run()
at.selectbox(key="ex_sort").set_value("Recency").run()
at.toggle(key="ex_asc").set_value(True).run()
at.selectbox(key="ex_page_size").set_value(25).run()
at.number_input(key="ex_page").set_value(3).run()
table = next(d.value for d in at.dataframe if "CustomerID" in d.value.columns and "Country" in d.value.columns)
exp_ids = rfm.sort_values(["Recency", "CustomerID"], kind="mergesort")["CustomerID"].iloc[50:75].tolist()
record("T15", "sorting (Recency ascending) and pagination (page 3, 25 rows) show the correct rows", table["CustomerID"].tolist() == exp_ids)
at.button(key="ex_reset").click().run()
at.multiselect(key="ex_segments").set_value([risk]).run()
at.slider(key="ex_r").set_value((100, 374)).run()
f = {"search": "", "segments": [risk], "countries": None, "recency": (100, 374), "frequency": None, "monetary": None}
exp_df = core.export_customers(core.filter_customers(ref.assign(Cluster=ref.Cluster.astype("Int64")), f))
back = pd.read_csv(io.StringIO(core.to_csv_bytes(exp_df).decode("utf-8")))
record("T16", "filtered export = customers matching the on-screen filters, with readable column headings",
       match_count(at) == len(exp_df) and list(back.columns)[:3] == ["CustomerID", "Cluster", "Segment_Name"], len(exp_df))
at.button(key="ex_reset").click().run()
at.selectbox(key="ex_profile").set_value(12346).run()
txt = md(at)
r12346 = rfm.loc[rfm.CustomerID == 12346].iloc[0]
record("T17", "customer profile shows actual RFM values and the cancellation note for CustomerID 12346",
       "Customer 12346" in txt and f"£{r12346.Monetary:,.2f}" in txt and "cancelled 16 minutes later" in txt)

# ------------------------------------------------------------------ evaluation + results + about
goto(at, "evaluation")
kv = re.findall(r'<div class="v">([^<]+)</div>', md(at))
exp = [str(int(k_best["K"])), f"{k_best['Silhouette Score']:.4f}", f"{k_best['Davies-Bouldin Index']:.4f}", f"{k_best['Calinski-Harabasz Score']:,.2f}"]
record("T18", "evaluation KPIs equal the saved metrics (3 · 0.4157 · 0.8249 · 4,393.73)", kv == exp, kv)
record("T18", "evaluation states all three metrics agree and keeps the 'not objectively correct' caveat",
       "all three agree" in md(at) and "do not prove" in md(at))
goto(at, "results")
txt = md(at)
hs = p7.set_index("Suggested_Segment_Name").loc["Recent, Frequent, High-Spend Customers"]
record("T19", "results headline uses the verified high-spend segment figures",
       f"{int(hs.Customers):,} customers ({hs.Customer_Percent:.1f}%)" in txt and f"£{hs.Monetary_Median:,.2f}" in txt)
dls = at.get("download_button")
record("T19", "results offers the documented downloads", len(dls) >= 6, len(dls))
record("T19", "recommendations are labelled as suggestions, not proven outcomes", "Suggestion" in txt and "no effect on revenue or retention is claimed" in txt)
goto(at, "about")
txt = md(at)
record("T20", "about page shows objectives, dataset citation, SDG 9 and limitations",
       "10.24432/C5BW33" in txt and "SDG 9" in txt and "Objectives" in txt and "Gross, not net" in txt)

# ------------------------------------------------------------------ business explanation panels
def panel(at, label):
    return next((e for e in at.expander if e.label == label), None)


def md_join(expander):
    """All markdown text inside an expander ('' if the expander is missing)."""
    return "\n".join(m.value for m in expander.markdown) if expander is not None else ""


goto(at, "dataset")
ex = panel(at, "Understand What These Statistics Tell Us")
ln = core.load_lines(DATA / "Online_Retail_Cleaned_Purchase_Lines.csv")
q, pr, lv = ln["Quantity"], ln["UnitPrice"], ln["LineValue"]
need = [f"{q.mean():.2f} units", f"{q.median():.2f} units", f"£{pr.mean():.2f}", f"£{pr.median():.2f}", f"£{lv.mean():.2f}",
        f"£{lv.median():.2f}", f"{q.std():.2f}", f"{q.skew():.2f}", f"{int(q.max()):,} units", f"£{pr.max():,.2f}"]
txt = md_join(ex)
record("E01", "Dataset & EDA: statistics explanation exists, starts collapsed and quotes the table's own values",
       ex is not None and not ex.proto.expanded and all(n in txt for n in need), {n: n in txt for n in need} if ex else "panel missing")
goto(at, "overview")
e1, e2 = panel(at, "Understand Customer Distribution"), panel(at, "Understand Recency, Frequency and Monetary Values")
t1, t2 = md_join(e1), md_join(e2)
need1 = [f"{int(r.Customers):,} customers ({r.Customer_Percent:.1f}% of customers)" for r in p7.itertuples()]
need1 += [f"{v:.1f}% of gross positive purchase value" for v in core.segment_profiles(cust)["Value_Share_%"]]
need2 = [f"{r.Recency_Median:.0f} days" for r in p7.itertuples()] + [f"£{r.Monetary_Median:,.0f}" for r in p7.itertuples()]
record("E02", "Overview: customer-distribution explanation exists, starts collapsed and matches the chart",
       e1 is not None and not e1.proto.expanded and all(n in t1 for n in need1), {n: n in t1 for n in need1} if e1 else "panel missing")
record("E03", "Overview: R/F/M explanation exists, starts collapsed, matches the charts and labels actions as possible strategies",
       e2 is not None and not e2.proto.expanded and all(n in t2 for n in need2) and "not guaranteed outcomes" in t2,
       {n: n in t2 for n in need2} if e2 else "panel missing")

# ------------------------------------------------------------------ robustness (CustomerID joins, missing/invalid files)
tmp = Path(tempfile.mkdtemp(prefix="dash_white_"))
try:
    for key, (name, _, _) in FILES.items():
        if (DATA / name).exists():
            shutil.copy(DATA / name, tmp / name)
    lab.sample(frac=1, random_state=1).to_csv(tmp / "Phase5_Customer_Clusters.csv", index=False)
    a2 = new_app(tmp)
    a2.run()
    goto(a2, "segmentation")
    a2.text_input(key="ex_search").set_value("12346").run()
    row = next(d.value for d in a2.dataframe if "Country" in d.value.columns and "CustomerID" in d.value.columns)
    record("T21", "shuffled label file: each customer keeps its own cluster (CustomerID join)",
           len(a2.exception) == 0 and len(row) == 1 and row["Segment"].iloc[0] == SEGMENT_SHORT[lab.loc[lab.CustomerID == 12346, "Segment"].iloc[0]])
    lab.iloc[5:].to_csv(tmp / "Phase5_Customer_Clusters.csv", index=False)
    a3 = new_app(tmp)
    a3.run()
    record("T22", "labels missing for 5 customers: a plain-language data warning appears, nothing is guessed",
           len(a3.exception) == 0 and "Some data consistency checks did not pass" in md(a3))
    pd.concat([lab, lab.iloc[:1]]).to_csv(tmp / "Phase5_Customer_Clusters.csv", index=False)
    a4 = new_app(tmp)
    a4.run()
    record("T22", "duplicate CustomerID in labels: app refuses to join and shows a setup message",
           len(a4.exception) == 0 and "could not be loaded" in md(a4))
    lab.to_csv(tmp / "Phase5_Customer_Clusters.csv", index=False)
    (tmp / "Online_Retail_Cleaned_Purchase_Lines.csv").unlink()
    a5 = new_app(tmp)
    a5.run()
    errs = {}
    for p in PAGES:
        goto(a5, p)
        if a5.exception:
            errs[p] = a5.exception[0].value[:150]
    record("T22", "transaction file missing: every page still loads", not errs, errs or None)
    (tmp / "Phase5_Customer_Clusters.csv").unlink()
    a6 = new_app(tmp)
    a6.run()
    record("T22", "required file missing: clear setup message, no crash", len(a6.exception) == 0 and "could not be loaded" in md(a6))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    os.environ.pop("SEGMENTATION_DATA_DIR", None)

# ------------------------------------------------------------------ removed features, stale links, dependencies, server
gone = [f for f in ["dashboard/ai.py", "dashboard/views/assistant.py", "dashboard/views/whatif.py", "dashboard/views/similarity.py"]
        if (APP.parent / f).exists()]
pat = re.compile(r"assistant|what-?if|similar[ _]customers|views\.similarity|gemini|campaign planning", re.I)
hits = [f"{f.relative_to(APP.parent)}:{n}" for f in [APP, *sorted((APP.parent / "dashboard").rglob("*.py"))]
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
record("T23", "removed features (AI assistant, what-if, similar customers, campaign planner) are absent", not gone and not hits, (gone + hits) or None)
at = new_app()
for k in ["explorer", "trends", "project", "assistant"]:
    at.session_state["page"] = k
    at.run()
record("T23", "old page links fall back to the overview", len(at.exception) == 0 and at.session_state["page"] == "overview")
req = (APP.parent / "requirements.txt").read_text(encoding="utf-8") if (APP.parent / "requirements.txt").exists() else ""
record("T24", "requirements.txt lists the dashboard dependencies", all(n in req for n in ["streamlit", "pandas", "numpy", "scikit-learn", "altair"]))
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(port),
                         "--server.address", "127.0.0.1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(APP.parent))
health = None
try:
    for _ in range(60):
        try:
            health = urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read().decode()
            break
        except Exception:
            time.sleep(1)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
record("T25", "streamlit run app.py starts a healthy server", health == "ok", health)

summary = {"app": APP.name, "results": results, "timings": timings,
           "n_pass": sum(r["status"] == "PASS" for r in results.values()),
           "n_fail": sum(r["status"] == "FAIL" for r in results.values()),
           "streamlit": __import__("streamlit").__version__, "pandas": pd.__version__, "python": sys.version.split()[0]}
print(json.dumps(summary, indent=1, default=str))
sys.exit(1 if summary["n_fail"] else 0)
