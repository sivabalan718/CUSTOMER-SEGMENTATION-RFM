# Customer Segmentation and Purchasing Behaviour Analysis Using Machine Learning (SegmentIQ)

SRM Institute of Science and Technology, Tiruchirappalli — 21CSC305P Machine Learning, Assessment Task 3
(case-based implementation, unsupervised learning). SDG 9 (Industry, Innovation and Infrastructure), Level 2.

**Team:** Siva Balan S (RA2411026050230) · Shabari Raj Kumar S (RA2411026050227) · Shailesh M (RA2411026050223) (see *Team contribution*).

## 1. Problem and objectives
An online retailer has thousands of customers but no structured view of how their purchasing behaviour differs. There are no
predefined customer types, so this is an **unsupervised clustering** problem.

1. Clean and audit the transaction data.
2. Describe each customer by **Recency, Frequency, Monetary (RFM)**.
3. Segment customers with **K-Means**, choosing K with internal validation metrics.
4. Interpret the segments and propose testable (not proven) business actions.

**Expected outcome:** a few interpretable segments with measured statistics. The project is descriptive; it does **not**
predict churn or future purchases.

## 2. Dataset
Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 — CC BY 4.0.
541,909 rows × 8 attributes (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country),
1 Dec 2010 – 9 Dec 2011, UK-based online retailer. Expected file: `data/raw/Online Retail.xlsx`
(SHA-256 `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`). The raw file is never modified.

## 3. Verified results
| Item | Value |
|---|---|
| Sequential cleaning | 541,909 → −5,268 duplicates → −135,037 no CustomerID → −8,872 cancellations → −0 qty ≤ 0 → −40 price ≤ 0 → **392,692** lines |
| Customers (RFM rows) | **4,338**; reference date 2011-12-10; Frequency = distinct invoices |
| Model inputs | log1p(Frequency), log1p(Monetary), Recency (days) → StandardScaler |
| Model | K-Means, K = 2–8 evaluated, random_state = 42, n_init = 10 → **K = 3** |
| K = 3 metrics | silhouette **0.4157**, Davies–Bouldin 0.8249, Calinski–Harabasz 4,393.73 (best of K = 2–8 on all three) |
| Segments | 2 · Recent, Frequent, High-Spend (1,320 · 30.4%) · 0 · Occasional / Lower-Spend (2,038 · 47.0%) · 1 · At-Risk / Inactive, Low-Frequency (980 · 22.6%) |

Categorical attributes (InvoiceNo, StockCode, Description, Country, CustomerID) are identifiers or context and are not model
features, so no encoding is needed. A train/test split does not apply to clustering (no target); stability across random seeds
is checked instead (ARI 0.9993).

## 4. Folder structure
```
TASKS 3/
├── app.py                      SegmentIQ dashboard entry point (streamlit run app.py)
├── requirements.txt            Python dependencies (version ranges)
├── README.md                   this file
├── .gitignore                  excludes .venv, caches, secrets, temporary files
├── .streamlit/config.toml      dashboard theme
├── dashboard/                  dashboard package: config, data loading, charts, products.py (product/basket/forecast logic), views/
├── data/
│   ├── raw/Online Retail.xlsx  UCI raw data (read-only, never modified)
│   └── processed/              all CSVs produced by the notebooks and scripts (read by the dashboard)
├── notebooks/                  Phase3 → Phase8 notebooks (run in order; they work inside data/processed)
├── artifacts/phase_outputs/    phase3_outputs … phase7_outputs: figures saved by the notebooks
├── scripts/product_eligibility.py   finds purchases later cancelled in full → data/processed/Product_Reversed_Purchase_Lines.csv
├── presentations/              SegmentIQ_7_Slide_Presentation.pptx + .pdf (source/: build_deck.js and slide images)
├── verification/               independent checks, sensitivity analysis, dashboard and feature tests, result logs (JSON)
└── reports/
    ├── final/                  Final_Project_Report.pdf + matching Final_Project_Report.docx  ← submission report
    ├── phase_reports/          Phase 3–9 Word reports, Phase9_README.txt
    └── source/
        ├── final_report_builder/   builds reports/final (chapter text in content/, figures/, screenshots in shots/)
        └── phase_report_builder/   rebuilds reports/phase_reports from the verified outputs
```
`.venv/` is a local environment only (not part of the submission); recreate it from `requirements.txt`.

## 5. Setup and running
Requires Python 3.13 (tested 3.13.7 on Windows 11).
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
**Dashboard (SegmentIQ):** from the project folder run `streamlit run app.py` (or `.venv\Scripts\python -m streamlit run app.py`)
→ http://localhost:8501. Data are read from `data/processed/` (override with the `SEGMENTATION_DATA_DIR` environment variable).
Tested with Streamlit 1.66 / pandas 3.0 (project `.venv`) and Streamlit 1.49 / pandas 2.3 (Python 3.11). It only reads saved
results, joins labels by CustomerID and never retrains the model. Twelve sections: Overview · Dataset & EDA · Preprocessing ·
RFM Analysis · Customer Segmentation · Model Evaluation · Results & Insights · Product Intelligence · Sales & Customer Insights ·
Market Basket Analysis · Demand Forecasting · About the Project.

**Notebooks:** open `notebooks/` and run Phase 3 → 8 in order; each notebook moves into `data/processed/` itself, reads the raw file
from `data/raw/` and saves figures to `artifacts/phase_outputs/`. Close the CSV files in Excel/WPS first — an open file blocks saving.

**Product eligibility file:** `python scripts/product_eligibility.py` (after Phase 3).

**Verification:** `python verification/run_all.py` re-runs every notebook and check (≈ 6 min); `--checks-only` skips the notebooks.
`python verification/test_product_features.py` checks the product, sales, basket and forecast features.
Latest results (2026-10-20): notebooks 6/6 without errors, core pipeline 15/15, Phases 6–7 24/24, sensitivity analysis completed,
dashboard 47/47, product features 40/40.

**Reports:** final report — `cd reports/source/final_report_builder`, `python build_report.py`, then
`powershell -File word_update.ps1 -Docx <full path to reports\final\Final_Project_Report.docx> -Pdf <…\Final_Project_Report.pdf>`
(needs Microsoft Word; updates page numbers and exports the PDF). Phase reports — `python reports/source/phase_report_builder/build_phase_reports.py`.

## 6. Limitations
- **Gross, not net:** Monetary is gross positive purchase value; cancellations are excluded, not subtracted (e.g. CustomerID 12346:
  £77,183.60 gross, £0 net). 21 customers have net spend ≤ £0. Using net values changes the clusters very little (ARI 0.984).
- **Service lines** (POST, DOT, C2, M, BANK CHARGES: 1,539 lines, £149,981) are included; excluding them barely matters (ARI 0.976).
- **Recency is not log-transformed** (modelling choice); log-transforming it gives best K = 2 and a different partition (ARI 0.318).
- 25% of raw rows lack a CustomerID; ~90% of customers are UK-based; about one year of data.
- Silhouette ≈ 0.42 = moderate separation; segment names are descriptive labels, not ground truth.

## 7. Team contribution
| Member | Register no. | Role | Contribution |
|---|---|---|---|
| Siva Balan S | RA2411026050230 | Data Preparation and Feature Engineering | Dataset sourcing and inspection, EDA, statistical summaries, data cleaning, preprocessing validation, construction and verification of the RFM features |
| Shabari Raj Kumar S | RA2411026050227 | Machine Learning and Model Evaluation | Feature transformation and scaling checks, K-Means workflow, candidate-K evaluation, validation metrics and stability checks |
| Shailesh M | RA2411026050223 | Dashboard Development and Business Interpretation | SegmentIQ dashboard development and testing, integration of verified outputs, segment visualisations, user explanations, business implications and limitations |

Shared: all three members contribute to final result verification, report review, presentation preparation and the project demonstration.

## Product intelligence, sales insights, market basket analysis and demand forecasting (added 2026-10-20)

Four dashboard sections were added after *Results & Insights*. They are **separate analyses** from the RFM / K-Means
segmentation, which is unchanged; they only reuse its final CustomerID-to-segment labels.

**One eligibility rule for all four sections.** They use the project's 392,692 retained purchase lines *minus* purchase lines
that a later cancellation reversed in full (same customer, StockCode, unit price and quantity; one-to-one, chronological).
`scripts/product_eligibility.py` finds them (2,769 lines, 223,470 units, £400,426.52) and writes `Product_Reversed_Purchase_Lines.csv`.
This stops cancelled bulk orders (e.g. 80,995 units cancelled 12 minutes later) from inflating product rankings and forecasts.
Result: 389,923 eligible lines, 18,394 invoices, 4,327 customers. All values are gross purchase value, not profit.
The segmentation keeps its own documented definition (all retained lines), so its totals differ by design.

| Section | Feature | Definition |
|---|---|---|
| Product Intelligence | Top products by units | Sum of Quantity per StockCode |
| | Most frequently purchased | Distinct invoices containing the StockCode |
| | Demand over time | Monthly Quantity sum, empty months = 0, Dec 2011 marked partial |
| | By customer segment | Lines joined to K-Means segment by CustomerID; rank by units, invoices or £ |
| | Repeat purchases | Customers buying the product in ≥ 2 distinct invoices ÷ customers buying it × 100 |
| Sales & Customer Insights | Top customers | Gross value per CustomerID, with invoices, units, last purchase, segment |
| | Average invoice value / units per invoice | Gross value (units) ÷ distinct invoices, overall and per month |
| | Country value | Gross value per Country (distinct from customers per country on Dataset & EDA) |
| | Monthly active customers | Distinct CustomerID per month |
| Market Basket Analysis | Apriori + association rules | One basket per invoice, product counted once; support, confidence, lift (standard definitions); itemsets up to 3 products; adjustable thresholds (defaults support 1.5 %, confidence 40 %, lift 2) |
| Demand Forecasting | Damped-trend Holt exponential smoothing | 12 full months (Dec 2010 – Nov 2011); holdout = last 3 full months, model fitted on the first 9 only; MAE/RMSE vs last-value and 3-month-average baselines; approximate 80 % interval; horizon 1–6 months; no forecast when < 8 active months or one month holds > 50 % of units |

Calculations live in `dashboard/products.py` (pure pandas/NumPy/SciPy, no new dependency); pages in `dashboard/views/`
(`product_intel.py`, `sales.py`, `basket.py`, `forecast.py`). Apriori is implemented directly and checked against
mlxtend in the tests (identical itemsets). Limitations: one year of data cannot establish yearly seasonality; association
rules show co-occurrence, not causation; forecasts are estimates to monitor, not guaranteed sales or stock levels.

Tests: `python verification/test_product_features.py` (40 checks: every metric recomputed independently, Apriori vs mlxtend,
forecast leakage test, page rendering) and `python verification/test_dashboard.py` (47 checks, now covering all 12 sections).
