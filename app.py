"""SegmentIQ — Customer Segmentation Intelligence (Streamlit dashboard).

Run from this folder:   streamlit run app.py
Data files are looked up in $SEGMENTATION_DATA_DIR, this folder, then ../03_Datasets_and_Results.
The app only reads the project's saved results; it never retrains the K-Means model.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make the local dashboard package importable on every Streamlit version
from dashboard import APP_NAME, APP_TITLE, ui  # noqa: E402
from dashboard.core import DataError  # noqa: E402
from dashboard.views import (about, basket, dataset, evaluation, forecast, overview, preprocessing, product_intel, results,  # noqa: E402
                             rfm, sales, segmentation)

st.set_page_config(page_title=f"{APP_NAME} · {APP_TITLE}", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

NAV = [("overview", "Overview", ":material/space_dashboard:", overview),
       ("dataset", "Dataset & EDA", ":material/table_chart:", dataset),
       ("preprocessing", "Preprocessing", ":material/filter_alt:", preprocessing),
       ("rfm", "RFM Analysis", ":material/bar_chart:", rfm),
       ("segmentation", "Customer Segmentation", ":material/scatter_plot:", segmentation),
       ("evaluation", "Model Evaluation", ":material/analytics:", evaluation),
       ("results", "Results & Insights", ":material/lightbulb:", results),
       ("products", "Product Intelligence", ":material/inventory_2:", product_intel),
       ("sales", "Sales & Customer Insights", ":material/payments:", sales),
       ("basket", "Market Basket Analysis", ":material/shopping_basket:", basket),
       ("forecast", "Demand Forecasting", ":material/trending_up:", forecast),
       ("about", "About the Project", ":material/info:", about)]
PAGES = {key: module for key, _, _, module in NAV}


def go(page):
    st.session_state["page"] = page


st.session_state.setdefault("page", "overview")
if st.session_state["page"] not in PAGES:
    st.session_state["page"] = "overview"
ui.brand(APP_NAME, f"{APP_TITLE}<br>UCI Online Retail · 2010–2011")
for key, label, icon, _ in NAV:
    st.sidebar.button(label, key=f"nav_{key}", icon=icon, on_click=go, args=(key,), **ui.FULL,
                      type="primary" if st.session_state["page"] == key else "tertiary")
ui.sidebar_footer("Machine Learning case study · RFM features and K-Means clustering")

try:
    from dashboard.state import get_context
    ctx = get_context()
except DataError as e:
    ui.header("Setup", "The project data could not be loaded")
    ui.callout(f"{e}. Keep the project's result files in the same folder as app.py (or in ../03_Datasets_and_Results) and restart "
               "the application.", "warn", label="Missing data")
    st.stop()

PAGES[st.session_state["page"]].render(ctx, go)
