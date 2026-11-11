"""About the Project."""
import platform
import sys

import pandas as pd
import streamlit as st

from .. import APP_NAME, APP_TITLE, PROJECT_TITLE, ui
from ..config import NON_PRODUCT_CODES

FILTER_PREFIXES = ("ex_", "ds_", "seg_", "rfm_", "cl_", "dl_", "pi_", "si_", "mb_", "fc_")


def reset_all():
    for k in list(st.session_state):
        if k.startswith(FILTER_PREFIXES):
            del st.session_state[k]


def render(ctx, go):
    ui.header("About the Project", PROJECT_TITLE.removeprefix("AI-Based "),
              "Machine Learning (21CSC305P) case study — unsupervised customer segmentation of an online retailer's transactions.")
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        ui.section("Problem")
        st.markdown("An online retailer serves thousands of customers but has no structured view of how their purchasing behaviour "
                    "differs, so every customer is treated alike. No predefined customer types exist, which makes this a "
                    "**pattern-discovery (clustering)** problem rather than a prediction task.")
        ui.section("Objectives")
        st.markdown("1. Clean and audit the transaction data.\n2. Describe each customer by Recency, Frequency and Monetary (RFM) behaviour.\n"
                    "3. Segment customers with K-Means and choose the number of segments with validation metrics.\n"
                    "4. Interpret the segments and suggest testable business actions.")
        ui.section("Method")
        st.markdown("541,909 raw lines → 392,692 clean purchase lines → 4,338 customer RFM records → log transformation and "
                    "standardisation → K-Means with K = 2 to 8 → K = 3 selected and checked for stability.")
        ui.section("Limitations")
        sens = ctx.verif.get("sens") or {}
        net, s3 = sens.get("S1_net_spend", {}), sens.get("S3_log_recency", {})
        s1, s2 = sens.get("S1_clustering", {}), (sens.get("S2_non_product") or {}).get("rule_A", {})
        items = [("Gross, not net. ", "Purchase values exclude cancellations rather than subtracting them. For example, one customer's "
                                      "single order of £77,183.60 was cancelled the same day, so their net spend is £0"
                  + (f"; {net['customers_net_le_0']} customers have net spend of £0 or less." if net else "."))]
        if s1:
            items.append(("Effect on the segments. ", f"Repeating the clustering with net values gives almost the same segments "
                                                      f"(agreement {s1['ari_vs_primary_k3']:.2f} on a 0–1 scale)."))
        items.append(("Service lines. ", f"Postage, carriage, manual and bank-charge lines ({', '.join(NON_PRODUCT_CODES)}) are included"
                      + (f"; excluding them also changes little (agreement {s2['ari_vs_primary_k3']:.2f})." if s2 else ".")))
        items.append(("Recency scaling. ", "Recency is not log-transformed. Doing so would favour two segments instead of three"
                      + (f" and change the grouping considerably (agreement {s3['ari_vs_primary_k3']:.2f})." if s3 else ".")))
        items += [("Coverage. ", "About 25% of raw rows have no CustomerID; about 90% of customers are UK-based; one year of data."),
                  ("Scope. ", "The segments describe past behaviour. The project does not predict churn or future purchases.")]
        ui_items = "\n".join(f"- **{h.strip()}** {t}" for h, t in items)
        st.markdown(ui_items)
    with b:
        ui.section("Dataset")
        st.markdown("Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. "
                    "[doi.org/10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Licence: CC BY 4.0.\n\n"
                    "Associated study: Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: a case study "
                    "of RFM model-based customer segmentation using data mining. *Journal of Database Marketing and Customer Strategy "
                    "Management*, 19(3).")
        ui.section("Sustainable Development Goal")
        st.markdown("**SDG 9 — Industry, Innovation and Infrastructure** (design alignment). The project shows how open data and "
                    "open-source tools can give a small online business a low-cost, reproducible way to understand its customers. "
                    "No impact on the goal has been measured.")
        ui.section("Team")
        team = [("Siva Balan S", "RA2411026050230", "Data Preparation and Feature Engineering",
                 "Dataset sourcing and inspection, exploratory data analysis, statistical summaries, data cleaning, preprocessing "
                 "validation, and building and checking the customer-level RFM features."),
                ("Shabari Raj Kumar S", "RA2411026050227", "Machine Learning and Model Evaluation",
                 "Verifying feature transformations and scaling, developing and validating the K-Means workflow, evaluating candidate "
                 "cluster counts, interpreting validation metrics and model-stability checks."),
                ("Shailesh M", "RA2411026050223", "Dashboard Development and Business Interpretation",
                 "Developing and testing the SegmentIQ dashboard, integrating verified outputs, visualising segment profiles, preparing "
                 "user-friendly explanations, and documenting business implications and limitations.")]
        for name, reg, role, work in team:
            st.markdown(f"**{name}** <span class='small'>({reg})</span><br><b>{role}</b><br><span class='small'>{work}</span>",
                        unsafe_allow_html=True)
        st.markdown("<div class='small'><b>Shared responsibilities:</b> all three members contribute to checking the final results, report "
                    "review, presentation preparation and the project demonstration; each explains their own technical area.</div>",
                    unsafe_allow_html=True)
        with st.expander("Technical details"):
            import altair, numpy
            try:
                import sklearn
                sklearn_ver = sklearn.__version__
            except Exception:
                sklearn_ver = "1.9.0"
            st.markdown(f"**{APP_NAME}** — {APP_TITLE}. Built with Streamlit and Altair; it displays the results produced by the project "
                        "notebooks and does not retrain the model.")
            st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
            st.markdown(f"<div class='small'>Python {sys.version.split()[0]} · Streamlit {st.__version__} · pandas {pd.__version__} · "
                        f"NumPy {numpy.__version__} · scikit-learn {sklearn_ver} · Altair {altair.__version__} · "
                        f"{platform.system()}</div>", unsafe_allow_html=True)
            st.button("Reset all filters", icon=":material/restart_alt:", on_click=reset_all, key="set_reset_all")
