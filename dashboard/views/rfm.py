"""RFM Analysis."""
import altair as alt
import pandas as pd
import streamlit as st

from .. import charts, core, ui
from ..config import UNITS


def render(ctx, go):
    c = ctx.customers
    ui.header("RFM Analysis", "Customer RFM Features",
              "Each customer's transactions are summarised by three behavioural features, which are the inputs to the clustering model.")
    ui.kpis([("Recency", "How recently?", "Days between the customer's last purchase and 10 Dec 2011, the day after the last invoice. "
                                           "Lower means more recent."),
             ("Frequency", "How often?", "Number of distinct invoices the customer placed (not the number of invoice lines)."),
             ("Monetary", "How much?", "Gross positive purchase value in GBP: the sum of quantity × unit price over retained lines.")], cols=3)
    st.markdown(f"<div class='small'>Modelling population: <b>{len(c):,} customers</b>, one row per customer.</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Distributions", "Relationships", "Model inputs"])
    with t1:
        f = st.segmented_control("Feature", core.CORE, default="Monetary", key="rfm_feat") or "Monetary"
        scale = st.segmented_control("Display scale", ["Original units", "Log scale (model)"], default="Log scale (model)",
                                     key="rfm_scale", help="The model uses log(1 + x) for Frequency and Monetary; Recency stays in days.")             or "Original units"
        use_log = scale != "Original units" and f != "Recency"
        charts.show(charts.histogram(ctx, c, f, f"Distribution of {f}{' (log scale)' if use_log else ''}", log=use_log, by_segment=True,
                                     x_title=f"log(1 + {f})" if use_log else f"{f} ({UNITS[f]})").properties(height=300), key="rfm_hist")
        if f == "Recency" and scale != "Original units":
            st.markdown("<div class='small'>Recency is not log-transformed by the model, so it is shown in days.</div>", unsafe_allow_html=True)
        sk = c[core.CORE].skew()
        ui.callout(f"Frequency and Monetary are very right-skewed (skewness {sk['Frequency']:.1f} and {sk['Monetary']:.1f}): most customers "
                   "order a few times and spend modestly, while a few order very often or spend very large amounts. Recency is less skewed.",
                   "observed")
        with st.expander("Summary statistics"):
            desc = c[core.CORE].describe().T[["mean", "50%", "std", "min", "max"]].rename(columns={"50%": "median"}).assign(skewness=sk)
            st.dataframe(desc.reset_index().rename(columns={"index": "Feature"}), hide_index=True, **ui.FULL,
                         column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ["mean", "median", "std", "min", "max", "skewness"]})
    with t2:
        a, b = st.columns(2, gap="medium")
        x = a.selectbox("Horizontal axis", core.CORE, index=0, key="rfm_x")
        y = b.selectbox("Vertical axis", core.CORE, index=2, key="rfm_y")
        d, e, _ = st.columns([1, 1, 2])
        logx = d.toggle("Log x-axis", value=x != "Recency", key="rfm_logx")
        logy = e.toggle("Log y-axis", value=y != "Recency", key="rfm_logy")
        charts.show(charts.scatter(ctx, c, x, y, f"{y} against {x}", logx=logx, logy=logy), key="rfm_scatter")
        corr = c[core.CORE].corr(method="spearman")
        long = corr.reset_index().melt(id_vars="index", var_name="B", value_name="r").rename(columns={"index": "A"})
        charts.show(charts.heatmap(long, "B", "A", "r", "Rank correlation (Spearman)", scheme="redblue", domain=[-1, 1]), key="rfm_heat")
        ui.callout(f"Customers who order more often also spend more (correlation {corr.loc['Frequency', 'Monetary']:.2f}). Both are "
                   f"negatively related to Recency ({corr.loc['Recency', 'Frequency']:.2f} and {corr.loc['Recency', 'Monetary']:.2f}): "
                   "frequent, high-spending customers tend to have bought recently. These are associations, not causes.", "calculated")
    with t3:
        Z, X, mean, std = core.model_features(c)
        st.markdown("Before clustering, the features are transformed in two steps so that no single feature dominates the distance "
                    "calculations:\n\n1. **Log transformation** of Frequency and Monetary, log(1 + x), to reduce extreme skew. "
                    "Recency stays in days.\n2. **Standardisation**: each feature is rescaled to mean 0 and standard deviation 1.")
        zl = Z.assign(Segment=c["Segment"].values).melt(id_vars="Segment", var_name="Feature", value_name="z")
        charts.show(alt.Chart(zl).mark_boxplot(extent=1.5, size=22, median={"color": "#182033"}).encode(
            x=alt.X("Feature:N", title=None, sort=core.CORE, axis=alt.Axis(labelAngle=0)), xOffset="Segment:N",
            y=alt.Y("z:Q", title="Standardised value"), color=alt.Color("Segment:N", scale=charts.seg_scale(ctx), legend=alt.Legend(title=None))
        ).properties(title="Standardised model inputs by segment", height=320), key="rfm_z")
        st.markdown("<div class='small'>Standardised values are unit-free; compare segments within a feature, not across features.</div>",
                    unsafe_allow_html=True)
        with st.expander("Technical details"):
            st.dataframe(pd.DataFrame({"Feature": core.CORE, "Transformation": ["none (days)", "log(1 + x)", "log(1 + x)"],
                                       "Mean": mean.values, "Std. deviation": std.values}),
                         hide_index=True, **ui.FULL,
                         column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ["Mean", "Std. deviation"]})
            st.markdown("<div class='small'>Mean and standard deviation are measured after the transformation and are the values "
                        "used for standardisation.</div>", unsafe_allow_html=True)
            cid = st.selectbox("Inspect one customer's model inputs", c["CustomerID"].tolist(), key="rfm_cid")
            i = c.index[c["CustomerID"] == cid][0]
            row = c.loc[i]
            st.dataframe(pd.DataFrame({"Feature": core.CORE, "Original value": [row[f] for f in core.CORE],
                                       "After log step": [X.loc[i, f] for f in core.CORE], "Standardised value": [Z.loc[i, f] for f in core.CORE]}),
                         hide_index=True, **ui.FULL,
                         column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["Original value", "After log step", "Standardised value"]})
            st.markdown(f"<div class='small'>Customer {int(cid)} belongs to <b>{row['Segment']}</b> (cluster {row['Cluster']}).</div>",
                        unsafe_allow_html=True)
            ui.callout("Recency is standardised but not log-transformed. This is a modelling choice: log-transforming Recency as well "
                       "produces a noticeably different segmentation (see About the Project → Limitations).", "warn", label="Modelling choice")
