"""Customer Segmentation."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, core, ui
from . import lookup


@st.cache_data(show_spinner=False)
def _pca(_key, _customers):
    return core.pca_projection(_customers)


def render(ctx, go):
    c, p, med = ctx.customers, ctx.profiles, ctx.medians
    ui.header("Customer Segmentation", "K-Means Customer Segments",
              "The output of the K-Means model: how many customers fall into each segment, where the segments sit and how they differ.")
    ui.lead(f"K-Means identified {p['Cluster'].nunique()} customer segments from {len(c):,} customer records.")
    cols = st.columns(len(p), gap="medium")
    for col, (_, row) in zip(cols, p.iterrows()):
        with col:
            ui.segment_card(row, ctx.colors[row["Segment"]])
    ui.callout("Segment names are descriptive interpretations of each cluster's median Recency, Frequency and Monetary values. They are "
               "not labels that exist in the dataset, and cluster numbers do not imply a ranking.", "observed", label="About the names")

    ui.section("Cluster visualisation", "Every customer, coloured by segment")
    coords, ratio2, loadings, ratio = _pca((len(c), int(c["CustomerID"].sum()), int(c["Cluster"].fillna(-1).sum())), c)
    d = c[["CustomerID", "Cluster", "Segment", "Recency", "Frequency", "Monetary"]].copy()
    d["Component 1"], d["Component 2"] = coords[:, 0], coords[:, 1]
    view = st.segmented_control("View", ["PCA projection", "Recency vs Monetary"], default="PCA projection", key="cl_view") or "PCA projection"
    v2, v3 = st.columns([2.2, 1], gap="medium", vertical_alignment="bottom")
    hl = v2.multiselect("Highlight segments", ctx.segments, default=ctx.segments, key="cl_hl", format_func=ui.short)
    show_c = v3.toggle("Segment centres", value=True, key="cl_centres")
    if view.startswith("PCA"):
        st.markdown(f"<div class='small'>The two components shown capture <b>{100 * ratio2.sum():.1f}%</b> of the variation in the "
                    f"three model features (component 1: {100 * ratio2[0]:.1f}%, component 2: {100 * ratio2[1]:.1f}%).</div>",
                    unsafe_allow_html=True)
    hl = hl or []
    if view.startswith("PCA"):
        ch = charts.scatter(ctx, d, "Component 1", "Component 2", "Customers in two dimensions", highlight=hl, size=12)
        centre_cols = ("Component 1", "Component 2")
    else:
        ch = charts.scatter(ctx, d, "Recency", "Monetary", "Recency (days) against Monetary (£, log scale)", logy=True, highlight=hl, size=12)
        centre_cols = ("Recency", "Monetary")
    if show_c:
        cen = d.groupby("Segment")[list(centre_cols)].median().reset_index()
        cen["Name"] = cen["Segment"].map(ui.short)
        base = alt.Chart(cen).encode(x=f"{centre_cols[0]}:Q", y=f"{centre_cols[1]}:Q")
        ch = ch + base.mark_point(shape="diamond", size=320, filled=True, stroke="#FFFFFF", strokeWidth=2).encode(
            color=alt.Color("Segment:N", scale=charts.seg_scale(ctx), legend=None), tooltip=[alt.Tooltip("Segment:N", title="Centre of")]) \
            + base.mark_text(dy=-16, fontSize=12, fontWeight=600, color="#182033").encode(text="Name:N")
    charts.show(ch, key="cl_scatter")
    ui.callout("The model clusters customers using all three features (Recency, log Frequency, log Monetary). The chart squeezes those "
               "three dimensions into two with principal component analysis (PCA) purely for display — PCA does not perform the "
               "clustering, and a flat projection can make segments look more or less separated than they are.", "observed", label="How to read")
    with st.expander("Technical details"):
        lt = loadings.reset_index().rename(columns={"index": "Model feature", "PC1": "Component 1", "PC2": "Component 2"})
        lt["Model feature"] = lt["Model feature"].map({"Recency": "Recency (standardised)", "Frequency": "log Frequency (standardised)",
                                                       "Monetary": "log Monetary (standardised)"})
        st.dataframe(lt, hide_index=True, **ui.FULL, column_config={"Component 1": st.column_config.NumberColumn(format="%+.3f"),
                                                                          "Component 2": st.column_config.NumberColumn(format="%+.3f")})
        st.markdown("<div class='small'>Loadings are the weights of each standardised feature in a component. Component 1 combines all "
                    "three features; component 2 is dominated by Recency. Centres shown are segment medians.</div>", unsafe_allow_html=True)

    ui.section("Segment profiles", "Measured statistics for each segment")
    rows = [("Cluster ID", lambda r: f"{int(r['Cluster'])}"), ("Customers", lambda r: f"{int(r['Customers']):,}"),
            ("Share of customers", lambda r: f"{r['Customer_Share_%']:.1f}%"), ("Median recency", lambda r: f"{r['Recency_Median']:.0f} days"),
            ("Median frequency", lambda r: f"{r['Frequency_Median']:.0f} invoices"),
            ("Median gross value", lambda r: f"£{r['Monetary_Median']:,.2f}"), ("Share of gross value", lambda r: f"{r['Value_Share_%']:.1f}%")]
    table = pd.DataFrame({"Measure": [n for n, _ in rows],
                          **{ui.short(r["Segment"]): [f(r) for _, f in rows] for _, r in p.iterrows()}})
    st.dataframe(table, hide_index=True, **ui.FULL)
    bcols = st.columns(3, gap="medium")
    for col, f in zip(bcols, core.CORE):
        with col:
            charts.show(charts.boxplot(ctx, c, f, {"Recency": "Recency (days)", "Frequency": "Frequency", "Monetary": "Monetary (£)"}[f],
                                       log=f != "Recency"), key=f"seg_box_{f}")
    st.markdown("<div class='small'>Boxes show the middle 50% of customers and the line the median. Frequency and Monetary use a "
                "compressed (symmetric log) scale.</div>", unsafe_allow_html=True)

    ui.section("Segment detail", "One segment compared with all other customers")
    seg = st.pills("Segment", ctx.segments, default=ctx.segments[0], key="seg_pick",
                   format_func=lambda s: f"{int(p.set_index('Segment').loc[s, 'Cluster'])} · {ui.short(s)}") or ctx.segments[0]
    row = p.set_index("Segment").loc[seg]
    color = ctx.colors[seg]
    st.markdown(f"<div class='small'>{ui.seg_label(seg, color)} · {int(row['Customers']):,} customers ({row['Customer_Share_%']:.1f}%)</div>",
                unsafe_allow_html=True)
    stats = pd.DataFrame({
        "Statistic": ["Mean", "Median", "25th percentile", "75th percentile", "All customers (median)"],
        "Recency (days)": [row["Recency_Mean"], row["Recency_Median"], row["Recency_P25"], row["Recency_P75"], med["Recency"]],
        "Frequency (invoices)": [row["Frequency_Mean"], row["Frequency_Median"], row["Frequency_P25"], row["Frequency_P75"], med["Frequency"]],
        "Monetary (£ gross)": [row["Monetary_Mean"], row["Monetary_Median"], row["Monetary_P25"], row["Monetary_P75"], med["Monetary"]]})
    st.dataframe(stats, hide_index=True, **ui.FULL, column_config={
        "Recency (days)": st.column_config.NumberColumn(format="%.1f"), "Frequency (invoices)": st.column_config.NumberColumn(format="%.2f"),
        "Monetary (£ gross)": st.column_config.NumberColumn(format="£%.2f")})
    l1, l2 = st.columns([1, 1.3], gap="large")
    with l1:
        st.markdown(ui.tag("Interpretation", "calculated"), unsafe_allow_html=True)
        for line in core.interpret_segment(row, med):
            st.markdown(f"<div class='small' style='margin:6px 0'>• {line}</div>", unsafe_allow_html=True)
    with l2:
        feat = st.segmented_control("Feature", core.CORE, default="Monetary", key="seg_feat") or "Monetary"
        dd = c.assign(Group=c["Segment"].where(c["Segment"] != seg, ui.short(seg)).where(c["Segment"] == seg, "All other customers"))
        log = feat != "Recency"
        dd["value"] = np.log1p(dd[feat]) if log else dd[feat]
        charts.show(alt.Chart(dd).transform_joinaggregate(total="count()", groupby=["Group"]).transform_calculate(pct="1 / datum.total")
                    .mark_bar(opacity=.8, binSpacing=0).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=35), title=f"log(1 + {feat})" if log else f"{feat} (days)"),
            y=alt.Y("sum(pct):Q", stack=None, title="Share of group", axis=alt.Axis(format="%")),
            color=alt.Color("Group:N", scale=alt.Scale(domain=[ui.short(seg), "All other customers"], range=[color, "#C9CFDC"]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("sum(pct):Q", format=".1%", title="Share")]).properties(height=280), key="seg_dist")

    ui.section("Customer lookup", "Search individual customers and their segment assignment")
    lookup.render(ctx)
