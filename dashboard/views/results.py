"""Results & Insights."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, core, ui
from .lookup import current_filters, describe_filters

PLAYBOOK = {
    "Recent, Frequent, High-Spend Customers": {
        "objective": "Retain the customers who generate most of the gross purchase value.",
        "action": "Loyalty recognition (priority service, early access to new ranges) and relevant cross-selling.",
        "why": "They bought recently, order often and account for the largest share of value.",
        "measure": "Repeat-order rate and gross value per customer, compared with a hold-out group.",
        "risk": "Includes a few customers whose large orders were later cancelled; check net value before rewarding individuals.",
    },
    "Occasional / Lower-Spend Customers": {
        "objective": "Encourage a further order from customers who buy occasionally and spend below the median.",
        "action": "Personalised product recommendations or a time-limited offer on the next order.",
        "why": "They are about as recent as the typical customer but order less often and spend less.",
        "measure": "Share placing another order within 60 days, compared with a hold-out group.",
        "risk": "Offers may reward orders that would have happened anyway; only a controlled test can tell.",
    },
    "At-Risk / Inactive, Low-Frequency Customers": {
        "objective": "Test whether inactive, low-frequency customers can be re-engaged at low cost.",
        "action": "A low-cost reminder or ‘new arrivals’ message to a random half, keeping the other half as a control group.",
        "why": "Their median last purchase was over eight months ago and most bought only once.",
        "measure": "Share purchasing again within 90 days, contacted group versus control.",
        "risk": "RFM cannot tell lapsed customers from naturally infrequent ones, so expect a low response.",
    },
}
PROFILE_COLUMNS = {"Cluster": "Cluster", "Segment": "Segment", "Customers": "Customers", "Customer_Share_%": "Share of customers (%)",
                   "Value_Share_%": "Share of gross value (%)", "Recency_Median": "Median recency (days)",
                   "Frequency_Median": "Median frequency (invoices)", "Monetary_Median": "Median gross value (GBP)",
                   "Monetary_Mean": "Mean gross value (GBP)"}


def summary_markdown(ctx):
    p = ctx.profiles
    lines = ["# Customer segmentation — summary of results", "",
             f"Customers segmented: {len(ctx.customers):,} · segments: {p['Cluster'].nunique()} (K-Means)", "",
             "| Segment | Customers | % customers | % gross value | Median recency (days) | Median frequency | Median gross value (£) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in p.iterrows():
        lines.append(f"| {r['Segment']} | {int(r['Customers']):,} | {r['Customer_Share_%']:.1f} | {r['Value_Share_%']:.1f} | "
                     f"{r['Recency_Median']:.0f} | {r['Frequency_Median']:.0f} | {r['Monetary_Median']:,.2f} |")
    if ctx.eval5 is not None:
        k = ctx.eval5.loc[ctx.eval5["Silhouette Score"].idxmax()]
        lines += ["", f"Model: K = {int(k['K'])}; silhouette {k['Silhouette Score']:.4f}; Davies–Bouldin {k['Davies-Bouldin Index']:.4f}; "
                      f"Calinski–Harabasz {k['Calinski-Harabasz Score']:,.2f}."]
    lines += ["", "## Key findings"] + [f"- {f}" for f in core.key_findings(ctx.customers, p, ctx.lines)]
    lines += ["", "## Notes", f"- {ui.GROSS_DISCLAIMER}", "- Segment names are descriptive labels, not ground-truth categories.",
              "- Suggested actions are hypotheses to test, not measured outcomes."]
    return "\n".join(lines) + "\n"


def render(ctx, go):
    c, p = ctx.customers, ctx.profiles
    ui.header("Results & Insights", "What the Segments Mean",
              "Connecting the clustering results to the retail problem: who the segments are, how they buy and what could be tested next.")
    top = p.sort_values("Value_Share_%", ascending=False).iloc[0]
    ui.lead(f"The {ui.short(top['Segment']).lower()} segment holds {int(top['Customers']):,} customers ({top['Customer_Share_%']:.1f}%) "
            f"with a median gross purchase value of £{top['Monetary_Median']:,.2f}, and accounts for {top['Value_Share_%']:.1f}% of all "
            "gross purchase value.")
    ui.section("Segment summary")
    cols = st.columns(len(p), gap="medium")
    for col, (_, r) in zip(cols, p.iterrows()):
        with col:
            ui.segment_card(r, ctx.colors[r["Segment"]])
    ui.section("Key findings")
    for f in core.key_findings(c, p, ctx.lines)[:3]:
        ui.callout(f, "calculated")

    if ctx.has_lines:
        ui.section("Purchasing behaviour by segment", "Transaction-level comparison")
        inv = ctx.invoices
        seg_cmp = inv.groupby("Segment").agg(Invoices=("InvoiceNo", "count"), Median_value=("Value", "median"),
                                             Median_items=("Items", "median"), Median_lines=("Lines", "median")).reindex(ctx.segments).reset_index()
        seg_cmp["Segment"] = seg_cmp["Segment"].map(ui.short)
        st.dataframe(seg_cmp, hide_index=True, **ui.FULL, column_config={
            "Invoices": st.column_config.NumberColumn(format="%d"),
            "Median_value": st.column_config.NumberColumn("Median invoice (£)", format="£%.2f", help="Gross positive value per invoice"),
            "Median_items": st.column_config.NumberColumn("Median units", format="%.0f", help="Units per invoice"),
            "Median_lines": st.column_config.NumberColumn("Median lines", format="%.0f", help="Product lines per invoice")})
        iv = inv.assign(logv=np.log10(inv["Value"].clip(lower=0.01)))
        charts.show(alt.Chart(iv).mark_bar(opacity=.85).encode(
            x=alt.X("logv:Q", bin=alt.Bin(maxbins=40), title="Invoice value (£, log scale: 2 = £100, 3 = £1,000)"),
            y=alt.Y("count():Q", title="Invoices", stack=True),
            color=alt.Color("Segment:N", scale=charts.seg_scale(ctx), legend=alt.Legend(title=None))
        ).properties(title="Distribution of invoice values", height=260), key="rs_inv")
        conc = core.concentration(c)
        lz = conc["lorenz"]
        eq = pd.DataFrame({"customer_share": [0, 1], "value_share": [0, 1]})
        charts.show((alt.Chart(lz).mark_area(color="#6655D9", opacity=.18, line={"color": "#6655D9"}).encode(
            x=alt.X("customer_share:Q", title="Share of customers (lowest spenders first)", axis=alt.Axis(format="%")),
            y=alt.Y("value_share:Q", title="Share of gross purchase value", axis=alt.Axis(format="%")))
            + alt.Chart(eq).mark_line(color="#9AA3B5", strokeDash=[4, 4]).encode(x="customer_share:Q", y="value_share:Q")
        ).properties(title="Spending concentration (Lorenz curve)", height=280), key="rs_lorenz")
        ui.callout(f"The top 10% of customers account for {100 * conc['top_share'][10]:.1f}% of gross purchase value and the top 1% "
                   f"for {100 * conc['top_share'][1]:.1f}% (Gini coefficient {conc['gini']:.2f}).", "calculated")

    ui.section("Possible retail applications", "Suggestions derived from each segment's observed behaviour — hypotheses to test, not proven outcomes")
    med = ctx.medians
    for _, r in p.iterrows():
        seg, color = r["Segment"], ctx.colors[r["Segment"]]
        play = PLAYBOOK.get(seg)
        with st.container(border=True):
            st.markdown(f"<div style='font-weight:650;font-size:15.5px;margin-bottom:6px'>{ui.seg_label(seg, color)}</div>",
                        unsafe_allow_html=True)
            a, b = st.columns([1, 1.25], gap="large")
            with a:
                st.markdown(ui.tag("Observed", "observed"), unsafe_allow_html=True)
                for line in core.interpret_segment(r, med):
                    st.markdown(f"<div class='small' style='margin:4px 0'>• {line}</div>", unsafe_allow_html=True)
            with b:
                if play is None:
                    st.markdown("<div class='small'>No suggestion is offered for this segment.</div>", unsafe_allow_html=True)
                    continue
                st.markdown(ui.tag("Suggestion", "recommendation"), unsafe_allow_html=True)
                st.markdown(f"**{play['action']}**\n\n<span class='small'>Objective: {play['objective']} Why: {play['why']} "
                            f"Measure: {play['measure']}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='small' style='color:#8F5E12;margin-top:6px'>Uncertainty: {play['risk']}</div>", unsafe_allow_html=True)
    ui.callout("No campaign has been run on this data, so no effect on revenue or retention is claimed.", "warn", label="Note")

    ui.section("Limitations")
    st.markdown("- The segmentation describes past behaviour (Dec 2010 – Dec 2011); it does not predict churn or future purchases.\n"
                f"- {ui.GROSS_DISCLAIMER}\n"
                "- Customers without an ID (about a quarter of raw rows) are not represented; about 90% of customers are UK-based.\n"
                "- A silhouette score of about 0.42 indicates moderate separation: customers near a boundary could reasonably belong to "
                "either neighbouring segment.")

    with st.expander("Download results"):
        f = current_filters(ctx)
        filtered = core.filter_customers(c, f)
        items = [("Customer segment assignments", "Every customer with segment and RFM values",
                  core.to_csv_bytes(core.export_customers(c)), "customer_segments.csv", "text/csv", "dl_all"),
                 ("Customers matching the lookup filters", describe_filters(f),
                  core.to_csv_bytes(core.export_customers(filtered)), "filtered_customers.csv", "text/csv", "dl_filtered"),
                 ("Segment profiles", "Counts, shares and median values per segment",
                  core.to_csv_bytes(p[list(PROFILE_COLUMNS)].rename(columns=PROFILE_COLUMNS)), "segment_profiles.csv", "text/csv", "dl_profiles")]
        if ctx.eval5 is not None:
            items.append(("Model evaluation (K = 2 to 8)", "Inertia and the three validation metrics",
                          core.to_csv_bytes(ctx.eval5), "kmeans_evaluation.csv", "text/csv", "dl_eval5"))
        items.append(("Summary of results", "Short written summary (Markdown)", summary_markdown(ctx).encode("utf-8"),
                      "results_summary.md", "text/markdown", "dl_summary"))
        for label, desc, data, fname, mime, key in items:
            a, b = st.columns([3, 1])
            a.markdown(f"**{label}**<br><span class='small'>{desc}</span>", unsafe_allow_html=True)
            b.download_button("Download", data, file_name=fname, mime=mime, key=key, icon=":material/download:", **ui.FULL)
