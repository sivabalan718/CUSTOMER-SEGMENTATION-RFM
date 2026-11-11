"""Overview."""
import altair as alt
import pandas as pd
import streamlit as st

from .. import charts, core, explain, ui
from ..config import EXPECTED


def render(ctx, go):
    c, p = ctx.customers, ctx.profiles
    ui.header("Overview", "Customer Segmentation Overview",
              "An analytical view of customer purchasing behaviour using Recency, Frequency, Monetary features and K-Means clustering.")
    if ctx.validation.failed:
        ui.callout("Some data consistency checks did not pass, so figures may differ from the documented results.", "warn", label="Data check")

    lines_n = len(ctx.lines) if ctx.has_lines else (int(ctx.audit["Rows"].iloc[-1]) if ctx.audit is not None else None)
    sil = k_sel = None
    if ctx.eval5 is not None:
        best = ctx.eval5.loc[ctx.eval5["Silhouette Score"].idxmax()]
        k_sel, sil = int(best["K"]), float(best["Silhouette Score"])
    ui.kpis([
        ("Identified customers", f"{len(c):,}", "Customers with an ID and at least one retained purchase",
         "Rows without a CustomerID (about 25% of the raw data) cannot be linked to a customer and are not segmented."),
        ("Customer segments", f"{p['Cluster'].nunique()}", "Selected from K = 2 to 8 using validation metrics",
         "K = 3 achieved the best silhouette, Davies–Bouldin and Calinski–Harabasz scores among the values tested."),
        ("Retained purchase lines", f"{lines_n:,}" if lines_n else "—", f"Cleaned from {EXPECTED['raw_rows']:,} raw transaction lines",
         "After removing duplicates, rows without CustomerID, cancellations and non-positive quantities or prices."),
        ("Silhouette score", f"{sil:.4f}" if sil is not None else "—", f"K-Means at K = {k_sel}; range −1 to 1, higher is better" if sil else "",
         "How much closer each customer is to its own segment than to the nearest other segment. About 0.42 indicates moderate separation."),
    ])
    if ctx.has_lines:
        l = ctx.lines
        ui.strip([("Invoices", f"{l['InvoiceNo'].nunique():,}"), ("Gross positive purchase value", f"£{l['LineValue'].sum():,.0f}"),
                  ("Date coverage", f"{l['InvoiceDate'].min():%d %b %Y} – {l['InvoiceDate'].max():%d %b %Y}"),
                  ("Countries", f"{l['Country'].nunique()}"),
                  ("Median customer", f"{ctx.medians['Frequency']:.0f} invoices · £{ctx.medians['Monetary']:,.2f}")])
    ui.callout(ui.GROSS_DISCLAIMER, "warn", label="About the figures")

    ui.section("How the analysis works")
    steps = [(f"{EXPECTED['raw_rows']:,}", "raw transaction lines (UCI Online Retail)")]
    if lines_n:
        steps.append((f"{lines_n:,}", "clean positive purchase lines"))
    steps += [(f"{len(c):,}", "customers described by Recency, Frequency, Monetary"),
              (f"K = {p['Cluster'].nunique()}", "K-Means segments, chosen from K = 2 to 8")]
    ui.pipeline(steps)

    ui.section("Customer distribution", "Customers per segment")
    d = p[["Segment", "Cluster", "Customers", "Customer_Share_%", "Value_Share_%"]].copy()
    d["Label"] = d.apply(lambda r: f"{int(r.Customers):,} · {r['Customer_Share_%']:.1f}%", axis=1)
    d["Short"] = d["Segment"].map(ui.short)
    y = alt.Y("Short:N", sort=[ui.short(s) for s in ctx.segments], title=None)
    base = alt.Chart(d).encode(y=y, color=alt.Color("Segment:N", scale=charts.seg_scale(ctx), legend=None),
                               tooltip=["Segment", "Cluster", alt.Tooltip("Customers:Q", format=","),
                                        alt.Tooltip("Customer_Share_%:Q", format=".1f", title="% of customers"),
                                        alt.Tooltip("Value_Share_%:Q", format=".1f", title="% of gross value")])
    bars = base.mark_bar(height=22, cornerRadiusEnd=4).encode(x=alt.X("Customers:Q", title="Customers", scale=alt.Scale(domainMax=float(d["Customers"].max()) * 1.55)))
    text = base.mark_text(align="left", dx=6, color="#4A5468", fontSize=12).encode(x="Customers:Q", text="Label:N")
    charts.show((bars + text).properties(height=140), key="ov_dist")
    share = " · ".join(f"{ui.short(r.Segment)} {r['Value_Share_%']:.1f}%" for _, r in d.iterrows())
    st.markdown(f"<div class='small'>Share of gross purchase value — {share}</div>", unsafe_allow_html=True)
    ui.explainer("Understand Customer Distribution", explain.customer_distribution(p))
    ui.section("Segment comparison", "Median Recency, Frequency and Monetary per segment, in original units")
    pp = p.assign(Short=p["Segment"].map(ui.short))
    order = [ui.short(s) for s in ctx.segments]
    pcols = st.columns(3, gap="medium")
    for i, (pc, (f, title, fmt)) in enumerate(zip(pcols, [("Recency_Median", "Recency (days)", ",.0f"),
                                                           ("Frequency_Median", "Frequency (invoices)", ",.0f"),
                                                           ("Monetary_Median", "Monetary (£ gross)", ",.0f")])):
        b = alt.Chart(pp).encode(
            y=alt.Y("Short:N", sort=order, title=None, axis=alt.Axis() if i == 0 else None),
            x=alt.X(f"{f}:Q", title=title, axis=alt.Axis(tickCount=3), scale=alt.Scale(domainMax=float(pp[f].max()) * 1.35)),
            color=alt.Color("Segment:N", scale=charts.seg_scale(ctx), legend=None),
            tooltip=["Segment", alt.Tooltip(f"{f}:Q", format=fmt, title=title)])
        with pc:
            charts.show((b.mark_bar(height=22, cornerRadiusEnd=4) +
                         b.mark_text(align="left", dx=4, color="#4A5468", fontSize=11.5).encode(text=alt.Text(f"{f}:Q", format=fmt)))
                        .properties(height=160), key=f"ov_cmp_{i}")
    st.markdown("<div class='small'>Lower Recency means a more recent purchase. Each panel has its own scale.</div>",
                unsafe_allow_html=True)
    ref = (f"{(ctx.lines['InvoiceDate'].max().normalize() + pd.Timedelta(days=1)):%d %B %Y}" if ctx.has_lines
           else EXPECTED["reference_date"])
    ui.explainer("Understand Recency, Frequency and Monetary Values", explain.rfm_comparison(p, ref))

    ui.section("Main findings")
    for f in core.key_findings(c, p, ctx.lines)[:3]:
        ui.callout(f, "calculated")
