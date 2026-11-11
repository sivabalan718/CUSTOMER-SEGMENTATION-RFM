"""Product Intelligence: quantity and invoice rankings, demand over time, segment preferences, repeat purchases."""
import streamlit as st

from .. import charts, products as P, ui
from ..config import SEGMENT_ORDER
from ..state import get_products


def eligibility_note(pdata):
    i = pdata.info
    if i["rule_applied"]:
        ui.callout(f"Based on {i['eligible_lines']:,} eligible purchase lines ({i['invoices']:,} invoices, {i['customers']:,} customers). "
                   f"{i['reversed_lines']:,} purchase lines ({i['reversed_units']:,} units) that were later cancelled in full are excluded, "
                   "so cancelled bulk orders cannot inflate these results. Values are gross purchase value, not profit.", "observed",
                   label="Data used")
    else:
        ui.callout("The cancelled-purchase file (Product_Reversed_Purchase_Lines.csv) is missing, so purchases that were later cancelled "
                   "are still included. Run product_eligibility.py to create it.", "warn")


def render(ctx, go):
    ui.header("Product Intelligence", "Which Products Customers Buy",
              "Product rankings by units and by invoices, demand over time, segment preferences and repeat purchasing.")
    pdata = get_products()
    if pdata is None:
        ui.unavailable("The purchase-line file")
        return
    e, names = pdata.lines, pdata.names
    eligibility_note(pdata)
    t1, t2, t3, t4 = st.tabs(["Top products", "Demand over time", "By customer segment", "Repeat purchases"])

    with t1:
        n = st.slider("Number of products", 5, 25, 10, key="pi_n")
        a, b = st.columns(2, gap="large")
        q = P.top_by_quantity(e, names, n)
        with a:
            charts.show(charts.hbar(q, "Product", "Units", f"Top {n} products by units purchased", "#6655D9", "Units purchased",
                                    tooltip=["Product", "StockCode", "Units", "Invoices"]), key="pi_q")
            ui.explainer("What does this chart show?", "This chart shows which products had the highest total recorded **units** "
                         "purchased between December 2010 and December 2011. It measures the number of units, not money earned or profit. "
                         "A product bought in very large wholesale quantities by a few customers can rank high here.")
        iv = P.top_by_invoices(e, names, n)
        with b:
            charts.show(charts.hbar(iv, "Product", "Invoices", f"Top {n} products by number of invoices", "#2E9BB0",
                                    "Invoices containing the product", tooltip=["Product", "StockCode", "Invoices", "Units"]), key="pi_i")
            ui.explainer("What does this chart show?", "Each bar counts the separate **invoices (orders)** that contained the product; "
                         "an invoice is counted once even if it lists the product several times. A higher value means the product was "
                         "ordered more often — not necessarily that more units were bought.")

    with t2:
        top = P.top_by_invoices(e, names, 300)
        opts = dict(zip(top["Product"] + "  ·  " + top["StockCode"], top["StockCode"]))
        label = st.selectbox("Product (300 most frequently ordered)", list(opts), key="pi_prod")
        m = P.monthly_product_units(e, opts[label])
        charts.show(charts.monthly_bars(m, "Units", f"{label.split('  ·  ')[0]} — units purchased per month, Dec 2010 – 9 Dec 2011",
                                        "Units purchased"), key="pi_month")
        ui.explainer("How to read this chart", f"Bars show the total units of this product on eligible purchase lines in each month "
                     f"(months with no purchases show zero). For this product the full months are **{P.describe_trend(m)}**. "
                     "December 2011 contains only 1–9 December, so it is lighter and should not be compared with full months. "
                     "One year of data cannot show whether a rise or fall is seasonal or will continue.")

    with t3:
        a, b = st.columns(2)
        seg = a.segmented_control("Segment", [s for s in SEGMENT_ORDER if s in ctx.segments], default=SEGMENT_ORDER[0],
                                  format_func=ui.short, key="pi_seg") or SEGMENT_ORDER[0]
        meas = b.segmented_control("Rank by", ["Units", "Invoices", "Value"], default="Units", key="pi_meas",
                                   format_func={"Units": "Units purchased", "Invoices": "Invoices", "Value": "Gross value (£)"}.get) or "Units"
        sp = P.segment_products(e, names, seg, meas, 10)
        title = {"Units": "Units purchased", "Invoices": "Invoices containing the product", "Value": "Gross purchase value (£)"}[meas]
        charts.show(charts.hbar(sp, "Product", meas, f"Top 10 products for {ui.short(seg)} customers — {title.lower()}",
                                ctx.colors.get(seg, "#6655D9"), title, fmt=",.0f",
                                tooltip=["Product", "StockCode", "Units", "Invoices", "Customers", alt_value()]), key="pi_segbar")
        ui.explainer("What does this chart show?", f"The products bought most by customers assigned to the **{ui.short(seg)}** segment by "
                     "the existing K-Means model (joined by CustomerID). It does not mean the segment buys only these products, or that "
                     "the segment caused the purchases; popular products tend to rank high in every segment.")

    with t4:
        a, b = st.columns(2)
        meas = a.segmented_control("Rank by", ["Repeat_Rate", "Repeat_Buyers"], default="Repeat_Rate", key="pi_rmeas",
                                   format_func={"Repeat_Rate": "Repeat-purchaser rate (%)", "Repeat_Buyers": "Repeat purchasers"}.get) or "Repeat_Rate"
        mb = b.slider("Minimum customers per product", 20, 300, 100, 10, key="pi_minb",
                      help="Products bought by few customers give unstable rates, so they are hidden.")
        rp = P.repeat_purchase(e, names, mb).sort_values([meas, "Buyers"], ascending=False).head(10)
        if rp.empty:
            ui.callout("No product meets the minimum number of customers.", "warn", label="No results")
        else:
            charts.show(charts.hbar(rp, "Product", meas, "Top 10 products by " + ("repeat-purchaser rate" if meas == "Repeat_Rate"
                                    else "number of repeat purchasers"), "#2F9E77",
                                    "Repeat-purchaser rate (%)" if meas == "Repeat_Rate" else "Customers buying it in 2+ invoices",
                                    fmt=".1f" if meas == "Repeat_Rate" else ",.0f",
                                    tooltip=["Product", "Buyers", "Repeat_Buyers", alt_rate(), "Median_Invoices"]), key="pi_rep")
            st.dataframe(rp[["Product", "Buyers", "Repeat_Buyers", "Repeat_Rate", "Median_Invoices"]], hide_index=True, **ui.FULL,
                         column_config={"Buyers": "Customers who bought it", "Repeat_Buyers": "Bought it in 2+ invoices",
                                        "Repeat_Rate": st.column_config.NumberColumn("Repeat-purchaser rate (%)", format="%.1f"),
                                        "Median_Invoices": st.column_config.NumberColumn("Median invoices per buyer", format="%.0f")})
        ui.explainer("What is a repeat purchase?", "Repeat-purchaser rate = customers who bought the product in **at least two separate "
                     "invoices** ÷ customers who bought it at least once × 100. Buying several units, or several lines, on one invoice "
                     "does not count as a repeat purchase. A high rate suggests a product customers come back for; it does not prove loyalty.")


def alt_value():
    import altair as alt
    return alt.Tooltip("Value:Q", format=",.2f", title="£ gross")


def alt_rate():
    import altair as alt
    return alt.Tooltip("Repeat_Rate:Q", format=".1f", title="Repeat rate %")
