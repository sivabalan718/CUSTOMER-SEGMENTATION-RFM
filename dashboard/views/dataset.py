"""Dataset & EDA."""
import altair as alt
import pandas as pd
import streamlit as st

from .. import charts, explain, ui
from ..config import EXPECTED

DICTIONARY = {
    "InvoiceNo": ("Identifier", "Invoice number; a leading ‘C’ marks a cancellation."),
    "StockCode": ("Identifier", "Product code (5 digits for products; letters for services such as POST)."),
    "Description": ("Text", "Product name."),
    "Quantity": ("Numeric", "Units on the invoice line (negative on cancellations)."),
    "InvoiceDate": ("Date-time", "When the invoice was generated."),
    "UnitPrice": ("Numeric", "Price per unit in GBP (£)."),
    "CustomerID": ("Identifier", "Customer number; missing for about a quarter of rows."),
    "Country": ("Categorical", "Country of the customer."),
}


def render(ctx, go):
    ui.header("Dataset & EDA", "Understanding the Data",
              "The UCI Online Retail dataset: transactions of a UK-based online retailer between December 2010 and December 2011.")
    ins = ctx.inspection
    if ins is not None:
        i = ins.set_index("Attribute")
        ui.kpis([("Raw transactions", f"{EXPECTED['raw_rows']:,}", "invoice lines in the original file"),
                 ("Attributes", f"{len(ins)}", "columns per transaction line"),
                 ("Missing CustomerIDs", f"{int(i.loc['CustomerID', 'missing']):,}", f"{i.loc['CustomerID', 'missing_pct']:.1f}% of raw rows"),
                 ("Unique customers", f"{int(i.loc['CustomerID', 'unique_values']):,}", "distinct CustomerIDs in the raw data",
                  "Counts every non-missing CustomerID in the raw file. The modelling population (4,338) is smaller because some "
                  "customers have only cancelled or invalid lines.")])
        ui.section("Missing values", "Raw rows with a missing value, by attribute")
        charts.show(alt.Chart(ins).mark_bar(color="#D08A2E", cornerRadiusEnd=4, height=18).encode(
            y=alt.Y("Attribute:N", sort="-x", title=None), x=alt.X("missing:Q", title="Missing values (rows)", axis=alt.Axis(labelExpr="datum.value == 0 ? '0' : format(datum.value, '~s')", tickCount=4)),
            tooltip=["Attribute", alt.Tooltip("missing:Q", format=","), alt.Tooltip("missing_pct:Q", format=".2f", title="% of rows")]
        ).properties(height=250), key="ds_missing")
        ui.section("Attribute summary", "Type, completeness and number of distinct values")
        t = pd.DataFrame({"Attribute": ins["Attribute"], "Type": [DICTIONARY.get(a, ("", ""))[0] for a in ins["Attribute"]],
                          "Missing": ins["missing"], "Missing %": ins["missing_pct"], "Distinct values": ins["unique_values"]})
        st.dataframe(t, hide_index=True, **ui.FULL, column_config={
            "Missing": st.column_config.NumberColumn(format="%d"), "Missing %": st.column_config.NumberColumn(format="%.2f%%"),
            "Distinct values": st.column_config.NumberColumn(format="%d")})
        ui.callout("The dataset documentation lists no missing values, but the file has 135,080 rows without a CustomerID and 1,454 "
                   "without a Description (all of which also lack a CustomerID).", "observed")
    else:
        ui.unavailable("The raw-data summary")

    if not ctx.has_lines:
        ui.unavailable("Transaction-level data")
        return
    l = ctx.lines
    ui.section("Purchasing activity", "Retained purchase lines of identified customers")
    t1, t2, t3 = st.tabs(["Monthly trend", "Countries & products", "Day and hour"])
    with t1:
        metric = st.segmented_control("Measure", ["Gross positive purchase value (£)", "Invoices", "Active customers"],
                                      default="Gross positive purchase value (£)", key="ds_metric") or "Gross positive purchase value (£)"
        col, fn = {"Gross positive purchase value (£)": ("LineValue", "sum"), "Invoices": ("InvoiceNo", "nunique"),
                   "Active customers": ("CustomerID", "nunique")}[metric]
        m = l.groupby(pd.Grouper(key="InvoiceDate", freq="MS"))[col].agg(fn).rename("Value").reset_index()
        m["Month"] = m["InvoiceDate"].dt.strftime("%b %y")
        m["Period"] = ["Partial month (1–9 Dec)" if d >= pd.Timestamp("2011-12-01") else "Full month" for d in m["InvoiceDate"]]
        charts.show(alt.Chart(m).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Month:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=11)),
            y=alt.Y("Value:Q", title=metric),
            color=alt.Color("Period:N", scale=alt.Scale(domain=["Full month", "Partial month (1–9 Dec)"], range=["#6655D9", "#C9C3F2"]),
                            legend=alt.Legend(title=None, orient="top-left", direction="horizontal")),
            tooltip=["Month", alt.Tooltip("Value:Q", format=",.0f", title=metric)]).properties(height=300), key="ds_month")
        peak = m[m["Period"] == "Full month"].sort_values("Value").iloc[-1]
        ui.callout(f"The busiest full month was {peak['Month']} ({peak['Value']:,.0f}). December 2011 covers only nine days. "
                   "This describes one year of history; it does not establish seasonality.", "observed")
    with t2:
        n = st.slider("Show top", 5, 20, 10, key="ds_topn")
        ctry = l.groupby("Country", observed=True).agg(Customers=("CustomerID", "nunique"), Value=("LineValue", "sum")).reset_index()
        uk = 100 * ctry.set_index("Country").loc["United Kingdom", "Customers"] / l["CustomerID"].nunique() if "United Kingdom" in set(ctry.Country) else 0
        charts.show(alt.Chart(ctry.nlargest(n, "Customers")).mark_bar(color="#6655D9", cornerRadiusEnd=4).encode(
            y=alt.Y("Country:N", sort="-x", title=None), x=alt.X("Customers:Q", title="Distinct customers"),
            tooltip=["Country", "Customers", alt.Tooltip("Value:Q", format=",.0f", title="£ gross")]
        ).properties(title=f"Top {n} countries by customers", height=26 * n + 30), key="ds_ctry")
        prod = l.groupby("Description", observed=True).agg(Value=("LineValue", "sum"), Units=("Quantity", "sum")).reset_index()
        charts.show(alt.Chart(prod.nlargest(n, "Value")).mark_bar(color="#2E9BB0", cornerRadiusEnd=4).encode(
            y=alt.Y("Description:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420)), x=alt.X("Value:Q", title="£ gross positive value", axis=alt.Axis(labelExpr="datum.value == 0 ? '0' : format(datum.value, '~s')")),
            tooltip=["Description", alt.Tooltip("Value:Q", format=",.2f"), alt.Tooltip("Units:Q", format=",")]
        ).properties(title=f"Top {n} products by gross value", height=26 * n + 30), key="ds_prod")
        ui.callout(f"{uk:.1f}% of customers are based in the United Kingdom, so patterns for other countries rest on small samples.", "observed")
    with t3:
        inv = l.groupby("InvoiceNo", observed=True)["InvoiceDate"].min().reset_index()
        inv["Day"], inv["Hour"] = inv["InvoiceDate"].dt.day_name(), inv["InvoiceDate"].dt.hour
        dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hm = inv.groupby(["Day", "Hour"]).size().reset_index(name="Invoices")
        charts.show(alt.Chart(hm).mark_rect(cornerRadius=3).encode(
            x=alt.X("Hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)), y=alt.Y("Day:O", sort=dow, title=None),
            color=alt.Color("Invoices:Q", scale=alt.Scale(scheme="purples"), legend=alt.Legend(title="Invoices", orient="right", direction="vertical")),
            tooltip=["Day", "Hour", "Invoices"]).properties(title="Invoices by weekday and hour", height=250), key="ds_heat")
        missing = sorted(set(dow) - set(hm["Day"]))
        if missing:
            ui.callout(f"No invoices are recorded on {', '.join(missing)}.", "observed")

    ui.section("Cleaned purchase lines — descriptive statistics", f"Summary of the {len(l):,} retained purchase lines")
    num = l[["Quantity", "UnitPrice", "LineValue"]].agg(["mean", "median", "std", "min", "max"]).T
    num["skewness"] = l[["Quantity", "UnitPrice", "LineValue"]].skew()
    num.index = ["Quantity (units)", "Unit price (£)", "Line value (£ gross)"]
    st.dataframe(num.reset_index().rename(columns={"index": "Variable"}), hide_index=True, **ui.FULL,
                 column_config={**{k: st.column_config.NumberColumn(format="%.2f") for k in ["mean", "median", "std", "max", "skewness"]},
                                "min": st.column_config.NumberColumn(format="%.3f")})
    st.markdown("<div class='small'>All three variables are extremely right-skewed: a few very large orders sit far above the typical line.</div>",
                unsafe_allow_html=True)
    ui.explainer("Understand What These Statistics Tell Us", explain.descriptive_stats(l))

    with st.expander("Data dictionary"):
        st.dataframe(pd.DataFrame([{"Attribute": k, "Description": v[1]} for k, v in DICTIONARY.items()]),
                     hide_index=True, **ui.FULL)
