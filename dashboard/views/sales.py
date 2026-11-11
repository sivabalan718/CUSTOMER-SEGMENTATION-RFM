"""Sales & Customer Insights: top customers, invoice metrics, country value, monthly active customers."""
import altair as alt
import streamlit as st

from .. import charts, products as P, ui
from ..state import get_products
from .product_intel import eligibility_note


def render(ctx, go):
    ui.header("Sales & Customer Insights", "Customers, Invoices and Countries",
              "Who buys the most, how large a typical invoice is, where purchase value comes from and how many customers buy each month.")
    pdata = get_products()
    if pdata is None:
        ui.unavailable("The purchase-line file")
        return
    e = pdata.lines
    eligibility_note(pdata)
    k = P.invoice_metrics(e)
    ui.kpis([("Gross purchase value", f"£{k['value']:,.0f}", "Eligible lines, Dec 2010 – 9 Dec 2011"),
             ("Invoices", f"{k['invoices']:,}", "Distinct purchase invoices"),
             ("Average invoice value", f"£{k['avg_invoice_value']:,.2f}", "Gross value ÷ distinct invoices"),
             ("Average units per invoice", f"{k['avg_units_per_invoice']:,.1f}", "Units ÷ distinct invoices")])
    ui.explainer("What do these figures mean?", "**Average invoice value** is the gross purchase amount of a typical order (total value ÷ "
                 "number of distinct invoices) — not the price of a product and not profit. **Average units per invoice** is total units ÷ "
                 "distinct invoices. Both are pulled up by large wholesale orders; the median invoice is smaller.")
    t1, t2, t3, t4 = st.tabs(["Top customers", "Invoice metrics by month", "Country value", "Monthly active customers"])

    with t1:
        n = st.slider("Number of customers", 5, 50, 15, key="si_n")
        tc = P.top_customers(e, n).assign(Segment=lambda d: d["Segment"].astype(str).map(ui.short),
                                          Customer=lambda d: "Customer " + d["CustomerID"].astype(str))
        charts.show(charts.hbar(tc, "Customer", "Value", f"Top {n} customers by gross purchase value", "#6655D9",
                                "Gross purchase value (£)", tooltip=["Customer", "Segment", alt.Tooltip("Value:Q", format=",.2f"),
                                                                       "Invoices", "Units", "Last_Purchase"]), key="si_cust")
        st.dataframe(tc[["CustomerID", "Segment", "Country", "Value", "Invoices", "Units", "Last_Purchase"]], hide_index=True, **ui.FULL,
                     column_config={"CustomerID": st.column_config.NumberColumn("CustomerID", format="%d"),
                                    "Value": st.column_config.NumberColumn("Gross value (£)", format="%.2f"),
                                    "Last_Purchase": st.column_config.DatetimeColumn("Last purchase", format="DD MMM YYYY")})
        ui.explainer("What does this ranking show?", "Customers with the highest recorded **gross positive purchase value** in the period, "
                     "with their number of invoices, units, last purchase and existing segment. It does not show profit or lifetime value, "
                     "which would need cost and longer-term data.")

    mm = P.monthly_invoice_metrics(e)
    with t2:
        a, b = st.columns(2, gap="large")
        with a:
            charts.show(charts.monthly_bars(mm, "Avg_Invoice_Value", "Average invoice value per month", "Average invoice value (£)",
                                            fmt=",.2f"), key="si_aiv")
        with b:
            charts.show(charts.monthly_bars(mm, "Avg_Units_per_Invoice", "Average units per invoice per month", "Average units per invoice",
                                            "#2E9BB0", fmt=",.1f"), key="si_aui")
        ui.explainer("How are these calculated?", "For each month: gross value ÷ distinct invoices issued that month, and units ÷ the same "
                     "invoices. Each invoice is counted once, in the month it was issued. December 2011 covers only 1–9 December.")

    with t3:
        n = st.slider("Number of countries", 5, 37, 10, key="si_cn")
        cv = P.country_value(e, n)
        charts.show(charts.hbar(cv, "Country", "Value", f"Top {n} countries by gross purchase value", "#C25E8A", "Gross purchase value (£)",
                                tooltip=["Country", alt.Tooltip("Value:Q", format=",.2f"), alt.Tooltip("Value_Share:Q", format=".1f",
                                         title="Share of value %"), "Customers", "Invoices"]), key="si_ctry")
        ui.explainer("What does this chart show?", "Countries ranked by the total gross purchase value of their customers' eligible lines. "
                     "This is different from the number of customers per country on the Dataset & EDA page: a country with few customers "
                     "can still contribute a lot of value. Postage and other service lines are included, as in the rest of the project.")

    with t4:
        ma = P.monthly_active_customers(e)
        charts.show(charts.monthly_bars(ma, "Active_Customers", "Customers with at least one purchase, per month", "Active customers",
                                        "#2F9E77"), key="si_active")
        ui.explainer("What is an active customer?", "A customer with at least one eligible purchase in that month, counted once however many "
                     "invoices they placed. It is not the size of the whole customer base and does not by itself measure loyalty.")
