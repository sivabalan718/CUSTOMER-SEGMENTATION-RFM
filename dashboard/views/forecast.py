"""Product Demand Forecasting: monthly unit demand with chronological validation against simple baselines."""
import altair as alt
import pandas as pd
import streamlit as st

from .. import charts, products as P, ui
from ..state import get_products
from .product_intel import eligibility_note


def render(ctx, go):
    ui.header("Demand Forecasting", "Product Demand Forecast",
              "Estimated future monthly units for one product, checked against simple baselines. A separate analysis from the segmentation.")
    pdata = get_products()
    if pdata is None:
        ui.unavailable("The purchase-line file")
        return
    e, names = pdata.lines, pdata.names
    eligibility_note(pdata)
    a, b = st.columns([3, 1])
    top = P.top_by_invoices(e, names, 300)
    opts = dict(zip(top["Product"] + "  ·  " + top["StockCode"], top["StockCode"]))
    label = a.selectbox("Product (300 most frequently ordered)", list(opts), key="fc_prod")
    horizon = b.slider("Months ahead", 1, 6, 3, key="fc_h", help="Limited to 6 because only 12 full months of history exist.")
    m = P.monthly_product_units(e, opts[label])
    res = P.forecast_product(m, horizon)
    name = label.split("  ·  ")[0]
    hist = m.assign(Series=m["Partial"].map({False: "Recorded units", True: "Recorded units (partial month)"}))
    enc = dict(x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %y")), y=alt.Y("Units:Q", title="Units"),
               tooltip=[alt.Tooltip("Month:T", format="%b %Y"), "Units", "Series"])
    base = alt.Chart(hist[~hist["Partial"]]).mark_line(point=True, color="#6655D9").encode(**enc)
    partial = alt.Chart(hist[hist["Partial"]]).mark_point(filled=True, size=60, color="#9AA3B5").encode(**enc)
    layers = [base, partial]
    if res["ok"]:
        fc = res["forecast"]
        joint = pd.concat([res["history"].tail(1).rename(columns={"Units": "Forecast"}).assign(Lower=lambda d: d["Forecast"],
                                                                                                    Upper=lambda d: d["Forecast"]),
                           fc])[["Month", "Forecast", "Lower", "Upper"]]
        layers += [alt.Chart(joint).mark_area(opacity=0.18, color="#D08A2E").encode(x="Month:T", y="Lower:Q", y2="Upper:Q"),
                   alt.Chart(joint).mark_line(point=True, strokeDash=[6, 4], color="#D08A2E").encode(
                       x="Month:T", y="Forecast:Q", tooltip=[alt.Tooltip("Month:T", format="%b %Y"), alt.Tooltip("Forecast:Q", format=",.0f"),
                                                              alt.Tooltip("Lower:Q", format=",.0f"), alt.Tooltip("Upper:Q", format=",.0f")]),
                   alt.Chart(pd.DataFrame({"Month": [res["history"]["Month"].iloc[-1]]})).mark_rule(color="#16845B", strokeDash=[3, 3])
                   .encode(x="Month:T")]
    charts.show(alt.layer(*layers).properties(title=f"{name} — monthly units and forecast", height=320), key="fc_chart")
    st.markdown("<div class='small'>Purple line: recorded units per full month · grey point: 1–9 Dec 2011 only (partial month) · orange dashed line and band: forecast with an approximate 80% interval · green dashed line: end of the history used.</div>", unsafe_allow_html=True)
    if not res["ok"]:
        ui.callout(f"No forecast is shown for this product: {res['reason']}.", "warn", label="Not enough data")
    else:
        fc = res["forecast"]
        ui.kpis([("Forecast horizon", f"{horizon} month{'s' if horizon > 1 else ''}", f"{fc['Month'].iloc[0]:%b %Y} – {fc['Month'].iloc[-1]:%b %Y}"),
                 ("Total forecast units", f"{fc['Forecast'].sum():,.0f}", "Sum of the monthly estimates"),
                 ("Holdout check", "Beats baselines" if res["beats_baselines"] else "Does not beat a baseline",
                  f"Last {res['holdout_months']} full months, model fitted on the first {res['train_months']}")], cols=3)
        st.dataframe(fc.assign(Month=fc["Month"].dt.strftime("%b %Y")), hide_index=True, **ui.FULL,
                     column_config={c: st.column_config.NumberColumn(f, format="%.0f") for c, f in
                                    [("Forecast", "Forecast units"), ("Lower", "Lower (80%)"), ("Upper", "Upper (80%)")]})
        hm = res["holdout"]["Month"]
        st.markdown(f"**Holdout validation** — each method predicted the last {len(hm)} full months ({hm.iloc[0]:%b}–{hm.iloc[-1]:%b %Y}) "
                    "using only the earlier months:")
        st.dataframe(res["validation"], hide_index=True, **ui.FULL,
                     column_config={"MAE": st.column_config.NumberColumn("MAE (units)", format="%.1f"),
                                    "RMSE": st.column_config.NumberColumn("RMSE (units)", format="%.1f")})
        if not res["beats_baselines"]:
            ui.callout("For this product the model was not more accurate than a simple baseline on the holdout months, so its forecast "
                       "should be treated with extra caution.", "warn")
    ui.explainer("How does this forecast work?", "Monthly units for the selected product (eligible purchase lines only) form a time series "
                 "of 12 full months, December 2010 – November 2011; the partial month 1–9 December 2011 is shown but not used. A damped-trend "
                 "exponential smoothing model (Holt) learns the recent level and trend, with the trend fading over time. To check it, the "
                 "model was fitted on the first nine months and asked to predict September–November 2011, and its errors (MAE, RMSE) are "
                 "compared with two simple baselines. The dashed line is the estimate of future units; the shaded band is an approximate "
                 "80% range based on past one-step errors.\n\n"
                 "One year of data cannot capture yearly seasonality such as the November peak repeating, and bulk orders make some products "
                 "irregular. Treat forecasts as estimates to monitor against actual purchases, not guaranteed sales or stock requirements.")
