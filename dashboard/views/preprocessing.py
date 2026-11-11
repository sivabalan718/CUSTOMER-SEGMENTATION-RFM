"""Preprocessing."""
import altair as alt
import streamlit as st

from .. import charts, ui
from ..config import NON_PRODUCT_CODES

RULES = [
    ("Remove exact duplicates", "Rows identical in all eight attributes are kept once."),
    ("Exclude rows without a CustomerID", "They cannot be linked to a customer, so no customer-level features can be built."),
    ("Exclude cancellations", "Invoice numbers beginning with ‘C’ record cancellations and returns, not purchases."),
    ("Exclude non-positive quantities", "A purchase line must have at least one unit."),
    ("Exclude non-positive unit prices", "Lines with a zero or negative price carry no purchase value."),
    ("Check required fields", "Invoice number, date and stock code must be present."),
    ("Compute line value", "Line value = Quantity × UnitPrice, in GBP."),
]


def render(ctx, go):
    ui.header("Preprocessing", "From Raw Transactions to Modelling Data",
              "How the original transaction lines were cleaned so that each remaining line is a valid purchase by an identified customer.")
    a = ctx.audit
    if a is None:
        ui.unavailable("The cleaning audit")
        return
    after = a.set_index("Step")["Rows_After"] if "Rows_After" in a.columns else None
    rows = a.set_index("Step")["Rows"]
    get = lambda key: int(after[[s for s in after.index if s.startswith(key)][0]]) if after is not None else None
    ui.pipeline([(f"{int(rows.iloc[0]):,}", "raw rows"),
                 (f"{get('Exact duplicate'):,}", "after duplicate removal"),
                 (f"{get('Rows without CustomerID'):,}", "identified customers only"),
                 (f"{get('Cancellation'):,}", "cancellations removed"),
                 (f"{int(rows.iloc[-1]):,}", "retained purchase lines")])

    ui.section("Cleaning audit", "Each step is applied to the rows left by the previous one, so no row is counted twice")
    t = a.iloc[1:-1][["Step", "Rows_Before", "Rows", "Rows_After"]].copy()
    t["Step"] = (t["Step"].str.replace(" excluded from RFM preparation", " excluded").str.replace("Rows failing required invoice/date/stock-code checks",
                                                                                                  "Missing required fields"))
    st.dataframe(t, hide_index=True, **ui.FULL, column_config={
        "Step": st.column_config.TextColumn("Step", width="medium"),
        "Rows_Before": st.column_config.NumberColumn("Rows before", format="%d"),
        "Rows": st.column_config.NumberColumn("Removed", format="%d"),
        "Rows_After": st.column_config.NumberColumn("Rows after", format="%d")})
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ui.section("Cleaning rules", "Applied in this order")
        st.markdown("\n".join(f"{i}. **{name}.** {why}" for i, (name, why) in enumerate(RULES, 1)))
    with right:
        ui.section("Outliers")
        ui.callout("Very large orders were reviewed (99.5th-percentile thresholds) but not removed: they are genuine recorded "
                   "purchases. Their influence is reduced later by the log transformation.", "observed", label="Outliers")

    ui.section("Rows removed per step")
    charts.show(alt.Chart(t).mark_bar(color="#6655D9", cornerRadiusEnd=4, height=20).encode(
        y=alt.Y("Step:N", sort=None, title=None, axis=alt.Axis(labelLimit=340)),
        x=alt.X("Rows:Q", title="Rows removed", axis=alt.Axis(format=",d", tickCount=5)),
        tooltip=["Step", alt.Tooltip("Rows:Q", format=",", title="Rows removed")]).properties(height=200), key="pp_bar")

    rc = (ctx.verif.get("core") or {})
    with st.expander("Data quality and limitations"):
        if rc:
            st.markdown(
                f"- **Why step counts differ from raw counts.** The raw file has {rc.get('raw_missing_customerid', 0):,} rows without a "
                f"CustomerID, but {rc.get('raw_missing_customerid_rows_that_are_duplicates', 0)} of them are duplicates removed first, "
                f"leaving 135,037. It has {rc.get('raw_cancel_rows', 0):,} cancellation rows, of which 8,872 remain once duplicates and "
                f"unidentified rows are gone. All {rc.get('raw_noncancel_nonpos_qty', 0):,} other rows with Quantity ≤ 0 lack a "
                "CustomerID, so the quantity step removes none.")
        st.markdown(
            "- **Gross, not net.** Cancellations are excluded rather than subtracted, so purchase values are gross positive values.\n"
            f"- **Service lines.** Postage, carriage, manual adjustments and bank charges ({', '.join(NON_PRODUCT_CODES)}) are kept as "
            "purchase lines; excluding them changes the segments very little.\n"
            "- **Coverage.** About a quarter of raw rows have no CustomerID, so the analysis describes identified customers only.\n"
            "- **Categorical attributes** (InvoiceNo, StockCode, Description, Country, CustomerID) are identifiers or context, not model "
            "features, so no encoding is needed.")
