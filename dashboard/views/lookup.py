"""Customer lookup component (used on the Customer Segmentation page) (search, filters, sort, pagination, profile, export)."""
import altair as alt
import pandas as pd
import streamlit as st

from .. import charts, core, ui

KEYS = ["ex_search", "ex_segments", "ex_countries", "ex_r", "ex_f", "ex_m_min", "ex_m_max", "ex_sort", "ex_asc", "ex_page", "ex_profile"]
SORTABLE = {"Monetary": "Monetary (£ gross)", "Frequency": "Frequency (invoices)", "Recency": "Recency (days)",
            "CustomerID": "CustomerID", "TotalItems": "Total items", "DistinctProducts": "Distinct products", "Cluster": "Cluster"}


def _first_page():
    st.session_state["ex_page"] = 1


def reset_filters():
    for k in KEYS:
        st.session_state.pop(k, None)


def current_filters(ctx) -> dict:
    """Read the explorer filters from session state (also used by the export module)."""
    c = ctx.customers
    s = st.session_state
    segs = s.get("ex_segments")
    r = s.get("ex_r")
    f = s.get("ex_f")
    m_min, m_max = s.get("ex_m_min"), s.get("ex_m_max")
    return {"search": s.get("ex_search", ""),
            "segments": segs if segs else None,
            "countries": s.get("ex_countries") or None,
            "recency": tuple(r) if r and tuple(r) != (int(c.Recency.min()), int(c.Recency.max())) else None,
            "frequency": tuple(f) if f and tuple(f) != (int(c.Frequency.min()), int(c.Frequency.max())) else None,
            "monetary": (m_min if m_min is not None else float(c.Monetary.min()), m_max if m_max is not None else float(c.Monetary.max()))
            if (m_min is not None or m_max is not None) else None}


def describe_filters(f) -> str:
    parts = []
    if f["search"]:
        parts.append(f"CustomerID contains '{f['search']}'")
    if f["segments"]:
        parts.append("segments: " + "; ".join(f["segments"]))
    if f["countries"]:
        parts.append("countries: " + ", ".join(f["countries"]))
    for k, u in [("recency", "days"), ("frequency", "invoices")]:
        if f[k]:
            parts.append(f"{k} {f[k][0]}–{f[k][1]} {u}")
    if f["monetary"]:
        parts.append(f"monetary £{f['monetary'][0]:,.2f}–£{f['monetary'][1]:,.2f}")
    return "; ".join(parts) or "no filters (all customers)"


def column_config():
    return {"CustomerID": st.column_config.NumberColumn("CustomerID", format="%d"),
            "Cluster": st.column_config.NumberColumn("Cluster", format="%d"),
            "Segment": st.column_config.TextColumn("Segment"),
            "Recency": st.column_config.NumberColumn("Recency (d)", format="%d", width="small", help="Days since last purchase"),
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d", width="small", help="Distinct invoices"),
            "Monetary": st.column_config.NumberColumn("Gross value (£)", format="£%.2f", help="Gross positive purchase value"),
            "TotalItems": st.column_config.NumberColumn("Items", format="%d"),
            "DistinctProducts": st.column_config.NumberColumn("Products", format="%d"),
            "Country": st.column_config.TextColumn("Country"),
            "LastPurchaseDate": st.column_config.TextColumn("Last purchase")}


def render(ctx):
    c = ctx.customers
    st.session_state.setdefault("ex_r", (int(c.Recency.min()), int(c.Recency.max())))
    st.session_state.setdefault("ex_f", (int(c.Frequency.min()), int(c.Frequency.max())))
    with st.container(border=True):
        st.markdown("<div class='small'>Filters apply to the table, the profile list below and the CSV export.</div>", unsafe_allow_html=True)
        a, b, d = st.columns([1.1, 2, 1.4])
        a.text_input("Search CustomerID", key="ex_search", placeholder="e.g. 12347", on_change=_first_page, help="Matches any part of the ID")
        b.multiselect("Cluster · segment", ctx.segments, key="ex_segments",
                      format_func=lambda s: f"{int(ctx.profiles.set_index('Segment').loc[s, 'Cluster'])} · {s}"
                      if s in set(ctx.profiles['Segment']) else s, placeholder="All segments", on_change=_first_page)
        d.multiselect("Country", sorted(c["Country"].dropna().unique()), key="ex_countries", placeholder="All countries", on_change=_first_page)
        r1, r2, r3, r4 = st.columns([1.3, 1.3, 0.8, 0.8])
        r1.slider("Recency — days since last purchase", int(c.Recency.min()), int(c.Recency.max()), key="ex_r", on_change=_first_page)
        r2.slider("Frequency — number of invoices", int(c.Frequency.min()), int(c.Frequency.max()), key="ex_f", on_change=_first_page)
        r3.number_input("Min gross value (£)", min_value=0.0, value=None, step=100.0, key="ex_m_min", placeholder="any", on_change=_first_page)
        r4.number_input("Max gross value (£)", min_value=0.0, value=None, step=100.0, key="ex_m_max", placeholder="any", on_change=_first_page)

    f = current_filters(ctx)
    d = core.filter_customers(c, f)
    s1, s2 = st.columns([1.6, 1], gap="medium", vertical_alignment="bottom")
    sort_col = s1.selectbox("Sort by", list(SORTABLE), format_func=SORTABLE.get, key="ex_sort")
    ascending = s2.toggle("Ascending order", value=False, key="ex_asc")
    s3, s4, s5 = st.columns([1, 1, 1.2], gap="medium", vertical_alignment="bottom")
    page_size = s3.selectbox("Rows / page", [25, 50, 100, 250], index=1, key="ex_page_size", on_change=_first_page)
    pages = max(1, -(-len(d) // page_size))
    st.session_state["ex_page"] = min(max(1, int(st.session_state.get("ex_page", 1))), pages)
    page = s4.number_input("Page", min_value=1, max_value=pages, key="ex_page")
    s5.button("Reset filters", icon=":material/restart_alt:", on_click=reset_filters, **ui.FULL, key="ex_reset")

    shown, pages, page = core.sort_and_page(d, sort_col, ascending, int(page), int(page_size))
    st.markdown(f"**{len(d):,}** of {len(c):,} customers match · page {page} of {pages} · "
                f"<span class='small'>{describe_filters(f)}</span>", unsafe_allow_html=True)
    if d.empty:
        ui.callout("No customers match the current filters. Widen a range or press Reset filters.", "warn", label="No results")
        return
    cols = ["CustomerID", "Segment", "Recency", "Frequency", "Monetary", "Country"]
    table = shown[cols].assign(Segment=shown["Segment"].map(ui.short))
    st.dataframe(table, hide_index=True, column_config=column_config(), height=min(36 * len(shown) + 40, 470), key="ex_table")
    st.markdown("<div class='small'>Segment column shows short names: " + " · ".join(f"{ui.short(s)} = {s}" for s in ctx.segments)
                + ". Gross value = gross positive purchase value (GBP).</div>", unsafe_allow_html=True)
    e1, e2 = st.columns([1, 3.2], vertical_alignment="center")
    e1.download_button("Export CSV", core.to_csv_bytes(core.export_customers(d.sort_values(sort_col, ascending=ascending))),
                       file_name="filtered_customers.csv", mime="text/csv", icon=":material/download:", key="ex_download")
    e2.caption(f"Export contains all {len(d):,} matching customers (not only this page), sorted as shown. "
               "Monetary = gross positive purchase value (GBP).")

    st.markdown("**Customer profile**")
    ids = d.sort_values(sort_col, ascending=ascending)["CustomerID"].tolist()
    default = st.session_state.get("ex_profile")
    idx = ids.index(default) if default in ids else 0
    cid = st.selectbox("Open profile for CustomerID", ids, index=idx, key="ex_profile")
    profile(ctx, cid)


def profile(ctx, cid):
    c = ctx.customers
    row = c.loc[c["CustomerID"] == cid].iloc[0]
    seg = row["Segment"]
    color = ctx.colors.get(seg, "#646a8c")
    st.markdown(f"<div class='card seg-card' style='--c:{color}'><h4>Customer {int(cid)} · {ui.seg_label(seg, color)}</h4>"
                f"<span class='small'>Cluster {row['Cluster']}</span></div>",
                unsafe_allow_html=True)
    segrow = ctx.profiles.set_index("Segment").loc[seg] if seg in set(ctx.profiles["Segment"]) else None
    items = []
    for f, unit, fmt in [("Recency", "days", "{:.0f}"), ("Frequency", "invoices", "{:.0f}"), ("Monetary", "£", "£{:,.2f}")]:
        val = row[f]
        pr = core.percentile_rank(c[f], val)
        sub = f"segment median {fmt.format(segrow[f + '_Median'])}" if segrow is not None else ""
        items.append((f"{f} ({unit})" if f != "Monetary" else "Gross purchase value (£)", fmt.format(val), f"{sub} · percentile {pr:.0f}"))
    ui.kpis(items, cols=3)
    ui.strip([("Items purchased", f"{int(row['TotalItems']):,}"), ("Distinct products", f"{int(row['DistinctProducts']):,}"),
              ("Country", str(row["Country"])), ("Last purchase", str(row["LastPurchaseDate"])[:10])])
    if cid == 12346:
        ui.callout("This customer's only order (74,215 units, £77,183.60) was cancelled 16 minutes later. Monetary counts the gross "
                   "purchase, so net spend is £0 — a documented limitation of the gross-value policy.", "warn")
    if ctx.has_lines:
        tx = ctx.lines[ctx.lines["CustomerID"] == cid]
        m = tx.set_index("InvoiceDate").resample("MS")["LineValue"].sum().reset_index()
        charts.show(alt.Chart(m).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("yearmonth(InvoiceDate):O", title=None), y=alt.Y("LineValue:Q", title="£ gross positive value"),
            tooltip=[alt.Tooltip("yearmonth(InvoiceDate):O", title="Month"), alt.Tooltip("LineValue:Q", format=",.2f", title="£")]
        ).properties(title="Monthly purchase value", height=230), key=f"prof_month_{cid}")
        top = (tx.groupby(["StockCode", "Description"], observed=True)
               .agg(Quantity=("Quantity", "sum"), Value=("LineValue", "sum")).reset_index()
               .sort_values("Value", ascending=False).head(8))
        st.markdown("**Top products by gross value**")
        st.dataframe(top, hide_index=True, column_config={"Value": st.column_config.NumberColumn("£ gross", format="£%.2f")},
                     key=f"prof_top_{cid}")
        inv = ctx.invoices[ctx.invoices["CustomerID"] == cid].sort_values("InvoiceDate", ascending=False).head(10)
        with st.expander(f"Recent invoices ({int(row['Frequency'])} in total)"):
            st.dataframe(inv[["InvoiceNo", "InvoiceDate", "Lines", "Items", "Value"]], hide_index=True,
                         column_config={"Value": st.column_config.NumberColumn("£ gross", format="£%.2f")})
