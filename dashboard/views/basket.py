"""Market Basket Analysis: products frequently bought together (Apriori + association rules)."""
import altair as alt
import streamlit as st

from .. import charts, products as P, ui
from ..state import get_products
from .product_intel import eligibility_note


@st.cache_data(show_spinner="Finding frequent product combinations…", max_entries=20)
def _rules(_pdata, token, min_support, min_conf, min_lift):
    X, _, prod = _pdata.basket
    res = P.apriori(X, prod, min_support, max_len=3)
    return P.association_rules(res, _pdata.names, min_conf, min_lift), res["n_baskets"], res["min_count"], len(res["counts"])


def render(ctx, go):
    ui.header("Market Basket Analysis", "Products Frequently Bought Together",
              "Association rules found with the Apriori algorithm on invoice baskets. A separate analysis from the customer segmentation.")
    pdata = get_products()
    if pdata is None:
        ui.unavailable("The purchase-line file")
        return
    eligibility_note(pdata)
    c1, c2, c3, c4 = st.columns(4)
    sup = c1.slider("Minimum support (%)", 0.5, 5.0, 1.5, 0.5, key="mb_sup",
                    help="Share of all invoices that must contain the whole product combination.") / 100
    conf = c2.slider("Minimum confidence (%)", 10, 90, 40, 5, key="mb_conf") / 100
    lift = c3.slider("Minimum lift", 1.0, 20.0, 2.0, 0.5, key="mb_lift")
    top = c4.slider("Rules shown", 5, 50, 15, 5, key="mb_top")
    rules, n, min_count, n_sets = _rules(pdata, pdata.info["eligible_lines"], sup, conf, lift)
    flt = st.multiselect("Only rules involving these products (optional)", sorted(set(rules["Antecedent"]) | set(rules["Consequent"])),
                         key="mb_prod")
    if flt:
        rules = rules[rules["Antecedent"].isin(flt) | rules["Consequent"].isin(flt)]
    ui.strip([("Invoice baskets", f"{n:,}"), ("Minimum invoices for an itemset", f"{min_count:,}"), ("Frequent itemsets", f"{n_sets:,}"),
              ("Rules meeting the settings", f"{len(rules):,}")])
    if rules.empty:
        ui.callout("No rules meet these settings. Lower the minimum support, confidence or lift, or clear the product filter.", "warn",
                   label="No results")
        return
    shown = rules.head(top).assign(Rule=lambda d: d["Antecedent"] + "  →  " + d["Consequent"])
    st.dataframe(shown[["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Invoices"]], hide_index=True, **ui.FULL,
                 column_config={"Antecedent": "If an invoice contains…", "Consequent": "…it often also contains",
                                "Support": st.column_config.NumberColumn("Support", format="%.3f"),
                                "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                                "Lift": st.column_config.NumberColumn("Lift", format="%.1f"),
                                "Invoices": st.column_config.NumberColumn("Invoices with all items", format="%d")})
    metric = st.segmented_control("Chart metric", ["Lift", "Confidence", "Support"], default="Lift", key="mb_metric") or "Lift"
    charts.show(charts.hbar(shown, "Rule", metric, f"Top {len(shown)} rules by {metric.lower()}", "#C9A227", metric,
                            fmt=".1f" if metric == "Lift" else ".3f", label_limit=520,
                            tooltip=["Rule", alt.Tooltip("Support:Q", format=".3f"), alt.Tooltip("Confidence:Q", format=".2f"),
                                     alt.Tooltip("Lift:Q", format=".1f"), "Invoices"]), key="mb_chart")
    best = shown.iloc[0]
    ui.callout(f"Example: {best['Confidence']:.0%} of invoices containing {best['Antecedent']} also contain {best['Consequent']} "
               f"({int(best['Invoices']):,} invoices, {best['Support']:.1%} of all invoices) — {best['Lift']:.1f} times more often than if "
               "the products were bought independently.", "calculated")
    ui.explainer("How to read support, confidence and lift", f"Each invoice is one basket; a product counts once per invoice.\n\n"
                 "- **Support**: share of all invoices that contain every product in the rule.\n"
                 "- **Confidence**: of the invoices containing the first product(s), the share that also contain the second.\n"
                 "- **Lift**: confidence ÷ the overall share of invoices containing the second product. Above 1 means the products appear "
                 "together more often than chance would suggest.\n\n"
                 "Rules can suggest bundles, product placement, recommendations or cross-selling tests. Buying together does not prove that "
                 "one product causes the other to be bought, and a bundle is not guaranteed to raise sales — test it first. Rules supported "
                 "by few invoices are weak evidence, which is why a minimum support is required.")
