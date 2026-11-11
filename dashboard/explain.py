"""Plain-English explanations for staff, built from the same values the dashboard displays (no hard-coded figures)."""
import pandas as pd

from .config import SEGMENT_SHORT


def _short(name):
    return SEGMENT_SHORT.get(name, name)


def _name(text):
    """Readable product name: title case, no stray space before commas (as found in the raw descriptions)."""
    return " ".join(str(text).replace(" ,", ",").split()).title()


def descriptive_stats(lines: pd.DataFrame) -> str:
    """Explanation of the 'Cleaned purchase lines — descriptive statistics' table on Dataset & EDA."""
    q, p, v = lines["Quantity"], lines["UnitPrice"], lines["LineValue"]
    big_q = lines.loc[q.idxmax()]
    big_p = lines.loc[p.idxmax()]
    small_p = lines.loc[p.idxmin()]
    n_small_p = int((p == p.min()).sum())
    share_q12 = 100 * (q <= 12).mean()
    share_v25 = 100 * (v <= 25).mean()
    sk = lines[["Quantity", "UnitPrice", "LineValue"]].skew()
    return f"""
**Quantity (units per purchase line).** The average line contains **{q.mean():.2f} units**, but the median is
**{q.median():.2f} units**. The median is the middle observation when all lines are sorted, so a typical line has about
{q.median():.0f} units. The average is higher because a small number of exceptionally large quantities pull it up —
{share_q12:.1f}% of lines contain 12 units or fewer.

**Unit price (£).** The average unit price is **£{p.mean():.2f}**, while the median is **£{p.median():.2f}**. A small number of
expensive entries raise the average above the typical price.

**Line value (£ gross).** Each line's value is quantity × unit price. The average is **£{v.mean():.2f}**, the median
**£{v.median():.2f}**: {share_v25:.1f}% of lines are worth £25 or less, and a few high-value lines raise the average.

**Standard deviation (std).** This shows how much values vary around the average. The large values here
(**{q.std():.2f}** units, **£{p.std():.2f}**, **£{v.std():.2f}**) mean purchases differ a great deal from one line to the next,
partly because of a few extreme observations.

**Minimum and maximum.** The smallest values are {q.min():.0f} unit and £{p.min():.3f}
(£{p.min():.3f} appears on {n_small_p} lines for "{_name(small_p['Description'])}"). The largest values deserve a closer look:
- the biggest single line is **{int(big_q['Quantity']):,} units** of "{_name(big_q['Description'])}", worth
  **£{big_q['LineValue']:,.2f}** — far beyond a typical order;
- the highest unit price, **£{big_p['UnitPrice']:,.2f}**, is a "{_name(big_p['Description'])}" line — a service charge
  rather than a product price.

These records are kept in the data because they are genuine recorded transactions, but they should be checked (for example,
whether a very large order was later cancelled) before drawing conclusions from them.

**Skewness.** All three variables are strongly right-skewed (skewness {sk['Quantity']:.2f}, {sk['UnitPrice']:.2f} and
{sk['LineValue']:.2f}; 0 would mean a balanced shape). In simple terms, most purchase lines are relatively small, while a few
extremely large ones stretch the upper end of the range.

**What this means for the business.** The average alone does not describe a typical purchase — the median is a better guide
to what most customers actually buy. Unusually large transactions should be reviewed before they are used to judge customers
or set targets.
""".strip()


def customer_distribution(profiles: pd.DataFrame) -> str:
    """Explanation of the 'Customer distribution' chart on the Overview."""
    p = profiles
    lines = [f"- **{_short(r['Segment'])}:** {int(r['Customers']):,} customers ({r['Customer_Share_%']:.1f}% of customers)"
             for _, r in p.iterrows()]
    biggest = p.loc[p["Customers"].idxmax()]
    value_lines = [f"- **{_short(r['Segment'])}:** {r['Value_Share_%']:.1f}% of gross positive purchase value" for _, r in p.iterrows()]
    top = p.loc[p["Value_Share_%"].idxmax()]
    return f"""
**How many customers are in each segment.** The bars count customers:
{chr(10).join(lines)}

The **{_short(biggest['Segment'])}** segment is the largest group by number of customers.

**How much each segment buys.** The line under the chart shows a different measure — each segment's share of all gross
positive purchase value:
{chr(10).join(value_lines)}

**Two different measures.** *Share of customers* tells you how many people are in a segment; *share of gross purchase value*
tells you how much of the recorded purchasing those people account for. The two can be very different.

**Business insight.** {_short(top['Segment'])} customers make up **{top['Customer_Share_%']:.1f}% of customers** but contribute
**{top['Value_Share_%']:.1f}% of gross positive purchase value**. Keeping these customers is therefore an important potential
business priority. (Gross positive purchase value is not net revenue or profit.)
""".strip()


def rfm_comparison(profiles: pd.DataFrame, reference_date: str) -> str:
    """Explanation of the three 'Segment comparison' charts on the Overview."""
    p = profiles.set_index("Segment")
    order = list(profiles["Segment"])
    # identify the segments from their measured profile, not from their position or ID
    hs = profiles.loc[profiles["Monetary_Median"].idxmax(), "Segment"]
    risk = profiles.loc[profiles["Recency_Median"].idxmax(), "Segment"]
    occ = next((s for s in order if s not in (hs, risk)), hs)

    def vals(col, fmt):
        return "\n".join(f"- **{_short(s)}:** {fmt(p.loc[s, col])}" for s in order)

    r = lambda s: f"{p.loc[s, 'Recency_Median']:.0f}"
    f = lambda s: f"{p.loc[s, 'Frequency_Median']:.0f}"
    m = lambda s: f"£{p.loc[s, 'Monetary_Median']:,.0f}"
    biggest = profiles.loc[profiles["Customers"].idxmax(), "Segment"]
    return f"""
Each chart shows the **median** customer of each segment — the middle customer when the segment is sorted — so a few
unusual customers do not distort the picture.

**A. Recency (days)**
{vals('Recency_Median', lambda x: f'{x:.0f} days')}

Recency is the number of days between a customer's last recorded purchase and the analysis reference date
({reference_date}). **Lower means a more recent purchase.** For example, the median Recency for the {_short(hs)} segment is
{r(hs)} days: the middle customer in this segment last purchased about {r(hs)} days before the reference date. For
{_short(risk)} customers it is {r(risk)} days, so their last purchase was much longer ago.

**B. Frequency (invoices)**
{vals('Frequency_Median', lambda x: f'{x:.0f} invoice' + ('' if x == 1 else 's'))}

Frequency is the number of purchase invoices (separate orders) a customer placed during the analysis period. **Higher means
more frequent purchasing.** For example, the median Frequency for {_short(hs)} customers is {f(hs)} invoices, compared with
{f(occ)} for {_short(occ)} customers — the typical {_short(hs)} customer ordered more often during the period.

**C. Monetary (gross £)**
{vals('Monetary_Median', lambda x: f'£{x:,.0f}')}

These are **median gross purchase values per customer** for the whole period — not product prices and not segment totals.
For example, the median Monetary value for {_short(hs)} customers is {m(hs)}, substantially higher than the {m(occ)} median
for {_short(occ)} customers, indicating a higher typical gross purchase value.

**Summary**
- **{_short(hs)}** customers bought more recently, more often and for more money than the other segments.
- **{_short(biggest)}** customers make up the largest share of the customer base.
- **{_short(risk)}** customers have the longest gap since their last purchase and the lowest purchasing frequency.

**Possible actions** *(potential strategies to test, not guaranteed outcomes)*: focus on retaining {_short(hs)} customers,
encourage repeat purchases among {_short(occ)} customers, and consider a re-engagement campaign for {_short(risk)} customers.
""".strip()
