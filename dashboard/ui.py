"""SegmentIQ design system — premium light theme.

Palette: background #F7F8FC · cards #FFFFFF · text #182033 · secondary text #5D6880 · borders #E5E9F2 ·
accent #6655D9 · positive #0F6E4B · warning #8F5E12 (text).
Type scale: 12 (labels) · 14 (small) · 15 (body) · 20 (section) · 30 (page title) · 30 (KPI value).
Spacing scale: 4 / 8 / 12 / 16 / 24 / 32 px.
"""
from html import escape

import streamlit as st
import inspect


def _full_width(fn):
    """Full-width keyword for a Streamlit element: width="stretch" where supported (newer releases), else the older
    use_container_width=True (e.g. st.altair_chart in Streamlit 1.49)."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


FULL = _full_width(st.button)          # buttons, download buttons and tables
CHART_FULL = _full_width(st.altair_chart)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#F7F8FC;--card:#FFFFFF;--text:#182033;--muted:#5D6880;--dim:#687188;--border:#E5E9F2;--border2:#D9DEEA;
--accent:#6655D9;--accent-soft:#EFEDFC;--good:#0F6E4B;--good-soft:#E8F5EF;--warn:#8F5E12;--warn-soft:#FBF3E4;
--shadow:0 1px 2px rgba(24,32,51,.04),0 1px 3px rgba(24,32,51,.06);}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp{background:var(--bg);color:var(--text);font-size:15px;}
[data-testid="stHeader"]{background:rgba(247,248,252,.96);backdrop-filter:blur(6px);border-bottom:1px solid #EDEFF5;}
.block-container{padding-top:4.6rem;padding-bottom:3rem;max-width:1200px;}
h1,h2,h3,h4,h5{color:var(--text)!important;letter-spacing:-.01em;}
p,li{color:var(--text);}
/* ---- sidebar ---- */
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:.6rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:3px;}
section[data-testid="stSidebar"] .stButton button{width:100%;justify-content:flex-start;text-align:left;padding:7px 12px;min-height:36px;
 border-radius:8px;font-size:14px;font-weight:500;color:var(--muted);border:1px solid transparent;background:transparent;}
section[data-testid="stSidebar"] .stButton button > div{justify-content:flex-start;}
section[data-testid="stSidebar"] .stButton button:hover{background:#F4F5FA;color:var(--text);}
section[data-testid="stSidebar"] .stButton button[kind="primary"]{background:var(--accent-soft);color:var(--accent);font-weight:600;
 border:1px solid transparent;box-shadow:inset 3px 0 0 var(--accent);}
section[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"]{font-size:18px;}
.brand{padding:4px 8px 16px 8px;margin-bottom:10px;border-bottom:1px solid var(--border);}
.brand-name{font-weight:700;font-size:19px;letter-spacing:.02em;color:var(--text);}
.brand-name span{color:var(--accent);}
.brand-sub{font-size:12.5px;color:var(--muted);margin-top:2px;line-height:1.35;}
.side-foot{font-size:12px;color:var(--dim);margin-top:18px;padding:12px 8px 0 8px;border-top:1px solid var(--border);line-height:1.5;}
/* ---- headers ---- */
.eyebrow{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);}
.page-title{font-weight:700;font-size:30px;letter-spacing:-.02em;color:var(--text);margin:6px 0 8px 0;line-height:1.2;}
.page-desc{color:var(--muted);font-size:15.5px;max-width:800px;line-height:1.6;}
.divider{height:1px;background:var(--border);margin:20px 0 24px 0;}
.section{font-size:20px;font-weight:650;color:var(--text);margin:28px 0 2px 0;letter-spacing:-.01em;}
.section-cap{font-size:14px;color:var(--muted);margin:2px 0 14px 0;}
.lead{font-size:18px;font-weight:600;color:var(--text);margin:0 0 6px 0;}
/* ---- KPI cards ---- */
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:0 0 16px 0;}
.kpi-grid.c3{grid-template-columns:repeat(3,minmax(0,1fr));}
.kpi-grid.c2{grid-template-columns:repeat(2,minmax(0,1fr));}
@media (max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:900px){.kpi-grid.c3{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:640px){.kpi-grid,.kpi-grid.c3,.kpi-grid.c2{grid-template-columns:1fr;}}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);min-height:124px;}
.kpi .l{font-size:13px;font-weight:600;color:var(--muted);display:flex;justify-content:space-between;gap:8px;}
.kpi .i{color:var(--dim);cursor:help;font-weight:500;}
.kpi .v{font-size:30px;font-weight:700;letter-spacing:-.02em;color:var(--text);margin-top:8px;line-height:1.15;}
.kpi .s{font-size:13px;color:var(--muted);margin-top:6px;line-height:1.45;}
.strip{display:flex;flex-wrap:wrap;gap:10px 36px;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 20px;
 margin:0 0 16px 0;box-shadow:var(--shadow);}
.strip div{font-size:12.5px;color:var(--muted);}
.strip b{display:block;font-size:15px;color:var(--text);font-weight:600;margin-top:2px;}
/* ---- pipeline ---- */
.pipe{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:4px 0 18px 0;}
.pipe .st{flex:1;min-width:130px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px;box-shadow:var(--shadow);}
.pipe .st .n{font-size:20px;font-weight:700;color:var(--text);}
.pipe .st .t{font-size:12.5px;color:var(--muted);margin-top:2px;line-height:1.35;}
.pipe .st.final{border-color:var(--accent);background:var(--accent-soft);}
.pipe .ar{display:flex;align-items:center;color:var(--dim);font-size:18px;}
/* ---- cards ---- */
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;margin-bottom:16px;box-shadow:var(--shadow);}
.card h4{font-size:16px;font-weight:650;margin:0 0 10px 0;}
.seg-card{border-top:3px solid var(--c);}
.seg-row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:2px 10px;font-size:13.5px;color:var(--muted);
 padding:6px 0;border-bottom:1px solid #F0F2F7;}
.seg-row:last-child{border-bottom:none;}
.seg-row span{white-space:nowrap;}
.seg-row b{margin-left:auto;}
.seg-row b{color:var(--text);font-weight:600;white-space:nowrap;text-align:right;}
.seg-title{font-size:16px;font-weight:650;color:var(--text);display:flex;align-items:center;}
.seg-sub{font-size:12.5px;color:var(--muted);margin:4px 0 8px 0;line-height:1.35;}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle;}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:600;vertical-align:middle;
 background:#F1F3F8;color:var(--muted);}
.tag.observed{background:#EAF1FD;color:#2F5FB3;}
.tag.calculated{background:var(--accent-soft);color:var(--accent);}
.tag.recommendation{background:var(--warn-soft);color:var(--warn);}
.tag.good{background:var(--good-soft);color:var(--good);}
.tag.warn{background:var(--warn-soft);color:var(--warn);}
.note{border:1px solid var(--border);border-left:3px solid var(--c,#6655D9);background:var(--card);border-radius:10px;padding:12px 16px;
 margin:6px 0 14px 0;color:var(--muted);font-size:14px;line-height:1.55;}
.note b{color:var(--text);}
.small{font-size:13px;color:var(--muted);}
div[data-testid="stExpander"] details{border:1px solid var(--border);border-radius:12px;background:var(--card);}
div[data-testid="stExpander"] summary p{font-weight:600;font-size:14.5px;}
div[data-testid="stExpander"] summary:hover p{color:var(--accent);}
div[data-testid="stExpander"] details[open] summary{border-bottom:1px solid var(--border);}
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li{margin-bottom:2px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-size:14.5px;font-weight:500;padding:10px 14px;}
label[data-testid="stWidgetLabel"] p{font-size:13.5px;color:var(--muted);font-weight:500;}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:10px;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def brand(name, sub):
    sub = "<br>".join(escape(part) for part in sub.split("<br>"))
    st.sidebar.markdown(f'<div class="brand"><div class="brand-name">{escape(name[:-2])}<span>{escape(name[-2:])}</span></div>'
                        f'<div class="brand-sub">{sub}</div></div>', unsafe_allow_html=True)


def sidebar_footer(text):
    st.sidebar.markdown(f'<div class="side-foot">{text}</div>', unsafe_allow_html=True)


def header(eyebrow, title, desc=None):
    html = f'<div class="eyebrow">{escape(eyebrow)}</div><div class="page-title">{escape(title)}</div>'
    if desc:
        html += f'<div class="page-desc">{escape(desc)}</div>'
    st.markdown(html + '<div class="divider"></div>', unsafe_allow_html=True)


def section(title, caption=None):
    st.markdown(f'<div class="section">{escape(title)}</div>' + (f'<div class="section-cap">{escape(caption)}</div>' if caption else
                                                                "<div style='height:10px'></div>"), unsafe_allow_html=True)


def lead(text):
    st.markdown(f'<div class="lead">{escape(text)}</div>', unsafe_allow_html=True)


def kpis(items, cols=None):
    """items: (label, value, sub) or (label, value, sub, tooltip)."""
    cells = []
    for it in items:
        label, value, sub = it[0], it[1], it[2]
        tip = it[3] if len(it) > 3 else None
        info = f'<span class="i" title="{escape(tip)}">ⓘ</span>' if tip else ""
        cells.append(f'<div class="kpi"><div class="l"><span>{escape(label)}</span>{info}</div><div class="v">{escape(value)}</div>'
                     f'<div class="s">{escape(sub or "")}</div></div>')
    n = cols or min(4, len(items))
    cls = {4: "", 3: " c3", 2: " c2"}.get(n, "")
    for i in range(0, len(cells), n):
        st.markdown(f'<div class="kpi-grid{cls}">{"".join(cells[i:i + n])}</div>', unsafe_allow_html=True)


def strip(items):
    st.markdown('<div class="strip">' + "".join(f'<div>{escape(l)}<b>{escape(v)}</b></div>' for l, v in items) + "</div>",
                unsafe_allow_html=True)


def pipeline(steps):
    """steps: list of (number_text, label); the last step is highlighted."""
    parts = []
    for i, (n, t) in enumerate(steps):
        cls = " final" if i == len(steps) - 1 else ""
        parts.append(f'<div class="st{cls}"><div class="n">{escape(n)}</div><div class="t">{escape(t)}</div></div>')
        if i < len(steps) - 1:
            parts.append('<div class="ar">→</div>')
    st.markdown(f'<div class="pipe">{"".join(parts)}</div>', unsafe_allow_html=True)


def tag(text, kind):
    return f'<span class="tag {escape(kind)}">{escape(text)}</span>'


def callout(text, kind="observed", label=None):
    label = label or {"observed": "Observed", "calculated": "Finding", "recommendation": "Suggestion", "warn": "Note",
                      "good": "Note"}.get(kind, kind)
    c = {"observed": "#2F5FB3", "calculated": "#6655D9", "recommendation": "#B7791F", "warn": "#B7791F", "good": "#16845B"}.get(kind, "#6655D9")
    st.markdown(f'<div class="note" style="--c:{c}">{tag(label, kind)}&nbsp; {text}</div>', unsafe_allow_html=True)


def explainer(label, markdown_text, key=None):
    """Collapsed-by-default explanation panel for non-technical staff (click to open, click again to close)."""
    with st.expander(label, expanded=False, icon=":material/help_outline:"):
        st.markdown(markdown_text)


def unavailable(what):
    callout(f"{what} is not available in this copy of the project.", "warn", label="Unavailable")


def seg_label(name, color):
    return f'<span class="dot" style="background:{color}"></span>{escape(name)}'


def segment_card(row, color, extra_rows=()):
    """Compact card that stays readable in a narrow column: short labels, values that never wrap."""
    from .config import SEGMENT_SHORT
    rows = [("Customers", f"{int(row['Customers']):,} · {row['Customer_Share_%']:.1f}%"),
            ("Median recency", f"{row['Recency_Median']:.0f} days"), ("Median invoices", f"{row['Frequency_Median']:.0f}"),
            ("Median value", f"£{row['Monetary_Median']:,.0f}"), ("Value share", f"{row['Value_Share_%']:.1f}%"), *extra_rows]
    body = "".join(f'<div class="seg-row"><span>{escape(k)}</span><b>{escape(v)}</b></div>' for k, v in rows)
    title = SEGMENT_SHORT.get(row["Segment"], row["Segment"])
    st.markdown(f'<div class="card seg-card" style="--c:{color}"><div class="seg-title"><span class="dot" style="background:{color}"></span>'
                f'{escape(title)}</div><div class="seg-sub">{escape(row["Segment"])} · cluster {int(row["Cluster"])}</div>{body}</div>',
                unsafe_allow_html=True)


def short(name):
    from .config import SEGMENT_SHORT
    return SEGMENT_SHORT.get(name, name)


GROSS_DISCLAIMER = ("Gross positive purchase value reflects retained positive purchase lines. It is not net revenue or profit "
                    "and may not account for the full effect of returns and cancellations.")
