"""Altair chart builders sharing one dark theme and one segment colour scale."""
import altair as alt
import pandas as pd
import streamlit as st

from .ui import CHART_FULL

FONT = "Inter, system-ui, sans-serif"


@alt.theme.register("segmentiq_light", enable=True)
def _theme():
    return alt.theme.ThemeConfig({
        "config": {
            "background": "#FFFFFF",
            "padding": {"left": 6, "right": 10, "top": 8, "bottom": 6},
            "autosize": {"type": "fit", "contains": "padding"},  # padding counted inside the container width -> no overflow
            "view": {"stroke": "transparent"},
            "font": FONT,
            "title": {"font": FONT, "fontSize": 14, "fontWeight": 600, "color": "#182033", "anchor": "start", "offset": 12},
            "axis": {"labelColor": "#5D6880", "titleColor": "#4A5468", "gridColor": "#EEF1F6", "domainColor": "#D9DEEA",
                     "tickColor": "#D9DEEA", "labelFont": FONT, "titleFont": FONT, "titleFontWeight": 500, "labelFontSize": 12,
                     "titleFontSize": 12.5},
            # hide colliding tick labels on numeric axes only (category labels must always stay visible)
            "axisQuantitative": {"labelOverlap": "greedy", "labelSeparation": 6},
            "legend": {"labelColor": "#4A5468", "titleColor": "#5D6880", "labelFont": FONT, "titleFont": FONT, "labelFontSize": 12,
                       "titleFontSize": 12, "orient": "bottom", "direction": "vertical", "labelLimit": 400, "symbolSize": 90},
            "header": {"labelColor": "#4A5468", "titleColor": "#4A5468", "labelFont": FONT},
            "range": {"category": ["#6655D9", "#2E9BB0", "#2F9E77", "#C9A227", "#D08A2E", "#C25E8A", "#4C6FD6", "#8B7FE0"]},
        }
    })


alt.data_transformers.disable_max_rows()


def seg_scale(ctx):
    return alt.Scale(domain=list(ctx.colors), range=list(ctx.colors.values()))


def show(chart, key=None):
    st.altair_chart(chart, theme=None, key=key, **CHART_FULL)


def bar_segments(ctx, df, value, title, fmt=",.0f", value_title=None):
    base = alt.Chart(df).encode(
        y=alt.Y("Segment:N", sort=list(ctx.colors), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{value}:Q", title=value_title or value),
        color=alt.Color("Segment:N", scale=seg_scale(ctx), legend=None),
        tooltip=[alt.Tooltip("Segment:N"), alt.Tooltip(f"{value}:Q", format=fmt, title=value_title or value)])
    text = base.mark_text(align="left", dx=4, color="#4A5468", fontSize=11).encode(text=alt.Text(f"{value}:Q", format=fmt))
    return (base.mark_bar(cornerRadiusEnd=4, height=22) + text).properties(title=title, height=46 * df["Segment"].nunique() + 30)


def histogram(ctx, df, col, title, log=False, by_segment=True, bins=40, x_title=None):
    d = df[[col, "Segment"]].copy()
    field = col
    if log:
        import numpy as np
        d[f"log1p_{col}"] = np.log1p(d[col])
        field = f"log1p_{col}"
    enc = dict(x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=x_title or (f"log1p({col})" if log else col),
                       axis=alt.Axis(tickCount=5, labelOverlap="greedy", labelExpr="datum.value == 0 ? '0' : format(datum.value, '~s')")),
               y=alt.Y("count():Q", title="Customers", stack=None))
    if by_segment:
        enc.update(color=alt.Color("Segment:N", scale=seg_scale(ctx), legend=alt.Legend(title=None)),
                   opacity=alt.value(0.75))
    else:
        enc.update(color=alt.value("#6655D9"))
    return alt.Chart(d).mark_bar().encode(**enc, tooltip=[alt.Tooltip("count():Q", title="Customers")]).properties(title=title, height=260)


def boxplot(ctx, df, col, title, log=False):
    from .config import SEGMENT_SHORT
    df = df[["Segment", col]].assign(Short=df["Segment"].map(lambda s: SEGMENT_SHORT.get(s, s)))
    return alt.Chart(df).mark_boxplot(extent=1.5, size=22, median={"color": "#182033"}).encode(
        y=alt.Y("Short:N", sort=[SEGMENT_SHORT.get(s, s) for s in ctx.colors], title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X(f"{col}:Q", scale=alt.Scale(type="symlog" if log else "linear"), title=None,
                axis=alt.Axis(labelExpr="datum.value == 0 ? '0' : format(datum.value, '~s')", tickCount=4)),
        color=alt.Color("Segment:N", scale=seg_scale(ctx), legend=None)).properties(title=title, height=170)


def _axis_title(field, log):
    unit = {"Recency": "days", "Frequency": "invoices", "Monetary": "£ gross"}.get(field)
    t = f"{field} ({unit})" if unit else field
    return f"{t}, log scale" if log else t


def scatter(ctx, df, x, y, title, logx=False, logy=False, highlight=None, tooltip_extra=(), size=14):
    sel_opacity = alt.condition(alt.FieldOneOfPredicate(field="Segment", oneOf=highlight), alt.value(0.75), alt.value(0.06)) \
        if highlight is not None else alt.value(0.6)
    return alt.Chart(df).mark_circle(size=size).encode(
        x=alt.X(f"{x}:Q", scale=alt.Scale(type="log" if logx else "linear", zero=False), title=_axis_title(x, logx)),
        y=alt.Y(f"{y}:Q", scale=alt.Scale(type="log" if logy else "linear", zero=False), title=_axis_title(y, logy)),
        color=alt.Color("Segment:N", scale=seg_scale(ctx), legend=alt.Legend(title=None)),
        opacity=sel_opacity,
        tooltip=[alt.Tooltip("CustomerID:N"), alt.Tooltip("Segment:N"), alt.Tooltip("Cluster:N"),
                 alt.Tooltip("Recency:Q", title="Recency (days)"), alt.Tooltip("Frequency:Q", title="Frequency (invoices)"),
                 alt.Tooltip("Monetary:Q", format=",.2f", title="Monetary (£ gross)"), *tooltip_extra]
    ).properties(title=title, height=420).interactive()


def heatmap(df_long, x, y, value, title, fmt=".2f", scheme="purpleblue", domain=None):
    base = alt.Chart(df_long).encode(x=alt.X(f"{x}:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, orient="top", labelOverlap=False)),
                                     y=alt.Y(f"{y}:N", title=None, sort=None, axis=alt.Axis(labelOverlap=False)))
    rect = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme=scheme, domain=domain), legend=alt.Legend(title=None, orient="right")),
        tooltip=[x, y, alt.Tooltip(f"{value}:Q", format=fmt)])
    # white numbers on strong (dark) cells, dark numbers on pale cells
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.Text(f"{value}:Q", format=fmt),
        color=alt.condition(f"abs(datum['{value}']) > 0.55", alt.value("#FFFFFF"), alt.value("#182033")))
    return (rect + text).properties(title=title, height=max(200, 56 * df_long[y].nunique()))


def line(df, x, y, title, color=None, ctx=None, y_title=None, x_type="T", fmt=",.0f", point=True):
    enc = dict(x=alt.X(f"{x}:{x_type}", title=None), y=alt.Y(f"{y}:Q", title=y_title or y),
               tooltip=[alt.Tooltip(f"{x}:{x_type}"), alt.Tooltip(f"{y}:Q", format=fmt, title=y_title or y)])
    if color:
        scale = seg_scale(ctx) if (ctx is not None and color == "Segment") else alt.Undefined
        enc["color"] = alt.Color(f"{color}:N", scale=scale, legend=alt.Legend(title=None))
        enc["tooltip"] = enc["tooltip"] + [alt.Tooltip(f"{color}:N")]
    else:
        enc["color"] = alt.value("#6655D9")
    return alt.Chart(df).mark_line(point=point, strokeWidth=2).encode(**enc).properties(title=title, height=300)


def grouped_bars(df, cat, group, value, title, color_scale=None, fmt=".2f", y_title=None):
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{cat}:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset(f"{group}:N"),
        y=alt.Y(f"{value}:Q", title=y_title or value),
        color=alt.Color(f"{group}:N", scale=color_scale or alt.Undefined, legend=alt.Legend(title=None, orient="bottom", columns=1)),
        tooltip=[cat, group, alt.Tooltip(f"{value}:Q", format=fmt)]).properties(title=title, height=300)


def metric_curve(df, value, title, best_k, better):
    base = alt.Chart(df).encode(x=alt.X("K:O", title="Number of clusters (K)", axis=alt.Axis(labelAngle=0)),
                                y=alt.Y(f"{value}:Q", title=value, scale=alt.Scale(zero=False)))
    color = "Algorithm:N" if "Algorithm" in df.columns else alt.value("#6655D9")
    lines = base.mark_line(point=True, strokeWidth=2).encode(color=alt.Color(color, legend=alt.Legend(title=None))
                                                              if isinstance(color, str) else color,
                                                              tooltip=["K:O", alt.Tooltip(f"{value}:Q", format=",.4f")] +
                                                              (["Algorithm:N"] if "Algorithm" in df.columns else []))
    rule = alt.Chart(pd.DataFrame({"K": [best_k]})).mark_rule(color="#16845B", strokeDash=[4, 4]).encode(x="K:O")
    return (lines + rule).properties(title=alt.TitleParams(title, subtitle=better.capitalize(), subtitleColor="#5D6880",
                                                           subtitleFontSize=12), height=260)


def hbar(df, y, x, title, color="#6655D9", x_title=None, fmt=",.0f", tooltip=None, label_limit=340):
    """Ranked horizontal bars with value labels (shared by the product, sales and basket pages)."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=label_limit, labelOverlap=False)),
        x=alt.X(f"{x}:Q", title=x_title or x),
        tooltip=tooltip or [alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt, title=x_title or x)])
    text = base.mark_text(align="left", dx=4, color="#4A5468", fontSize=11).encode(text=alt.Text(f"{x}:Q", format=fmt))
    return (base.mark_bar(color=color, cornerRadiusEnd=4) + text).properties(title=title, height=max(150, 27 * len(df) + 40))


def monthly_bars(df, value, title, y_title, color="#6655D9", fmt=",.0f"):
    """Monthly bars; the partial final month (data end 9 Dec 2011) is drawn lighter and labelled in the tooltip."""
    d = df.assign(Period=df["Partial"].map({True: "Partial month (1–9 Dec 2011)", False: "Full month"}))
    return alt.Chart(d).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("yearmonth(Month):O", title=None, axis=alt.Axis(labelAngle=-40, format="%b %y")),
        y=alt.Y(f"{value}:Q", title=y_title),
        color=alt.Color("Period:N", scale=alt.Scale(domain=["Full month", "Partial month (1–9 Dec 2011)"], range=[color, "#C9C3F0"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("yearmonth(Month):O", title="Month"), alt.Tooltip(f"{value}:Q", format=fmt, title=y_title), "Period"]
    ).properties(title=title, height=280)
