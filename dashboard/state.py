"""Cached, validated application context shared by every view."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from . import core
from .config import FILES, SEGMENT_COLORS, EXTRA_COLORS, data_dir, verification_dir


@dataclass
class Context:
    data_dir: Path
    files: dict
    validation: core.Validation
    customers: pd.DataFrame
    profiles: pd.DataFrame
    segments: list
    colors: dict
    lines: pd.DataFrame | None
    invoices: pd.DataFrame | None
    eval5: pd.DataFrame | None
    audit: pd.DataFrame | None
    inspection: pd.DataFrame | None
    verif: dict

    @property
    def medians(self):
        return self.customers[core.CORE].median()

    @property
    def has_lines(self):
        return self.lines is not None


def _fingerprint(ddir: Path) -> tuple:
    out = []
    for name, _, _ in FILES.values():
        p = ddir / name
        out.append((name, p.stat().st_mtime_ns if p.exists() else None))
    vdir = verification_dir()
    for rel in ["independent_check_output.json", "check_segments_output.json", "sensitivity/sensitivity_summary.json"]:
        p = vdir / rel
        out.append((rel, p.stat().st_mtime_ns if p.exists() else None))
    return tuple(out)


@st.cache_resource(show_spinner="Loading and validating project data…", max_entries=2)
def _load(ddir_str: str, fingerprint: tuple) -> Context:
    ddir = Path(ddir_str)
    files = core.file_status(ddir)
    v = core.Validation()
    for key, stt in files.items():
        if stt.required and not stt.ok:
            raise core.DataError(f"Required file {stt.name}: {stt.message}")

    def opt(key, loader=core.read_csv):
        stt = files[key]
        if not stt.ok:
            return None
        df = loader(stt.path)
        stt.rows = len(df)
        return df

    rfm = opt("rfm")
    labels = opt("labels")
    customers = core.build_customers(rfm, labels, v)
    lines = opt("lines", core.load_lines)
    if lines is not None:
        seg_map = customers.set_index("CustomerID")["Segment"]
        lines["Segment"] = lines["CustomerID"].map(seg_map).fillna("Unlabelled").astype("category")
        unmatched = int((lines["Segment"] == "Unlabelled").sum())
        v.add("Every purchase line maps to a labelled customer (CustomerID)", unmatched == 0, f"{unmatched} unmatched lines")
    profiles = core.segment_profiles(customers)
    core.validate_profiles(customers, profiles, opt("profiles7"), v)
    audit = opt("audit")
    core.validate_against_expected(customers, lines, audit, v)
    segs = core.segment_order(customers)
    colors = {s: SEGMENT_COLORS.get(s, EXTRA_COLORS[i % len(EXTRA_COLORS)]) for i, s in enumerate(segs)}
    return Context(ddir, files, v, customers, profiles, segs, colors, lines,
                   core.invoices_table(lines) if lines is not None else None,
                   opt("eval5"), audit, opt("inspection"),
                   core.load_verification(verification_dir()))


def get_context() -> Context:
    ddir = data_dir()
    return _load(str(ddir), _fingerprint(ddir))


def clear_cache():
    _load.clear()
    st.cache_data.clear()


@dataclass
class ProductData:
    lines: pd.DataFrame          # eligible purchase lines (see dashboard/products.py)
    info: dict
    names: pd.Series             # StockCode -> readable product name
    basket: tuple                # (invoice x product presence matrix, invoice index, product index)


@st.cache_resource(show_spinner="Preparing product data…", max_entries=2)
def _load_products(ddir_str: str, fingerprint: tuple) -> ProductData:
    from . import products
    ctx = get_context()
    rev = None
    stt = ctx.files.get("reversed")
    if stt is not None and stt.ok:
        rev = pd.read_csv(stt.path, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    e, info = products.eligible_lines(ctx.lines, rev)
    return ProductData(e, info, products.product_names(e), products.basket_matrix(e))


def get_products() -> ProductData | None:
    ctx = get_context()
    if not ctx.has_lines:
        return None
    ddir = data_dir()
    return _load_products(str(ddir), _fingerprint(ddir))
