#!/usr/bin/env python3
"""
Feature Engineering Module for Cluster-Level ERP<->Gateway Matching.

Extracts aggregated financial, temporal, and token-similarity NLP features
for evaluating candidate ERP clusters against Gateway settlement records.

Key differences from GW<->Bank features.py:
- No UTR matching (ERP<->GW doesn't carry bank UTR)
- Invoice matching: ERP.invoice_number vs GW.invoices (list field)
- Amount: ERP.gross_amount vs GW.gross_amount (pure match, no fee noise)
- Temporal: ERP.entry_date vs GW.settled_at
- Extra features: invoice_token_overlap, exact_invoice_match
"""

import ast
from datetime import datetime
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "cluster_size",
    "gross_diff_abs",
    "gross_diff_pct",
    "time_delta_min_hrs",
    "time_delta_max_hrs",
    "time_span_hrs",
    "best_invoice_fuzz",
    "invoice_prefix_match",
    "invoice_token_overlap",
    "exact_invoice_match",
    "is_single_day",
]


def _parse_invoices(raw_invoices) -> List[str]:
    """Parse GW invoices field which may be a list, JSON string, or raw string."""
    if raw_invoices is None:
        return []
    if isinstance(raw_invoices, (list, tuple, np.ndarray)):
        return [str(inv) for inv in raw_invoices if inv and not pd.isna(inv)]
    if isinstance(raw_invoices, float) and pd.isna(raw_invoices):
        return []
    if isinstance(raw_invoices, str):
        val = raw_invoices.strip()
        if not val:
            return []
        if val.startswith("["):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)):
                    return [str(inv) for inv in parsed if inv and not pd.isna(inv)]
            except (ValueError, SyntaxError):
                pass
        return [val.replace('"', "").replace("'", "")]
    return []


def _parse_dt(val) -> datetime:
    if isinstance(val, datetime):
        return val
    s = str(val)[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime(2026, 1, 1)


def _token_overlap(a: str, b: str) -> float:
    """Token-level Jaccard overlap between two strings."""
    set_a = set(a.upper().split())
    set_b = set(b.upper().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def extract_cluster_features(
    erp_rows: Sequence[Union[dict, pd.Series]],
    gw_row: Union[dict, pd.Series],
) -> Dict[str, float]:
    """
    Computes flattened 1D numerical feature vector for a candidate ERP cluster
    against a Gateway settlement record.

    Args:
        erp_rows: List of ERP ledger records forming the candidate cluster.
        gw_row:   Single Gateway settlement record being matched against.

    Returns:
        Dictionary of feature_name -> float (aligned with FEATURE_COLUMNS).
    """
    if not erp_rows:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    gw_gross = float(gw_row.get("_gross", gw_row.get("gross_amount", 0.0)))
    gw_dt = gw_row.get("_dt") if "_dt" in gw_row else _parse_dt(gw_row.get("settled_at", "2026-01-01"))
    gw_invoices = gw_row.get("_invoices") if "_invoices" in gw_row else _parse_invoices(gw_row.get("invoices"))

    erp_dts = []
    erp_grosses = []
    erp_invoices = []
    dates_set = set()

    for e in erp_rows:
        dt = e.get("_dt") if "_dt" in e else _parse_dt(e.get("entry_date", "2026-01-01"))
        erp_dts.append(dt)
        dates_set.add(dt.strftime("%Y-%m-%d"))

        gross = float(e.get("_gross", e.get("gross_amount", 0.0)))
        erp_grosses.append(gross)

        inv = str(e.get("invoice_number") or "").strip()
        if inv:
            erp_invoices.append(inv)

    cluster_size = len(erp_rows)
    sum_erp_gross = sum(erp_grosses)
    gross_diff_abs = abs(gw_gross - sum_erp_gross)
    gross_diff_pct = gross_diff_abs / (gw_gross + 1e-5)

    # Temporal features (ERP entry_date vs GW settled_at)
    time_deltas = [(gw_dt - dt).total_seconds() / 3600.0 for dt in erp_dts]
    time_delta_min_hrs = min(time_deltas)
    time_delta_max_hrs = max(time_deltas)
    time_span_hrs = (max(erp_dts) - min(erp_dts)).total_seconds() / 3600.0 if len(erp_dts) > 1 else 0.0
    is_single_day = 1.0 if len(dates_set) <= 1 else 0.0

    # Invoice NLP features: compare ERP invoice_number(s) against GW invoices list
    best_inv_fuzz = 0.0
    inv_prefix_match = 0.0
    inv_token_overlap = 0.0
    exact_inv_match = 0.0

    for erp_inv in erp_invoices:
        erp_clean = erp_inv.upper()
        erp_stripped = erp_clean.replace("INV-", "").replace("ORD-", "").replace("INV_", "")

        for gw_inv in gw_invoices:
            gw_clean = str(gw_inv).upper()
            gw_stripped = gw_clean.replace("INV-", "").replace("ORD-", "").replace("INV_", "")

            # Exact match check
            if erp_clean == gw_clean or (erp_stripped and gw_stripped and erp_stripped == gw_stripped):
                exact_inv_match = 1.0
                best_inv_fuzz = 1.0
                inv_prefix_match = 1.0
                inv_token_overlap = 1.0
                break

            # Prefix match
            if (len(erp_stripped) >= 6 and len(gw_stripped) >= 6 and
                    (erp_stripped.startswith(gw_stripped[:6]) or gw_stripped.startswith(erp_stripped[:6]))):
                inv_prefix_match = 1.0
                best_inv_fuzz = max(best_inv_fuzz, 0.85)

            # Token overlap
            overlap = _token_overlap(erp_clean, gw_clean)
            inv_token_overlap = max(inv_token_overlap, overlap)

            # Fuzzy ratio
            ratio = fuzz.ratio(erp_clean, gw_clean) / 100.0
            best_inv_fuzz = max(best_inv_fuzz, ratio)

        if exact_inv_match == 1.0:
            break

    return {
        "cluster_size": float(cluster_size),
        "gross_diff_abs": round(gross_diff_abs, 4),
        "gross_diff_pct": round(gross_diff_pct, 6),
        "time_delta_min_hrs": round(time_delta_min_hrs, 2),
        "time_delta_max_hrs": round(time_delta_max_hrs, 2),
        "time_span_hrs": round(time_span_hrs, 2),
        "best_invoice_fuzz": round(best_inv_fuzz, 4),
        "invoice_prefix_match": inv_prefix_match,
        "invoice_token_overlap": round(inv_token_overlap, 4),
        "exact_invoice_match": exact_inv_match,
        "is_single_day": is_single_day,
    }
