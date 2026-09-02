#!/usr/bin/env python3
"""
Feature Engineering Module for Cluster-Level Gateway↔Bank Matching.

Extracts aggregated financial, temporal, and token-similarity NLP features
for evaluating candidate Gateway clusters against Bank statement records.
"""

import ast
from datetime import datetime
from typing import Dict, List, Sequence, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "cluster_size",
    "amount_diff_abs",
    "amount_diff_pct",
    "time_delta_min_hrs",
    "time_delta_max_hrs",
    "time_span_hrs",
    "best_utr_fuzz",
    "utr_prefix_match",
    "best_invoice_fuzz",
    "invoice_prefix_match",
    "is_single_day",
]


def _parse_invoices(raw_invoices) -> List[str]:
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


def extract_cluster_features(
    gw_rows: Sequence[Union[dict, pd.Series]],
    bank_row: Union[dict, pd.Series],
) -> Dict[str, float]:
    """
    Computes flattened 1D numerical feature vector for a candidate Gateway cluster
    against a Bank statement deposit.
    """
    if not gw_rows:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    bank_credit = float(bank_row.get("credit_amount", 0.0))
    bank_dt = bank_row.get("_dt") if "_dt" in bank_row else _parse_dt(bank_row.get("value_date", "2026-01-01"))
    remittance = str(bank_row.get("remittance_info") or "")
    remittance_upper = remittance.upper()

    gw_dts = []
    gw_nets = []
    all_utrs = []
    all_invoices = []
    dates_set = set()

    for g in gw_rows:
        dt = g.get("_dt") if "_dt" in g else _parse_dt(g.get("settled_at", "2026-01-01"))
        gw_dts.append(dt)
        dates_set.add(dt.strftime("%Y-%m-%d"))

        net = float(g.get("_net", g.get("net_settled", 0.0)))
        gw_nets.append(net)

        utr = str(g.get("bank_utr") or "").strip()
        if utr:
            all_utrs.append(utr)

        invs = g.get("_invoices") if "_invoices" in g else _parse_invoices(g.get("invoices"))
        all_invoices.extend(invs)

    cluster_size = len(gw_rows)
    sum_gw_net = sum(gw_nets)
    amount_diff_abs = abs(bank_credit - sum_gw_net)
    amount_diff_pct = amount_diff_abs / (bank_credit + 1e-5)

    # Temporal features
    time_deltas = [(bank_dt - dt).total_seconds() / 3600.0 for dt in gw_dts]
    time_delta_min_hrs = min(time_deltas)
    time_delta_max_hrs = max(time_deltas)
    time_span_hrs = (max(gw_dts) - min(gw_dts)).total_seconds() / 3600.0
    is_single_day = 1.0 if len(dates_set) <= 1 else 0.0

    # UTR NLP features
    best_utr_fuzz = 0.0
    utr_prefix_match = 0.0
    for utr in all_utrs:
        utr_clean = utr.upper()
        if utr_clean and utr_clean in remittance_upper:
            best_utr_fuzz = 1.0
            utr_prefix_match = 1.0
            break
        if len(utr_clean) >= 6:
            prefix = utr_clean[:8] if len(utr_clean) >= 8 else utr_clean[:6]
            if prefix in remittance_upper:
                utr_prefix_match = 1.0
                best_utr_fuzz = max(best_utr_fuzz, 0.8)
        if best_utr_fuzz < 0.8:
            ratio = fuzz.ratio(utr_clean, remittance_upper) / 100.0
            best_utr_fuzz = max(best_utr_fuzz, ratio)

    # Invoice NLP features
    best_inv_fuzz = 0.0
    inv_prefix_match = 0.0
    for inv in all_invoices:
        inv_clean = inv.upper()
        stripped_inv = inv_clean.replace("INV-", "").replace("ORD-", "")
        if stripped_inv and (stripped_inv in remittance_upper or inv_clean in remittance_upper):
            inv_prefix_match = 1.0
            best_inv_fuzz = 1.0
            break
        if best_inv_fuzz < 0.8:
            ratio = fuzz.ratio(inv_clean, remittance_upper) / 100.0
            best_inv_fuzz = max(best_inv_fuzz, ratio)

    return {
        "cluster_size": float(cluster_size),
        "amount_diff_abs": round(amount_diff_abs, 4),
        "amount_diff_pct": round(amount_diff_pct, 6),
        "time_delta_min_hrs": round(time_delta_min_hrs, 2),
        "time_delta_max_hrs": round(time_delta_max_hrs, 2),
        "time_span_hrs": round(time_span_hrs, 2),
        "best_utr_fuzz": round(best_utr_fuzz, 4),
        "utr_prefix_match": utr_prefix_match,
        "best_invoice_fuzz": round(best_inv_fuzz, 4),
        "invoice_prefix_match": inv_prefix_match,
        "is_single_day": is_single_day,
    }
