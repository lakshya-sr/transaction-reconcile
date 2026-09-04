#!/usr/bin/env python3
"""
Feature Engineering for both GW↔Bank and ERP↔GW matching models.

Contains:
- GATEWAY_BANK_FEATURES: Features for gateway to bank matching
- ERP_GATEWAY_FEATURES: Features for ERP to gateway matching
- extract_gateway_bank_features(): Feature extraction for GW↔Bank
- extract_erp_gateway_features(): Feature extraction for ERP↔GW
"""

import ast
import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Sequence, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ============================================================================
# FEATURE COLUMN DEFINITIONS
# ============================================================================

GATEWAY_BANK_FEATURES = [
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

ERP_GATEWAY_FEATURES = [
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

# Backward compatibility alias
FEATURE_COLUMNS = GATEWAY_BANK_FEATURES

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@lru_cache(maxsize=5000)
def _parse_dt_cached(val_str: str) -> datetime:
    """Cached datetime parsing."""
    if not val_str or val_str == 'nan':
        return datetime(2026, 1, 1)
    s = val_str[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime(2026, 1, 1)


def _parse_dt(val) -> datetime:
    """Parse datetime with caching."""
    if isinstance(val, datetime):
        return val
    return _parse_dt_cached(str(val))


@lru_cache(maxsize=5000)
def _parse_invoices_cached(invoices_str: str) -> tuple:
    """Cached invoice parsing."""
    if not invoices_str or invoices_str == 'nan':
        return tuple()
    
    if invoices_str.startswith('['):
        try:
            parsed = ast.literal_eval(invoices_str)
            if isinstance(parsed, (list, tuple)):
                return tuple(str(inv) for inv in parsed if inv and str(inv) != 'nan')
        except (ValueError, SyntaxError):
            pass
    
    return (invoices_str.replace('"', '').replace("'", ""),)


def parse_invoices(raw_invoices) -> List[str]:
    """Parse invoices field which may be list, JSON string, or raw string."""
    if raw_invoices is None:
        return []
    if isinstance(raw_invoices, (list, tuple, np.ndarray)):
        return [str(inv) for inv in raw_invoices if inv and not pd.isna(inv)]
    if isinstance(raw_invoices, float) and pd.isna(raw_invoices):
        return []
    if isinstance(raw_invoices, str):
        return list(_parse_invoices_cached(raw_invoices.strip()))
    return []


@lru_cache(maxsize=10000)
def _fuzz_ratio_cached(str1: str, str2: str) -> float:
    """Cached fuzzy ratio calculation."""
    return fuzz.ratio(str1, str2) / 100.0


def _token_overlap(a: str, b: str) -> float:
    """Token-level Jaccard overlap between two strings."""
    set_a = set(a.upper().split())
    set_b = set(b.upper().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


# ============================================================================
# GATEWAY ↔ BANK FEATURE EXTRACTION
# ============================================================================

def extract_gateway_bank_features(
    gw_rows: Sequence[Union[dict, pd.Series]],
    bank_row: Union[dict, pd.Series],
) -> Dict[str, float]:
    """
    Extract features for gateway cluster against bank deposit.
    
    Args:
        gw_rows: List of gateway records forming the candidate cluster.
        bank_row: Single bank deposit record being matched against.
    
    Returns:
        Dictionary of feature_name -> float (aligned with GATEWAY_BANK_FEATURES).
    """
    if not gw_rows:
        return {col: 0.0 for col in GATEWAY_BANK_FEATURES}

    # Bank data
    bank_credit = float(bank_row.get("credit_amount", 0.0))
    bank_dt = bank_row.get("_dt") if "_dt" in bank_row else _parse_dt(bank_row.get("value_date", "2026-01-01"))
    remittance = str(bank_row.get("remittance_info") or "")
    remittance_upper = remittance.upper()

    # Gateway data
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
        
        invs = g.get("_invoices") if "_invoices" in g else parse_invoices(g.get("invoices"))
        all_invoices.extend(invs)

    # Amount features
    cluster_size = len(gw_rows)
    sum_gw_net = sum(gw_nets)
    amount_diff_abs = abs(bank_credit - sum_gw_net)
    amount_diff_pct = amount_diff_abs / (bank_credit + 1e-5)

    # Temporal features
    time_deltas = [(bank_dt - dt).total_seconds() / 3600.0 for dt in gw_dts]
    time_delta_min_hrs = min(time_deltas)
    time_delta_max_hrs = max(time_deltas)
    time_span_hrs = (max(gw_dts) - min(gw_dts)).total_seconds() / 3600.0 if len(gw_dts) > 1 else 0.0
    is_single_day = 1.0 if len(dates_set) <= 1 else 0.0

    # UTR features
    best_utr_fuzz = 0.0
    utr_prefix_match = 0.0
    
    for utr in all_utrs:
        utr_clean = utr.upper()
        
        # Fast exact match
        if utr_clean and utr_clean in remittance_upper:
            best_utr_fuzz = 1.0
            utr_prefix_match = 1.0
            break
        
        # Fast prefix match
        if len(utr_clean) >= 6:
            prefix8 = utr_clean[:8] if len(utr_clean) >= 8 else utr_clean[:6]
            if prefix8 in remittance_upper:
                utr_prefix_match = 1.0
                best_utr_fuzz = max(best_utr_fuzz, 0.8)
                continue
        
        # Fuzzy match if no good match found
        if best_utr_fuzz < 0.8:
            ratio = _fuzz_ratio_cached(utr_clean, remittance_upper)
            best_utr_fuzz = max(best_utr_fuzz, ratio)

    # Invoice features
    best_inv_fuzz = 0.0
    inv_prefix_match = 0.0
    
    for inv in all_invoices:
        inv_clean = inv.upper()
        stripped_inv = inv_clean.replace("INV-", "").replace("ORD-", "")
        
        # Fast exact match
        if stripped_inv and (stripped_inv in remittance_upper or inv_clean in remittance_upper):
            inv_prefix_match = 1.0
            best_inv_fuzz = 1.0
            break
        
        # Fuzzy match if needed
        if best_inv_fuzz < 0.8:
            ratio = _fuzz_ratio_cached(inv_clean, remittance_upper)
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


# ============================================================================
# ERP ↔ GATEWAY FEATURE EXTRACTION
# ============================================================================

def extract_erp_gateway_features(
    erp_rows: Sequence[Union[dict, pd.Series]],
    gw_row: Union[dict, pd.Series],
) -> Dict[str, float]:
    """
    Extract features for ERP cluster against gateway settlement.
    
    Args:
        erp_rows: List of ERP ledger records forming the candidate cluster.
        gw_row: Single gateway settlement record being matched against.
    
    Returns:
        Dictionary of feature_name -> float (aligned with ERP_GATEWAY_FEATURES).
    """
    if not erp_rows:
        return {col: 0.0 for col in ERP_GATEWAY_FEATURES}

    # Gateway data
    gw_gross = float(gw_row.get("_gross", gw_row.get("gross_amount", 0.0)))
    gw_dt = gw_row.get("_dt") if "_dt" in gw_row else _parse_dt(gw_row.get("settled_at", "2026-01-01"))
    gw_invoices = gw_row.get("_invoices") if "_invoices" in gw_row else parse_invoices(gw_row.get("invoices"))

    # ERP data
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

    # Amount features
    cluster_size = len(erp_rows)
    sum_erp_gross = sum(erp_grosses)
    gross_diff_abs = abs(gw_gross - sum_erp_gross)
    gross_diff_pct = gross_diff_abs / (gw_gross + 1e-5)

    # Temporal features
    time_deltas = [(gw_dt - dt).total_seconds() / 3600.0 for dt in erp_dts]
    time_delta_min_hrs = min(time_deltas)
    time_delta_max_hrs = max(time_deltas)
    time_span_hrs = (max(erp_dts) - min(erp_dts)).total_seconds() / 3600.0 if len(erp_dts) > 1 else 0.0
    is_single_day = 1.0 if len(dates_set) <= 1 else 0.0

    # Invoice NLP features
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


# Backward compatibility alias
extract_cluster_features = extract_gateway_bank_features