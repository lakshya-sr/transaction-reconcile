#!/usr/bin/env python3
"""
Candidate Block Generator for ERP<->Gateway Cluster Matching.

Generates candidate ERP clusters for each orphaned Gateway record using:
1. Temporal windowing (ERP entry_date near GW settled_at)
2. Amount proximity (ERP.gross_amount sums close to GW.gross_amount)
3. Invoice prefix grouping (shared invoice prefix across ERPs)
4. N:1 bundled checkout grouping (multiple ERPs per one GW settlement)

Adapted from block_generator.py (GW<->Bank). Key differences:
- Target node is GW (not Bank), source nodes are ERP (not GW)
- Amount field: gross_amount (no fee deduction noise)
- Date field: ERP uses entry_date, GW uses settled_at
- No UTR; invoice prefix grouping replaces UTR prefix grouping
"""

import ast
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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


class ERPGWCandidateBlockGenerator:
    """
    Generates candidate ERP clusters for each orphaned Gateway record.

    Each candidate block contains:
        - gw_row: the target Gateway record
        - gw_id: Gateway payment_id
        - erp_rows: list of ERP records forming the candidate cluster
        - erp_ids: list of ERP entry_ids in the cluster
        - erp_ids_frozenset: frozenset for deduplication
        - block_type: one of '1:1', 'BundledCart', 'SequentialChunk', 'InvoicePrefix'
    """

    def __init__(
        self,
        max_delay_days: int = 5,
        max_amount_diff_pct: float = 0.20,
    ):
        self.max_delay_days = max_delay_days
        self.max_amount_diff_pct = max_amount_diff_pct

    def generate_blocks(
        self,
        orphan_erps: List[Dict[str, Any]],
        orphan_gws: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate ERP cluster blocks for each orphaned GW record.

        Args:
            orphan_erps: List of unmatched ERP ledger records (dicts).
            orphan_gws:  List of unmatched Gateway settlement records (dicts).

        Returns:
            List of candidate block dicts.
        """
        if not orphan_erps or not orphan_gws:
            return []

        # Pre-parse ERP records
        erp_parsed = []
        for e in orphan_erps:
            dt = _parse_dt(e.get("entry_date", "2026-01-01"))
            gross = float(e.get("gross_amount", 0.0))
            e_id = str(e.get("erp_entry_id", ""))
            inv = str(e.get("invoice_number") or "").strip().upper()
            erp_parsed.append({
                **e,
                "_pid": e_id,
                "_gross": gross,
                "_dt": dt,
                "_inv": inv,
            })

        # Sort ERPs by datetime for bisect windowing
        erp_parsed.sort(key=lambda x: x["_dt"])
        erp_dts = [e["_dt"] for e in erp_parsed]

        # Index ERPs by date string for daily grouping
        erp_by_date: Dict[str, List[dict]] = defaultdict(list)
        for e in erp_parsed:
            erp_by_date[e["_dt"].strftime("%Y-%m-%d")].append(e)

        # Index ERPs by invoice prefix (first 6 chars) for invoice grouping
        erp_by_inv_prefix: Dict[str, List[dict]] = defaultdict(list)
        for e in erp_parsed:
            inv = e["_inv"]
            inv_stripped = inv.replace("INV-", "").replace("ORD-", "").replace("INV_", "")
            if len(inv_stripped) >= 6:
                prefix = inv_stripped[:6]
                erp_by_inv_prefix[prefix].append(e)

        candidate_blocks = []
        seen_keys = set()

        for gw in orphan_gws:
            g_id = str(gw.get("payment_id", ""))
            gw_gross = float(gw.get("gross_amount", 0.0))
            gw_dt = _parse_dt(gw.get("settled_at", "2026-01-01"))
            gw_invoices = _parse_invoices(gw.get("invoices"))

            gw_rec = {
                **gw,
                "_pid": g_id,
                "_gross": gw_gross,
                "_dt": gw_dt,
                "_invoices": gw_invoices,
            }

            # Temporal window: ERPs within [gw_dt - max_delay_days, gw_dt + 1 day]
            window_start = gw_dt - timedelta(days=self.max_delay_days)
            window_end = gw_dt + timedelta(days=1)
            start_idx = bisect_left(erp_dts, window_start)
            end_idx = bisect_right(erp_dts, window_end)
            window_erps = erp_parsed[start_idx:end_idx]

            # 1. 1:1 Candidate Blocks
            for e in window_erps:
                amt_diff = abs(gw_gross - e["_gross"])
                if amt_diff / (gw_gross + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 5.0:
                    erp_set = frozenset([e["_pid"]])
                    key = (g_id, erp_set)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidate_blocks.append({
                            "block_type": "1:1",
                            "gw_row": gw_rec,
                            "gw_id": g_id,
                            "erp_rows": [e],
                            "erp_ids": [e["_pid"]],
                            "erp_ids_frozenset": erp_set,
                        })

            # 2. Bundled Cart / Daily Rollup (N:1): multiple ERPs per day summing to GW gross
            dates_in_window = set(e["_dt"].strftime("%Y-%m-%d") for e in window_erps)
            for d_str in dates_in_window:
                day_erps = [e for e in erp_by_date[d_str] if e in window_erps]
                if len(day_erps) >= 2:
                    total_gross = sum(e["_gross"] for e in day_erps)
                    amt_diff = abs(gw_gross - total_gross)
                    if amt_diff / (gw_gross + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 20.0:
                        erp_set = frozenset(e["_pid"] for e in day_erps)
                        key = (g_id, erp_set)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            candidate_blocks.append({
                                "block_type": "BundledCart",
                                "gw_row": gw_rec,
                                "gw_id": g_id,
                                "erp_rows": day_erps,
                                "erp_ids": [e["_pid"] for e in day_erps],
                                "erp_ids_frozenset": erp_set,
                            })

            # 3. Sequential Chunk Blocks (2 to 6 ERPs within each date)
            for d_str in dates_in_window:
                day_erps = [e for e in erp_by_date[d_str] if e in window_erps]
                n_day = len(day_erps)
                if n_day >= 3:
                    for k in range(2, min(7, n_day)):
                        for si in range(0, n_day - k + 1):
                            chunk = day_erps[si: si + k]
                            total_gross = sum(e["_gross"] for e in chunk)
                            amt_diff = abs(gw_gross - total_gross)
                            if amt_diff / (gw_gross + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 20.0:
                                erp_set = frozenset(e["_pid"] for e in chunk)
                                key = (g_id, erp_set)
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    candidate_blocks.append({
                                        "block_type": "SequentialChunk",
                                        "gw_row": gw_rec,
                                        "gw_id": g_id,
                                        "erp_rows": chunk,
                                        "erp_ids": [e["_pid"] for e in chunk],
                                        "erp_ids_frozenset": erp_set,
                                    })

            # 4. Invoice Prefix Grouping: ERPs sharing an invoice prefix with GW invoices
            for gw_inv in gw_invoices:
                gw_inv_stripped = str(gw_inv).upper().replace("INV-", "").replace("ORD-", "").replace("INV_", "")
                if len(gw_inv_stripped) >= 6:
                    prefix = gw_inv_stripped[:6]
                    inv_group = [e for e in erp_by_inv_prefix.get(prefix, []) if e in window_erps]
                    if len(inv_group) >= 1:
                        total_gross = sum(e["_gross"] for e in inv_group)
                        amt_diff = abs(gw_gross - total_gross)
                        if amt_diff / (gw_gross + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 20.0:
                            erp_set = frozenset(e["_pid"] for e in inv_group)
                            key = (g_id, erp_set)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                candidate_blocks.append({
                                    "block_type": "InvoicePrefix",
                                    "gw_row": gw_rec,
                                    "gw_id": g_id,
                                    "erp_rows": inv_group,
                                    "erp_ids": [e["_pid"] for e in inv_group],
                                    "erp_ids_frozenset": erp_set,
                                })

        return candidate_blocks
