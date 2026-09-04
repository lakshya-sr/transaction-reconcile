#!/usr/bin/env python3
"""
Candidate generation for both GW↔Bank and ERP↔GW matching.

This file combines:
- block_generator.py (GatewayBankCandidateGenerator)
- erp_gw_block_generator.py (ERPGatewayCandidateGenerator)

Logic is unchanged from the original files.
"""

import ast
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import pandas as pd


# ============================================================================
# SHARED UTILITY FUNCTIONS (from both original files)
# ============================================================================

@lru_cache(maxsize=10000)
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


def _parse_invoices(raw_invoices) -> List[str]:
    """Parse invoices field which may be list, JSON string, or raw string."""
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


# ============================================================================
# GATEWAY ↔ BANK CANDIDATE GENERATOR (from block_generator.py - FastCandidateBlockGenerator)
# ============================================================================

class GatewayBankCandidateGenerator:
    """
    Optimized candidate block generator that preserves ALL valid candidates.
    (Originally FastCandidateBlockGenerator in block_generator.py)
    """
    
    def __init__(
        self, 
        max_delay_days: int = 5, 
        max_amount_diff_pct: float = 0.20,
        max_candidates_per_bank: int = 100
    ):
        self.max_delay_days = max_delay_days
        self.max_amount_diff_pct = max_amount_diff_pct
        self.max_candidates_per_bank = max_candidates_per_bank

    def _preprocess_gateways(self, gws: List[Dict]) -> Dict:
        """Pre-index gateway records for fast lookup."""
        gw_data = {
            'by_date': defaultdict(list),
            'by_utr_prefix': defaultdict(list),
            'by_setl_prefix': defaultdict(list),
            'parsed': []
        }
        
        for g in gws:
            dt = _parse_dt(g.get("settled_at"))
            date_str = dt.strftime("%Y-%m-%d")
            net = float(g.get("net_settled", 0.0))
            pid = str(g.get("payment_id") or g.get("gw_id"))
            
            rec = dict(g)
            rec["_dt"] = dt
            rec["_net"] = net
            rec["_pid"] = pid
            
            gw_data['parsed'].append(rec)
            gw_data['by_date'][date_str].append(rec)
            
            utr = str(g.get("bank_utr") or "").strip()
            if len(utr) >= 6:
                gw_data['by_utr_prefix'][utr[:6]].append(rec)
            
            setl = str(g.get("settlement_id") or "").strip()
            if len(setl) >= 6:
                gw_data['by_setl_prefix'][setl[:6]].append(rec)
        
        # Sort by datetime for binary search
        gw_data['parsed'].sort(key=lambda x: x["_dt"])
        gw_data['dts'] = [g["_dt"] for g in gw_data['parsed']]
        gw_data['amounts'] = np.array([g["_net"] for g in gw_data['parsed']])
        
        return gw_data

    def generate_blocks(
        self,
        unmatched_gws: List[Dict],
        unmatched_banks: List[Dict],
    ) -> List[Dict]:
        """
        Generate candidate blocks with optimizations while preserving accuracy.
        (Original logic unchanged from FastCandidateBlockGenerator)
        """
        if not unmatched_gws or not unmatched_banks:
            return []

        # Preprocess gateways once
        gw_data = self._preprocess_gateways(unmatched_gws)
        if not gw_data['parsed']:
            return []
        
        gw_parsed = gw_data['parsed']
        gw_dts = gw_data['dts']
        gw_amounts = gw_data['amounts']
        
        candidate_blocks = []
        seen_keys: Set[Tuple[str, frozenset]] = set()
        
        for bank in unmatched_banks:
            b_id = str(bank.get("bank_entry_id") or bank.get("bank_id"))
            b_credit = float(bank.get("credit_amount", 0.0))
            b_dt = _parse_dt(bank.get("value_date"))
            
            # Quick skip for zero amount
            if b_credit <= 0:
                continue
            
            # Temporal window using binary search
            window_start = b_dt - timedelta(days=self.max_delay_days)
            window_end = b_dt + timedelta(days=1)
            
            start_idx = bisect_left(gw_dts, window_start)
            end_idx = bisect_right(gw_dts, window_end)
            
            if start_idx >= end_idx:
                continue
            
            window_gws = gw_parsed[start_idx:end_idx]
            window_amounts = gw_amounts[start_idx:end_idx]
            
            # 1. 1:1 Candidates (preserved exactly as original)
            for g in window_gws:
                amt_diff = abs(b_credit - g["_net"])
                if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 10.0:
                    gw_set = frozenset([g["_pid"]])
                    key = (b_id, gw_set)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidate_blocks.append({
                            "block_type": "1:1",
                            "bank_row": bank,
                            "bank_id": b_id,
                            "gw_rows": [g],
                            "gw_ids": [g["_pid"]],
                            "gw_ids_frozenset": gw_set,
                        })
            
            # 2. Daily Rollup Candidates (preserved)
            dates_in_window = set(g["_dt"].strftime("%Y-%m-%d") for g in window_gws)
            for d_str in dates_in_window:
                day_gws = [g for g in gw_data['by_date'][d_str] if g in window_gws]
                if len(day_gws) >= 2:
                    total_net = sum(g["_net"] for g in day_gws)
                    amt_diff = abs(b_credit - total_net)
                    if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 25.0:
                        gw_set = frozenset(g["_pid"] for g in day_gws)
                        key = (b_id, gw_set)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            candidate_blocks.append({
                                "block_type": "DailyRollup",
                                "bank_row": bank,
                                "bank_id": b_id,
                                "gw_rows": day_gws,
                                "gw_ids": [g["_pid"] for g in day_gws],
                                "gw_ids_frozenset": gw_set,
                            })
            
            # 3. Chronological Sequential Chunk Blocks (preserved - up to 6 records)
            for d_str in dates_in_window:
                day_gws = [g for g in gw_data['by_date'][d_str] if g in window_gws]
                if len(day_gws) >= 3:
                    n_day = len(day_gws)
                    for k in range(2, min(7, n_day)):  # Same as original: 2 to 6
                        for si in range(0, n_day - k + 1):
                            chunk = day_gws[si : si + k]
                            total_net = sum(g["_net"] for g in chunk)
                            amt_diff = abs(b_credit - total_net)
                            if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 25.0:
                                gw_set = frozenset(g["_pid"] for g in chunk)
                                key = (b_id, gw_set)
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    candidate_blocks.append({
                                        "block_type": "SequentialChunk",
                                        "bank_row": bank,
                                        "bank_id": b_id,
                                        "gw_rows": chunk,
                                        "gw_ids": [g["_pid"] for g in chunk],
                                        "gw_ids_frozenset": gw_set,
                                    })
            
            # 4. Identifier Prefix Candidate Blocks (preserved)
            remittance = str(bank.get("remittance_info") or "").upper()
            
            # UTR prefix groups
            for prefix, group in gw_data['by_utr_prefix'].items():
                if len(group) >= 2 and (prefix in remittance):
                    valid_group = [g for g in group if g in window_gws]
                    if len(valid_group) >= 2:
                        total_net = sum(g["_net"] for g in valid_group)
                        amt_diff = abs(b_credit - total_net)
                        if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 25.0:
                            gw_set = frozenset(g["_pid"] for g in valid_group)
                            key = (b_id, gw_set)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                candidate_blocks.append({
                                    "block_type": "IdentifierPrefix",
                                    "bank_row": bank,
                                    "bank_id": b_id,
                                    "gw_rows": valid_group,
                                    "gw_ids": [g["_pid"] for g in valid_group],
                                    "gw_ids_frozenset": gw_set,
                                })
            
            # Settlement ID prefix groups
            for prefix, group in gw_data['by_setl_prefix'].items():
                if len(group) >= 2 and (prefix in remittance):
                    valid_group = [g for g in group if g in window_gws]
                    if len(valid_group) >= 2:
                        total_net = sum(g["_net"] for g in valid_group)
                        amt_diff = abs(b_credit - total_net)
                        if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 25.0:
                            gw_set = frozenset(g["_pid"] for g in valid_group)
                            key = (b_id, gw_set)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                candidate_blocks.append({
                                    "block_type": "IdentifierPrefix",
                                    "bank_row": bank,
                                    "bank_id": b_id,
                                    "gw_rows": valid_group,
                                    "gw_ids": [g["_pid"] for g in valid_group],
                                    "gw_ids_frozenset": gw_set,
                                })
        
        return candidate_blocks

    # Alias method so both generate() and generate_blocks() work
    def generate(self, *args, **kwargs):
        """Alias for generate_blocks to maintain compatibility."""
        return self.generate_blocks(*args, **kwargs)


# ============================================================================
# ERP ↔ GATEWAY CANDIDATE GENERATOR (from erp_gw_block_generator.py - ERPGWCandidateBlockGenerator)
# ============================================================================

class ERPGatewayCandidateGenerator:
    """
    Generates candidate ERP clusters for each orphaned Gateway record.
    (Originally ERPGWCandidateBlockGenerator in erp_gw_block_generator.py)
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
        orphan_erps: List[Dict[str, any]],
        orphan_gws: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Generate candidate ERP cluster blocks for each orphaned GW record.
        (Original logic unchanged from ERPGWCandidateBlockGenerator)
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

    # Alias method so both generate() and generate_blocks() work
    def generate(self, *args, **kwargs):
        """Alias for generate_blocks to maintain compatibility."""
        return self.generate_blocks(*args, **kwargs)


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Aliases so existing code doesn't break
CandidateBlockGenerator = GatewayBankCandidateGenerator
FastCandidateBlockGenerator = GatewayBankCandidateGenerator
ERPGWCandidateBlockGenerator = ERPGatewayCandidateGenerator