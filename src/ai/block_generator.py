#!/usr/bin/env python3
"""
Candidate Cluster Block Generator.

Generates candidate Gateway clusters (1:1, daily rollups, chronological sequence
chunks, and identifier prefix groups) for candidate Bank statement deposits.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Union
import pandas as pd


def _parse_dt(val) -> datetime:
    if isinstance(val, datetime):
        return val
    s = str(val)[:19]
    if len(s) == 10:
        try:
            return datetime(int(s[:4]), int(s[5:7]), int(s[8:10]))
        except ValueError:
            pass
    elif len(s) >= 19:
        try:
            return datetime(int(s[:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:19]))
        except ValueError:
            pass
    return datetime(2026, 1, 1)


class CandidateBlockGenerator:
    def __init__(self, max_delay_days: int = 5, max_amount_diff_pct: float = 0.20):
        self.max_delay_days = max_delay_days
        self.max_amount_diff_pct = max_amount_diff_pct

    def generate_blocks(
        self,
        unmatched_gws: List[Dict],
        unmatched_banks: List[Dict],
    ) -> List[Dict]:
        """
        Returns a list of candidate blocks:
        {
            "block_type": "1:1" | "DailyRollup" | "SequentialChunk" | "IdentifierPrefix",
            "bank_row": dict,
            "bank_id": str,
            "gw_rows": List[dict],
            "gw_ids": List[str],
            "gw_ids_frozenset": frozenset(gw_ids),
        }
        """
        if not unmatched_gws or not unmatched_banks:
            return []

        # Pre-process gateway records
        gw_by_date = defaultdict(list)
        gw_by_utr_prefix = defaultdict(list)
        gw_by_setl_prefix = defaultdict(list)
        gw_parsed = []

        for g in unmatched_gws:
            dt = _parse_dt(g.get("settled_at"))
            date_str = dt.strftime("%Y-%m-%d")
            net = float(g.get("net_settled", 0.0))
            pid = str(g.get("payment_id") or g.get("gw_id"))

            rec = dict(g)
            rec["_dt"] = dt
            rec["_net"] = net
            rec["_pid"] = pid

            gw_parsed.append(rec)
            gw_by_date[date_str].append(rec)

            utr = str(g.get("bank_utr") or "").strip()
            if len(utr) >= 6:
                gw_by_utr_prefix[utr[:6]].append(rec)

            setl = str(g.get("settlement_id") or "").strip()
            if len(setl) >= 6:
                gw_by_setl_prefix[setl[:6]].append(rec)

        gw_parsed.sort(key=lambda x: x["_dt"])
        gw_dts = [g["_dt"] for g in gw_parsed]

        candidate_blocks = []
        seen_keys: Set[Tuple[str, frozenset]] = set()

        for b in unmatched_banks:
            b_id = str(b.get("bank_entry_id") or b.get("bank_id"))
            b_credit = float(b.get("credit_amount", 0.0))
            b_dt = _parse_dt(b.get("value_date"))
            window_start = b_dt - timedelta(days=self.max_delay_days)
            window_end = b_dt + timedelta(days=1)

            # Logarithmic binary search for temporal window slicing
            start_idx = bisect_left(gw_dts, window_start)
            end_idx = bisect_right(gw_dts, window_end)
            window_gws = gw_parsed[start_idx:end_idx]

            # 1. 1:1 Candidate Blocks
            for g in window_gws:
                amt_diff = abs(b_credit - g["_net"])
                if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 10.0:
                    gw_set = frozenset([g["_pid"]])
                    key = (b_id, gw_set)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidate_blocks.append({
                            "block_type": "1:1",
                            "bank_row": b,
                            "bank_id": b_id,
                            "gw_rows": [g],
                            "gw_ids": [g["_pid"]],
                            "gw_ids_frozenset": gw_set,
                        })

            # 2. Daily Rollup Candidate Blocks (N:1)
            # Find distinct dates within the window
            dates_in_window = set(g["_dt"].strftime("%Y-%m-%d") for g in window_gws)
            for d_str in dates_in_window:
                day_gws = [g for g in gw_by_date[d_str] if g in window_gws]
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
                                "bank_row": b,
                                "bank_id": b_id,
                                "gw_rows": day_gws,
                                "gw_ids": [g["_pid"] for g in day_gws],
                                "gw_ids_frozenset": gw_set,
                            })

            # 3. Chronological Sequential Chunk Blocks (2 to 6 records)
            for d_str in dates_in_window:
                day_gws = [g for g in gw_by_date[d_str] if g in window_gws]
                if len(day_gws) >= 3:
                    n_day = len(day_gws)
                    for k in range(2, min(7, n_day)):
                        for start_idx in range(0, n_day - k + 1):
                            chunk = day_gws[start_idx : start_idx + k]
                            total_net = sum(g["_net"] for g in chunk)
                            amt_diff = abs(b_credit - total_net)
                            if amt_diff / (b_credit + 1e-5) <= self.max_amount_diff_pct or amt_diff <= 25.0:
                                gw_set = frozenset(g["_pid"] for g in chunk)
                                key = (b_id, gw_set)
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    candidate_blocks.append({
                                        "block_type": "SequentialChunk",
                                        "bank_row": b,
                                        "bank_id": b_id,
                                        "gw_rows": chunk,
                                        "gw_ids": [g["_pid"] for g in chunk],
                                        "gw_ids_frozenset": gw_set,
                                    })

            # 4. Identifier Prefix Candidate Blocks
            remittance = str(b.get("remittance_info") or "").upper()
            for prefix, group in gw_by_utr_prefix.items():
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
                                    "bank_row": b,
                                    "bank_id": b_id,
                                    "gw_rows": valid_group,
                                    "gw_ids": [g["_pid"] for g in valid_group],
                                    "gw_ids_frozenset": gw_set,
                                })

        return candidate_blocks
