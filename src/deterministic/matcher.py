#!/usr/bin/env python3
"""
Multi-Source Reconciliation Matching Engine with Graph Edge Assembly.

Matches transactions across ERP, Gateway, and Bank sources using:
1. Identifier-based exact matching (invoice, UTR, settlement ID)
2. Subset sum combinatorial matching for batch settlements
3. Invoice-keyed matching with connected-component balancing
4. Partial split detection for reserve splits
5. Amount + temporal matching for no-invoice records
6. Fuzzy similarity matching for residual cases
"""

import ast
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import (
    DB_PATH,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_BULK,
    MATCH_TYPE_FUZZY,
    MATCH_STAGE_IDENTIFIER,
    MATCH_STAGE_SUBSET_SUM,
    MATCH_STAGE_SUBSET_SUM_SPLIT,
    STAGE_EXACT_ERP_GW,
    STAGE_FUZZY_ERP_GW,
    STAGE_FUZZY_GW_BANK,
    TABLE_BANK,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_ERP_GW_PRED,
    TABLE_GW_BANK_PRED,
)
from src.core.database import get_connection, clear_graph_edges, save_graph_edges


def extract_invoice_number(remittance_info: str) -> Optional[str]:
    """Extract invoice number from remittance info string."""
    if not remittance_info or not isinstance(remittance_info, str):
        return None
    match = re.search(r"(INV-[a-zA-Z0-9]+|ORD-[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_utr_number(remittance_info: str) -> Optional[str]:
    """Extract UTR number from remittance info string."""
    if not remittance_info or not isinstance(remittance_info, str):
        return None
    match = re.search(r"(UTR\d{12}|UTR[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_settlement_id(remittance_info: str) -> Optional[str]:
    """Extract settlement ID from remittance info string."""
    if not remittance_info or not isinstance(remittance_info, str):
        return None
    match = re.search(r"(setl_[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1) if match else None


def parse_invoices(val) -> List[str]:
    """Parse invoices field which may be list, JSON string, or raw string."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if x]
    if isinstance(val, float) and pd.isna(val):
        return []
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if val.startswith('['):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x]
            except (ValueError, SyntaxError):
                pass
        return [val.replace('"', '').replace("'", "")]
    return []


def fetch_unmatched_records(db_path: Path = DB_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch all raw records from database."""
    conn = get_connection(db_path)
    try:
        df_erp = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn)
        df_gateway = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
        df_bank = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}", conn)
    finally:
        conn.close()
    return df_erp, df_gateway, df_bank


def _invoice_keys(raw: str) -> List[str]:
    """Generate normalized lookup keys for an invoice string."""
    keys = []
    clean = raw.strip().upper()
    if not clean or clean == "NAN":
        return keys
    keys.append(clean)
    for prefix in ("INV-", "INV_", "ORD-", "ORD_", "INV", "ORD"):
        if clean.startswith(prefix):
            stripped = clean[len(prefix):]
            if stripped:
                keys.append(stripped)
            break
    alnum = re.sub(r"[^A-Z0-9]", "", clean)
    if alnum and alnum not in keys:
        keys.append(alnum)
    return keys


def _parse_erp_dt(record: dict) -> Optional[datetime]:
    """Parse ERP entry_date into datetime object."""
    raw = str(record.get("entry_date", ""))
    for fmt, length in [("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d", 10)]:
        try:
            return datetime.strptime(raw[:length], fmt)
        except ValueError:
            continue
    return None


def _parse_gw_dt(record: dict) -> Optional[datetime]:
    """Parse Gateway settled_at into datetime object."""
    raw = str(record.get("settled_at", ""))
    for fmt, length in [("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d", 10)]:
        try:
            return datetime.strptime(raw[:length], fmt)
        except ValueError:
            continue
    return None


class ReconciliationEngine:
    """Matching engine for reconciling transactions across three sources."""

    def __init__(self, df_erp: pd.DataFrame, df_gateway: pd.DataFrame, df_bank: pd.DataFrame):
        self.df_erp = df_erp
        self.df_gateway = df_gateway
        self.df_bank = df_bank
        
        # Record lists and dictionary indexes
        self.erp_records = df_erp.to_dict("records")
        self.gw_records = df_gateway.to_dict("records")
        self.bank_records = df_bank.to_dict("records")

        self.erp_by_id = {r["erp_entry_id"]: r for r in self.erp_records}
        self.gw_by_id = {r["payment_id"]: r for r in self.gw_records}
        self.bank_by_id = {r["bank_entry_id"]: r for r in self.bank_records}

        self.erp_by_inv = {str(r["invoice_number"]): r for r in self.erp_records if pd.notna(r.get("invoice_number"))}

        # Tracking sets for matched records
        self.matched_erp_entries = set()
        self.matched_bank_entries = set()
        self.gw_linked_to_erp = set()
        self.gw_linked_to_bank = set()
        
        # Link and stage tracking
        self.gw_bank_links = {}
        self.gw_to_erp_links = {}
        self.gw_bank_stage_map = {}
        self.erp_gw_stage_map = {}

        # Final edge lists
        self.erp_gw_edges = []
        self.gw_bank_edges = []

    # ========================================================================
    # GATEWAY ↔ BANK MATCHING METHODS
    # ========================================================================

    def match_gateway_to_bank_by_identifier(self):
        """Match gateway payments to bank deposits using settlement IDs and UTRs."""
        self._match_by_settlement_id()
        self._match_by_utr()
        self._match_reserve_splits()

    def _match_by_settlement_id(self):
        """Match gateway to bank using settlement ID."""
        gw_by_setl = {}
        for row in self.gw_records:
            g_id = row["payment_id"]
            if g_id not in self.gw_linked_to_bank:
                setl_id = row.get("settlement_id")
                if setl_id and pd.notna(setl_id):
                    gw_by_setl.setdefault(setl_id, []).append(row)

        bank_by_setl = {}
        for row in self.bank_records:
            b_id = row["bank_entry_id"]
            if b_id not in self.matched_bank_entries:
                setl_id = extract_settlement_id(row.get("remittance_info", ""))
                if setl_id:
                    bank_by_setl.setdefault(setl_id, []).append(row)

        for setl_id, b_rows in bank_by_setl.items():
            if setl_id in gw_by_setl:
                g_rows = gw_by_setl[setl_id]
                sum_bank = round(sum(float(r["credit_amount"]) for r in b_rows), 2)
                sum_gw = round(sum(float(r["net_settled"]) for r in g_rows), 2)

                if abs(sum_bank - sum_gw) < 0.01:
                    b_ids = [b["bank_entry_id"] for b in b_rows]
                    for b in b_rows:
                        self.matched_bank_entries.add(b["bank_entry_id"])

                    for g in g_rows:
                        gw_pid = g["payment_id"]
                        self.gw_linked_to_bank.add(gw_pid)
                        self.gw_bank_links[gw_pid] = {
                            "bank_ids": b_ids,
                            "utr": g.get("bank_utr"),
                            "match_type": MATCH_TYPE_BULK if len(g_rows) > 1 else MATCH_TYPE_EXACT,
                            "matching_stage": MATCH_STAGE_IDENTIFIER,
                            "score": 1.00,
                            "note": f"Identifier match: Settlement ID ({setl_id})."
                        }

    def _match_by_utr(self):
        """Match gateway to bank using UTR number."""
        gw_by_utr = {}
        gw_by_utr_prefix = {}
        for row in self.gw_records:
            g_id = row["payment_id"]
            if g_id not in self.gw_linked_to_bank:
                utr = row.get("bank_utr")
                if utr and pd.notna(utr):
                    gw_by_utr.setdefault(utr, []).append(row)
                    if len(utr) > 3:
                        gw_by_utr_prefix.setdefault(utr[:-3], []).append(row)

        bank_by_utr = {}
        for row in self.bank_records:
            b_id = row["bank_entry_id"]
            if b_id not in self.matched_bank_entries:
                utr = extract_utr_number(row.get("remittance_info", ""))
                if utr:
                    bank_by_utr.setdefault(utr, []).append(row)

        for utr, b_rows in bank_by_utr.items():
            g_rows = gw_by_utr.get(utr)
            if not g_rows and utr in gw_by_utr_prefix:
                g_rows = gw_by_utr_prefix[utr]

            if g_rows:
                sum_bank = round(sum(float(r["credit_amount"]) for r in b_rows), 2)
                sum_gw = round(sum(float(r["net_settled"]) for r in g_rows), 2)

                if abs(sum_bank - sum_gw) < 0.01:
                    b_ids = [b["bank_entry_id"] for b in b_rows]
                    for b in b_rows:
                        self.matched_bank_entries.add(b["bank_entry_id"])

                    for g in g_rows:
                        gw_pid = g["payment_id"]
                        self.gw_linked_to_bank.add(gw_pid)
                        self.gw_bank_links[gw_pid] = {
                            "bank_ids": b_ids,
                            "utr": g.get("bank_utr", utr),
                            "match_type": MATCH_TYPE_BULK if len(g_rows) > 1 else MATCH_TYPE_EXACT,
                            "matching_stage": MATCH_STAGE_IDENTIFIER,
                            "score": 1.00,
                            "note": f"Identifier match: UTR ({utr})."
                        }

    def _match_reserve_splits(self):
        """Match 1:N reserve splits (1 gateway payment → 2 bank entries)."""
        main_banks = [r for r in self.bank_records if r["bank_entry_id"] not in self.matched_bank_entries and str(r["bank_entry_id"]).endswith("-MAIN")]
        rsv_banks = [r for r in self.bank_records if r["bank_entry_id"] not in self.matched_bank_entries and str(r["bank_entry_id"]).endswith("-RSV")]

        rsv_by_utr: Dict[str, dict] = {}
        for rb in rsv_banks:
            u = extract_utr_number(rb.get("remittance_info", ""))
            if u:
                rsv_by_utr[u] = rb

        for mb in main_banks:
            u = extract_utr_number(mb.get("remittance_info", ""))
            if not u:
                continue

            rb = rsv_by_utr.get(u)
            if rb is None and len(u) > 3 and u[:-3] in rsv_by_utr:
                rb = rsv_by_utr[u[:-3]]
            if rb is None:
                for ru, r_row in rsv_by_utr.items():
                    if u.startswith(ru) or ru.startswith(u):
                        rb = r_row
                        break

            if rb is not None:
                tot_credit = round(float(mb["credit_amount"]) + float(rb["credit_amount"]), 2)
                
                # Find matching gateway
                gw_by_utr = {}
                for row in self.gw_records:
                    g_id = row["payment_id"]
                    if g_id not in self.gw_linked_to_bank:
                        utr = row.get("bank_utr")
                        if utr and pd.notna(utr):
                            gw_by_utr.setdefault(utr, []).append(row)
                            if len(utr) > 3:
                                gw_by_utr.setdefault(utr[:-3], []).append(row)

                gw_rows = gw_by_utr.get(u)
                if not gw_rows and u in gw_by_utr:
                    gw_rows = gw_by_utr[u]

                if gw_rows and len(gw_rows) == 1:
                    gw_row = gw_rows[0]
                    gw_pid = gw_row["payment_id"]
                    if gw_pid not in self.gw_linked_to_bank:
                        gw_net = round(float(gw_row["net_settled"]), 2)
                        if abs(tot_credit - gw_net) < 0.01:
                            b_ids = [mb["bank_entry_id"], rb["bank_entry_id"]]
                            self.matched_bank_entries.add(mb["bank_entry_id"])
                            self.matched_bank_entries.add(rb["bank_entry_id"])
                            self.gw_linked_to_bank.add(gw_pid)
                            self.gw_bank_links[gw_pid] = {
                                "bank_ids": b_ids,
                                "utr": gw_row.get("bank_utr", u),
                                "match_type": MATCH_TYPE_EXACT,
                                "matching_stage": MATCH_STAGE_IDENTIFIER,
                                "score": 1.00,
                                "note": f"Reserve split match ({mb['bank_entry_id']} + {rb['bank_entry_id']})."
                            }

    def match_gateway_to_bank_exact(self):
        """Alias for identifier-based matching."""
        self.match_gateway_to_bank_by_identifier()

    def match_gateway_batches_by_subset_sum(self, max_delay_days=4, max_batch_size=6):
        """Match multiple gateway payments to single bank deposit using subset sum."""
        unmatched_gws = [r for r in self.gw_records if r["payment_id"] not in self.gw_linked_to_bank]
        unmatched_banks = [r for r in self.bank_records if r["bank_entry_id"] not in self.matched_bank_entries]
        if not unmatched_gws or not unmatched_banks:
            return

        unmatched_banks.sort(key=lambda x: str(x.get("value_date", "")))

        gw_pool = []
        for gw in unmatched_gws:
            gw_dt = _parse_gw_dt(gw)
            if gw_dt is None:
                continue
            cents = int(round(float(gw.get("net_settled", 0.0)) * 100))
            utr = str(gw.get("bank_utr", ""))
            invs = parse_invoices(gw.get("invoices"))
            gw_pool.append((gw_dt, gw, cents, utr, invs))

        for b in unmatched_banks:
            b_id = b["bank_entry_id"]
            if b_id in self.matched_bank_entries:
                continue

            b_date_str = str(b.get("value_date", ""))[:10]
            try:
                b_dt = datetime.strptime(b_date_str, "%Y-%m-%d")
            except Exception:
                continue
            target_cents = int(round(float(b.get("credit_amount", 0.0)) * 100))

            window_start = b_dt - timedelta(days=max_delay_days)
            window_end = b_dt + timedelta(days=1)

            valid_gws = [
                item for item in gw_pool
                if window_start.date() <= item[0].date() <= window_end.date() 
                and item[1]["payment_id"] not in self.gw_linked_to_bank
            ]

            if not valid_gws:
                continue

            remittance = str(b.get("remittance_info", ""))
            utr_in_rem = extract_utr_number(remittance)

            anchor_gws = []
            for dt, gw, cents, utr, invs in valid_gws:
                if utr_in_rem:
                    if utr and (utr == utr_in_rem or utr.startswith(utr_in_rem) or utr[:-3] == utr_in_rem):
                        anchor_gws.append((dt, gw, cents))
                else:
                    if utr and len(utr) >= 6 and (utr in remittance or utr[:-3] in remittance):
                        anchor_gws.append((dt, gw, cents))
                    elif any(inv and (inv in remittance or inv.replace("INV-", "") in remittance) for inv in invs):
                        anchor_gws.append((dt, gw, cents))

            if not anchor_gws:
                continue

            found_subsets = []

            for a_dt, anchor, a_cents in anchor_gws:
                if a_cents == target_cents:
                    found_subsets.append([anchor])
                    continue

                rem_target = target_cents - a_cents
                if rem_target <= 0:
                    continue

                nearby_gws = [
                    (dt, gw, cents) for dt, gw, cents, _, _ in valid_gws
                    if abs((dt - a_dt).total_seconds()) <= 64800
                    and gw["payment_id"] != anchor["payment_id"]
                    and cents <= rem_target
                ]

                if sum(c for _, _, c in nearby_gws) < rem_target:
                    continue

                nearby_gws.sort(key=lambda x: x[2], reverse=True)
                if len(nearby_gws) > 12:
                    nearby_gws = nearby_gws[:12]

                def find_subsets(start, current_sum, path):
                    if current_sum == rem_target and len(path) >= 1:
                        found_subsets.append([anchor] + list(path))
                        return
                    if len(path) >= max_batch_size - 1 or len(found_subsets) >= 2 or start >= len(nearby_gws):
                        return

                    for i in range(start, len(nearby_gws)):
                        item_val = nearby_gws[i][2]
                        if current_sum + item_val > rem_target:
                            continue
                        find_subsets(i + 1, current_sum + item_val, path + [nearby_gws[i][1]])

                find_subsets(0, 0, [])

            selected_subset = found_subsets[0] if len(found_subsets) == 1 else None

            if selected_subset:
                self.matched_bank_entries.add(b_id)
                is_bulk = len(selected_subset) > 1
                for g in selected_subset:
                    g_id = g["payment_id"]
                    self.gw_linked_to_bank.add(g_id)
                    self.gw_bank_links[g_id] = {
                        "bank_ids": [b_id],
                        "utr": g.get("bank_utr"),
                        "match_type": MATCH_TYPE_BULK if is_bulk else MATCH_TYPE_EXACT,
                        "matching_stage": MATCH_STAGE_SUBSET_SUM,
                        "score": 1.00,
                        "note": f"Subset sum match (Batch size: {len(selected_subset)})."
                    }

    def match_gateway_batches_combinatorial(self, max_delay_days=10):
        """Alias for subset sum matching with wider window."""
        self.match_gateway_batches_by_subset_sum(max_delay_days)

    # ========================================================================
    # ERP ↔ GATEWAY MATCHING METHODS
    # ========================================================================

    def match_erp_to_gateway_by_invoice(self):
        """Match ERP orders to gateway payments using invoice numbers."""
        erp_inv_index: Dict[str, str] = {}
        for r in self.erp_records:
            raw = str(r.get("invoice_number") or "")
            if not raw or raw == "nan":
                continue
            erp_id = r["erp_entry_id"]
            for key in _invoice_keys(raw):
                erp_inv_index[key] = erp_id

        adj: Dict[str, List[str]] = {}
        erp_amounts: Dict[str, float] = {}

        for row in self.erp_records:
            node_id = row["erp_entry_id"]
            adj[node_id] = []
            erp_amounts[node_id] = float(row["gross_amount"])

        gw_amounts: Dict[str, float] = {}
        for row in self.gw_records:
            gw_id = row["payment_id"]
            adj[gw_id] = []
            gw_amounts[gw_id] = float(row["gross_amount"])
            for inv in parse_invoices(row.get("invoices")):
                for key in _invoice_keys(str(inv)):
                    erp_id = erp_inv_index.get(key)
                    if erp_id:
                        adj[gw_id].append(erp_id)
                        adj[erp_id].append(gw_id)
                        break

        visited: set = set()
        for node in list(adj.keys()):
            if node not in visited:
                comp, stack = [], [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.append(curr)
                        stack.extend(adj.get(curr, []))

                erp_nodes = [n for n in comp if n.startswith("ERP-")]
                gw_nodes = [n for n in comp if n.startswith("GW-")]

                if erp_nodes and gw_nodes:
                    sum_erp = sum(erp_amounts[n] for n in erp_nodes)
                    sum_gw = sum(gw_amounts[n] for n in gw_nodes)

                    if abs(sum_erp - sum_gw) < 0.05:
                        for e in erp_nodes:
                            self.matched_erp_entries.add(e)
                        for g in gw_nodes:
                            self.gw_linked_to_erp.add(g)
                            self.gw_to_erp_links.setdefault(g, []).extend(erp_nodes)
                            for e in erp_nodes:
                                self.erp_gw_stage_map.setdefault(g, {})[e] = STAGE_EXACT_ERP_GW

    def match_partial_invoice_splits(self):
        """Match ERP orders split across multiple gateway payments where only one has invoice."""
        erp_inv_index: Dict[str, str] = {}
        for r in self.erp_records:
            raw = str(r.get("invoice_number") or "")
            if not raw or raw == "nan":
                continue
            for key in _invoice_keys(raw):
                erp_inv_index[key] = r["erp_entry_id"]

        unmatched_gw_pool = [
            r for r in self.gw_records if r["payment_id"] not in self.gw_linked_to_erp
        ]

        for anchor_gw in list(unmatched_gw_pool):
            anchor_id = anchor_gw["payment_id"]
            if anchor_id in self.gw_linked_to_erp:
                continue

            erp_id = None
            for inv in parse_invoices(anchor_gw.get("invoices")):
                for key in _invoice_keys(str(inv)):
                    candidate = erp_inv_index.get(key)
                    if candidate and candidate not in self.matched_erp_entries:
                        erp_id = candidate
                        break
                if erp_id:
                    break
            if not erp_id:
                continue

            erp_rec = self.erp_by_id.get(erp_id)
            if not erp_rec:
                continue

            erp_gross = float(erp_rec["gross_amount"])
            anchor_gross = float(anchor_gw["gross_amount"])

            if anchor_gross >= erp_gross - 0.05:
                continue

            remainder = erp_gross - anchor_gross
            anchor_dt = _parse_gw_dt(anchor_gw)
            if not anchor_dt:
                continue

            window_start = anchor_dt - timedelta(hours=2)
            window_end = anchor_dt + timedelta(hours=2)

            valid_partners = []
            for p in unmatched_gw_pool:
                if p["payment_id"] == anchor_id or p["payment_id"] in self.gw_linked_to_erp:
                    continue
                p_dt = _parse_gw_dt(p)
                if not p_dt or not (window_start <= p_dt <= window_end):
                    continue
                p_gross = float(p["gross_amount"])
                if p_gross > remainder + 0.05:
                    continue
                valid_partners.append((p_gross, p))

            if not valid_partners:
                continue

            target_rem_cents = int(round(remainder * 100))
            partner_map: Dict[int, List[str]] = {0: []}
            for p_gross, p_rec in sorted(valid_partners, key=lambda x: x[0], reverse=True):
                p_cents = int(round(p_gross * 100))
                for cur_sum, path in list(partner_map.items()):
                    cand = cur_sum + p_cents
                    if cand > target_rem_cents + 5:
                        continue
                    if cand not in partner_map:
                        partner_map[cand] = path + [p_rec["payment_id"]]

            best_key = None
            for k in partner_map:
                if abs(k - target_rem_cents) <= 5 and partner_map[k]:
                    if best_key is None or abs(k - target_rem_cents) < abs(best_key - target_rem_cents):
                        best_key = k

            if best_key is None:
                continue

            partner_gw_ids = partner_map[best_key]
            all_gw_ids = [anchor_id] + partner_gw_ids

            total_gw = anchor_gross + sum(
                float(self.gw_by_id[g]["gross_amount"])
                for g in partner_gw_ids if g in self.gw_by_id
            )
            if abs(total_gw - erp_gross) > 0.05:
                continue

            self.matched_erp_entries.add(erp_id)
            for g_id in all_gw_ids:
                self.gw_linked_to_erp.add(g_id)
                self.gw_to_erp_links.setdefault(g_id, []).append(erp_id)
                self.erp_gw_stage_map.setdefault(g_id, {})[erp_id] = MATCH_STAGE_SUBSET_SUM_SPLIT

    def match_bundled_erp_to_single_gateway(self):
        """Match multiple ERP orders to single gateway payment (N:1 bundled cart)."""
        unmatched_erps = [r for r in self.erp_records if r["erp_entry_id"] not in self.matched_erp_entries]
        unmatched_gws = [r for r in self.gw_records if r["payment_id"] not in self.gw_linked_to_erp]
        if not unmatched_erps or not unmatched_gws:
            return

        erp_pool = []
        for e in unmatched_erps:
            dt = _parse_erp_dt(e)
            if dt is None:
                continue
            cents = int(round(float(e.get("gross_amount", 0.0)) * 100))
            erp_pool.append((dt, e, cents))

        for gw in unmatched_gws:
            gw_id = gw["payment_id"]
            if gw_id in self.gw_linked_to_erp:
                continue
            gw_dt = _parse_gw_dt(gw)
            if gw_dt is None:
                continue
            target_cents = int(round(float(gw.get("gross_amount", 0.0)) * 100))

            gw_inv_keys: set = set()
            for inv in parse_invoices(gw.get("invoices")):
                gw_inv_keys.update(_invoice_keys(str(inv)))

            if not gw_inv_keys:
                continue

            window_start = gw_dt - timedelta(hours=24)
            window_end = gw_dt + timedelta(hours=6)

            valid_erps = [
                (dt, e, cents) for dt, e, cents in erp_pool
                if window_start <= dt <= window_end
                and e["erp_entry_id"] not in self.matched_erp_entries
                and cents < target_cents
            ]
            if not valid_erps:
                continue

            def erp_inv_in_gw(erp_rec):
                raw = str(erp_rec.get("invoice_number") or "").strip()
                if not raw or raw.lower() == "nan":
                    return False
                return bool(set(_invoice_keys(raw)) & gw_inv_keys)

            valid_erps = [(dt, e, cents) for dt, e, cents in valid_erps if erp_inv_in_gw(e)]
            if len(valid_erps) < 2:
                continue

            valid_erps.sort(key=lambda x: x[2], reverse=True)
            if sum(c for _, _, c in valid_erps) < target_cents - 2:
                continue

            sum_map: Dict[int, List[str]] = {0: []}
            for _, e, cents in valid_erps:
                for current_sum, path in list(sum_map.items()):
                    candidate_sum = current_sum + cents
                    if candidate_sum > target_cents + 2:
                        continue
                    if candidate_sum not in sum_map:
                        sum_map[candidate_sum] = path + [e["erp_entry_id"]]

            best_key = None
            for k in sum_map:
                if abs(k - target_cents) <= 2 and len(sum_map[k]) >= 2:
                    if best_key is None or abs(k - target_cents) < abs(best_key - target_cents):
                        best_key = k

            if best_key is not None:
                erp_ids = sum_map[best_key]
                self.gw_linked_to_erp.add(gw_id)
                for e_id in erp_ids:
                    self.matched_erp_entries.add(e_id)
                self.gw_to_erp_links.setdefault(gw_id, []).extend(erp_ids)
                for e_id in erp_ids:
                    self.erp_gw_stage_map.setdefault(gw_id, {})[e_id] = MATCH_STAGE_SUBSET_SUM

    def match_split_erp_to_multiple_gateways(self):
        """Match single ERP order to multiple gateway payments (1:N split)."""
        unmatched_erps = [r for r in self.erp_records if r["erp_entry_id"] not in self.matched_erp_entries]
        unmatched_gws = [r for r in self.gw_records if r["payment_id"] not in self.gw_linked_to_erp]
        if not unmatched_erps or not unmatched_gws:
            return

        gw_pool = []
        for gw in unmatched_gws:
            dt = _parse_gw_dt(gw)
            if dt is None:
                continue
            cents = int(round(float(gw.get("gross_amount", 0.0)) * 100))
            invs = parse_invoices(gw.get("invoices"))
            gw_pool.append((dt, gw, cents, invs))

        for erp in unmatched_erps:
            erp_id = erp["erp_entry_id"]
            if erp_id in self.matched_erp_entries:
                continue
            erp_dt = _parse_erp_dt(erp)
            if erp_dt is None:
                continue
            target_cents = int(round(float(erp.get("gross_amount", 0.0)) * 100))
            erp_inv_raw = str(erp.get("invoice_number") or "").upper().strip()
            erp_inv_keys = set(_invoice_keys(erp_inv_raw)) if erp_inv_raw else set()

            window_start = erp_dt - timedelta(hours=6)
            window_end = erp_dt + timedelta(hours=48)

            valid_gws = [
                (dt, gw, cents, invs) for dt, gw, cents, invs in gw_pool
                if window_start <= dt <= window_end
                and gw["payment_id"] not in self.gw_linked_to_erp
                and cents < target_cents
            ]
            if not valid_gws:
                continue

            if any(abs(cents - target_cents) <= 2 for _, _, cents, _ in valid_gws):
                continue

            if erp_inv_keys:
                def gw_has_invoice(invs):
                    for inv in invs:
                        if set(_invoice_keys(str(inv))) & erp_inv_keys:
                            return True
                    return False
                if not any(gw_has_invoice(invs) for _, _, _, invs in valid_gws):
                    continue

            valid_gws_simple = [(dt, gw, cents) for dt, gw, cents, _ in valid_gws]
            valid_gws_simple.sort(key=lambda x: x[2], reverse=True)
            if sum(c for _, _, c in valid_gws_simple) < target_cents - 2:
                continue

            sum_map: Dict[int, List[str]] = {0: []}
            for _, gw, cents in valid_gws_simple:
                for current_sum, path in list(sum_map.items()):
                    candidate_sum = current_sum + cents
                    if candidate_sum > target_cents + 2:
                        continue
                    if candidate_sum not in sum_map:
                        sum_map[candidate_sum] = path + [gw["payment_id"]]

            best_key = None
            for k in sum_map:
                if abs(k - target_cents) <= 2 and 2 <= len(sum_map[k]) <= 3:
                    if best_key is None or abs(k - target_cents) < abs(best_key - target_cents):
                        best_key = k

            if best_key is not None:
                gw_ids = sum_map[best_key]
                self.matched_erp_entries.add(erp_id)
                for gw_id in gw_ids:
                    self.gw_linked_to_erp.add(gw_id)
                    self.gw_to_erp_links.setdefault(gw_id, []).append(erp_id)
                    self.erp_gw_stage_map.setdefault(gw_id, {})[erp_id] = MATCH_STAGE_SUBSET_SUM_SPLIT

    def match_erp_to_gateway_by_amount_time(self):
        """Match ERP to gateway using amount and temporal proximity (no invoice)."""
        unmatched_erps = [r for r in self.erp_records if r["erp_entry_id"] not in self.matched_erp_entries]
        unmatched_gws = [
            r for r in self.gw_records
            if r["payment_id"] not in self.gw_linked_to_erp
            and not parse_invoices(r.get("invoices"))
        ]
        if not unmatched_erps or not unmatched_gws:
            return

        erp_by_cents: Dict[int, List[Tuple]] = {}
        for e in unmatched_erps:
            dt = _parse_erp_dt(e)
            if dt is None:
                continue
            cents = int(round(float(e.get("gross_amount", 0.0)) * 100))
            erp_by_cents.setdefault(cents, []).append((dt, e))

        for gw in unmatched_gws:
            gw_id = gw["payment_id"]
            if gw_id in self.gw_linked_to_erp:
                continue
            gw_dt = _parse_gw_dt(gw)
            if gw_dt is None:
                continue
            target_cents = int(round(float(gw.get("gross_amount", 0.0)) * 100))

            window_start = gw_dt - timedelta(hours=72)
            window_end = gw_dt + timedelta(hours=6)

            seen_eids: set = set()
            candidates = []
            for delta_c in range(-2, 3):
                for dt, e in erp_by_cents.get(target_cents + delta_c, []):
                    eid = e["erp_entry_id"]
                    if eid not in seen_eids and window_start <= dt <= window_end and eid not in self.matched_erp_entries:
                        seen_eids.add(eid)
                        candidates.append((dt, e))

            if len(candidates) == 1:
                _, erp = candidates[0]
                erp_id = erp["erp_entry_id"]
                self._link_erp_to_gateway(erp_id, gw_id, STAGE_FUZZY_ERP_GW)

            elif 2 <= len(candidates) <= 3:
                tight_start = gw_dt - timedelta(hours=24)
                tight_end = gw_dt + timedelta(hours=4)
                tight = [(dt, e) for dt, e in candidates if tight_start <= dt <= tight_end]
                if len(tight) == 1:
                    _, erp = tight[0]
                    erp_id = erp["erp_entry_id"]
                    self._link_erp_to_gateway(erp_id, gw_id, STAGE_FUZZY_ERP_GW)

    # ========================================================================
    # FUZZY MATCHING METHODS
    # ========================================================================

    def match_gateway_to_bank_fuzzy(self):
        """Match gateway to bank using fuzzy UTR similarity."""
        unmatched_gws = [r for r in self.gw_records if r["payment_id"] not in self.gw_linked_to_bank]
        unmatched_banks = [r for r in self.bank_records if r["bank_entry_id"] not in self.matched_bank_entries]

        bank_by_amt = {}
        for b in unmatched_banks:
            amt = round(float(b["credit_amount"]), 2)
            bank_by_amt.setdefault(amt, []).append(b)

        for gw_row in unmatched_gws:
            gw_id = gw_row["payment_id"]
            gw_net = round(float(gw_row["net_settled"]), 2)
            gw_utr = str(gw_row.get("bank_utr", "")).strip()
            if not gw_utr or len(gw_utr) < 6:
                continue
            candidate_banks = bank_by_amt.get(gw_net, [])
            for bank_row in candidate_banks:
                bank_id = bank_row["bank_entry_id"]
                if bank_id in self.matched_bank_entries:
                    continue
                remittance = str(bank_row.get("remittance_info", ""))
                
                if (len(gw_utr) >= 10 and gw_utr[:-3] in remittance) or (gw_utr in remittance):
                    self.matched_bank_entries.add(bank_id)
                    self.gw_linked_to_bank.add(gw_id)
                    self.gw_bank_links[gw_id] = {
                        "bank_ids": [bank_id],
                        "utr": gw_utr,
                        "match_type": MATCH_TYPE_FUZZY,
                        "matching_stage": STAGE_FUZZY_GW_BANK,
                        "score": 0.85,
                        "note": "Fuzzy gateway-bank match."
                    }
                    break

    def match_erp_to_gateway_fuzzy(self):
        """Match ERP to gateway using fuzzy invoice similarity."""
        unmatched_gws = [r for r in self.gw_records if r["payment_id"] not in self.gw_linked_to_erp]
        unmatched_erps = [r for r in self.erp_records if r["erp_entry_id"] not in self.matched_erp_entries]

        erp_by_amt = {}
        for e in unmatched_erps:
            amt = round(float(e["gross_amount"]), 2)
            erp_by_amt.setdefault(amt, []).append(e)

        for gw_row in unmatched_gws:
            gw_id = gw_row["payment_id"]
            gw_gross = round(float(gw_row["gross_amount"]), 2)
            invoices = parse_invoices(gw_row.get("invoices"))
            linked_bank = self.gw_bank_links.get(gw_id)
            remittance_ctx = ""
            if linked_bank:
                b_ids = linked_bank.get("bank_ids", [])
                b_rows = [self.bank_by_id[b] for b in b_ids if b in self.bank_by_id]
                remittance_ctx = " ".join(str(r.get("remittance_info", "")) for r in b_rows)

            candidate_erps = erp_by_amt.get(gw_gross, [])
            for erp_row in candidate_erps:
                erp_id = erp_row["erp_entry_id"]
                if erp_id in self.matched_erp_entries:
                    continue
                stripped_inv = str(erp_row["invoice_number"]).replace("INV-", "")
                
                is_fuzzy = False
                note_add = ""
                if invoices and any(stripped_inv in str(inv) for inv in invoices if inv):
                    is_fuzzy, note_add = True, "Invoice prefix missing."
                elif stripped_inv and stripped_inv in remittance_ctx:
                    is_fuzzy, note_add = True, "Gateway invoice recovered from bank."

                if is_fuzzy:
                    self._link_erp_to_gateway(erp_id, gw_id, STAGE_FUZZY_ERP_GW)

                    if gw_id in self.gw_bank_links:
                        self.gw_bank_links[gw_id]["match_type"] = MATCH_TYPE_FUZZY
                        self.gw_bank_links[gw_id]["matching_stage"] = STAGE_FUZZY_GW_BANK
                        self.gw_bank_links[gw_id]["score"] = 0.85
                    else:
                        self.gw_bank_links[gw_id] = {
                            "bank_ids": [],
                            "utr": gw_row.get("bank_utr"),
                            "match_type": MATCH_TYPE_FUZZY,
                            "matching_stage": STAGE_FUZZY_GW_BANK,
                            "score": 0.80,
                            "note": note_add,
                        }
                    break

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _link_erp_to_gateway(self, erp_id: str, gw_id: str, stage: str):
        """Link an ERP entry to a gateway payment."""
        self.matched_erp_entries.add(erp_id)
        self.gw_linked_to_erp.add(gw_id)
        self.gw_to_erp_links.setdefault(gw_id, []).append(erp_id)
        self.erp_gw_stage_map.setdefault(gw_id, {})[erp_id] = stage

    def build_matched_edges(self):
        """Assemble final edge lists from tracked links."""
        for gw_id, erp_ids in self.gw_to_erp_links.items():
            gw_row = self.gw_by_id.get(gw_id, {})
            gw_gross = float(gw_row.get("gross_amount", 0.0))
            split_amt = round(gw_gross / len(erp_ids), 2) if erp_ids else 0.0
            stage_map = self.erp_gw_stage_map.get(gw_id, {})

            for erp_id in erp_ids:
                stage = stage_map.get(erp_id, STAGE_EXACT_ERP_GW if len(erp_ids) == 1 else STAGE_FUZZY_ERP_GW)
                self.erp_gw_edges.append({
                    "erp_order_id": erp_id,
                    "gateway_payment_id": gw_id,
                    "allocated_amount": split_amt,
                    "match_type": MATCH_TYPE_BULK if len(erp_ids) > 1 else MATCH_TYPE_EXACT,
                    "matching_stage": stage,
                    "confidence_score": 1.00,
                    "notes": f"Edge: ERP linked to Gateway {gw_id}."
                })

        for gw_id, bank_info in self.gw_bank_links.items():
            bank_ids = bank_info.get("bank_ids", [])
            gw_row = self.gw_by_id.get(gw_id, {})
            gw_net = float(gw_row.get("net_settled", 0.0))

            for b_id in bank_ids:
                self.gw_bank_edges.append({
                    "gateway_payment_id": gw_id,
                    "bank_entry_id": b_id,
                    "allocated_amount": gw_net,
                    "match_type": bank_info.get("match_type", MATCH_TYPE_EXACT),
                    "matching_stage": bank_info.get("matching_stage", MATCH_STAGE_IDENTIFIER),
                    "confidence_score": bank_info.get("score", 1.00),
                    "notes": bank_info.get("note", "Edge: Gateway linked to Bank.")
                })

    def run_matching_pipeline(self, include_fuzzy: bool = True) -> Dict:
        """Execute the complete matching pipeline in order."""
        # ERP ↔ Gateway matching
        self.match_erp_to_gateway_by_invoice()
        self.match_partial_invoice_splits()
        self.match_bundled_erp_to_single_gateway()
        self.match_erp_to_gateway_by_amount_time()
        self.match_split_erp_to_multiple_gateways()
        
        # Gateway ↔ Bank matching
        self.match_gateway_to_bank_exact()
        self.match_gateway_batches_combinatorial()
        
        # Fuzzy matching (optional)
        if include_fuzzy:
            self.match_gateway_to_bank_fuzzy()
            self.match_erp_to_gateway_fuzzy()
        
        # Build final edges
        self.build_matched_edges()
        
        return self.collect_results()

    def collect_results(self) -> Dict:
        """Collect and return matching results."""
        unmatched_erp = self.df_erp[~self.df_erp["erp_entry_id"].isin(self.matched_erp_entries)].copy()
        unmatched_gw = self.df_gateway[~self.df_gateway["payment_id"].isin(self.gw_linked_to_bank)].copy()
        unmatched_bank = self.df_bank[~self.df_bank["bank_entry_id"].isin(self.matched_bank_entries)].copy()

        return {
            "total_erp": len(self.df_erp),
            "total_gateway": len(self.df_gateway),
            "total_bank": len(self.df_bank),
            "matches_count": len(self.erp_gw_edges) + len(self.gw_bank_edges),
            "erp_gw_edges": self.erp_gw_edges,
            "gw_bank_edges": self.gw_bank_edges,
            "unmatched_erp": unmatched_erp,
            "unmatched_gateway": unmatched_gw,
            "unmatched_bank": unmatched_bank,
        }


def run_exact_matching(db_path: Path = DB_PATH, deterministic_only: bool = True) -> Dict[str, any]:
    """Run the matching pipeline and save results to database."""
    clear_graph_edges(db_path)
    df_erp, df_gw, df_bank = fetch_unmatched_records(db_path)
    engine = ReconciliationEngine(df_erp, df_gw, df_bank)
    results = engine.run_matching_pipeline(include_fuzzy=not deterministic_only)
    save_graph_edges(results["erp_gw_edges"], results["gw_bank_edges"], db_path)
    return results
