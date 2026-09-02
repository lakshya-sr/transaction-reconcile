#!/usr/bin/env python3
"""
Phase 3 & 4: Exact + Fuzzy Matching Engine Module with Graph Edge Assembly.
Includes full complex tracking and subset sum matching optimizations.
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
    STAGE_T1_IDENTIFIER,
    STAGE_T2_SUBSET_SUM,
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
    if not remittance_info or not isinstance(remittance_info, str): return None
    match = re.search(r"(INV-[a-zA-Z0-9]+|ORD-[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None

def extract_utr_number(remittance_info: str) -> Optional[str]:
    if not remittance_info or not isinstance(remittance_info, str): return None
    match = re.search(r"(UTR\d{12}|UTR[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None

def extract_settlement_id(remittance_info: str) -> Optional[str]:
    if not remittance_info or not isinstance(remittance_info, str): return None
    match = re.search(r"(setl_[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1) if match else None

def parse_invoices(val) -> List[str]:
    if pd.isna(val) or not val: return []
    if isinstance(val, list): return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('['):
            try: return ast.literal_eval(val)
            except (ValueError, SyntaxError): pass
        return [val.replace('"', '').replace("'", "")]
    return []

def fetch_unmatched_records(db_path: Path = DB_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = get_connection(db_path)
    try:
        df_erp = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn)
        df_gateway = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
        df_bank = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}", conn)
    finally:
        conn.close()
    return df_erp, df_gateway, df_bank


class ReconciliationEngine:
    def __init__(self, df_erp: pd.DataFrame, df_gateway: pd.DataFrame, df_bank: pd.DataFrame):
        self.df_erp = df_erp
        self.df_gateway = df_gateway
        self.df_bank = df_bank
        
        self.matched_erp_entries = set()
        self.matched_bank_entries = set()
        # Separated gateway tracking to prevent state cannibalization between layers
        self.gw_linked_to_erp = set()
        self.gw_linked_to_bank = set()
        
        self.gw_bank_links = {}
        self.gw_to_erp_links = {}
        self.gw_bank_stage_map = {}
        self.erp_gw_stage_map = {}

        self.erp_gw_edges = []
        self.gw_bank_edges = []

    def match_tier1_identifier_clusters_gateway_to_bank(self):
        gw_by_setl = {}
        for _, row in self.df_gateway.iterrows():
            g_id = row["payment_id"]
            if g_id not in self.gw_linked_to_bank:
                setl_id = row.get("settlement_id")
                if setl_id and pd.notna(setl_id):
                    gw_by_setl.setdefault(setl_id, []).append(row)

        bank_by_setl = {}
        for _, row in self.df_bank.iterrows():
            b_id = row["bank_entry_id"]
            if b_id not in self.matched_bank_entries:
                setl_id = extract_settlement_id(row["remittance_info"])
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
                            "matching_stage": STAGE_T1_IDENTIFIER,
                            "score": 1.00,
                            "note": f"Tier 1: Settlement ID match ({setl_id})."
                        }

        gw_by_utr = {}
        for _, row in self.df_gateway.iterrows():
            g_id = row["payment_id"]
            if g_id not in self.gw_linked_to_bank:
                utr = row.get("bank_utr")
                if utr and pd.notna(utr):
                    gw_by_utr.setdefault(utr, []).append(row)

        bank_by_utr = {}
        for _, row in self.df_bank.iterrows():
            b_id = row["bank_entry_id"]
            if b_id not in self.matched_bank_entries:
                utr = extract_utr_number(row["remittance_info"])
                if utr:
                    bank_by_utr.setdefault(utr, []).append(row)

        for utr, b_rows in bank_by_utr.items():
            if utr in gw_by_utr:
                g_rows = gw_by_utr[utr]
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
                            "utr": utr,
                            "match_type": MATCH_TYPE_BULK if len(g_rows) > 1 else MATCH_TYPE_EXACT,
                            "matching_stage": STAGE_T1_IDENTIFIER,
                            "score": 1.00,
                            "note": f"Tier 1: Bank UTR match ({utr})."
                        }

        unmatched_gws = [r for _, r in self.df_gateway.iterrows() if r["payment_id"] not in self.gw_linked_to_bank]
        unmatched_banks = [r for _, r in self.df_bank.iterrows() if r["bank_entry_id"] not in self.matched_bank_entries]

        rem_by_setl = {}
        for g in unmatched_gws:
            setl = g.get("settlement_id")
            if setl:
                rem_by_setl.setdefault(setl, []).append(g)

        for setl_id, g_rows in rem_by_setl.items():
            sum_gw = round(sum(float(r["net_settled"]) for r in g_rows), 2)
            all_invoices = set()
            for g in g_rows:
                for inv in parse_invoices(g.get("invoices")):
                    all_invoices.add(inv)

            for b in unmatched_banks:
                b_id = b["bank_entry_id"]
                if b_id in self.matched_bank_entries:
                    continue
                bank_credit = round(float(b["credit_amount"]), 2)
                if abs(sum_gw - bank_credit) < 0.01:
                    remittance = str(b.get("remittance_info", ""))
                    if any(inv in remittance or inv.replace("INV-", "") in remittance for inv in all_invoices if inv):
                        self.matched_bank_entries.add(b_id)
                        for g in g_rows:
                            gw_pid = g["payment_id"]
                            self.gw_linked_to_bank.add(gw_pid)
                            self.gw_bank_links[gw_pid] = {
                                "bank_ids": [b_id],
                                "utr": g.get("bank_utr"),
                                "match_type": MATCH_TYPE_BULK if len(g_rows) > 1 else MATCH_TYPE_EXACT,
                                "matching_stage": STAGE_T1_IDENTIFIER,
                                "score": 1.00,
                                "note": f"Tier 1: Batch invoice cross-ref match."
                            }
                        break

    def match_exact_gateway_to_bank(self):
        self.match_tier1_identifier_clusters_gateway_to_bank()

    def match_tier2_3_4_bounded_subset_sum_gateway_to_bank(self, max_delay_days=4, max_batch_size=8):
        """
        Tier 2: Temporal Window Partitioning
        Tier 3: Bounded Branch & Bound Subset Sum (integer-cents arithmetic)
        Tier 4: Narrative Anchor & Uniqueness Verification
        """
        unmatched_gws = [row for _, row in self.df_gateway.iterrows() if row["payment_id"] not in self.gw_linked_to_bank]
        unmatched_banks = [row for _, row in self.df_bank.iterrows() if row["bank_entry_id"] not in self.matched_bank_entries]
        unmatched_banks.sort(key=lambda x: str(x["value_date"]))

        if not unmatched_gws or not unmatched_banks:
            return

        # Pre-parse gateway dates and amounts into integer cents
        gw_pool = []
        for gw in unmatched_gws:
            dt_str = str(gw["settled_at"])[:10]
            gw_dt = datetime.strptime(dt_str, "%Y-%m-%d")
            cents = int(round(float(gw["net_settled"]) * 100))
            gw_pool.append((gw_dt, gw, cents))

        for b in unmatched_banks:
            b_id = b["bank_entry_id"]
            if b_id in self.matched_bank_entries:
                continue

            b_date_str = str(b["value_date"])[:10]
            b_dt = datetime.strptime(b_date_str, "%Y-%m-%d")
            target_cents = int(round(float(b["credit_amount"]) * 100))

            # Tier 2: Temporal Window (Gateways within lookback window up to bank value date)
            window_start = b_dt - timedelta(days=max_delay_days)
            window_end = b_dt + timedelta(days=1)

            valid_gws = [
                (dt, gw, cents) for dt, gw, cents in gw_pool
                if window_start <= dt <= window_end and gw["payment_id"] not in self.gw_linked_to_bank
            ]

            if not valid_gws:
                continue

            # Sort ascending by amount (cents) for branch-and-bound
            valid_gws.sort(key=lambda x: x[2])

            # Extract candidate anchor gateway records mentioned in bank remittance
            remittance = str(b.get("remittance_info", ""))
            anchor_gws = []
            for dt, gw, cents in valid_gws:
                utr = str(gw.get("bank_utr", ""))
                invs = parse_invoices(gw.get("invoices"))
                if utr and len(utr) >= 6 and (utr in remittance or utr[:-3] in remittance):
                    anchor_gws.append((dt, gw, cents))
                elif any(inv and (inv in remittance or inv.replace("INV-", "") in remittance) for inv in invs):
                    anchor_gws.append((dt, gw, cents))

            # Tier 3: Bounded Branch & Bound Subset Sum
            found_subsets = []

            def find_subsets(start, current_sum, path, pool):
                if current_sum == target_cents and len(path) >= 1:
                    found_subsets.append(list(path))
                    return
                if len(path) >= max_batch_size or len(found_subsets) >= 3:
                    return

                for i in range(start, len(pool)):
                    item_val = pool[i][2]
                    if current_sum + item_val > target_cents:
                        break
                    find_subsets(i + 1, current_sum + item_val, path + [pool[i][1]], pool)

            # If an anchor gateway transaction exists, anchor the search around its time window
            if anchor_gws:
                for a_dt, anchor, a_cents in anchor_gws:
                    if a_cents == target_cents:
                        found_subsets.append([anchor])
                        break

                    # Search nearby gateways within ±24 hours of anchor
                    nearby_gws = [
                        (dt, gw, cents) for dt, gw, cents in valid_gws
                        if abs((dt - a_dt).total_seconds()) <= 86400 and gw["payment_id"] != anchor["payment_id"]
                    ]
                    nearby_gws.sort(key=lambda x: x[2])

                    find_subsets(0, a_cents, [anchor], nearby_gws)
                    if found_subsets:
                        break

            # Tier 4: Select unique confirmed subset with narrative validation
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
                        "matching_stage": STAGE_T2_SUBSET_SUM,
                        "score": 1.00 if is_bulk else 1.00,
                        "note": f"Tier 2-4: Bounded subset sum with token verification (Batch size: {len(selected_subset)})."
                    }

    def match_combinatorial_gateway_to_bank(self, max_delay_days=10):
        self.match_tier2_3_4_bounded_subset_sum_gateway_to_bank(max_delay_days)

    def match_exact_erp_to_gateway(self):
        adj, erp_amounts, gw_amounts = {}, {}, {}
        for _, row in self.df_erp.iterrows():
            node_id = row["erp_entry_id"]
            adj[node_id] = []
            erp_amounts[node_id] = float(row["gross_amount"])
            
        for _, row in self.df_gateway.iterrows():
            gw_id = row["payment_id"]
            adj[gw_id] = []
            gw_amounts[gw_id] = float(row["gross_amount"])
            for inv in parse_invoices(row.get("invoices")):
                erp_matches = self.df_erp[self.df_erp["invoice_number"] == inv]
                if not erp_matches.empty:
                    erp_id = erp_matches.iloc[0]["erp_entry_id"]
                    adj[gw_id].append(erp_id)
                    adj.setdefault(erp_id, []).append(gw_id)

        visited = set()
        for node in adj.keys():
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

                    if abs(sum_erp - sum_gw) < 0.01:
                        for e in erp_nodes: self.matched_erp_entries.add(e)
                        for g in gw_nodes:
                            self.gw_linked_to_erp.add(g)
                            self.gw_to_erp_links.setdefault(g, []).extend(erp_nodes)
                            for e in erp_nodes:
                                self.erp_gw_stage_map.setdefault(g, {})[e] = STAGE_EXACT_ERP_GW

    def match_fuzzy_gateway_to_bank(self):
        unmatched_gws = [row for _, row in self.df_gateway.iterrows() if row["payment_id"] not in self.gw_linked_to_bank]
        unmatched_banks = [row for _, row in self.df_bank.iterrows() if row["bank_entry_id"] not in self.matched_bank_entries]

        for gw_row in unmatched_gws:
            gw_id, gw_net, gw_utr = gw_row["payment_id"], round(float(gw_row["net_settled"]), 2), str(gw_row["bank_utr"])
            for bank_row in unmatched_banks:
                bank_id = bank_row["bank_entry_id"]
                if bank_id in self.matched_bank_entries: continue
                bank_credit = round(float(bank_row["credit_amount"]), 2)
                remittance = str(bank_row["remittance_info"])
                
                if abs(gw_net - bank_credit) < 0.01:
                    if (len(gw_utr) >= 10 and gw_utr[:-3] in remittance) or (gw_utr in remittance):
                        self.matched_bank_entries.add(bank_id)
                        self.gw_linked_to_bank.add(gw_id)
                        self.gw_bank_links[gw_id] = {
                            "bank_ids": [bank_id], "utr": gw_utr,
                            "match_type": MATCH_TYPE_FUZZY,
                            "matching_stage": STAGE_FUZZY_GW_BANK,
                            "score": 0.85,
                            "note": "Fuzzy Bank-Gateway match."
                        }
                        break

    def match_fuzzy_erp_to_gateway(self):
        unmatched_gws = [row for _, row in self.df_gateway.iterrows() if row["payment_id"] not in self.gw_linked_to_erp]
        unmatched_erps = [row for _, row in self.df_erp.iterrows() if row["erp_entry_id"] not in self.matched_erp_entries]

        for gw_row in unmatched_gws:
            gw_id, gw_gross = gw_row["payment_id"], round(float(gw_row["gross_amount"]), 2)
            invoices = parse_invoices(gw_row.get("invoices"))
            linked_bank = self.gw_bank_links.get(gw_id)
            remittance_ctx = ""
            if linked_bank:
                b_rows = self.df_bank[self.df_bank["bank_entry_id"].isin(linked_bank["bank_ids"])]
                remittance_ctx = " ".join(b_rows["remittance_info"].fillna("").values)

            for erp_row in unmatched_erps:
                erp_id = erp_row["erp_entry_id"]
                if erp_id in self.matched_erp_entries: continue
                erp_gross = round(float(erp_row["gross_amount"]), 2)
                stripped_inv = str(erp_row["invoice_number"]).replace("INV-", "")
                
                if abs(gw_gross - erp_gross) < 0.01:
                    is_fuzzy = False
                    if any(stripped_inv in str(inv) for inv in invoices):
                        is_fuzzy, note_add = True, "Invoice prefix missing."
                    elif stripped_inv in remittance_ctx:
                        is_fuzzy, note_add = True, "Gateway invoice recovered from Bank."
                    elif not invoices:
                        is_fuzzy, note_add = True, "IDs missing, matched via isolated exact amount."

                    if is_fuzzy:
                        self.matched_erp_entries.add(erp_id)
                        self.gw_linked_to_erp.add(gw_id)
                        self.gw_to_erp_links.setdefault(gw_id, []).append(erp_id)
                        self.erp_gw_stage_map.setdefault(gw_id, {})[erp_id] = STAGE_FUZZY_ERP_GW

                        if gw_id in self.gw_bank_links:
                            self.gw_bank_links[gw_id]["match_type"] = MATCH_TYPE_FUZZY
                            self.gw_bank_links[gw_id]["matching_stage"] = STAGE_FUZZY_GW_BANK
                            self.gw_bank_links[gw_id]["score"] = 0.85
                        else:
                            self.gw_bank_links[gw_id] = {
                                "bank_ids": [], "utr": gw_row["bank_utr"],
                                "match_type": MATCH_TYPE_FUZZY,
                                "matching_stage": STAGE_FUZZY_GW_BANK,
                                "score": 0.80,
                                "note": note_add,
                            }
                        break

    def assemble_graph_edges(self):
        for gw_id, erp_ids in self.gw_to_erp_links.items():
            gw_row = self.df_gateway[self.df_gateway["payment_id"] == gw_id]
            gw_gross = float(gw_row.iloc[0]["gross_amount"]) if not gw_row.empty else 0.0
            split_amt = round(gw_gross / len(erp_ids), 2)
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
                    "notes": f"Graph edge: ERP linked to Gateway {gw_id}."
                })

        for gw_id, bank_info in self.gw_bank_links.items():
            bank_ids = bank_info.get("bank_ids", [])
            gw_row = self.df_gateway[self.df_gateway["payment_id"] == gw_id]
            gw_net = float(gw_row.iloc[0]["net_settled"]) if not gw_row.empty else 0.0

            for b_id in bank_ids:
                self.gw_bank_edges.append({
                    "gateway_payment_id": gw_id,
                    "bank_entry_id": b_id,
                    "allocated_amount": gw_net,
                    "match_type": bank_info.get("match_type", MATCH_TYPE_EXACT),
                    "matching_stage": bank_info.get("matching_stage", STAGE_T1_IDENTIFIER),
                    "confidence_score": bank_info.get("score", 1.00),
                    "notes": bank_info.get("note", "Graph edge: Gateway linked to Bank.")
                })

    def run(self, deterministic_only: bool = False):
        self.match_exact_gateway_to_bank()
        self.match_exact_erp_to_gateway()
        self.match_combinatorial_gateway_to_bank()
        if not deterministic_only:
            self.match_fuzzy_gateway_to_bank()
            self.match_fuzzy_erp_to_gateway()
        self.assemble_graph_edges()
        
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

def run_exact_matching(db_path: Path = DB_PATH, deterministic_only: bool = False) -> Dict[str, any]:
    clear_graph_edges(db_path)
    df_erp, df_gw, df_bank = fetch_unmatched_records(db_path)
    engine = ReconciliationEngine(df_erp, df_gw, df_bank)
    results = engine.run(deterministic_only=deterministic_only)
    save_graph_edges(results["erp_gw_edges"], results["gw_bank_edges"], db_path)
    return results