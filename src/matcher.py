#!/usr/bin/env python3
"""
Phase 3 & 4: Exact + Fuzzy Matching Engine Module with Graph Edge Assembly.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import (
    DB_PATH, MATCH_TYPE_EXACT, MATCH_TYPE_BULK,
    MATCH_TYPE_FUZZY, TABLE_BANK, TABLE_ERP,
    TABLE_GATEWAY
)
from src.database import get_connection

def extract_invoice_number(remittance_info: str) -> Optional[str]:
    if not remittance_info or not isinstance(remittance_info, str): return None
    match = re.search(r"(INV-[a-zA-Z0-9]+|ORD-[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None

def extract_utr_number(remittance_info: str) -> Optional[str]:
    if not remittance_info or not isinstance(remittance_info, str): return None
    match = re.search(r"(UTR\d{12}|UTR[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None

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
        self.matched_gw_payments = set()
        self.matched_bank_entries = set()
        
        self.gw_bank_links = {} 
        self.gw_to_erp_links = {}
        
        self.erp_gw_edges = []
        self.gw_bank_edges = []

    def match_exact_gateway_to_bank(self):
        gw_by_utr = {}
        for _, row in self.df_gateway.iterrows():
            gw_by_utr.setdefault(row["bank_utr"], []).append(row)

        bank_by_utr = {}
        for _, row in self.df_bank.iterrows():
            utr = extract_utr_number(row["remittance_info"])
            if utr: bank_by_utr.setdefault(utr, []).append(row)

        for utr, bank_rows in bank_by_utr.items():
            if utr in gw_by_utr:
                gw_rows = gw_by_utr[utr]
                sum_bank = sum(float(r["credit_amount"]) for r in bank_rows)
                sum_gw = sum(float(r["net_settled"]) for r in gw_rows)

                if abs(sum_bank - sum_gw) < 0.01:
                    for b in bank_rows: self.matched_bank_entries.add(b["bank_entry_id"])
                    for g in gw_rows:
                        gw_pid = g["payment_id"]
                        self.matched_gw_payments.add(gw_pid)
                        self.gw_bank_links[gw_pid] = {
                            "bank_ids": [b["bank_entry_id"] for b in bank_rows],
                            "utr": utr,
                            "match_type": MATCH_TYPE_EXACT if len(bank_rows) == 1 else MATCH_TYPE_BULK,
                            "score": 1.00,
                            "note": f"Exact Bank-Gateway match. Bank sum: ₹{sum_bank:.2f}."
                        }

    def match_combinatorial_gateway_to_bank(self):
        unmatched_gws = [row for _, row in self.df_gateway.iterrows() if row["payment_id"] not in self.matched_gw_payments]
        unmatched_banks = [row for _, row in self.df_bank.iterrows() if row["bank_entry_id"] not in self.matched_bank_entries]

        gw_by_date = {}
        for gw in unmatched_gws:
            gw_date = str(gw["settled_at"])[:10]
            gw_by_date.setdefault(gw_date, []).append(gw)

        bank_by_date = {}
        for b in unmatched_banks:
            b_date = str(b["value_date"])[:10]
            bank_by_date.setdefault(b_date, []).append(b)

        for b_date, banks in bank_by_date.items():
            gws_for_date = gw_by_date.get(b_date, [])
            if not gws_for_date: continue

            for b in banks:
                target_cents = int(round(float(b["credit_amount"]) * 100))
                items = [(g, int(round(float(g["net_settled"]) * 100))) for g in gws_for_date]
                items.sort(key=lambda x: x[1])
                
                suffix_sums = [0] * len(items)
                curr_sum = 0
                for i in range(len(items)-1, -1, -1):
                    curr_sum += items[i][1]
                    suffix_sums[i] = curr_sum

                def find_subset(start, current_sum, path):
                    if current_sum == target_cents: return path
                    if len(path) >= 6: return None
                    for i in range(start, len(items)):
                        item_val = items[i][1]
                        if current_sum + item_val > target_cents: break
                        if current_sum + suffix_sums[i] < target_cents: continue
                        res = find_subset(i + 1, current_sum + item_val, path + [items[i][0]])
                        if res: return res
                    return None

                matched_combo = find_subset(0, 0, [])
                if matched_combo:
                    b_id = b["bank_entry_id"]
                    self.matched_bank_entries.add(b_id)
                    for g in matched_combo:
                        g_id = g["payment_id"]
                        self.matched_gw_payments.add(g_id)
                        self.gw_bank_links[g_id] = {
                            "bank_ids": [b_id],
                            "utr": g["bank_utr"],
                            "match_type": MATCH_TYPE_BULK if len(matched_combo) > 1 else MATCH_TYPE_EXACT,
                            "score": 1.00,
                            "note": f"Combinatorial subset sum match. Batch size: {len(matched_combo)}."
                        }
                    matched_ids = {g["payment_id"] for g in matched_combo}
                    gws_for_date = [g for g in gws_for_date if g["payment_id"] not in matched_ids]

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
                            self.matched_gw_payments.add(g)
                            self.gw_to_erp_links.setdefault(g, []).extend(erp_nodes)

    def match_fuzzy_gateway_to_bank(self):
        unmatched_gws = [row for _, row in self.df_gateway.iterrows() if row["payment_id"] not in self.gw_bank_links]
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
                        self.matched_gw_payments.add(gw_id)
                        self.gw_bank_links[gw_id] = {
                            "bank_ids": [bank_id], "utr": gw_utr,
                            "match_type": MATCH_TYPE_FUZZY, "score": 0.85,
                            "note": "Fuzzy Bank-Gateway match."
                        }
                        break

    def match_fuzzy_erp_to_gateway(self):
        unmatched_gws = [row for _, row in self.df_gateway.iterrows() if row["payment_id"] not in self.gw_to_erp_links]
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
                        self.matched_gw_payments.add(gw_id)
                        self.gw_to_erp_links.setdefault(gw_id, []).append(erp_id)
                        
                        if gw_id in self.gw_bank_links:
                            self.gw_bank_links[gw_id]["match_type"] = MATCH_TYPE_FUZZY
                            self.gw_bank_links[gw_id]["score"] = 0.85
                        else:
                            self.gw_bank_links[gw_id] = {
                                "bank_ids": [], "utr": gw_row["bank_utr"],
                                "match_type": MATCH_TYPE_FUZZY, "score": 0.80, "note": note_add
                            }
                        break

    def assemble_graph_edges(self):
        """Builds explicit 1:N, N:1, and M:N relationship graph edges."""
        # 1. Build ERP <-> Gateway Edges
        for gw_id, erp_ids in self.gw_to_erp_links.items():
            gw_row = self.df_gateway[self.df_gateway["payment_id"] == gw_id]
            gw_gross = float(gw_row.iloc[0]["gross_amount"]) if not gw_row.empty else 0.0
            split_amt = round(gw_gross / len(erp_ids), 2)
            
            for erp_id in erp_ids:
                self.erp_gw_edges.append({
                    "erp_order_id": erp_id,
                    "gateway_payment_id": gw_id,
                    "allocated_amount": split_amt,
                    "match_type": MATCH_TYPE_BULK if len(erp_ids) > 1 else MATCH_TYPE_EXACT,
                    "confidence_score": 1.00,
                    "notes": f"Graph edge: ERP linked to Gateway {gw_id}."
                })

        # 2. Build Gateway <-> Bank Edges
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
                    "confidence_score": bank_info.get("score", 1.00),
                    "notes": bank_info.get("note", "Graph edge: Gateway linked to Bank.")
                })

    def run(self):
        self.match_exact_gateway_to_bank()
        # self.match_combinatorial_gateway_to_bank()
        self.match_exact_erp_to_gateway()
        self.match_fuzzy_gateway_to_bank()
        self.match_fuzzy_erp_to_gateway()
        self.assemble_graph_edges()
        
        unmatched_erp = self.df_erp[~self.df_erp["erp_entry_id"].isin(self.matched_erp_entries)].copy()
        unmatched_gw = self.df_gateway[~self.df_gateway["payment_id"].isin(self.matched_gw_payments)].copy()
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

def run_exact_matching(db_path: Path = DB_PATH) -> Dict[str, any]:
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM erp_to_gateway_edges;")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'erp_to_gateway_edges';")
        cursor.execute("DELETE FROM gateway_to_bank_edges;")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'gateway_to_bank_edges';")
        conn.commit()
    finally:
        conn.close()

    df_erp, df_gw, df_bank = fetch_unmatched_records(db_path)
    engine = ReconciliationEngine(df_erp, df_gw, df_bank)
    results = engine.run()
    
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        for edge in results["erp_gw_edges"]:
            cursor.execute("""
                INSERT INTO erp_to_gateway_edges 
                (erp_order_id, gateway_payment_id, allocated_amount, match_type, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (edge["erp_order_id"], edge["gateway_payment_id"], edge["allocated_amount"], edge["match_type"], edge["confidence_score"], edge["notes"]))

        for edge in results["gw_bank_edges"]:
            cursor.execute("""
                INSERT INTO gateway_to_bank_edges 
                (gateway_payment_id, bank_entry_id, allocated_amount, match_type, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (edge["gateway_payment_id"], edge["bank_entry_id"], edge["allocated_amount"], edge["match_type"], edge["confidence_score"], edge["notes"]))
        conn.commit()
    finally:
        conn.close()

    return results
