#!/usr/bin/env python3
"""
Diagnostic Visualizer: Unreconciled Records Viewer.

Identifies and displays all records from ERP, Gateway, and Bank that
failed to find complete connections in the graph-based reconciliation engine.
"""

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from src.core.config import (
    DB_PATH,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_ERP_GW_PRED,
    TABLE_GW_BANK_PRED,
)
from src.core.database import get_connection


def truncate_text(text, max_len=40):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len-3] + "..."


def main():
    conn = get_connection(DB_PATH)
    
    # 1. Load Raw Tables
    df_erp = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn)
    df_gw = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
    df_bank = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}", conn)
    
    # 2. Load Graph Edges
    try:
        df_eg_edges = pd.read_sql_query(f"SELECT erp_order_id, gateway_payment_id FROM {TABLE_ERP_GW_PRED}", conn)
        df_gb_edges = pd.read_sql_query(f"SELECT gateway_payment_id, bank_entry_id FROM {TABLE_GW_BANK_PRED}", conn)
    except Exception as e:
        print("[!] Could not read edge tables. Have you run Phase 3 matching yet?")
        conn.close()
        return
        
    conn.close()

    # 3. Identify Matched IDs from Graph Edges
    matched_erp = set(df_eg_edges["erp_order_id"].dropna())
    matched_bank = set(df_gb_edges["bank_entry_id"].dropna())
    
    # A Gateway payment is fully reconciled ONLY if it links to BOTH an ERP and a Bank entry
    gw_with_erp = set(df_eg_edges["gateway_payment_id"].dropna())
    gw_with_bank = set(df_gb_edges["gateway_payment_id"].dropna())
    fully_matched_gw = gw_with_erp.intersection(gw_with_bank)
    
    # 4. Filter Unmatched Records
    unmatched_erp = df_erp[~df_erp["erp_entry_id"].isin(matched_erp)].copy()
    unmatched_gw = df_gw[~df_gw["payment_id"].isin(fully_matched_gw)].copy()
    unmatched_bank = df_bank[~df_bank["bank_entry_id"].isin(matched_bank)].copy()

    print("=" * 100)
    print("  UNRECONCILED RECORDS VIEWER (Graph Database)")
    print("=" * 100)
    print(f"Total ERP Records       : {len(df_erp):<4} | Unmatched: {len(unmatched_erp)}")
    print(f"Total Gateway Records   : {len(df_gw):<4} | Unmatched: {len(unmatched_gw)}")
    print(f"Total Bank Records      : {len(df_bank):<4} | Unmatched: {len(unmatched_bank)}")
    print("=" * 100 + "\n")

    # 5. Display ERP Table
    if not unmatched_erp.empty:
        print(f"--- UNRECONCILED ERP INVOICES ({len(unmatched_erp)}) ---")
        display_erp = unmatched_erp[["erp_entry_id", "invoice_number", "gross_amount", "entry_date"]]
        print(tabulate(display_erp, headers=["ERP ID", "Invoice", "Gross ₹", "Date"], tablefmt="fancy_grid", showindex=False))
        print("\n")
    else:
        print("--- UNRECONCILED ERP INVOICES (0) ---\n[✔] All ERP records fully matched.\n")

    # 6. Display Gateway Table
    if not unmatched_gw.empty:
        print(f"--- UNRECONCILED GATEWAY PAYMENTS ({len(unmatched_gw)}) ---")
        display_gw = unmatched_gw[["payment_id", "gross_amount", "net_settled", "bank_utr", "invoices"]].copy()
        display_gw["invoices"] = display_gw["invoices"].apply(truncate_text)
        
        def get_missing_side(gw_id):
            missing = []
            if gw_id not in gw_with_erp: missing.append("ERP")
            if gw_id not in gw_with_bank: missing.append("Bank")
            return "Missing: " + " & ".join(missing)
            
        display_gw["status"] = display_gw["payment_id"].apply(get_missing_side)
        
        print(tabulate(display_gw, headers=["Gateway ID", "Gross ₹", "Net ₹", "UTR", "Invoices", "Graph Status"], tablefmt="fancy_grid", showindex=False))
        print("\n")
    else:
        print("--- UNRECONCILED GATEWAY PAYMENTS (0) ---\n[✔] All Gateway records fully matched.\n")

    # 7. Display Bank Table
    if not unmatched_bank.empty:
        print(f"--- UNRECONCILED BANK DEPOSITS ({len(unmatched_bank)}) ---")
        display_bank = unmatched_bank[["bank_entry_id", "value_date", "credit_amount", "remittance_info"]].copy()
        display_bank["remittance_info"] = display_bank["remittance_info"].apply(lambda x: truncate_text(x, 60))
        print(tabulate(display_bank, headers=["Bank ID", "Date", "Credit ₹", "Remittance Narrative"], tablefmt="fancy_grid", showindex=False))
        print("\n")
    else:
        print("--- UNRECONCILED BANK DEPOSITS (0) ---\n[✔] All Bank records fully matched.\n")


if __name__ == "__main__":
    main()