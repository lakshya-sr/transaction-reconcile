#!/usr/bin/env python3
"""
Diagnostic Visualizer: Ground Truth vs. Matcher Failures.

Provides a compact summary view of transactions that the deterministic
matcher failed to reconcile, exposing the exact numerical discrepancies.
"""

import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DB_PATH, TABLE_ERP, TABLE_GATEWAY, 
    TABLE_BANK, TABLE_RESULTS, DATA_DIR
)
from src.database import get_connection


def main():
    truth_csv = DATA_DIR / "ground_truth.csv"
    if not truth_csv.exists():
        print(f"[!] Ground truth CSV not found at {truth_csv}. Please run generate_data.py first.")
        return

    # 1. Load Data
    df_truth = pd.read_csv(truth_csv)
    conn = get_connection(DB_PATH)
    df_results = pd.read_sql_query(f"SELECT * FROM {TABLE_RESULTS}", conn)
    df_erp = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn)
    df_gw = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
    df_bank = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}", conn)
    conn.close()

    # 2. Identify Unmatched Gateway Hubs
    matched_gw_ids = set(df_results["gateway_payment_id"].dropna())
    unmatched_gw_ids = df_truth[~df_truth["gw_id"].isin(matched_gw_ids)]["gw_id"].dropna().unique()

    print("=" * 100)
    print("  UNMATCHED RECORDS VISUALIZER (Compact Ground Truth Comparison)")
    print("=" * 100)
    print(f"Total Gateway Records Generated : {len(df_gw)}")
    print(f"Successfully Matched Records    : {len(matched_gw_ids)}")
    print(f"Failed Records (To Analyze)     : {len(unmatched_gw_ids)}\n")

    if len(unmatched_gw_ids) == 0:
        print("[✔] Awesome! All records were perfectly matched by the engine.")
        return

    # 3. Build Compact Summary Table
    summary_table = []
    
    for gw_id in unmatched_gw_ids:
        truth_subset = df_truth[df_truth["gw_id"] == gw_id]
        erp_ids = truth_subset["erp_id"].dropna().unique()
        bank_ids = truth_subset["bank_id"].dropna().unique()
        
        erp_rows = df_erp[df_erp["erp_entry_id"].isin(erp_ids)]
        gw_row = df_gw[df_gw["payment_id"] == gw_id]
        bank_rows = df_bank[df_bank["bank_entry_id"].isin(bank_ids)]
        
        # Calculate Totals
        erp_gross = round(erp_rows["gross_amount"].astype(float).sum(), 2) if not erp_rows.empty else 0.0
        gw_gross = round(float(gw_row.iloc[0]["gross_amount"]), 2) if not gw_row.empty else 0.0
        gw_net = round(float(gw_row.iloc[0]["net_settled"]), 2) if not gw_row.empty else 0.0
        bank_credit = round(bank_rows["credit_amount"].astype(float).sum(), 2) if not bank_rows.empty else 0.0
        
        # Diagnose the mismatch
        diff_erp_gw = round(erp_gross - gw_gross, 2)
        diff_gw_bank = round(gw_net - bank_credit, 2)
        
        diagnosis_tags = []
        if diff_erp_gw != 0:
            diagnosis_tags.append(f"ERP/GW Diff: ₹{abs(diff_erp_gw):.2f}")
        if not bank_rows.empty and diff_gw_bank != 0:
            diagnosis_tags.append(f"GW/Bank Diff: ₹{abs(diff_gw_bank):.2f}")
        elif bank_rows.empty:
            diagnosis_tags.append("Bank Record Missing/Delayed")
            
        diag_str = " | ".join(diagnosis_tags) if diagnosis_tags else "Unknown structural failure"

        summary_table.append([
            gw_id,
            f"₹ {erp_gross:.2f}",
            f"₹ {gw_gross:.2f}",
            f"₹ {gw_net:.2f}",
            f"₹ {bank_credit:.2f}",
            diag_str
        ])

    # 4. Render Table
    headers = ["Gateway ID", "ERP Gross (True)", "GW Gross", "GW Net", "Bank Credit (True)", "Primary Discrepancy"]
    print(tabulate(summary_table, headers=headers, tablefmt="fancy_grid"))
    print("\n")


if __name__ == "__main__":
    main()