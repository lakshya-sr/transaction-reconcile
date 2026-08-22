#!/usr/bin/env python3
"""
Phase 3 Entrypoint: Exact Matching Engine.
Enterprise Multi-Source Reconciliation for Razorpay AI Buildathon.

Run:
    python exact_matcher.py
    or
    uv run exact_matcher.py
"""

import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.matcher import run_exact_matching
from src.config import DB_PATH, TABLE_RESULTS
from src.database import get_connection


def main():
    print("=" * 85)
    print("  PHASE 3: EXACT MATCHING ENGINE (Razorpay AI Buildathon)")
    print("=" * 85)
    print(f"[*] Connecting to database: {DB_PATH.name}")
    print("[*] Running 1:1 deterministic 3-way reconciliation (ERP ↔ Gateway ↔ Bank)...")

    stats = run_exact_matching(DB_PATH)

    # 1. Summary Statistics Table
    summary_data = [
        ["Total ERP Invoices Ingested", stats["total_erp"]],
        ["Total Gateway Settlements Ingested", stats["total_gateway"]],
        ["Total Bank Statement Credits Ingested", stats["total_bank"]],
        ["1:1 Exact Matches Found ('Exact 1:1')", f"{stats['exact_matches_count']} (Confidence: 1.00)"],
        ["Unmatched ERP Records (TDS/Pending)", stats["unmatched_erp_count"]],
        ["Unmatched Gateway Settlements", stats["unmatched_gateway_count"]],
        ["Unmatched Bank Statements (CAMT.053)", stats["unmatched_bank_count"]],
    ]

    print("\n[+] Reconciliation Summary Statistics:")
    print(tabulate(summary_data, headers=["Reconciliation Metric", "Count / Status"], tablefmt="fancy_grid"))

    # 2. Preview Sample Exact Matches
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT match_id, erp_order_id, gateway_payment_id, bank_utr, match_type, confidence_score, notes
        FROM {TABLE_RESULTS}
        LIMIT 3
    """)
    sample_matches = [list(row) for row in cursor.fetchall()]
    conn.close()

    print("\n[+] Audit Trail Preview: Records in 'reconciliation_results':")
    headers_matches = ["Match_ID", "ERP_Entry_ID", "Gateway_Payment_ID", "Bank_UTR", "Match_Type", "Score", "Notes"]
    print(tabulate(sample_matches, headers=headers_matches, tablefmt="fancy_grid", maxcolwidths=[None, 16, 20, 16, 12, 6, 35]))

    # 3. Unmatched Records Diagnostic Breakdown (Candidates for LLM Agent)
    print("\n[!] UNMATCHED RECORDS DIAGNOSTICS (Ready for LLM Fuzzy Resolution):")
    
    if not stats["unmatched_bank"].empty:
        print("\n  --> Bank Records Requiring Fuzzy / Semantic LLM Matching (Sample):")
        unmatched_bank_sample = stats["unmatched_bank"][["bank_entry_id", "value_date", "remittance_info", "credit_amount", "diagnosis"]].head(5)
        print(tabulate(unmatched_bank_sample.values, headers=["Bank Entry ID", "Value Date", "Remittance Info", "Credit ₹", "Diagnosis"], tablefmt="grid", maxcolwidths=[18, 12, 30, 10, 35]))

    if not stats["unmatched_erp"].empty:
        print("\n  --> ERP Invoices Requiring Exception Handling / TDS Matching (Sample):")
        unmatched_erp_sample = stats["unmatched_erp"][["erp_entry_id", "invoice_number", "gross_amount", "tds_expected", "status"]].head(5)
        print(tabulate(unmatched_erp_sample.values, headers=["ERP Entry ID", "Invoice #", "Gross ₹", "TDS Expected ₹", "Status"], tablefmt="grid"))

    print("\n[✔] Phase 3 Completed Successfully.")
    print(f"[✔] Reconciliation Results Persisted in '{DB_PATH.name}' (Table: {TABLE_RESULTS}).")
    print("=" * 85)


if __name__ == "__main__":
    main()
