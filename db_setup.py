#!/usr/bin/env python3
"""
Phase 2 Entrypoint: Database Setup & Data Ingestion.
Enterprise schemas: erp_ledger, gateway_settlements, bank_statement, reconciliation_results.

Run:
    python db_setup.py
    or
    uv run db_setup.py
"""

import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database import init_database, ingest_data, get_table_counts, get_connection
from src.config import (
    DB_PATH,
    ERP_LEDGER_PATH,
    GATEWAY_SETTLEMENTS_PATH,
    BANK_STATEMENT_PATH,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_RESULTS,
)


def main():
    print("=" * 80)
    print("  PHASE 2: DATABASE SETUP & INGESTION (Razorpay AI Buildathon)")
    print("=" * 80)
    print(f"[*] Target SQLite Database: {DB_PATH.name}")

    # 1. Initialize DB schemas
    print("[*] Creating tables: erp_ledger, gateway_settlements, bank_statement, reconciliation_results...")
    init_database(DB_PATH)

    # 2. Ingest datasets
    print(f"[*] Ingesting datasets into SQLite...")
    print(f"    - ERP Ledger JSON           : {ERP_LEDGER_PATH.name}")
    print(f"    - Gateway Settlements JSON  : {GATEWAY_SETTLEMENTS_PATH.name}")
    print(f"    - Bank Statement CSV (CAMT) : {BANK_STATEMENT_PATH.name}")

    counts = ingest_data(
        erp_path=ERP_LEDGER_PATH,
        gateway_path=GATEWAY_SETTLEMENTS_PATH,
        bank_path=BANK_STATEMENT_PATH,
        db_path=DB_PATH,
    )

    # 3. Verification table
    table_summary = [
        ["1", TABLE_ERP, "ERPNext / BenchRec", counts.get(TABLE_ERP, 0), "erp_entry_id, customer_account_id, invoice_number, gross_amount, tds_expected, currency, entry_date, status, allocation_key"],
        ["2", TABLE_GATEWAY, "Razorpay Payload", counts.get(TABLE_GATEWAY, 0), "payment_id, settlement_id, gateway_status, gross_amount, fee_deducted, tax_on_fee, net_settled, amount_reversed, settled_at, bank_utr"],
        ["3", TABLE_BANK, "ISO 20022 CAMT.053", counts.get(TABLE_BANK, 0), "bank_entry_id, value_date, transaction_type, credit_amount, debit_amount, running_balance, remittance_info, reversal_indicator"],
        ["4", TABLE_RESULTS, "Reconciliation Audit Ledger", 0, "match_id, erp_order_id, gateway_payment_id, bank_utr, match_type, confidence_score, notes"],
    ]

    print("\n[+] Database Ingestion & Schema Summary:")
    print(tabulate(table_summary, headers=["#", "Table Name", "Standard/Source", "Records", "Columns"], tablefmt="fancy_grid", maxcolwidths=[None, 22, 20, 10, 35]))

    # 4. Preview sample rows
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    
    print("\n[+] Verification Row Previews:")
    
    cursor.execute(f"SELECT erp_entry_id, invoice_number, gross_amount, tds_expected, status FROM {TABLE_ERP} LIMIT 1")
    row_erp = dict(cursor.fetchone())
    print(f"    [ERP Ledger]        : {row_erp}")

    cursor.execute(f"SELECT payment_id, settlement_id, gross_amount, fee_deducted, tax_on_fee, net_settled, bank_utr FROM {TABLE_GATEWAY} WHERE fee_deducted > 0 LIMIT 1")
    row_gw = dict(cursor.fetchone())
    print(f"    [Gateway Settlement]: {row_gw}")

    cursor.execute(f"SELECT bank_entry_id, value_date, credit_amount, running_balance, remittance_info FROM {TABLE_BANK} LIMIT 1")
    row_bank = dict(cursor.fetchone())
    print(f"    [Bank Statement]    : {row_bank}")

    conn.close()

    print("\n[✔] Phase 2 Completed Successfully. Ready for Phase 3 (exact_matcher.py).")
    print("=" * 80)


if __name__ == "__main__":
    main()
