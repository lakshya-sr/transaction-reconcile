#!/usr/bin/env python3
"""
Phase 1 Entrypoint: Synthetic Data Generation.
Enterprise schemas: ERPNext/BenchRec, Razorpay Payload, ISO 20022 CAMT.053.

Run:
    python generate_data.py
    or
    uv run generate_data.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.generator import generate_dataset
from src.config import ERP_LEDGER_PATH, GATEWAY_SETTLEMENTS_PATH, BANK_STATEMENT_PATH


def main():
    print("=" * 75)
    print("  PHASE 1: SYNTHETIC DATA GENERATION (Razorpay AI Buildathon)")
    print("=" * 75)
    print("[*] Generating 100 linked multi-source transaction records...")
    print("    - ERP Schema: BenchRec/ERPNext standard (with TDS & Allocation Keys)")
    print("    - Gateway Schema: Razorpay JSON Payload (with 2% MDR & 18% GST)")
    print("    - Bank Schema: ISO 20022 CAMT.053 standard (with Value Date delays)")
    
    erp_records, gateway_records, bank_records = generate_dataset(total_transactions=100, seed=42)
    
    print("\n[+] Dataset Generation Summary:")
    print(f"  1. ERP Ledger Records          : {len(erp_records):>3} -> {ERP_LEDGER_PATH.name}")
    print(f"  2. Gateway Settlements Records : {len(gateway_records):>3} -> {GATEWAY_SETTLEMENTS_PATH.name}")
    print(f"  3. Bank Statement (CAMT.053)   : {len(bank_records):>3} -> {BANK_STATEMENT_PATH.name}")
    
    # Sample previews
    print("\n[+] Sample ERP Ledger Record (ERPNext format):")
    print(f"    {erp_records[0]}")
    
    print("\n[+] Sample Gateway Settlement Record (Razorpay Payload format):")
    fee_sample = next((r for r in gateway_records if r["fee_deducted"] > 0), gateway_records[0])
    print(f"    {fee_sample}")
    
    print("\n[+] Sample Bank Statement Record (ISO 20022 CAMT.053 format):")
    print(f"    {bank_records[0]}")
    
    print("\n[✔] Phase 1 Completed Successfully. Ready for Phase 2 (db_setup.py).")
    print("=" * 75)


if __name__ == "__main__":
    main()
