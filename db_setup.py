#!/usr/bin/env python3
"""
Phase 2 Entrypoint: Database Setup & Data Ingestion.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database import init_database, ingest_data, get_connection
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
    init_database(DB_PATH)
    counts = ingest_data(
        erp_path=ERP_LEDGER_PATH,
        gateway_path=GATEWAY_SETTLEMENTS_PATH,
        bank_path=BANK_STATEMENT_PATH,
        db_path=DB_PATH,
    )
    print(f"[✔] DB Ingested: {counts.get(TABLE_ERP, 0)} ERP, {counts.get(TABLE_GATEWAY, 0)} GW, {counts.get(TABLE_BANK, 0)} Bank records.")


if __name__ == "__main__":
    main()
