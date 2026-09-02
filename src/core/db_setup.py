#!/usr/bin/env python3
"""
Phase 2 Entrypoint: Database Setup & Data Ingestion.
"""

from src.core.database import init_database, ingest_data, get_connection
from src.core.config import (
    DB_PATH,
    ERP_LEDGER_PATH,
    GATEWAY_SETTLEMENTS_PATH,
    BANK_STATEMENT_PATH,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_ERP_GW_TRUE,
    TABLE_GW_BANK_TRUE,
)


def main():
    init_database(DB_PATH)
    counts = ingest_data(
        erp_path=ERP_LEDGER_PATH,
        gateway_path=GATEWAY_SETTLEMENTS_PATH,
        bank_path=BANK_STATEMENT_PATH,
        db_path=DB_PATH,
    )
    print(f"[✔] DB Ingested:")
    print(f"    - {TABLE_ERP:<22}: {counts.get(TABLE_ERP, 0)} records")
    print(f"    - {TABLE_GATEWAY:<22}: {counts.get(TABLE_GATEWAY, 0)} records")
    print(f"    - {TABLE_BANK:<22}: {counts.get(TABLE_BANK, 0)} records")
    print(f"    - {TABLE_ERP_GW_TRUE:<22}: {counts.get(TABLE_ERP_GW_TRUE, 0)} edges")
    print(f"    - {TABLE_GW_BANK_TRUE:<22}: {counts.get(TABLE_GW_BANK_TRUE, 0)} edges")


if __name__ == "__main__":
    main()

