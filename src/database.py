"""
Phase 2: Database Setup & Ingestion Module.

Implements database schemas for:
1. `erp_ledger`: ERPNext / BenchRec standard
2. `gateway_settlements`: Razorpay JSON payload standard
3. `bank_statement`: ISO 20022 CAMT.053 standard
4. `reconciliation_results`: Multi-source audit trail ledger
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import (
    BANK_STATEMENT_PATH,
    DB_PATH,
    ERP_LEDGER_PATH,
    GATEWAY_PAYOUTS_PATH,
    GATEWAY_SETTLEMENTS_PATH,
    TABLE_BANK,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_RESULTS,
)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path = DB_PATH) -> None:
    """
    Initialize SQLite database tables strictly adhering to enterprise schemas.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    try:
        # 1. ERP Ledger Table (BenchRec / ERPNext standard)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_ERP} (
                erp_entry_id VARCHAR(50) PRIMARY KEY,
                customer_account_id VARCHAR(50),
                invoice_number VARCHAR(50),
                gross_amount DECIMAL(15, 2),
                tds_expected DECIMAL(15, 2) DEFAULT 0,
                currency VARCHAR(3) DEFAULT 'INR',
                entry_date TIMESTAMP,
                status VARCHAR(20),
                allocation_key VARCHAR(100)
            );
        """)

        # 2. Gateway Settlements Table (Razorpay Payload standard)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_GATEWAY} (
                payment_id VARCHAR(50) PRIMARY KEY,
                settlement_id VARCHAR(50),
                gateway_status VARCHAR(20),
                gross_amount DECIMAL(15, 2),
                fee_deducted DECIMAL(10, 2),
                tax_on_fee DECIMAL(10, 2),
                net_settled DECIMAL(15, 2),
                amount_reversed DECIMAL(15, 2) DEFAULT 0,
                settled_at TIMESTAMP,
                bank_utr VARCHAR(50)
            );
        """)

        # 3. Bank Statement Table (ISO 20022 CAMT.053 standard)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_BANK} (
                bank_entry_id VARCHAR(50) PRIMARY KEY,
                value_date DATE,
                transaction_type VARCHAR(10),
                credit_amount DECIMAL(15, 2),
                debit_amount DECIMAL(15, 2),
                running_balance DECIMAL(15, 2),
                remittance_info TEXT,
                reversal_indicator BOOLEAN DEFAULT FALSE
            );
        """)

        # 4. Reconciliation Results Table (Audit Trail Ledger)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_RESULTS} (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                erp_order_id VARCHAR(50),
                gateway_payment_id VARCHAR(50),
                bank_utr VARCHAR(50),
                match_type VARCHAR(20),
                confidence_score DECIMAL(3, 2),
                notes TEXT
            );
        """)

        # Indexes for fast lookup and relational reconciliation
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_erp_invoice ON {TABLE_ERP}(invoice_number);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_gw_settlement ON {TABLE_GATEWAY}(settlement_id);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_gw_utr ON {TABLE_GATEWAY}(bank_utr);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_res_erp ON {TABLE_RESULTS}(erp_order_id);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_res_gw ON {TABLE_RESULTS}(gateway_payment_id);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_res_bank ON {TABLE_RESULTS}(bank_utr);")

        conn.commit()
    finally:
        conn.close()


def ingest_data(
    erp_path: Path = ERP_LEDGER_PATH,
    gateway_path: Path = GATEWAY_SETTLEMENTS_PATH,
    bank_path: Path = BANK_STATEMENT_PATH,
    db_path: Path = DB_PATH,
) -> Dict[str, int]:
    """
    Read JSON and CSV data files using pandas and ingest them into SQLite tables.
    
    Returns:
        Dictionary with record counts ingested per table.
    """
    if not erp_path.exists():
        raise FileNotFoundError(f"ERP ledger file not found at: {erp_path}")
    
    # Support both gateway_settlements.json and gateway_payouts.json
    actual_gw_path = gateway_path if gateway_path.exists() else GATEWAY_PAYOUTS_PATH
    if not actual_gw_path.exists():
        raise FileNotFoundError(f"Gateway settlements file not found at: {gateway_path}")
        
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank statement file not found at: {bank_path}")

    # Ensure tables exist
    init_database(db_path)

    # 1. Load ERP Ledger (JSON)
    df_erp = pd.read_json(erp_path)
    
    # 2. Load Gateway Settlements (JSON)
    df_gateway = pd.read_json(actual_gw_path)
    
    # 3. Load Bank Statement (CSV)
    df_bank = pd.read_csv(bank_path)

    conn = get_connection(db_path)
    counts = {}

    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {TABLE_ERP};")
        cursor.execute(f"DELETE FROM {TABLE_GATEWAY};")
        cursor.execute(f"DELETE FROM {TABLE_BANK};")
        cursor.execute(f"DELETE FROM {TABLE_RESULTS};")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name = ?;", (TABLE_RESULTS,))
        conn.commit()

        # Ingest using pandas
        df_erp.to_sql(TABLE_ERP, conn, if_exists="append", index=False)
        df_gateway.to_sql(TABLE_GATEWAY, conn, if_exists="append", index=False)
        df_bank.to_sql(TABLE_BANK, conn, if_exists="append", index=False)

        counts[TABLE_ERP] = len(df_erp)
        counts[TABLE_GATEWAY] = len(df_gateway)
        counts[TABLE_BANK] = len(df_bank)

        conn.commit()
    finally:
        conn.close()

    return counts


def get_table_counts(db_path: Path = DB_PATH) -> Dict[str, int]:
    """Query and return record counts from all tables."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    counts = {}
    for table in [TABLE_ERP, TABLE_GATEWAY, TABLE_BANK, TABLE_RESULTS]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cursor.fetchone()[0]
    conn.close()
    return counts
