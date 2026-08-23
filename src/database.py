import sqlite3
import json
import pandas as pd
from pathlib import Path
from src.config import TABLE_ERP, TABLE_GATEWAY, TABLE_BANK

def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_ERP} (
            erp_entry_id TEXT PRIMARY KEY,
            customer_account_id TEXT,
            invoice_number TEXT,
            gross_amount REAL,
            tds_expected REAL,
            currency TEXT,
            entry_date TEXT,
            status TEXT,
            allocation_key TEXT
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_GATEWAY} (
            payment_id TEXT PRIMARY KEY,
            settlement_id TEXT,
            gateway_status TEXT,
            gross_amount REAL,
            fee_deducted REAL,
            tax_on_fee REAL,
            net_settled REAL,
            amount_reversed REAL,
            settled_at TEXT,
            bank_utr TEXT,
            invoices TEXT
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_BANK} (
            bank_entry_id TEXT PRIMARY KEY,
            value_date TEXT,
            transaction_type TEXT,
            credit_amount REAL,
            debit_amount REAL,
            running_balance REAL,
            remittance_info TEXT,
            reversal_indicator BOOLEAN
        )
    """)

    # New Graph Edge Tables replacing the flat audit ledger
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS erp_to_gateway_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            erp_order_id TEXT NOT NULL,
            gateway_payment_id TEXT NOT NULL,
            allocated_amount REAL NOT NULL,
            match_type TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gateway_to_bank_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            gateway_payment_id TEXT NOT NULL,
            bank_entry_id TEXT NOT NULL,
            allocated_amount REAL NOT NULL,
            match_type TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()

def ingest_data(erp_path: Path, gateway_path: Path, bank_path: Path, db_path: Path) -> dict:
    conn = get_connection(db_path)
    counts = {}

    if erp_path.exists():
        with open(erp_path, "r", encoding="utf-8") as f:
            erp_data = json.load(f)
        df_erp = pd.DataFrame(erp_data)
        df_erp.to_sql(TABLE_ERP, conn, if_exists="replace", index=False)
        counts[TABLE_ERP] = len(df_erp)

    if gateway_path.exists():
        with open(gateway_path, "r", encoding="utf-8") as f:
            gw_data = json.load(f)
        for row in gw_data:
            if "invoices" in row and isinstance(row["invoices"], list):
                row["invoices"] = json.dumps(row["invoices"])
        df_gw = pd.DataFrame(gw_data)
        df_gw.to_sql(TABLE_GATEWAY, conn, if_exists="replace", index=False)
        counts[TABLE_GATEWAY] = len(df_gw)

    if bank_path.exists():
        df_bank = pd.read_csv(bank_path)
        df_bank.to_sql(TABLE_BANK, conn, if_exists="replace", index=False)
        counts[TABLE_BANK] = len(df_bank)

    conn.commit()
    conn.close()
    return counts