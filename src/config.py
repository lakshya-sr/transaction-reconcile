"""
Configuration and constants for the Razorpay Multi-Source Reconciliation Agent.
Upgraded to enterprise ERPNext/BenchRec, Razorpay Payload, and ISO 20022 CAMT.053 standards.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data File Paths
ERP_LEDGER_PATH = DATA_DIR / "erp_ledger.json"
GATEWAY_PAYOUTS_PATH = DATA_DIR / "gateway_payouts.json"
GATEWAY_SETTLEMENTS_PATH = DATA_DIR / "gateway_settlements.json"  # Canonical alias
BANK_STATEMENT_PATH = DATA_DIR / "bank_statement.csv"
DB_PATH = DATA_DIR / "reconciliation.db"

# Root DB fallback
ROOT_DB_PATH = BASE_DIR / "reconciliation.db"

# Schema Table Names (Industry standard)
TABLE_ERP = "erp_ledger"
TABLE_GATEWAY = "gateway_settlements"
TABLE_BANK = "bank_statement"
TABLE_RESULTS = "reconciliation_results"

# Standard Match Types
MATCH_TYPE_EXACT = "Exact 1:1"
MATCH_TYPE_FUZZY = "Fuzzy Net Match"
MATCH_TYPE_BULK = "Many:1 Bulk"
MATCH_TYPE_TDS = "TDS Exception"

# Regex Patterns for extracting invoice numbers, ERP entry IDs, and UTRs from CAMT.053 remittance info
INVOICE_REGEX = r"(INV[-_/]?(?:2026[-_/])?\d{4,6}|ORD[-_]?\d{4,6})"
UTR_REGEX = r"(UTR\d{12})"
SETTLEMENT_REGEX = r"(setl_[a-zA-Z0-9]+)"
