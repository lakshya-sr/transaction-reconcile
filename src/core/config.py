"""
Configuration and constants for the Razorpay Multi-Source Reconciliation Agent.
Upgraded to enterprise ERPNext/BenchRec, Razorpay Payload, and ISO 20022 CAMT.053 standards.
"""

from pathlib import Path

# Base Paths (Points to repository root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR = DATA_DIR / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# Visual Output Paths
RECONCILIATION_GRAPH_PATH = VISUALS_DIR / "reconciliation_graph.html"
ALL_DATA_GRAPH_PATH = VISUALS_DIR / "all_data_graph.html"

# Data File Paths
ERP_LEDGER_PATH = DATA_DIR / "erp_ledger.json"
GATEWAY_PAYOUTS_PATH = DATA_DIR / "gateway_payouts.json"
GATEWAY_SETTLEMENTS_PATH = DATA_DIR / "gateway_settlements.json"
BANK_STATEMENT_PATH = DATA_DIR / "bank_statement.csv"

# Ground Truth Edge File Paths (Split into two graph layers)
GROUND_TRUTH_ERP_GW_PATH = DATA_DIR / "ground_truth_erp_gw.csv"
GROUND_TRUTH_ERP_GW_JSON_PATH = DATA_DIR / "ground_truth_erp_gw.json"
GROUND_TRUTH_GW_BANK_PATH = DATA_DIR / "ground_truth_gw_bank.csv"
GROUND_TRUTH_GW_BANK_JSON_PATH = DATA_DIR / "ground_truth_gw_bank.json"

# Legacy alias for backward compatibility
GROUND_TRUTH_CSV_PATH = GROUND_TRUTH_ERP_GW_PATH

# Database Path
DB_PATH = DATA_DIR / "reconciliation.db"
ROOT_DB_PATH = BASE_DIR / "reconciliation.db"

# Schema Table Names
TABLE_ERP = "erp_ledger"
TABLE_GATEWAY = "gateway_settlements"
TABLE_BANK = "bank_statement"

# Ground Truth Edge Tables
TABLE_ERP_GW_TRUE = "erp_gw_true"
TABLE_GW_BANK_TRUE = "gw_bank_true"

# Predicted Edge Tables (Reconciliation Engine Results)
TABLE_ERP_GW_PRED = "erp_gw_pred"
TABLE_GW_BANK_PRED = "gw_bank_pred"

# Legacy alias
TABLE_RESULTS = "erp_gw_pred"

# Standard Match Types
MATCH_TYPE_EXACT = "Exact 1:1"
MATCH_TYPE_FUZZY = "Fuzzy Net Match"
MATCH_TYPE_BULK  = "Many:1 Bulk"
MATCH_TYPE_TDS   = "TDS Exception"
MATCH_TYPE_CLUSTER = "Cluster XGB"

# Matching Stage Labels (which pipeline phase produced the edge)
STAGE_T1_IDENTIFIER = "Tier1:Identifier"
STAGE_T1_SETL_ID    = STAGE_T1_IDENTIFIER
STAGE_T1_UTR        = STAGE_T1_IDENTIFIER
STAGE_T1_INVOICE    = STAGE_T1_IDENTIFIER
STAGE_T2_SUBSET_SUM = "Tier2-4:SubsetSum"
STAGE_EXACT_ERP_GW  = "Exact:ERP-GW"
STAGE_FUZZY_GW_BANK = "Fuzzy:GW-Bank"
STAGE_FUZZY_ERP_GW  = "Fuzzy:ERP-GW"
STAGE_CLUSTER_SEED_XGB = "Cluster:Seed-XGB"
STAGE_CLUSTER_EXPAND_XGB = "Cluster:Expand-XGB"

# Regex Patterns for extracting invoice numbers, ERP entry IDs, and UTRs from CAMT.053 remittance info
INVOICE_REGEX = r"(INV[-_/]?(?:2026[-_/])?[a-zA-Z0-9]{4,10}|ORD[-_]?[a-zA-Z0-9]{4,10})"
UTR_REGEX = r"(UTR\d{12}|UTR[a-zA-Z0-9]+)"
SETTLEMENT_REGEX = r"(setl_[a-zA-Z0-9]+)"
