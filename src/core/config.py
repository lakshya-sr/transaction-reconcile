"""
Configuration and constants for the Razorpay Multi-Source Reconciliation Agent.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
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

# Ground Truth Paths
GROUND_TRUTH_ERP_GW_PATH = DATA_DIR / "ground_truth_erp_gw.csv"
GROUND_TRUTH_ERP_GW_JSON_PATH = DATA_DIR / "ground_truth_erp_gw.json"
GROUND_TRUTH_GW_BANK_PATH = DATA_DIR / "ground_truth_gw_bank.csv"
GROUND_TRUTH_GW_BANK_JSON_PATH = DATA_DIR / "ground_truth_gw_bank.json"

# Database
DB_PATH = DATA_DIR / "reconciliation.db"

# Table Names
TABLE_ERP = "erp_ledger"
TABLE_GATEWAY = "gateway_settlements"
TABLE_BANK = "bank_statement"
TABLE_ERP_GW_TRUE = "erp_gw_true"
TABLE_GW_BANK_TRUE = "gw_bank_true"
TABLE_ERP_GW_PRED = "erp_gw_pred"
TABLE_GW_BANK_PRED = "gw_bank_pred"

# Match Types (Relationship types)
MATCH_TYPE_ONE_TO_ONE = "One-to-One"
MATCH_TYPE_MANY_TO_ONE = "Many-to-One"
MATCH_TYPE_ONE_TO_MANY = "One-to-Many"
MATCH_TYPE_FUZZY = "Fuzzy Similarity"
MATCH_TYPE_CLUSTER = "ML Cluster"
MATCH_TYPE_TDS = "TDS Exception"

# Backward compatibility aliases
MATCH_TYPE_EXACT = MATCH_TYPE_ONE_TO_ONE
MATCH_TYPE_BULK = MATCH_TYPE_MANY_TO_ONE

# Match Stages (Methods used)
MATCH_STAGE_IDENTIFIER = "Identifier Match"
MATCH_STAGE_SUBSET_SUM = "Subset Sum"
MATCH_STAGE_SUBSET_SUM_SPLIT = "Subset Sum Split"
MATCH_STAGE_AMOUNT_TEMPORAL = "Amount & Temporal"
MATCH_STAGE_FUZZY_ERP_GW = "Fuzzy ERP-Gateway"
MATCH_STAGE_FUZZY_GW_BANK = "Fuzzy Gateway-Bank"
MATCH_STAGE_ML_CLUSTER = "ML Cluster"
MATCH_STAGE_ML_SEED = "ML Seed"
MATCH_STAGE_ML_EXPAND = "ML Expand"

# Backward compatibility aliases
MATCH_STAGE_IDENTIFIER = MATCH_STAGE_IDENTIFIER
MATCH_STAGE_SUBSET_SUM = MATCH_STAGE_SUBSET_SUM
STAGE_EXACT_ERP_GW = MATCH_STAGE_IDENTIFIER
STAGE_FUZZY_ERP_GW = MATCH_STAGE_FUZZY_ERP_GW
STAGE_FUZZY_GW_BANK = MATCH_STAGE_FUZZY_GW_BANK
MATCH_STAGE_ML_CLUSTER = MATCH_STAGE_ML_CLUSTER
STAGE_CLUSTER_SEED_XGB = MATCH_STAGE_ML_SEED
STAGE_CLUSTER_EXPAND_XGB = MATCH_STAGE_ML_EXPAND

# Regex Patterns
INVOICE_REGEX = r"(INV[-_/]?(?:2026[-_/])?[a-zA-Z0-9]{4,10}|ORD[-_]?[a-zA-Z0-9]{4,10})"
UTR_REGEX = r"(UTR\d{12}|UTR[a-zA-Z0-9]+)"
SETTLEMENT_REGEX = r"(setl_[a-zA-Z0-9]+)"
