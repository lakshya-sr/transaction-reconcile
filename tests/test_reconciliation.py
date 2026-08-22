"""
Unit and Integration Tests for Multi-Source Reconciliation Pipeline (Enterprise Schemas).
"""

import sqlite3
import pytest
from pathlib import Path
import pandas as pd

from src.config import (
    BASE_DIR,
    DATA_DIR,
    ERP_LEDGER_PATH,
    GATEWAY_SETTLEMENTS_PATH,
    BANK_STATEMENT_PATH,
    DB_PATH,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_RESULTS,
    MATCH_TYPE_EXACT,
)
from src.generator import generate_dataset
from src.matcher import (
    extract_invoice_number,
    extract_utr_number,
    extract_settlement_id,
    run_exact_matching,
)
from src.database import init_database, ingest_data, get_connection, get_table_counts


def test_regex_extractors():
    """Test regex extraction of Invoices, UTRs, and Settlement IDs from CAMT.053 narrations."""
    sample_narrative_1 = "NEFT-RAZORPAY-INV-10016-UTR342160733754-UPI"
    assert extract_invoice_number(sample_narrative_1) == "INV-10016"
    assert extract_utr_number(sample_narrative_1) == "UTR342160733754"

    sample_narrative_2 = "UPI/RPAY/setl_998124/INV-10023/DIRECT"
    assert extract_invoice_number(sample_narrative_2) == "INV-10023"
    assert extract_settlement_id(sample_narrative_2) == "setl_998124"

    sample_narrative_3 = "IMPS-P2A-INV-10005-SETTLEMENT-setl_ab12cd34"
    assert extract_invoice_number(sample_narrative_3) == "INV-10005"
    assert extract_settlement_id(sample_narrative_3) == "setl_ab12cd34"

    # Obscured should return None
    assert extract_invoice_number("NEFT-BULK-CR-SETTLE-491201/NODAL-POOL") is None
    assert extract_utr_number("NEFT-BULK-CR-SETTLE-491201/NODAL-POOL") is None


def test_data_generation_enterprise_rules(tmp_path):
    """Test synthetic data generation rules, 2% MDR fee, 18% GST, and CAMT.053 tags."""
    erp, gw, bank = generate_dataset(total_transactions=100, seed=42, output_dir=tmp_path)
    
    assert len(erp) == 100
    assert len(gw) == 100
    assert len(bank) == 100

    # 1. Test fee deduction ratio (30% of records)
    fee_records = [r for r in gw if r["fee_deducted"] > 0]
    assert len(fee_records) == 30

    for r in fee_records:
        expected_fee = round(r["gross_amount"] * 0.02, 2)
        expected_gst = round(expected_fee * 0.18, 2)
        assert abs(r["fee_deducted"] - expected_fee) <= 0.01
        assert abs(r["tax_on_fee"] - expected_gst) <= 0.01
        expected_net = round(r["gross_amount"] - (r["fee_deducted"] + r["tax_on_fee"]), 2)
        assert abs(r["net_settled"] - expected_net) <= 0.01

    # 2. Test CAMT.053 tags and fields
    for b in bank:
        assert "bank_entry_id" in b
        assert "value_date" in b
        assert "remittance_info" in b
        assert "running_balance" in b
        assert b["transaction_type"] == "CRDT"


def test_database_ingestion_and_exact_matching(tmp_path):
    """Test end-to-end SQLite ingestion and deterministic exact matching engine."""
    test_db = tmp_path / "test_reconciliation.db"
    
    # 1. Generate
    erp, gw, bank = generate_dataset(total_transactions=100, seed=42, output_dir=tmp_path)

    # 2. Ingest
    counts = ingest_data(
        erp_path=tmp_path / "erp_ledger.json",
        gateway_path=tmp_path / "gateway_settlements.json",
        bank_path=tmp_path / "bank_statement.csv",
        db_path=test_db,
    )
    assert counts[TABLE_ERP] == 100
    assert counts[TABLE_GATEWAY] == 100
    assert counts[TABLE_BANK] == 100

    # 3. Match
    stats = run_exact_matching(test_db)
    assert stats["exact_matches_count"] == 75  # 75 clean Exact 1:1 matches
    assert stats["unmatched_erp_count"] == 25  # 5 TDS + 10 obscured + 5 amount diff + 5 orphan

    # 4. Verify results table in DB
    conn = get_connection(test_db)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_RESULTS} WHERE match_type = '{MATCH_TYPE_EXACT}' AND confidence_score = 1.0")
    exact_in_db = cursor.fetchone()[0]
    conn.close()

    assert exact_in_db == 75
