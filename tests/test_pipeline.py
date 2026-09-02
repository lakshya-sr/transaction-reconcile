import pytest
import sqlite3
from pathlib import Path
import pandas as pd

from src.core.config import DB_PATH
from src.core.database import init_database, ingest_data, get_connection
from src.simulation.generate_data import run_continuous_simulation
from src.deterministic.matcher import ReconciliationEngine, run_exact_matching
from src.reporting.evaluate import calculate_metrics


def test_calculate_metrics():
    # Perfect match
    precision, recall, f1 = calculate_metrics(tp=10, fp=0, fn=0)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

    # Half precision
    precision, recall, f1 = calculate_metrics(tp=5, fp=5, fn=0)
    assert precision == 0.5
    assert recall == 1.0
    assert pytest.approx(f1, 0.01) == 0.666

    # Zero cases
    precision, recall, f1 = calculate_metrics(tp=0, fp=0, fn=0)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_simulation_and_matcher(tmp_path):
    erp_df, gw_df, bank_df, eg_true, gb_true = run_continuous_simulation(days=1, seed=42)

    assert not erp_df.empty
    assert not gw_df.empty
    assert not bank_df.empty
    assert not eg_true.empty
    assert not gb_true.empty

    # Test Reconciliation Engine
    engine = ReconciliationEngine(erp_df, gw_df, bank_df)
    results = engine.run()

    assert "erp_gw_edges" in results
    assert "gw_bank_edges" in results
    assert len(results["erp_gw_edges"]) > 0
    assert len(results["gw_bank_edges"]) > 0


def test_db_setup_and_matching_pipeline(tmp_path):
    test_db = tmp_path / "test_recon.db"
    init_database(test_db)
    assert test_db.exists()

    conn = get_connection(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "erp_ledger" in tables
    assert "gateway_settlements" in tables
    assert "bank_statement" in tables
    assert "erp_gw_pred" in tables
    assert "gw_bank_pred" in tables
