#!/usr/bin/env python3
"""
Demo script for video presentation.

Runs the reconciliation pipeline step-by-step with pauses for explanation.
"""

import subprocess
import sys
import time
import json
import csv
import os
from pathlib import Path

# Colors for terminal output
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
BOLD = '\033[1m'
NC = '\033[0m'

SIMULATION_DAYS = 3
BENCHMARK_SEED = 42


def print_banner(title: str, color: str = BLUE):
    """Print a formatted banner."""
    print(f"\n{color}{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}{NC}\n")


def wait_for_enter():
    """Wait for user to press Enter."""
    input()


def run_command(cmd: list, description: str):
    """Run a command and show its output."""
    print(f"{CYAN}▶ {description}{NC}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"{RED}✗ Command failed{NC}")
        sys.exit(1)
    
    return result


def count_records():
    """Count records in database."""
    import sqlite3
    from src.core.config import DB_PATH
    
    if not DB_PATH.exists():
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    counts = {}
    for table in ['erp_ledger', 'gateway_settlements', 'bank_statement', 
                  'erp_gw_pred', 'gw_bank_pred', 'erp_gw_true', 'gw_bank_true']:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        except:
            counts[table] = 0
    
    conn.close()
    return counts


def show_reconciliation_progress():
    """Show current reconciliation progress."""
    counts = count_records()
    if not counts:
        return
    
    total_true = counts.get('erp_gw_true', 0) + counts.get('gw_bank_true', 0)
    total_pred = counts.get('erp_gw_pred', 0) + counts.get('gw_bank_pred', 0)
    match_pct = (total_pred / total_true * 100) if total_true > 0 else 0
    
    print(f"\n{MAGENTA}📊 CURRENT STATUS:{NC}")
    print(f"  • ERP Orders: {counts.get('erp_ledger', 0)}")
    print(f"  • Gateway Payments: {counts.get('gateway_settlements', 0)}")
    print(f"  • Bank Deposits: {counts.get('bank_statement', 0)}")
    print(f"  • ERP↔GW Matches: {counts.get('erp_gw_pred', 0)} / {counts.get('erp_gw_true', 0)}")
    print(f"  • GW↔Bank Matches: {counts.get('gw_bank_pred', 0)} / {counts.get('gw_bank_true', 0)}")
    print(f"  • Overall Match Rate: {match_pct:.1f}%")
    print()


def main():
    print_banner("🎬 FINANCIAL RECONCILIATION AGENT - LIVE DEMO", BOLD)
    
    print(f"{GREEN}Multi-stage reconciliation pipeline with deterministic and AI matching.{NC}")
    
    wait_for_enter()
    
    # =========================================================================
    # STAGE 1: DATA GENERATION + DATABASE INGESTION
    # =========================================================================
    print_banner("📦 STAGE 1: DATA GENERATION", BLUE)
    
    print(f"{CYAN}Simulating {SIMULATION_DAYS} days of financial transactions:{NC}")
    print(f"  • ERP Ledger (Orders)")
    print(f"  • Gateway Settlements (Payments)")
    print(f"  • Bank Statement (Deposits)")
    print(f"  • Realistic noise: MDR fees, delayed settlements, corrupted UTRs")
    
    # Delete existing database
    from src.core.config import DB_PATH, DATA_DIR
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    # Generate data
    run_command(
        [sys.executable, "main.py", "--generate", "--generate-days", str(SIMULATION_DAYS), "--quiet"],
        f"Generating {SIMULATION_DAYS} days of data..."
    )
    
    # Ingest into database silently
    subprocess.run(
        [sys.executable, "main.py", "--setup-db", "--quiet"],
        capture_output=True, text=True
    )
    
    # Count generated records
    erp_count = 0
    gw_count = 0
    bank_count = 0
    
    erp_file = DATA_DIR / "erp_ledger.json"
    gw_file = DATA_DIR / "gateway_settlements.json"
    bank_file = DATA_DIR / "bank_statement.csv"
    
    if erp_file.exists():
        with open(erp_file, 'r') as f:
            erp_count = len(json.load(f))
    if gw_file.exists():
        with open(gw_file, 'r') as f:
            gw_count = len(json.load(f))
    if bank_file.exists():
        with open(bank_file, 'r') as f:
            bank_count = sum(1 for _ in f) - 1
    
    print(f"\n{GREEN}✅ Generated {SIMULATION_DAYS} days: {erp_count} ERP, {gw_count} Gateway, {bank_count} Bank records{NC}")
    
    wait_for_enter()
    
    # =========================================================================
    # STAGE 2: DETERMINISTIC MATCHING
    # =========================================================================
    print_banner("🎯 STAGE 2: DETERMINISTIC MATCHING", BLUE)
    
    print(f"{CYAN}Exact identifier matching:{NC}")
    print(f"  • Invoice Number")
    print(f"  • UTR")
    print(f"  • Settlement ID")
    print(f"  • Connected-component sum balancing")
    print(f"  • Subset sum for batch settlements")
    
    run_command(
        [sys.executable, "main.py", "--match", "--deterministic-only", "--quiet"],
        "Running deterministic matching..."
    )
    
    print(f"\n{GREEN}✅ Deterministic matching complete{NC}")
    show_reconciliation_progress()
    
    wait_for_enter()
    
    # =========================================================================
    # STAGE 3: AI INFERENCE
    # =========================================================================
    print_banner("🤖 STAGE 3: AI INFERENCE", MAGENTA)
    
    print(f"{CYAN}XGBoost models for remaining unmatched records:{NC}")
    print(f"  • Gateway↔Bank matching")
    print(f"  • ERP↔Gateway matching")
    print(f"  • Candidate cluster generation")
    print(f"  • 11 features extracted per candidate")
    print(f"  • Batch prediction")
    
    run_command(
        [sys.executable, "main.py", "--infer", "--quiet"],
        "Running AI inference..."
    )
    
    print(f"\n{GREEN}✅ AI inference complete{NC}")
    show_reconciliation_progress()
    
    wait_for_enter()
    
    # =========================================================================
    # STAGE 4: EVALUATION
    # =========================================================================
    print_banner("📊 STAGE 4: EVALUATION", BLUE)
    
    run_command(
        [sys.executable, "main.py", "--evaluate"],
        "Evaluating accuracy against ground truth..."
    )
    
    wait_for_enter()
    
    # =========================================================================
    # STAGE 5: UNRECONCILED RECORDS
    # =========================================================================
    print_banner("⚠️  STAGE 5: UNRECONCILED RECORDS", RED)
    
    run_command(
        [sys.executable, "main.py", "--unmatched"],
        "Showing unreconciled records..."
    )
    
    wait_for_enter()
    
    # =========================================================================
    # STAGE 6: BENCHMARK
    # =========================================================================
    print_banner("⚡ STAGE 6: PERFORMANCE BENCHMARK", BLUE)
    
    print(f"{CYAN}Measuring time per stage with {SIMULATION_DAYS} days, seed {BENCHMARK_SEED}...{NC}")
    
    run_command(
        [
            sys.executable, "main.py", "--benchmark",
            "--benchmark-days", str(SIMULATION_DAYS),
            "--benchmark-seed", str(BENCHMARK_SEED),
        ],
        "Running benchmark..."
    )
    
    wait_for_enter()
    
    # =========================================================================
    # LAUNCH DASHBOARD
    # =========================================================================
    print_banner("🚀 LAUNCHING DASHBOARD", BOLD)
    
    print(f"{CYAN}Starting Streamlit dashboard...{NC}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + (os.pathsep + env.get("PYTHONPATH", ""))
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/ui/app.py"],
        env=env,
        cwd=str(Path(__file__).parent)
    )


if __name__ == "__main__":
    main()