#!/usr/bin/env python3
"""
End-to-End Orchestrator: Multi-Source Reconciliation Agent Pipeline.

Executes all phases sequentially:
1. Synthetic Data Generation with Noise (generate_data.py)
2. Database Setup & Ingestion (db_setup.py)
3. Exact & Fuzzy Matching Engine (exact_matcher.py)

Usage:
    python run_pipeline.py
    or
    uv run run_pipeline.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import generate_data
import db_setup
import exact_matcher


def run_pipeline():
    start_time = time.time()
    print("[*] Running Pipeline: Data Gen -> DB Setup -> Matching...")

    generate_data.main()
    db_setup.main()
    exact_matcher.main()

    elapsed = time.time() - start_time
    print(f"[✔] Pipeline completed in {elapsed:.2f}s")


if __name__ == "__main__":
    run_pipeline()