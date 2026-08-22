#!/usr/bin/env python3
"""
End-to-End Orchestrator: Multi-Source Reconciliation Agent Pipeline.

Executes all 3 phases sequentially:
1. Synthetic Data Generation (generate_data.py)
2. Database Setup & Ingestion (db_setup.py)
3. Exact Matching Engine (exact_matcher.py)

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
    print("=" * 80)
    print("  RAZORPAY AI BUILDATHON: MULTI-SOURCE RECONCILIATION AGENT PIPELINE")
    print("=" * 80)

    print("\n>>> STEP 1 / 3: Executing Synthetic Data Generation...")
    generate_data.main()

    print("\n>>> STEP 2 / 3: Executing Database Setup & Ingestion...")
    db_setup.main()

    print("\n>>> STEP 3 / 3: Executing Exact Matching Engine...")
    exact_matcher.main()

    elapsed = time.time() - start_time
    print(f"\n[★] Entire Pipeline Completed Successfully in {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
