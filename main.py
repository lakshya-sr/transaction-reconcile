#!/usr/bin/env python3
"""
Unified CLI for the Multi-Source Reconciliation Agent.

Usage:
    python main.py --help
    python main.py --all
    python main.py -g -d -m
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in path just in case
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import modules based on the new modular architecture
try:
    from src.simulation import generate_data
    from src.core import db_setup
    from src.deterministic import exact_matcher
    from src.ai import inference as ai_inference
    from src.reporting import evaluate
    from src.reporting import reconciled_records
    from src.reporting import all_records_visualizer
    from src.reporting import show_unreconciled_records
except ImportError as e:
    print(f"[!] Import Error: {e}")
    print("Ensure you have moved all scripts into their respective src/ folders.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Source Reconciliation Agent CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Core Pipeline Phases
    parser.add_argument("--generate", "-g", action="store_true", help="Phase 1: Generate synthetic noisy data (JSON/CSV)")
    parser.add_argument("--setup-db", "-d", action="store_true", help="Phase 2: Initialize SQLite DB and ingest raw data")
    parser.add_argument("--match", "-m", action="store_true", help="Phase 3: Run the deterministic + fuzzy + cluster-XGBoost matching engine")
    parser.add_argument("--infer", "-i", action="store_true", help="Phase 4: Run the residual XGBoost inference pass on unmatched Gateway↔Bank pairs after the fuzzy residual layer")

    # Diagnostics & Reporting
    parser.add_argument("--evaluate", "-e", action="store_true", help="Phase 5: Run strict ID graph accuracy evaluator")
    parser.add_argument("--visualize", "-v", action="store_true", help="Diagnostic: Generate static HTML graph for reconciled predictions (erp_gw_pred, gw_bank_pred)")
    parser.add_argument("--visualize-all", "-va", action="store_true", help="Diagnostic: Generate static HTML graph for GROUND TRUTH (erp_gw_true, gw_bank_true)")
    parser.add_argument("--unmatched", "-u", action="store_true", help="Diagnostic: Show isolated/unmatched records")

    # Run All & Control Flags
    parser.add_argument("--all", "-a", action="store_true", help="Run the full pipeline (Gen -> DB -> Match -> Infer -> Eval)")
    parser.add_argument("--deterministic-only", "--no-ai", action="store_true", help="Skip the AI inference pass and run only the deterministic reconciliation engine")
    parser.add_argument("--verbose", "-V", action="store_true", help="Enable verbose step-by-step logging and intermediate outputs")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run quietly, suppressing banners and intermediate logs")

    args = parser.parse_args()

    # Show help if no arguments are passed
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    from src.core.logging_config import set_verbose, suppress_stdout

    is_verbose = args.verbose
    is_quiet = args.quiet
    set_verbose(is_verbose)

    # Suppress intermediate module stdout unless verbose is explicitly enabled
    suppress_internal = not is_verbose

    start_time = time.time()

    def print_banner(title: str):
        if not is_quiet:
            print("\n" + "=" * 50)
            print(f"[*] {title}")
            print("=" * 50)

    # Execute selected arguments sequentially
    if args.all or args.generate:
        print_banner("PHASE 1: SYNTHETIC DATA GENERATION")
        with suppress_stdout(suppress_internal):
            generate_data.main()
        if not is_quiet and not is_verbose:
            print("[✔] Phase 1 Completed: Generated synthetic datasets.")

    if args.all or args.setup_db:
        print_banner("PHASE 2: DATABASE SETUP & INGESTION")
        with suppress_stdout(suppress_internal):
            db_setup.main()
        if not is_quiet and not is_verbose:
            print("[✔] Phase 2 Completed: Ingested records into SQLite.")

    if args.all or args.match:
        banner = "PHASE 3: MATCHING ENGINE (DETERMINISTIC ONLY)" if args.deterministic_only else "PHASE 3: MATCHING ENGINE (DETERMINISTIC)"
        print_banner(banner)
        with suppress_stdout(suppress_internal):
            exact_matcher.main(deterministic_only=args.deterministic_only)
        if not is_quiet and not is_verbose:
            print("[✔] Phase 3 Completed: Deterministic matching finished.")

    if (args.all or args.infer) and not args.deterministic_only:
        print_banner("PHASE 4: RESIDUAL XGBOOST INFERENCE")
        with suppress_stdout(suppress_internal):
            ai_inference.main()
        if not is_quiet and not is_verbose:
            print("[✔] Phase 4 Completed: Residual AI cluster matching finished.")

    if args.all or args.evaluate:
        print_banner("PHASE 5: EVALUATION METRICS")
        evaluate.main()

    if args.visualize:
        print_banner("DIAGNOSTIC: GRAPH VISUALIZATION (RECONCILED)")
        with suppress_stdout(suppress_internal):
            reconciled_records.main()
        if not is_quiet and not is_verbose:
            print("[✔] Reconciled predictions graph generated.")

    if args.visualize_all:
        print_banner("DIAGNOSTIC: GRAPH VISUALIZATION (ALL DATA)")
        with suppress_stdout(suppress_internal):
            all_records_visualizer.main()
        if not is_quiet and not is_verbose:
            print("[✔] Ground truth all-data graph generated.")

    if args.unmatched:
        print_banner("DIAGNOSTIC: UNMATCHED RECORDS")
        show_unreconciled_records.main()

    elapsed = time.time() - start_time
    if not is_quiet:
        print(f"\n[✔] Execution finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()