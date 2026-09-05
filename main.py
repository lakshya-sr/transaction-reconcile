#!/usr/bin/env python3
"""
Unified CLI for the Multi-Source Reconciliation Agent.

Usage:
    python main.py --help
    python main.py --all
    python main.py -g -d -m
    python main.py --benchmark
    python main.py --profile-inference
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
    parser.add_argument("--generate", "-g", action="store_true", help="Generate synthetic noisy data (JSON/CSV)")
    parser.add_argument("--generate-days", type=int, default=5, help="Number of days to simulate (default: 5)")
    parser.add_argument("--setup-db", "-d", action="store_true", help="Initialize SQLite DB and ingest raw data")
    parser.add_argument("--match", "-m", action="store_true", help="Run the deterministic matching engine")
    parser.add_argument("--infer", "-i", action="store_true", help="Run AI inference on unmatched records")
    
    # AI Training
    parser.add_argument("--build-gw-bank-dataset", action="store_true", help="Generate GW↔Bank training dataset")
    parser.add_argument("--train-gw-bank", action="store_true", help="Train GW↔Bank XGBoost classifier")
    parser.add_argument("--build-erp-gw-dataset", action="store_true", help="Generate ERP↔GW training dataset")
    parser.add_argument("--train-erp-gw", action="store_true", help="Train ERP↔GW XGBoost classifier")
    parser.add_argument("--train-all", action="store_true", help="Train both models (dataset + training)")

    # Diagnostics & Reporting
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run accuracy evaluator")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate graph for reconciled predictions")
    parser.add_argument("--visualize-all", "-va", action="store_true", help="Generate graph for ground truth")
    parser.add_argument("--unmatched", "-u", action="store_true", help="Show isolated/unmatched records")
    parser.add_argument("--dashboard", "--ui", action="store_true", help="Launch Streamlit dashboard")

    # Benchmarking & Profiling
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run performance benchmarking suite")
    parser.add_argument("--benchmark-days", type=int, default=4, help="Days to simulate for benchmark (default: 4)")
    parser.add_argument("--benchmark-seed", type=int, default=42, help="Random seed for benchmark (default: 42)")
    parser.add_argument("--benchmark-compare", action="store_true", help="Compare with previous baseline")
    parser.add_argument("--profile-inference", "-pi", action="store_true", help="Profile inference pipeline")

    # Run All & Control Flags
    parser.add_argument("--all", "-a", action="store_true", help="Run full pipeline (Gen → DB → Match → Infer → Eval)")
    parser.add_argument("--deterministic-only", "--no-ai", action="store_true", help="Skip AI inference")
    parser.add_argument("--verbose", "-V", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run quietly")

    args = parser.parse_args()

    # Show help if no arguments are passed
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    from src.core.logging_config import set_verbose, suppress_stdout

    is_verbose = args.verbose
    is_quiet = args.quiet
    set_verbose(is_verbose)
    suppress_internal = not is_verbose

    start_time = time.time()

    def print_banner(title: str):
        if not is_quiet:
            print("\n" + "=" * 50)
            print(f"[*] {title}")
            print("=" * 50)

    # =========================================================================
    # INFERENCE PROFILING MODE
    # =========================================================================
    if args.profile_inference:
        print_banner("INFERENCE PROFILING")
        try:
            profile_script = BASE_DIR / "src" / "profiling" / "profile_inference.py"
            if profile_script.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("profile_inference", profile_script)
                profile_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(profile_module)
                profile_module.profile_inference()
            else:
                _run_inline_inference_profiling()
        except Exception as e:
            print(f"[!] Profiling failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        elapsed = time.time() - start_time
        if not is_quiet:
            print(f"\n[✔] Profiling completed in {elapsed:.2f}s")
        return

    # =========================================================================
    # BENCHMARK MODE
    # =========================================================================
    if args.benchmark:
        print_banner("PERFORMANCE BENCHMARKING")
        try:
            from src.profiling.benchmark import (
                ReconciliationProfiler,
                print_profile_report,
                compare_benchmarks,
                BENCHMARK_DIR,
            )
            import json
            
            if args.benchmark_compare:
                baseline_file = BENCHMARK_DIR / "benchmark_baseline.json"
                if baseline_file.exists():
                    with open(baseline_file, "r", encoding="utf-8") as f:
                        baseline = json.load(f)
                    print("[*] Running benchmark for comparison...")
                    profiler = ReconciliationProfiler(days=args.benchmark_days, seed=args.benchmark_seed)
                    current = profiler.run_profile()
                    print_profile_report(current, title="CURRENT BENCHMARK RESULTS")
                    compare_benchmarks(baseline, current)
                else:
                    print("[!] No baseline found. Running first benchmark...")
                    profiler = ReconciliationProfiler(days=args.benchmark_days, seed=args.benchmark_seed)
                    profile = profiler.run_profile()
                    print_profile_report(profile)
                    with open(BENCHMARK_DIR / "benchmark_baseline.json", "w") as f:
                        json.dump(profile, f, indent=2)
            else:
                profiler = ReconciliationProfiler(days=args.benchmark_days, seed=args.benchmark_seed)
                profile = profiler.run_profile()
                print_profile_report(profile)
                with open(BENCHMARK_DIR / "benchmark_latest.json", "w") as f:
                    json.dump(profile, f, indent=2)
                print(f"[✔] Benchmark saved to {BENCHMARK_DIR / 'benchmark_latest.json'}")
            
        except ImportError as e:
            print(f"[!] Benchmark module not found: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[!] Benchmark failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        elapsed = time.time() - start_time
        if not is_quiet:
            print(f"\n[✔] Benchmark completed in {elapsed:.2f}s")
        return

    # =========================================================================
    # NORMAL PIPELINE EXECUTION
    # =========================================================================
    
    # Phase 1: Generate data
    if args.all or args.generate:
        print_banner(f"PHASE 1: SYNTHETIC DATA GENERATION ({args.generate_days} days)")
        with suppress_stdout(suppress_internal):
            generate_data.main()
        if not is_quiet and not is_verbose:
            print("[✔] Data generation completed.")

    # Phase 2: Setup database
    if args.all or args.setup_db:
        print_banner("PHASE 2: DATABASE SETUP & INGESTION")
        with suppress_stdout(suppress_internal):
            db_setup.main()
        if not is_quiet and not is_verbose:
            print("[✔] Database setup completed.")

    # Phase 3: Matching
    if args.all or args.match:
        banner = "PHASE 3: MATCHING (DETERMINISTIC ONLY)" if args.deterministic_only else "PHASE 3: MATCHING"
        print_banner(banner)
        with suppress_stdout(suppress_internal):
            exact_matcher.main(deterministic_only=args.deterministic_only)
        if not is_quiet and not is_verbose:
            print("[✔] Matching completed.")

    # Phase 4: AI Inference
    if (args.all or args.infer) and not args.deterministic_only:
        print_banner("PHASE 4: AI INFERENCE (GW↔Bank + ERP↔GW)")
        with suppress_stdout(suppress_internal):
            ai_inference.main()
        if not is_quiet and not is_verbose:
            print("[✔] AI inference completed.")

    # Phase 5: Evaluation
    if args.all or args.evaluate:
        print_banner("PHASE 5: EVALUATION")
        evaluate.main()

    # AI Training
    if args.build_gw_bank_dataset:
        print_banner("GENERATING GW↔BANK TRAINING DATASET")
        from src.ai.dataset_builder import generate_gateway_bank_dataset
        generate_gateway_bank_dataset()

    if args.train_gw_bank:
        print_banner("TRAINING GW↔BANK MODEL")
        from src.ai.training import train_gateway_bank_model
        train_gateway_bank_model()

    if args.build_erp_gw_dataset:
        print_banner("GENERATING ERP↔GW TRAINING DATASET")
        from src.ai.dataset_builder import generate_erp_gateway_dataset
        generate_erp_gateway_dataset()

    if args.train_erp_gw:
        print_banner("TRAINING ERP↔GW MODEL")
        from src.ai.training import train_erp_gateway_model
        train_erp_gateway_model()

    if args.train_all:
        print_banner("TRAINING ALL MODELS")
        from src.ai.dataset_builder import generate_gateway_bank_dataset, generate_erp_gateway_dataset
        from src.ai.training import train_gateway_bank_model, train_erp_gateway_model
        
        print("[*] Generating datasets...")
        generate_gateway_bank_dataset()
        generate_erp_gateway_dataset()
        
        print("[*] Training models...")
        train_gateway_bank_model()
        train_erp_gateway_model()

    # Visualization
    if args.visualize:
        print_banner("GRAPH VISUALIZATION (RECONCILED)")
        with suppress_stdout(suppress_internal):
            reconciled_records.main()
        if not is_quiet and not is_verbose:
            from src.core.config import RECONCILIATION_GRAPH_PATH
            from src.reporting.visualizer import open_html_in_browser
            open_html_in_browser(RECONCILIATION_GRAPH_PATH)

    if args.visualize_all:
        print_banner("GRAPH VISUALIZATION (ALL DATA)")
        with suppress_stdout(suppress_internal):
            all_records_visualizer.main()
        if not is_quiet and not is_verbose:
            from src.core.config import ALL_DATA_GRAPH_PATH
            from src.reporting.visualizer import open_html_in_browser
            open_html_in_browser(ALL_DATA_GRAPH_PATH)

    if args.unmatched:
        print_banner("UNMATCHED RECORDS")
        show_unreconciled_records.main()

    if args.dashboard:
        print_banner("LAUNCHING STREAMLIT DASHBOARD")
        import os
        import subprocess

        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        app_path = str(BASE_DIR / "src" / "ui" / "app.py")
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        try:
            subprocess.run(cmd, env=env, cwd=str(BASE_DIR))
        except KeyboardInterrupt:
            print("\n[✔] Streamlit dashboard closed.")

    elapsed = time.time() - start_time
    if not is_quiet:
        print(f"\n[✔] Execution finished in {elapsed:.2f}s")


def _run_inline_inference_profiling():
    """Fallback inline profiling for inference pipeline."""
    import json
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    from src.ai.candidate_generation import GatewayBankCandidateGenerator
    from src.ai.features import GATEWAY_BANK_FEATURES, extract_gateway_bank_features
    from src.core.config import DB_PATH, TABLE_BANK, TABLE_GATEWAY, TABLE_GW_BANK_PRED
    from src.core.database import get_connection
    
    timings = {}
    counts = {}
    
    def timer(name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                timings[name] = timings.get(name, 0.0) + elapsed
                return result
            return wrapper
        return decorator
    
    @timer("1. Data Loading")
    def load_data():
        conn = get_connection(DB_PATH)
        try:
            df_gw = pd.read_sql_query(f"SELECT payment_id, net_settled, settled_at, bank_utr, invoices FROM {TABLE_GATEWAY}", conn)
            df_bank = pd.read_sql_query(f"SELECT bank_entry_id, credit_amount, value_date, remittance_info FROM {TABLE_BANK}", conn)
            df_pred = pd.read_sql_query(f"SELECT gateway_payment_id, bank_entry_id FROM {TABLE_GW_BANK_PRED}", conn)
        finally:
            conn.close()
        counts["Total Gateways"] = len(df_gw)
        counts["Total Banks"] = len(df_bank)
        return df_gw, df_bank, df_pred
    
    @timer("2. Model Loading")
    def load_model():
        model_path = BASE_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank.json"
        threshold_path = BASE_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank_threshold.json"
        model = XGBClassifier()
        model.load_model(str(model_path))
        model.set_params(n_jobs=-1)
        with open(threshold_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.95)
        return model, threshold
    
    @timer("3. Orphan Filtering")
    def filter_orphans(df_gw, df_bank, df_pred):
        claimed_gw = set(df_pred["gateway_payment_id"].astype(str).dropna()) if not df_pred.empty else set()
        claimed_bank = set(df_pred["bank_entry_id"].astype(str).dropna()) if not df_pred.empty else set()
        orphan_gws = [r.to_dict() for _, r in df_gw.iterrows() if str(r["payment_id"]) not in claimed_gw]
        orphan_banks = [r.to_dict() for _, r in df_bank.iterrows() if str(r["bank_entry_id"]) not in claimed_bank]
        counts["Orphan Gateways"] = len(orphan_gws)
        counts["Orphan Banks"] = len(orphan_banks)
        return orphan_gws, orphan_banks
    
    @timer("4. Candidate Generation")
    def generate_candidates(orphan_gws, orphan_banks):
        gen = GatewayBankCandidateGenerator(max_delay_days=5, max_amount_diff_pct=0.20)
        blocks = gen.generate(orphan_gws, orphan_banks)
        counts["Candidate Blocks"] = len(blocks)
        return blocks
    
    @timer("5. Feature Extraction")
    def extract_features(candidate_blocks):
        features = []
        for block in candidate_blocks:
            feats = extract_gateway_bank_features(block["gw_rows"], block["bank_row"])
            features.append(feats)
        df = pd.DataFrame(features)[GATEWAY_BANK_FEATURES]
        counts["Feature Rows"] = len(df)
        return df
    
    @timer("6. XGBoost Prediction")
    def predict(model, df_features):
        feature_array = df_features[GATEWAY_BANK_FEATURES].values.astype(np.float32)
        return model.predict_proba(feature_array)[:, 1]
    
    # Run profiling
    print("[*] Loading data...")
    df_gw, df_bank, df_pred = load_data()
    
    print("[*] Loading model...")
    model, threshold = load_model()
    
    print("[*] Filtering orphans...")
    orphan_gws, orphan_banks = filter_orphans(df_gw, df_bank, df_pred)
    print(f"  • Orphan Gateways: {len(orphan_gws)}")
    print(f"  • Orphan Banks: {len(orphan_banks)}")
    
    if not orphan_gws or not orphan_banks:
        print("[!] No orphan records. Run inference first.")
        return
    
    print("[*] Generating candidates...")
    candidate_blocks = generate_candidates(orphan_gws, orphan_banks)
    print(f"  • Candidate Blocks: {len(candidate_blocks)}")
    
    if not candidate_blocks:
        print("[!] No candidates generated.")
        return
    
    print("[*] Extracting features...")
    df_features = extract_features(candidate_blocks)
    
    print("[*] Running prediction...")
    probs = predict(model, df_features)
    
    # Print results
    total = sum(timings.values())
    print("\n" + "=" * 70)
    print("  INFERENCE PIPELINE - PERFORMANCE PROFILE")
    print("=" * 70)
    
    print(f"\n📊 COUNTS:")
    for name, count in counts.items():
        print(f"  • {name}: {count:,}")
    
    print(f"\n⏱️  TIMINGS:")
    for name, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = (elapsed / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  • {name:<30} {elapsed:>8.3f}s  {pct:>5.1f}%  {bar}")
    
    print(f"\n  {'TOTAL':<30} {total:>8.3f}s  100.0%")
    print("=" * 70)
    
    # Bottleneck analysis
    clustering_time = timings.get('4. Candidate Generation', 0)
    xgboost_time = timings.get('6. XGBoost Prediction', 0)
    feature_time = timings.get('5. Feature Extraction', 0)
    max_time = max(clustering_time, xgboost_time, feature_time)
    
    if max_time > 0:
        print(f"\n🎯 BOTTLENECK:")
        if clustering_time == max_time:
            print(f"  → Candidate Generation: {clustering_time:.3f}s")
        elif xgboost_time == max_time:
            print(f"  → XGBoost Prediction: {xgboost_time:.3f}s")
        else:
            print(f"  → Feature Extraction: {feature_time:.3f}s")


if __name__ == "__main__":
    main()
