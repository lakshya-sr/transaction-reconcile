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
    from src.ai import erp_gw_inference as erp_gw_ai_inference
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
    parser.add_argument("--build-erp-gw-dataset", action="store_true", help="AI Training: Generate ERP↔GW XGBoost training dataset from simulation")
    parser.add_argument("--train-erp-gw", action="store_true", help="AI Training: Train the ERP↔GW XGBoost classifier and export artifacts")
    parser.add_argument("--build-gw-bank-dataset", action="store_true", help="AI Training: Generate GW↔Bank XGBoost training dataset from simulation")
    parser.add_argument("--train-gw-bank", action="store_true", help="AI Training: Train the GW↔Bank XGBoost classifier and export artifacts")

    # Diagnostics & Reporting
    parser.add_argument("--evaluate", "-e", action="store_true", help="Phase 5: Run strict ID graph accuracy evaluator")
    parser.add_argument("--visualize", "-v", action="store_true", help="Diagnostic: Generate static HTML graph for reconciled predictions (erp_gw_pred, gw_bank_pred)")
    parser.add_argument("--visualize-all", "-va", action="store_true", help="Diagnostic: Generate static HTML graph for GROUND TRUTH (erp_gw_true, gw_bank_true)")
    parser.add_argument("--unmatched", "-u", action="store_true", help="Diagnostic: Show isolated/unmatched records")
    parser.add_argument("--dashboard", "--ui", action="store_true", help="Launch the interactive Streamlit Reconciliation Dashboard")

    # Benchmarking & Profiling
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run performance profiling and benchmarking suite")
    parser.add_argument("--benchmark-days", type=int, default=4, help="Number of days to simulate for benchmarking (default: 4)")
    parser.add_argument("--benchmark-seed", type=int, default=42, help="Random seed for benchmark reproducibility (default: 42)")
    parser.add_argument("--benchmark-compare", action="store_true", help="Compare current benchmark with previous baseline if available")
    parser.add_argument("--profile-inference", "-pi", action="store_true", help="Profile GW↔Bank inference pipeline to identify bottlenecks")

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

    # =========================================================================
    # INFERENCE PROFILING MODE
    # =========================================================================
    if args.profile_inference:
        print_banner("GW↔BANK INFERENCE PROFILING")
        
        try:
            # Try to import external profiling script first
            profile_script = BASE_DIR / "profile_inference.py"
            
            if profile_script.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("profile_inference", profile_script)
                profile_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(profile_module)
                profile_module.profile_inference()
            else:
                # Use inline profiling
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
        print_banner("PERFORMANCE BENCHMARKING & PROFILING")
        
        try:
            from src.profiling.benchmark import (
                ReconciliationProfiler,
                print_profile_report,
                compare_benchmarks,
                BENCHMARK_DIR,
            )
            import json
            
            # Check if we should compare with previous baseline
            if args.benchmark_compare:
                baseline_file = BENCHMARK_DIR / "benchmark_baseline.json"
                if baseline_file.exists():
                    with open(baseline_file, "r", encoding="utf-8") as f:
                        baseline = json.load(f)
                    print("[*] Running benchmark for comparison with baseline...")
                    profiler = ReconciliationProfiler(
                        days=args.benchmark_days, 
                        seed=args.benchmark_seed
                    )
                    current = profiler.run_profile()
                    print_profile_report(current, title="CURRENT BENCHMARK RESULTS")
                    print("\n[*] Comparing with baseline...")
                    compare_benchmarks(baseline, current)
                else:
                    print("[!] No baseline benchmark found. Running first benchmark...")
                    profiler = ReconciliationProfiler(
                        days=args.benchmark_days, 
                        seed=args.benchmark_seed
                    )
                    profile = profiler.run_profile()
                    print_profile_report(profile)
                    
                    # Save as baseline for future comparisons
                    with open(BENCHMARK_DIR / "benchmark_baseline.json", "w", encoding="utf-8") as f:
                        json.dump(profile, f, indent=2)
                    print(f"[✔] Baseline saved to {BENCHMARK_DIR / 'benchmark_baseline.json'}")
            else:
                # Run single benchmark
                profiler = ReconciliationProfiler(
                    days=args.benchmark_days, 
                    seed=args.benchmark_seed
                )
                profile = profiler.run_profile()
                print_profile_report(profile)
                
                # Save latest benchmark
                with open(BENCHMARK_DIR / "benchmark_latest.json", "w", encoding="utf-8") as f:
                    json.dump(profile, f, indent=2)
                print(f"[✔] Latest benchmark saved to {BENCHMARK_DIR / 'benchmark_latest.json'}")
            
        except ImportError as e:
            print(f"[!] Benchmark module not found: {e}")
            print("Make sure src/profiling/benchmark.py exists and dependencies are installed.")
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
        print_banner("PHASE 4a: RESIDUAL XGBOOST INFERENCE (GW↔Bank)")
        with suppress_stdout(suppress_internal):
            ai_inference.main()
        if not is_quiet and not is_verbose:
            print("[✔] Phase 4a Completed: GW↔Bank residual AI cluster matching finished.")

        print_banner("PHASE 4b: RESIDUAL XGBOOST INFERENCE (ERP↔Gateway)")
        with suppress_stdout(suppress_internal):
            erp_gw_ai_inference.main()
        if not is_quiet and not is_verbose:
            print("[✔] Phase 4b Completed: ERP↔GW residual AI cluster matching finished.")

    if args.all or args.evaluate:
        print_banner("PHASE 5: EVALUATION METRICS")
        evaluate.main()

    if args.visualize:
        print_banner("DIAGNOSTIC: GRAPH VISUALIZATION (RECONCILED)")
        with suppress_stdout(suppress_internal):
            reconciled_records.main()
        if not is_quiet and not is_verbose:
            from src.core.config import RECONCILIATION_GRAPH_PATH
            from src.reporting.visualizer import open_html_in_browser
            open_html_in_browser(RECONCILIATION_GRAPH_PATH)

    if args.visualize_all:
        print_banner("DIAGNOSTIC: GRAPH VISUALIZATION (ALL DATA)")
        with suppress_stdout(suppress_internal):
            all_records_visualizer.main()
        if not is_quiet and not is_verbose:
            from src.core.config import ALL_DATA_GRAPH_PATH
            from src.reporting.visualizer import open_html_in_browser
            open_html_in_browser(ALL_DATA_GRAPH_PATH)

    if args.unmatched:
        print_banner("DIAGNOSTIC: UNMATCHED RECORDS")
        show_unreconciled_records.main()


    if args.build_erp_gw_dataset:
        print_banner("ERP↔GW AI: GENERATING TRAINING DATASET")
        from src.ai import erp_gw_dataset_builder
        erp_gw_dataset_builder.main()

    if args.train_erp_gw:
        print_banner("ERP↔GW AI: TRAINING XGBOOST MODEL")
        from src.ai import erp_gw_train_model
        erp_gw_train_model.main()

    if args.build_gw_bank_dataset:
        print_banner("GW↔Bank AI: GENERATING TRAINING DATASET")
        from src.ai import dataset_builder
        dataset_builder.main()

    if args.train_gw_bank:
        print_banner("GW↔Bank AI: TRAINING XGBOOST MODEL")
        from src.ai import train_model
        train_model.main()

    if args.dashboard:
        print_banner("LAUNCHING STREAMLIT RECONCILIATION DASHBOARD")
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
    """Fallback inline profiling for GW↔Bank inference pipeline."""
    import json
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    from src.ai.features import FEATURE_COLUMNS, extract_cluster_features
    from src.core.config import DB_PATH, TABLE_BANK, TABLE_GATEWAY, TABLE_GW_BANK_PRED
    from src.core.database import get_connection
    
    # Try to import the correct block generator
    BlockGenerator = None
    try:
        from src.ai.block_generator import FastCandidateBlockGenerator
        BlockGenerator = FastCandidateBlockGenerator
        print("[*] Using FastCandidateBlockGenerator")
    except ImportError:
        pass
    
    if BlockGenerator is None:
        try:
            from src.ai.block_generator import CandidateBlockGenerator
            BlockGenerator = CandidateBlockGenerator
            print("[*] Using CandidateBlockGenerator")
        except ImportError:
            pass
    
    if BlockGenerator is None:
        # Check what's actually in the module
        import src.ai.block_generator as bg
        print(f"[!] Available classes in block_generator: {[x for x in dir(bg) if 'Generator' in x]}")
        print("[!] No block generator found!")
        return
    
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
            df_gw = pd.read_sql_query(
                f"SELECT payment_id, net_settled, settled_at, bank_utr, invoices FROM {TABLE_GATEWAY}", 
                conn
            )
            df_bank = pd.read_sql_query(
                f"SELECT bank_entry_id, credit_amount, value_date, remittance_info FROM {TABLE_BANK}", 
                conn
            )
            df_pred = pd.read_sql_query(
                f"SELECT gateway_payment_id, bank_entry_id FROM {TABLE_GW_BANK_PRED}", 
                conn
            )
        finally:
            conn.close()
        counts["Total Gateways"] = len(df_gw)
        counts["Total Banks"] = len(df_bank)
        counts["Existing Predictions"] = len(df_pred)
        return df_gw, df_bank, df_pred
    
    @timer("2. Model Loading")
    def load_model():
        model_path = BASE_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank.json"
        threshold_path = BASE_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank_threshold.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = XGBClassifier()
        model.load_model(str(model_path))
        model.set_params(n_jobs=-1)
        
        threshold = 0.95  # default
        if threshold_path.exists():
            with open(threshold_path, "r", encoding="utf-8") as f:
                threshold_data = json.load(f)
            threshold = float(threshold_data.get("threshold", 0.95))
        
        return model, threshold
    
    @timer("3. Orphan Filtering")
    def filter_orphans(df_gw, df_bank, df_pred):
        claimed_gw_ids = set(df_pred["gateway_payment_id"].astype(str).dropna()) if not df_pred.empty else set()
        claimed_bank_ids = set(df_pred["bank_entry_id"].astype(str).dropna()) if not df_pred.empty else set()
        
        orphan_gws = [
            row.to_dict() for _, row in df_gw.iterrows()
            if str(row["payment_id"]) not in claimed_gw_ids
        ]
        orphan_banks = [
            row.to_dict() for _, row in df_bank.iterrows()
            if str(row["bank_entry_id"]) not in claimed_bank_ids
        ]
        counts["Orphan Gateways"] = len(orphan_gws)
        counts["Orphan Banks"] = len(orphan_banks)
        return orphan_gws, orphan_banks
    
    @timer("4. Block Generation (Clustering)")
    def generate_blocks(orphan_gws, orphan_banks):
        try:
            block_gen = BlockGenerator(
                max_delay_days=5, 
                max_amount_diff_pct=0.20
            )
        except TypeError:
            # Fallback for different constructor signature
            block_gen = BlockGenerator()
        
        blocks = block_gen.generate_blocks(orphan_gws, orphan_banks)
        counts["Candidate Blocks"] = len(blocks)
        return blocks
    
    @timer("5. Feature Extraction")
    def extract_features(candidate_blocks):
        features = []
        for block in candidate_blocks:
            feats = extract_cluster_features(block["gw_rows"], block["bank_row"])
            features.append(feats)
        df = pd.DataFrame(features)[FEATURE_COLUMNS]
        counts["Feature Rows"] = len(df)
        return df
    
    @timer("6. XGBoost Prediction")
    def predict(model, df_features):
        feature_array = df_features[FEATURE_COLUMNS].values.astype(np.float32)
        return model.predict_proba(feature_array)[:, 1]
    
    @timer("7. Filtering & Assignment")
    def filter_and_assign(candidate_blocks, probs, threshold, df_features):
        amount_diff_pct = df_features["amount_diff_pct"].values
        valid_mask = (probs >= threshold) & (amount_diff_pct <= 0.10)
        valid_indices = np.where(valid_mask)[0]
        sorted_indices = valid_indices[np.argsort(-probs[valid_indices])]
        
        assigned_edges = []
        assigned_gw_ids = set()
        assigned_bank_ids = set()
        
        for idx in sorted_indices:
            block = candidate_blocks[idx]
            bank_id = block["bank_id"]
            gw_ids = block["gw_ids"]
            
            if bank_id in assigned_bank_ids:
                continue
            
            conflict = False
            for g_id in gw_ids:
                if g_id in assigned_gw_ids:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            assigned_bank_ids.add(bank_id)
            assigned_gw_ids.update(gw_ids)
            
            for gw_rec in block["gw_rows"]:
                assigned_edges.append({
                    "gateway_payment_id": gw_rec["_pid"],
                    "bank_entry_id": bank_id,
                    "allocated_amount": float(gw_rec["_net"]),
                })
        
        counts["Assigned Edges"] = len(assigned_edges)
        return assigned_edges
    
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
        print("[!] No orphan records to profile. Run inference first with: python main.py --infer")
        return
    
    print("[*] Generating candidate blocks...")
    candidate_blocks = generate_blocks(orphan_gws, orphan_banks)
    print(f"  • Candidate Blocks: {len(candidate_blocks)}")
    
    if not candidate_blocks:
        print("[!] No candidate blocks generated.")
        return
    
    print("[*] Extracting features...")
    df_features = extract_features(candidate_blocks)
    print(f"  • Feature Rows: {len(df_features)}")
    
    print("[*] Running XGBoost prediction...")
    probs = predict(model, df_features)
    
    print("[*] Filtering and assigning...")
    assigned_edges = filter_and_assign(candidate_blocks, probs, threshold, df_features)
    print(f"  • Assigned Edges: {len(assigned_edges)}")
    
    # Print results
    total = sum(timings.values())
    
    print("\n" + "=" * 70)
    print("  GW↔BANK INFERENCE PIPELINE - PERFORMANCE PROFILE")
    print("=" * 70)
    
    print(f"\n📊 COUNTS:")
    for name, count in counts.items():
        print(f"  • {name}: {count:,}")
    
    print(f"\n⏱️  TIMINGS:")
    for name, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = (elapsed / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  • {name:<35} {elapsed:>8.3f}s  {pct:>5.1f}%  {bar}")
    
    print(f"\n  {'TOTAL':<35} {total:>8.3f}s  100.0%")
    print("=" * 70)
    
    # Additional analysis
    print(f"\n🔍 ANALYSIS:")
    if len(candidate_blocks) > 0:
        print(f"  • Blocks per orphan bank: {len(candidate_blocks) / max(1, len(orphan_banks)):.1f}")
        print(f"  • XGBoost prediction per block: {(timings.get('6. XGBoost Prediction', 0) / len(candidate_blocks) * 1000):.3f} ms")
        print(f"  • Feature extraction per block: {(timings.get('5. Feature Extraction', 0) / len(candidate_blocks) * 1000):.3f} ms")
    
    if len(orphan_banks) > 0:
        print(f"  • Block generation per bank: {(timings.get('4. Block Generation (Clustering)', 0) / len(orphan_banks) * 1000):.3f} ms")
    
    # Identify bottleneck
    clustering_time = timings.get('4. Block Generation (Clustering)', 0)
    xgboost_time = timings.get('6. XGBoost Prediction', 0)
    feature_time = timings.get('5. Feature Extraction', 0)
    
    max_time = max(clustering_time, xgboost_time, feature_time)
    
    if max_time > 0:
        print(f"\n🎯 BOTTLENECK ANALYSIS:")
        if clustering_time == max_time:
            print(f"  → Block Generation (Clustering) is the slowest: {clustering_time:.3f}s")
            print(f"    Suggestion: Optimize candidate generation with pre-indexing")
        elif xgboost_time == max_time:
            print(f"  → XGBoost Prediction is the slowest: {xgboost_time:.3f}s")
            print(f"    Suggestion: Use DMatrix or reduce feature dimensions")
        else:
            print(f"  → Feature Extraction is the slowest: {feature_time:.3f}s")
            print(f"    Suggestion: Cache computations or parallelize extraction")


if __name__ == "__main__":
    main()