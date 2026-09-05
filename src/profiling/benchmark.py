#!/usr/bin/env python3
"""
Performance & Throughput Profiling Framework for Multi-Source Reconciliation.
"""

import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tabulate import tabulate
import argparse
import random

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import DB_PATH
from src.core.database import get_connection, init_database
from src.simulation.generate_data import run_continuous_simulation, save_datasets
from src.core.db_setup import main as run_db_setup
from src.deterministic.matcher import ReconciliationEngine
from src.ai.inference import run_residual_ai_inference
from src.reporting.evaluate import calculate_metrics

BENCHMARK_DIR = ROOT_DIR / "data" / "benchmarks"


class StageTimer:
    """Context manager for timing and memory tracking."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0
        self.elapsed_ms = 0.0
        self.peak_mem_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mem_mb = peak / (1024.0 * 1024.0)


class ReconciliationProfiler:
    """Profiles each stage of the reconciliation pipeline."""
    
    def __init__(self, days: int = 4, seed: int = 42):
        self.days = days
        self.seed = seed
        self.stage_timings: Dict[str, float] = {}
        self.stage_memory: Dict[str, float] = {}
        self.record_counts: Dict[str, int] = {}
        self.metrics: Dict[str, float] = {}

    def run_profile(self) -> Dict:
        """Run complete profiling suite."""
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Data Generation
        with StageTimer("1. Data Generation") as t:
            df_erp, df_gw, df_bank, df_eg_true, df_gb_true = run_continuous_simulation(
                days=self.days, seed=self.seed
            )
            save_datasets(df_erp, df_gw, df_bank, df_eg_true, df_gb_true)
        self.stage_timings["1_data_gen_ms"] = t.elapsed_ms
        self.stage_memory["1_data_gen_mem_mb"] = t.peak_mem_mb

        total_txns = len(df_erp) + len(df_gw) + len(df_bank)
        self.record_counts = {
            "erp_records": len(df_erp),
            "gateway_records": len(df_gw),
            "bank_records": len(df_bank),
            "total_transactions": total_txns,
            "true_erp_gw_edges": len(df_eg_true),
            "true_gw_bank_edges": len(df_gb_true),
        }

        # 2. Database Setup
        with StageTimer("2. DB Ingestion") as t:
            run_db_setup()
        self.stage_timings["2_db_ingestion_ms"] = t.elapsed_ms
        self.stage_memory["2_db_ingestion_mem_mb"] = t.peak_mem_mb

        # 3. Deterministic Matching
        with StageTimer("3. Deterministic Matching") as t:
            from src.deterministic.matcher import run_exact_matching
            run_exact_matching(DB_PATH)
        self.stage_timings["3_deterministic_match_ms"] = t.elapsed_ms
        self.stage_memory["3_deterministic_match_mem_mb"] = t.peak_mem_mb

        # 4. AI Inference
        with StageTimer("4. AI Inference") as t:
            ai_edges_count = run_residual_ai_inference()
        self.stage_timings["4_ai_inference_ms"] = t.elapsed_ms
        self.stage_memory["4_ai_inference_mem_mb"] = t.peak_mem_mb

        # 5. Accuracy Metrics
        conn = get_connection(DB_PATH)
        try:
            df_eg_t = pd.read_sql_query("SELECT erp_id, gw_id FROM erp_gw_true", conn)
            df_gb_t = pd.read_sql_query("SELECT gw_id, bank_id FROM gw_bank_true", conn)
            df_eg_p = pd.read_sql_query("SELECT erp_order_id, gateway_payment_id FROM erp_gw_pred", conn)
            df_gb_p = pd.read_sql_query("SELECT gateway_payment_id, bank_entry_id FROM gw_bank_pred", conn)
        finally:
            conn.close()

        true_eg = set(zip(df_eg_t["erp_id"], df_eg_t["gw_id"]))
        true_gb = set(zip(df_gb_t["gw_id"], df_gb_t["bank_id"]))
        pred_eg = set(zip(df_eg_p["erp_order_id"], df_eg_p["gateway_payment_id"]))
        pred_gb = set(zip(df_gb_p["gateway_payment_id"], df_gb_p["bank_entry_id"]))

        p1, r1, f1_1 = calculate_metrics(
            len(pred_eg & true_eg), len(pred_eg - true_eg), len(true_eg - pred_eg)
        )
        p2, r2, f1_2 = calculate_metrics(
            len(pred_gb & true_gb), len(pred_gb - true_gb), len(true_gb - pred_gb)
        )

        total_match_time_ms = self.stage_timings["3_deterministic_match_ms"] + self.stage_timings["4_ai_inference_ms"]
        total_e2e_time_ms = sum(self.stage_timings.values())

        throughput_match_tps = (total_txns / (total_match_time_ms / 1000.0)) if total_match_time_ms > 0 else 0.0
        throughput_e2e_tps = (total_txns / (total_e2e_time_ms / 1000.0)) if total_e2e_time_ms > 0 else 0.0

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "days": self.days,
            "seed": self.seed,
            "counts": self.record_counts,
            "timings_ms": self.stage_timings,
            "memory_mb": self.stage_memory,
            "total_match_time_ms": round(total_match_time_ms, 2),
            "total_e2e_time_ms": round(total_e2e_time_ms, 2),
            "throughput_matching_txns_per_sec": round(throughput_match_tps, 1),
            "throughput_e2e_txns_per_sec": round(throughput_e2e_tps, 1),
            "layer1_precision": round(p1 * 100, 2),
            "layer1_recall": round(r1 * 100, 2),
            "layer1_f1": round(f1_1 * 100, 2),
            "layer2_precision": round(p2 * 100, 2),
            "layer2_recall": round(r2 * 100, 2),
            "layer2_f1": round(f1_2 * 100, 2),
            "ai_edges_count": ai_edges_count,
        }


def print_profile_report(profile: Dict, title: str = "RECONCILIATION PROFILER REPORT"):
    """Print formatted profiling report."""
    timings = profile["timings_ms"]
    total_ms = profile["total_e2e_time_ms"]
    counts = profile["counts"]

    table_data = [
        ["1. Data Generation", f"{timings['1_data_gen_ms']:.2f} ms", f"{(timings['1_data_gen_ms']/total_ms)*100:.1f}%", f"{profile['memory_mb']['1_data_gen_mem_mb']:.2f} MB"],
        ["2. DB Ingestion", f"{timings['2_db_ingestion_ms']:.2f} ms", f"{(timings['2_db_ingestion_ms']/total_ms)*100:.1f}%", f"{profile['memory_mb']['2_db_ingestion_mem_mb']:.2f} MB"],
        ["3. Deterministic Matching", f"{timings['3_deterministic_match_ms']:.2f} ms", f"{(timings['3_deterministic_match_ms']/total_ms)*100:.1f}%", f"{profile['memory_mb']['3_deterministic_match_mem_mb']:.2f} MB"],
        ["4. AI Inference", f"{timings['4_ai_inference_ms']:.2f} ms", f"{(timings['4_ai_inference_ms']/total_ms)*100:.1f}%", f"{profile['memory_mb']['4_ai_inference_mem_mb']:.2f} MB"],
    ]

    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(f"[*] Workload: {counts['total_transactions']} Total Transactions "
          f"({counts['erp_records']} ERP, {counts['gateway_records']} GW, {counts['bank_records']} Bank)")
    print(f"[*] Simulation Days: {profile['days']}, Seed: {profile['seed']}")
    print(tabulate(table_data, headers=["Stage", "Latency (ms)", "Share (%)", "Peak Memory"], tablefmt="fancy_grid"))

    print(f"\n[⚡] Matching Throughput: {profile['throughput_matching_txns_per_sec']:,.1f} txns/sec")
    print("=" * 90 + "\n")


def compare_benchmarks(before: Dict, after: Dict):
    """Compare two benchmark results."""
    total_speedup = before["total_e2e_time_ms"] / after["total_e2e_time_ms"] if after["total_e2e_time_ms"] > 0 else 1.0
    match_speedup = before["total_match_time_ms"] / after["total_match_time_ms"] if after["total_match_time_ms"] > 0 else 1.0

    comp_table = [
        ["Total Matching Time", f"{before['total_match_time_ms']:.2f} ms", f"{after['total_match_time_ms']:.2f} ms", f"{match_speedup:.2f}x Faster"],
        ["End-to-End Time", f"{before['total_e2e_time_ms']:.2f} ms", f"{after['total_e2e_time_ms']:.2f} ms", f"{total_speedup:.2f}x Faster"],
        ["Matching Throughput", f"{before['throughput_matching_txns_per_sec']:,.1f} tps", f"{after['throughput_matching_txns_per_sec']:,.1f} tps", f"+{((after['throughput_matching_txns_per_sec'] - before['throughput_matching_txns_per_sec'])/before['throughput_matching_txns_per_sec'])*100:.1f}%"],
    ]

    print("\n" + "=" * 90)
    print("  COMPARATIVE PERFORMANCE GAIN ANALYSIS")
    print("=" * 90)
    print(tabulate(comp_table, headers=["Metric", "Baseline", "Optimized", "Improvement"], tablefmt="fancy_grid"))
    print("=" * 90 + "\n")


def main():
    """Run benchmark and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    print(f"Running benchmark:")
    print(f"  • Days: {args.days}")
    print(f"  • Seed: {seed}")

    profiler = ReconciliationProfiler(days=args.days, seed=seed)
    profile = profiler.run_profile()
    print_profile_report(profile)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_DIR / "benchmark_latest.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


if __name__ == "__main__":
    main()