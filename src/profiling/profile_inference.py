#!/usr/bin/env python3
"""
Profile the GW↔Bank inference pipeline to identify bottlenecks.

Usage:
    python profile_inference.py
"""

import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.block_generator import CandidateBlockGenerator
from src.ai.features import FEATURE_COLUMNS, extract_cluster_features
from src.core.config import (
    DB_PATH,
    TABLE_BANK,
    TABLE_GATEWAY,
    TABLE_GW_BANK_PRED,
)
from src.core.database import get_connection

ARTIFACT_DIR = ROOT_DIR / "src" / "ai" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_gw_bank.json"
THRESHOLD_PATH = ARTIFACT_DIR / "xgb_gw_bank_threshold.json"


class InferenceProfiler:
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    def start_timer(self, name: str):
        self._current_timer = name
        self._start_time = time.perf_counter()
    
    def stop_timer(self, name: str = None):
        if name is None:
            name = self._current_timer
        elapsed = time.perf_counter() - self._start_time
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        return elapsed
    
    def add_count(self, name: str, count: int):
        self.counts[name] = self.counts.get(name, 0) + count
    
    def print_summary(self):
        total_time = sum(self.timings.values())
        
        print("\n" + "=" * 70)
        print("  GW↔BANK INFERENCE PIPELINE - PERFORMANCE PROFILE")
        print("=" * 70)
        
        print(f"\n📊 COUNTS:")
        for name, count in self.counts.items():
            print(f"  • {name}: {count:,}")
        
        print(f"\n⏱️  TIMINGS:")
        for name, elapsed in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  • {name:<30} {elapsed:>8.3f}s  {pct:>5.1f}%  {bar}")
        
        print(f"\n  {'TOTAL':<30} {total_time:>8.3f}s  100.0%")
        print("=" * 70)


def profile_inference(db_path: Path = DB_PATH):
    """Profile each stage of the inference pipeline."""
    profiler = InferenceProfiler()
    
    print("[*] Starting inference profiling...")
    
    # 1. Load data
    profiler.start_timer("1. Data Loading")
    conn = get_connection(db_path)
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
    profiler.stop_timer("1. Data Loading")
    profiler.add_count("Total Gateways", len(df_gw))
    profiler.add_count("Total Banks", len(df_bank))
    profiler.add_count("Existing Predictions", len(df_pred))
    
    # 2. Load model
    profiler.start_timer("2. Model Loading")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    model.set_params(n_jobs=-1)
    
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    threshold = float(threshold_data.get("threshold", 0.95))
    profiler.stop_timer("2. Model Loading")
    
    # 3. Filter orphans
    profiler.start_timer("3. Orphan Filtering")
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
    profiler.stop_timer("3. Orphan Filtering")
    profiler.add_count("Orphan Gateways", len(orphan_gws))
    profiler.add_count("Orphan Banks", len(orphan_banks))
    
    if not orphan_gws or not orphan_banks:
        profiler.print_summary()
        return
    
    # 4. Block Generation (Clustering)
    profiler.start_timer("4. Block Generation (Clustering)")
    block_gen = CandidateBlockGenerator(
        max_delay_days=5, 
        max_amount_diff_pct=0.20
    )
    candidate_blocks = block_gen.generate_blocks(orphan_gws, orphan_banks)
    profiler.stop_timer("4. Block Generation (Clustering)")
    profiler.add_count("Candidate Blocks", len(candidate_blocks))
    
    if not candidate_blocks:
        profiler.print_summary()
        return
    
    # 5. Feature Extraction
    profiler.start_timer("5. Feature Extraction")
    feature_rows = []
    for block in candidate_blocks:
        feats = extract_cluster_features(block["gw_rows"], block["bank_row"])
        feature_rows.append(feats)
    
    df_features = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    profiler.stop_timer("5. Feature Extraction")
    profiler.add_count("Feature Rows", len(df_features))
    
    # 6. XGBoost Prediction
    profiler.start_timer("6. XGBoost Prediction")
    feature_array = df_features[FEATURE_COLUMNS].values.astype(np.float32)
    probs = model.predict_proba(feature_array)[:, 1]
    profiler.stop_timer("6. XGBoost Prediction")
    
    # 7. Filtering & Assignment
    profiler.start_timer("7. Filtering & Assignment")
    amount_diff_pct = df_features["amount_diff_pct"].values
    valid_mask = (probs >= threshold) & (amount_diff_pct <= 0.10)
    valid_indices = np.where(valid_mask)[0]
    sorted_indices = valid_indices[np.argsort(-probs[valid_indices])]
    
    assigned_edges = []
    assigned_gw_ids: Set[str] = set()
    assigned_bank_ids: Set[str] = set()
    
    for idx in sorted_indices:
        block = candidate_blocks[idx]
        bank_id = block["bank_id"]
        gw_ids = block["gw_ids"]
        prob = float(probs[idx])
        
        if bank_id in claimed_bank_ids or bank_id in assigned_bank_ids:
            continue
        
        conflict = False
        for g_id in gw_ids:
            if g_id in claimed_gw_ids or g_id in assigned_gw_ids:
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
    profiler.stop_timer("7. Filtering & Assignment")
    profiler.add_count("Assigned Edges", len(assigned_edges))
    
    # Print results
    profiler.print_summary()
    
    # Additional breakdown
    print(f"\n🔍 ANALYSIS:")
    print(f"  • Blocks per orphan bank: {len(candidate_blocks) / max(1, len(orphan_banks)):.1f}")
    print(f"  • Features per block: {len(FEATURE_COLUMNS)}")
    print(f"  • XGBoost prediction time per block: {(profiler.timings.get('6. XGBoost Prediction', 0) / max(1, len(candidate_blocks)) * 1000):.3f} ms")
    print(f"  • Block generation time per bank: {(profiler.timings.get('4. Block Generation (Clustering)', 0) / max(1, len(orphan_banks)) * 1000):.3f} ms")
    print(f"  • Feature extraction per block: {(profiler.timings.get('5. Feature Extraction', 0) / max(1, len(candidate_blocks)) * 1000):.3f} ms")
    
    # Identify bottleneck
    clustering_time = profiler.timings.get('4. Block Generation (Clustering)', 0)
    xgboost_time = profiler.timings.get('6. XGBoost Prediction', 0)
    feature_time = profiler.timings.get('5. Feature Extraction', 0)
    
    if clustering_time > xgboost_time and clustering_time > feature_time:
        print(f"\n🎯 BOTTLENECK: Block Generation (Clustering) is the slowest stage!")
        print(f"   Suggestion: Optimize candidate block generation with pre-indexing")
    elif xgboost_time > clustering_time and xgboost_time > feature_time:
        print(f"\n🎯 BOTTLENECK: XGBoost Prediction is the slowest stage!")
        print(f"   Suggestion: Use batch prediction or reduce feature dimensions")
    else:
        print(f"\n🎯 BOTTLENECK: Feature Extraction is the slowest stage!")
        print(f"   Suggestion: Cache feature computations or parallelize extraction")
    
    return profiler


if __name__ == "__main__":
    profile_inference()