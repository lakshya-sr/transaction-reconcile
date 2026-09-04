#!/usr/bin/env python3
"""
Residual XGBoost Cluster Inference Engine - OPTIMIZED VERSION.

Key optimizations:
1. Batch prediction instead of row-by-row
2. Parallel feature extraction
3. Vectorized filtering
4. Pre-indexed data structures
5. Cached datetime parsing
6. Optimized database queries
7. Reduced memory footprint
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.block_generator import FastCandidateBlockGenerator
from src.ai.features import FEATURE_COLUMNS, extract_cluster_features
from src.core.config import (
    DB_PATH,
    MATCH_TYPE_BULK,
    MATCH_TYPE_EXACT,
    STAGE_AI_CLUSTER,
    TABLE_BANK,
    TABLE_GATEWAY,
    TABLE_GW_BANK_PRED,
)
from src.core.database import get_connection

ARTIFACT_DIR = ROOT_DIR / "src" / "ai" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_gw_bank.json"
THRESHOLD_PATH = ARTIFACT_DIR / "xgb_gw_bank_threshold.json"

# Cache for datetime parsing
@lru_cache(maxsize=10000)
def _parse_dt_cached(date_str: str) -> pd.Timestamp:
    """Cached datetime parsing."""
    return pd.to_datetime(date_str, errors='coerce')


def load_model_and_threshold() -> Tuple[XGBClassifier, float]:
    """Load model and threshold with optimizations."""
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Model artifacts missing in {ARTIFACT_DIR}. Run train_model.py first.")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    
    # Enable parallel prediction (CPU only - safe)
    model.set_params(n_jobs=-1)
    
    # Don't try GPU - use CPU hist which is always available
    try:
        model.set_params(tree_method='hist')
    except:
        pass  # Keep default

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data.get("threshold", 0.95))
    return model, threshold


def _batch_extract_features(candidate_blocks: List[Dict]) -> pd.DataFrame:
    """Extract features in parallel for all candidate blocks."""
    if not candidate_blocks:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    
    # Determine optimal number of workers
    import os
    num_workers = min(os.cpu_count() or 4, 8)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        feature_rows = list(executor.map(
            lambda block: extract_cluster_features(block["gw_rows"], block["bank_row"]),
            candidate_blocks
        ))
    
    return pd.DataFrame(feature_rows)[FEATURE_COLUMNS]


def run_residual_ai_inference(db_path: Path = DB_PATH) -> int:
    """
    Optimized residual AI matching stage.
    
    Returns:
        Number of new edges written.
    """
    start_time = time.time()
    print("[*] Starting optimized GW↔Bank AI inference...")
    
    # 1. Load data with optimized queries (only needed columns)
    t0 = time.time()
    conn = get_connection(db_path)
    try:
        df_gw = pd.read_sql_query(
            f"SELECT payment_id, net_settled, settled_at, bank_utr, invoices "
            f"FROM {TABLE_GATEWAY}", 
            conn
        )
        df_bank = pd.read_sql_query(
            f"SELECT bank_entry_id, credit_amount, value_date, remittance_info "
            f"FROM {TABLE_BANK}", 
            conn
        )
        df_pred = pd.read_sql_query(
            f"SELECT gateway_payment_id, bank_entry_id "
            f"FROM {TABLE_GW_BANK_PRED}", 
            conn
        )
    finally:
        conn.close()
    print(f"  Data loading: {time.time() - t0:.3f}s")

    if df_gw.empty or df_bank.empty:
        print("[!] Gateway or Bank tables are empty; skipping AI inference.")
        return 0

    # 2. Load model once
    t0 = time.time()
    try:
        model, threshold = load_model_and_threshold()
    except Exception as e:
        print(f"[!] Unable to load XGBoost artifacts: {e}. Skipping AI inference.")
        return 0
    print(f"  Model loading: {time.time() - t0:.3f}s")

    # 3. Vectorized orphan filtering
    t0 = time.time()
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
    print(f"  Orphan filtering: {time.time() - t0:.3f}s ({len(orphan_gws)} GWs, {len(orphan_banks)} Banks)")

    if not orphan_gws or not orphan_banks:
        print("[*] No orphaned Gateway or Bank records to process.")
        return 0

    # 4. Generate candidate blocks with optimized generator
    t0 = time.time()
    block_gen = FastCandidateBlockGenerator(
        max_delay_days=5, 
        max_amount_diff_pct=0.20,
        max_candidates_per_bank=15  # Limit to prevent explosion
    )
    candidate_blocks = block_gen.generate_blocks(orphan_gws, orphan_banks)
    print(f"  Block generation: {time.time() - t0:.3f}s ({len(candidate_blocks)} candidates)")

    if not candidate_blocks:
        print("[*] No candidate blocks generated.")
        return 0

    # 5. Parallel feature extraction
    t0 = time.time()
    df_features = _batch_extract_features(candidate_blocks)
    print(f"  Feature extraction: {time.time() - t0:.3f}s")

    # 6. Batch prediction (vectorized)
    t0 = time.time()
    # Convert to numpy for faster prediction
    feature_array = df_features[FEATURE_COLUMNS].values.astype(np.float32)
    
    # Use predict_proba but handle potential issues
    if len(feature_array) > 0:
        # Batch prediction
        probs = model.predict_proba(feature_array)[:, 1]
    else:
        probs = np.array([])
    
    print(f"  Batch prediction: {time.time() - t0:.3f}s")

    # 7. Vectorized filtering
    t0 = time.time()
    amount_diff_pct = df_features["amount_diff_pct"].values
    valid_mask = (probs >= threshold) & (amount_diff_pct <= 0.10)
    valid_indices = np.where(valid_mask)[0]
    
    # Sort by probability descending
    sorted_indices = valid_indices[np.argsort(-probs[valid_indices])]
    print(f"  Filtering: {time.time() - t0:.3f}s ({len(sorted_indices)} valid candidates)")

    # 8. Greedy assignment with fast lookups
    t0 = time.time()
    assigned_edges = []
    assigned_gw_ids: Set[str] = set()
    assigned_bank_ids: Set[str] = set()

    for idx in sorted_indices:
        block = candidate_blocks[idx]
        bank_id = block["bank_id"]
        gw_ids = block["gw_ids"]
        prob = float(probs[idx])

        # Fast conflict check using set operations
        if bank_id in claimed_bank_ids or bank_id in assigned_bank_ids:
            continue
        
        # Check if any gateway is already claimed
        conflict = False
        for g_id in gw_ids:
            if g_id in claimed_gw_ids or g_id in assigned_gw_ids:
                conflict = True
                break
        
        if conflict:
            continue

        # Claim IDs
        assigned_bank_ids.add(bank_id)
        assigned_gw_ids.update(gw_ids)

        is_bulk = len(gw_ids) > 1
        for gw_rec in block["gw_rows"]:
            g_id = gw_rec["_pid"]
            gw_net = float(gw_rec["_net"])
            assigned_edges.append({
                "gateway_payment_id": g_id,
                "bank_entry_id": bank_id,
                "allocated_amount": gw_net,
                "match_type": MATCH_TYPE_BULK if is_bulk else MATCH_TYPE_EXACT,
                "matching_stage": STAGE_AI_CLUSTER,
                "confidence_score": round(prob, 4),
                "notes": f"AI XGBoost cluster match (Score: {prob:.4f}, Batch size: {len(gw_ids)})."
            })
    print(f"  Greedy assignment: {time.time() - t0:.3f}s ({len(assigned_edges)} edges)")

    # 9. Batch insert using executemany
    t0 = time.time()
    if assigned_edges:
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()
            # Use executemany for batch insert
            cursor.executemany(f"""
                INSERT INTO {TABLE_GW_BANK_PRED}
                (gateway_payment_id, bank_entry_id, allocated_amount, match_type, 
                 matching_stage, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    edge["gateway_payment_id"], edge["bank_entry_id"],
                    edge["allocated_amount"], edge["match_type"],
                    edge["matching_stage"], edge["confidence_score"],
                    edge["notes"]
                )
                for edge in assigned_edges
            ])
            conn.commit()
        finally:
            conn.close()
    print(f"  Database insert: {time.time() - t0:.3f}s")

    total_time = time.time() - start_time
    print(f"[✔] AI Stage Completed in {total_time:.2f}s: {len(assigned_edges)} edges across {len(assigned_bank_ids)} deposits.")
    return len(assigned_edges)


def main():
    run_residual_ai_inference()


if __name__ == "__main__":
    main()