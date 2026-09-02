#!/usr/bin/env python3
"""
Residual XGBoost Cluster Inference Engine.

Runs post-deterministic inference on remaining orphaned Gateway and Bank records.
Uses candidate cluster blocking, feature aggregation, XGBoost scoring, and greedy
conflict-free bipartite assignment.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.block_generator import CandidateBlockGenerator
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


def load_model_and_threshold() -> Tuple[XGBClassifier, float]:
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Model or threshold artifacts missing in {ARTIFACT_DIR}. Run train_model.py first.")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data.get("threshold", 0.95))
    return model, threshold


def run_residual_ai_inference(db_path: Path = DB_PATH) -> int:
    """
    Executes the residual AI matching stage and appends newly predicted edges
    to the gw_bank_pred table.
    """
    conn = get_connection(db_path)
    try:
        df_gw = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
        df_bank = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}", conn)
        df_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_GW_BANK_PRED}", conn)
    finally:
        conn.close()

    if df_gw.empty or df_bank.empty:
        print("[!] Gateway or Bank tables are empty; skipping AI inference.")
        return 0

    try:
        model, threshold = load_model_and_threshold()
    except Exception as e:
        print(f"[!] Unable to load XGBoost artifacts: {e}. Skipping AI inference.")
        return 0

    claimed_gw_ids: Set[str] = set(df_pred["gateway_payment_id"].astype(str).dropna()) if not df_pred.empty else set()
    claimed_bank_ids: Set[str] = set(df_pred["bank_entry_id"].astype(str).dropna()) if not df_pred.empty else set()

    orphan_gws = [
        row.to_dict() for _, row in df_gw.iterrows()
        if str(row["payment_id"]) not in claimed_gw_ids
    ]
    orphan_banks = [
        row.to_dict() for _, row in df_bank.iterrows()
        if str(row["bank_entry_id"]) not in claimed_bank_ids
    ]

    if not orphan_gws or not orphan_banks:
        print("[*] No orphaned Gateway or Bank records to process for AI stage.")
        return 0

    print(f"[*] AI Residual Pass: {len(orphan_gws)} orphaned Gateways, {len(orphan_banks)} orphaned Bank entries.")

    # 1. Generate Candidate Blocks
    block_gen = CandidateBlockGenerator(max_delay_days=5, max_amount_diff_pct=0.20)
    candidate_blocks = block_gen.generate_blocks(orphan_gws, orphan_banks)

    if not candidate_blocks:
        print("[*] No candidate cluster blocks generated within temporal & amount windows.")
        return 0

    # 2. Extract Features
    feature_rows = []
    valid_blocks = []

    for block in candidate_blocks:
        feats = extract_cluster_features(block["gw_rows"], block["bank_row"])
        feature_rows.append(feats)
        valid_blocks.append(block)

    df_features = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]

    # 3. Model Scoring
    probs = model.predict_proba(df_features)[:, 1]

    # 4. Filter & Sort High-Confidence Candidates
    scored_candidates = []
    for idx, block in enumerate(valid_blocks):
        prob = float(probs[idx])
        amt_diff_pct = float(feature_rows[idx]["amount_diff_pct"])
        if prob >= threshold and amt_diff_pct <= 0.10:
            scored_candidates.append({
                "probability": prob,
                "block": block,
                "features": feature_rows[idx],
            })

    # Sort descending by confidence score
    scored_candidates.sort(key=lambda x: x["probability"], reverse=True)

    # 5. Greedy Conflict-Free Bipartite Matching
    assigned_edges = []
    assigned_gw_ids: Set[str] = set()
    assigned_bank_ids: Set[str] = set()

    for cand in scored_candidates:
        block = cand["block"]
        bank_id = block["bank_id"]
        gw_ids = block["gw_ids"]
        prob = cand["probability"]

        # Conflict check: ensure neither bank nor any gateway was claimed
        if bank_id in claimed_bank_ids or bank_id in assigned_bank_ids:
            continue
        if any(g_id in claimed_gw_ids or g_id in assigned_gw_ids for g_id in gw_ids):
            continue

        # Claim IDs
        assigned_bank_ids.add(bank_id)
        for g_id in gw_ids:
            assigned_gw_ids.add(g_id)

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

    if not assigned_edges:
        print("[✔] No AI cluster predictions met the strict calibrated threshold.")
        return 0

    # 6. Persist to SQLite
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        for edge in assigned_edges:
            cursor.execute(f"""
                INSERT INTO {TABLE_GW_BANK_PRED}
                (gateway_payment_id, bank_entry_id, allocated_amount, match_type, matching_stage, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge["gateway_payment_id"], edge["bank_entry_id"],
                edge["allocated_amount"], edge["match_type"],
                edge["matching_stage"], edge["confidence_score"],
                edge["notes"]
            ))
        conn.commit()
    finally:
        conn.close()

    print(f"[✔] AI Stage Completed: {len(assigned_edges)} edges established across {len(assigned_bank_ids)} Bank deposits.")
    return len(assigned_edges)


def main():
    run_residual_ai_inference()


if __name__ == "__main__":
    main()
