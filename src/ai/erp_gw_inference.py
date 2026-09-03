#!/usr/bin/env python3
"""
Residual XGBoost Inference Engine for ERP<->Gateway Matching.

Runs post-deterministic inference on remaining orphaned ERP and Gateway records.
Uses ERP-GW candidate cluster blocking, feature aggregation, XGBoost scoring, and
greedy conflict-free bipartite assignment.

Adapted from inference.py (GW<->Bank). Key differences:
- Source: orphaned ERP records (erp_entry_id, invoice_number, gross_amount, entry_date)
- Target: orphaned GW records (payment_id, gross_amount, settled_at, invoices)
- Model: xgb_erp_gw.json + xgb_erp_gw_threshold.json
- Output table: TABLE_ERP_GW_PRED (erp_order_id, gateway_payment_id, allocated_amount, ...)
- Amount filter: gross_diff_pct instead of amount_diff_pct
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

from src.ai.erp_gw_block_generator import ERPGWCandidateBlockGenerator
from src.ai.erp_gw_features import FEATURE_COLUMNS, extract_cluster_features
from src.core.config import (
    DB_PATH,
    MATCH_TYPE_BULK,
    MATCH_TYPE_EXACT,
    STAGE_AI_CLUSTER,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_ERP_GW_PRED,
)
from src.core.database import get_connection

ARTIFACT_DIR = ROOT_DIR / "src" / "ai" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_erp_gw.json"
THRESHOLD_PATH = ARTIFACT_DIR / "xgb_erp_gw_threshold.json"


def load_model_and_threshold() -> Tuple[XGBClassifier, float]:
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"ERP<->GW model or threshold artifacts missing in {ARTIFACT_DIR}. "
            "Run erp_gw_train_model.py first."
        )
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data.get("threshold", 0.95))
    return model, threshold


def run_erp_gw_ai_inference(db_path: Path = DB_PATH) -> int:
    """
    Executes the residual AI matching stage for ERP<->Gateway and appends
    newly predicted edges to the erp_gw_pred table.

    Returns:
        Number of new edges written.
    """
    conn = get_connection(db_path)
    try:
        df_erp = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn)
        df_gw = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
        df_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP_GW_PRED}", conn)
    finally:
        conn.close()

    if df_erp.empty or df_gw.empty:
        print("[!] ERP or Gateway tables are empty; skipping ERP<->GW AI inference.")
        return 0

    try:
        model, threshold = load_model_and_threshold()
    except Exception as e:
        print(f"[!] Unable to load ERP<->GW XGBoost artifacts: {e}. Skipping AI inference.")
        return 0

    # Collect already-claimed IDs from deterministic + prior AI passes
    claimed_erp_ids: Set[str] = set(df_pred["erp_order_id"].astype(str).dropna()) if not df_pred.empty else set()
    claimed_gw_ids: Set[str] = set(df_pred["gateway_payment_id"].astype(str).dropna()) if not df_pred.empty else set()

    orphan_erps = [
        row.to_dict() for _, row in df_erp.iterrows()
        if str(row["erp_entry_id"]) not in claimed_erp_ids
    ]
    orphan_gws = [
        row.to_dict() for _, row in df_gw.iterrows()
        if str(row["payment_id"]) not in claimed_gw_ids
    ]

    if not orphan_erps or not orphan_gws:
        print("[*] No orphaned ERP or Gateway records to process for AI stage.")
        return 0

    print(f"[*] ERP<->GW AI Residual Pass: {len(orphan_erps)} orphaned ERPs, {len(orphan_gws)} orphaned GW records.")

    # 1. Generate Candidate Blocks
    block_gen = ERPGWCandidateBlockGenerator(max_delay_days=5, max_amount_diff_pct=0.20)
    candidate_blocks = block_gen.generate_blocks(orphan_erps, orphan_gws)

    if not candidate_blocks:
        print("[*] No ERP<->GW candidate cluster blocks generated within temporal & amount windows.")
        return 0

    # 2. Extract Features
    feature_rows = []
    valid_blocks = []

    for block in candidate_blocks:
        feats = extract_cluster_features(block["erp_rows"], block["gw_row"])
        feature_rows.append(feats)
        valid_blocks.append(block)

    df_features = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]

    # 3. Model Scoring
    probs = model.predict_proba(df_features)[:, 1]

    # 4. Filter & Sort High-Confidence Candidates
    scored_candidates = []
    for idx, block in enumerate(valid_blocks):
        prob = float(probs[idx])
        gross_diff_pct = float(feature_rows[idx]["gross_diff_pct"])
        if prob >= threshold and gross_diff_pct <= 0.05:  # strict: ERP<->GW is pure gross match
            scored_candidates.append({
                "probability": prob,
                "block": block,
                "features": feature_rows[idx],
            })

    # Sort descending by confidence
    scored_candidates.sort(key=lambda x: x["probability"], reverse=True)

    # 5. Greedy Conflict-Free Bipartite Matching
    assigned_edges = []
    assigned_erp_ids: Set[str] = set()
    assigned_gw_ids: Set[str] = set()

    for cand in scored_candidates:
        block = cand["block"]
        gw_id = block["gw_id"]
        erp_ids = block["erp_ids"]
        prob = cand["probability"]

        # Conflict check
        if gw_id in claimed_gw_ids or gw_id in assigned_gw_ids:
            continue
        if any(e_id in claimed_erp_ids or e_id in assigned_erp_ids for e_id in erp_ids):
            continue

        # Claim IDs
        assigned_gw_ids.add(gw_id)
        for e_id in erp_ids:
            assigned_erp_ids.add(e_id)

        is_bulk = len(erp_ids) > 1
        gw_gross = float(block["gw_row"].get("gross_amount", 0.0))
        for erp_rec in block["erp_rows"]:
            e_id = erp_rec.get("_pid", erp_rec.get("erp_entry_id", ""))
            erp_gross = float(erp_rec.get("_gross", erp_rec.get("gross_amount", 0.0)))
            assigned_edges.append({
                "erp_order_id": e_id,
                "gateway_payment_id": gw_id,
                "allocated_amount": erp_gross,
                "match_type": MATCH_TYPE_BULK if is_bulk else MATCH_TYPE_EXACT,
                "matching_stage": STAGE_AI_CLUSTER,
                "confidence_score": round(prob, 4),
                "notes": (
                    f"ERP<->GW AI XGBoost cluster match "
                    f"(Score: {prob:.4f}, Cluster size: {len(erp_ids)})."
                ),
            })

    if not assigned_edges:
        print("[✔] No ERP<->GW AI cluster predictions met the strict calibrated threshold.")
        return 0

    # 6. Persist to SQLite
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        for edge in assigned_edges:
            cursor.execute(
                f"""
                INSERT INTO {TABLE_ERP_GW_PRED}
                (erp_order_id, gateway_payment_id, allocated_amount,
                 match_type, matching_stage, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge["erp_order_id"],
                    edge["gateway_payment_id"],
                    edge["allocated_amount"],
                    edge["match_type"],
                    edge["matching_stage"],
                    edge["confidence_score"],
                    edge["notes"],
                ),
            )
        conn.commit()
    finally:
        conn.close()

    unique_gws = len({e["gateway_payment_id"] for e in assigned_edges})
    print(f"[✔] ERP<->GW AI Stage Completed: {len(assigned_edges)} edges across {unique_gws} Gateway records.")
    return len(assigned_edges)


def main():
    run_erp_gw_ai_inference()


if __name__ == "__main__":
    main()
