#!/usr/bin/env python3
"""
Unified inference module for both GW↔Bank and ERP↔GW matching.

Contains:
- run_gateway_bank_inference(): AI inference for Gateway↔Bank matching
- run_erp_gateway_inference(): AI inference for ERP↔Gateway matching
"""

import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.candidate_generation import (
    GatewayBankCandidateGenerator,
    ERPGatewayCandidateGenerator,
)
from src.ai.features import (
    GATEWAY_BANK_FEATURES,
    ERP_GATEWAY_FEATURES,
    extract_gateway_bank_features,
    extract_erp_gateway_features,
)
from src.core.config import (
    DB_PATH,
    MATCH_TYPE_BULK,
    MATCH_TYPE_EXACT,
    MATCH_STAGE_ML_CLUSTER,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_ERP_GW_PRED,
    TABLE_GW_BANK_PRED,
)
from src.core.database import get_connection

ARTIFACT_DIR = ROOT_DIR / "src" / "ai" / "artifacts"


def _load_model(model_name: str) -> Tuple[XGBClassifier, float]:
    """Load model and threshold for given model name."""
    model_path = ARTIFACT_DIR / f"xgb_{model_name}.json"
    threshold_path = ARTIFACT_DIR / f"xgb_{model_name}_threshold.json"
    
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError(f"Model artifacts missing in {ARTIFACT_DIR} for {model_name}")
    
    model = XGBClassifier()
    model.load_model(str(model_path))
    model.set_params(n_jobs=-1)
    try:
        model.set_params(tree_method='hist')
    except:
        pass
    
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    
    threshold = float(threshold_data.get("threshold", 0.95))
    return model, threshold


def _load_orphans(db_path: Path, table_name: str, pred_table: str, id_col: str, pred_id_col: str) -> Tuple[List[Dict], Set[str]]:
    """Load orphaned records from database."""
    conn = get_connection(db_path)
    try:
        df_records = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df_pred = pd.read_sql_query(f"SELECT {pred_id_col} FROM {pred_table}", conn)
    finally:
        conn.close()
    
    claimed_ids = set(df_pred[pred_id_col].astype(str).dropna()) if not df_pred.empty else set()
    
    orphan_records = [
        row.to_dict() for _, row in df_records.iterrows()
        if str(row[id_col]) not in claimed_ids
    ]
    
    return orphan_records, claimed_ids


def _save_edges(db_path: Path, table_name: str, edges: List[Dict], columns: List[str]):
    """Save matched edges to database."""
    if not edges:
        return
    
    placeholders = ", ".join(["?"] * len(columns))
    column_names = ", ".join(columns)
    
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.executemany(
            f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})",
            [
                tuple(edge[col] for col in columns)
                for edge in edges
            ]
        )
        conn.commit()
    finally:
        conn.close()


def run_gateway_bank_inference(db_path: Path = DB_PATH) -> int:
    """Run AI inference for Gateway↔Bank matching."""
    start_time = time.time()
    print("[*] Starting Gateway↔Bank AI inference...")
    
    # Load data
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
    
    if df_gw.empty or df_bank.empty:
        print("[!] Gateway or Bank tables are empty.")
        return 0
    
    # Load model
    try:
        model, threshold = _load_model("gw_bank")
    except Exception as e:
        print(f"[!] Unable to load model: {e}")
        return 0
    
    # Filter orphans
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
    
    if not orphan_gws or not orphan_banks:
        print("[*] No orphaned records to process.")
        return 0
    
    print(f"  • Orphan Gateways: {len(orphan_gws)}")
    print(f"  • Orphan Banks: {len(orphan_banks)}")
    
    # Generate candidates
    block_gen = GatewayBankCandidateGenerator(max_delay_days=5, max_amount_diff_pct=0.20)
    candidate_blocks = block_gen.generate(orphan_gws, orphan_banks)
    
    if not candidate_blocks:
        print("[*] No candidate blocks generated.")
        return 0
    
    print(f"  • Candidate Blocks: {len(candidate_blocks)}")
    
    # Extract features
    feature_rows = []
    for block in candidate_blocks:
        feats = extract_gateway_bank_features(block["gw_rows"], block["bank_row"])
        feature_rows.append(feats)
    
    df_features = pd.DataFrame(feature_rows)[GATEWAY_BANK_FEATURES]
    
    # Predict
    feature_array = df_features[GATEWAY_BANK_FEATURES].values.astype(np.float32)
    probs = model.predict_proba(feature_array)[:, 1]
    
    # Filter and assign
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
        
        is_bulk = len(gw_ids) > 1
        for gw_rec in block["gw_rows"]:
            g_id = gw_rec["_pid"]
            gw_net = float(gw_rec["_net"])
            assigned_edges.append({
                "gateway_payment_id": g_id,
                "bank_entry_id": bank_id,
                "allocated_amount": gw_net,
                "match_type": MATCH_TYPE_BULK if is_bulk else MATCH_TYPE_EXACT,
                "matching_stage": MATCH_STAGE_ML_CLUSTER,
                "confidence_score": round(prob, 4),
                "notes": f"ML cluster match (Score: {prob:.4f}, Batch: {len(gw_ids)})."
            })
    
    # Save edges
    if assigned_edges:
        _save_edges(
            db_path,
            TABLE_GW_BANK_PRED,
            assigned_edges,
            ["gateway_payment_id", "bank_entry_id", "allocated_amount", "match_type", "matching_stage", "confidence_score", "notes"]
        )
    
    elapsed = time.time() - start_time
    print(f"[✔] Gateway↔Bank inference completed in {elapsed:.2f}s: {len(assigned_edges)} edges")
    return len(assigned_edges)


def run_erp_gateway_inference(db_path: Path = DB_PATH) -> int:
    """Run AI inference for ERP↔Gateway matching."""
    start_time = time.time()
    print("[*] Starting ERP↔Gateway AI inference...")
    
    # Load data
    conn = get_connection(db_path)
    try:
        df_erp = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn)
        df_gw = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
        df_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP_GW_PRED}", conn)
    finally:
        conn.close()
    
    if df_erp.empty or df_gw.empty:
        print("[!] ERP or Gateway tables are empty.")
        return 0
    
    # Load model
    try:
        model, threshold = _load_model("erp_gw")
    except Exception as e:
        print(f"[!] Unable to load model: {e}")
        return 0
    
    # Filter orphans
    claimed_erp_ids = set(df_pred["erp_order_id"].astype(str).dropna()) if not df_pred.empty else set()
    claimed_gw_ids = set(df_pred["gateway_payment_id"].astype(str).dropna()) if not df_pred.empty else set()
    
    orphan_erps = [
        row.to_dict() for _, row in df_erp.iterrows()
        if str(row["erp_entry_id"]) not in claimed_erp_ids
    ]
    orphan_gws = [
        row.to_dict() for _, row in df_gw.iterrows()
        if str(row["payment_id"]) not in claimed_gw_ids
    ]
    
    if not orphan_erps or not orphan_gws:
        print("[*] No orphaned records to process.")
        return 0
    
    print(f"  • Orphan ERPs: {len(orphan_erps)}")
    print(f"  • Orphan Gateways: {len(orphan_gws)}")
    
    # Generate candidates
    block_gen = ERPGatewayCandidateGenerator(max_delay_days=5, max_amount_diff_pct=0.20)
    candidate_blocks = block_gen.generate(orphan_erps, orphan_gws)
    
    if not candidate_blocks:
        print("[*] No candidate blocks generated.")
        return 0
    
    print(f"  • Candidate Blocks: {len(candidate_blocks)}")
    
    # Extract features
    feature_rows = []
    for block in candidate_blocks:
        feats = extract_erp_gateway_features(block["erp_rows"], block["gw_row"])
        feature_rows.append(feats)
    
    df_features = pd.DataFrame(feature_rows)[ERP_GATEWAY_FEATURES]
    
    # Predict
    feature_array = df_features[ERP_GATEWAY_FEATURES].values.astype(np.float32)
    probs = model.predict_proba(feature_array)[:, 1]
    
    # Filter and assign
    gross_diff_pct = df_features["gross_diff_pct"].values
    valid_mask = (probs >= threshold) & (gross_diff_pct <= 0.05)
    valid_indices = np.where(valid_mask)[0]
    sorted_indices = valid_indices[np.argsort(-probs[valid_indices])]
    
    assigned_edges = []
    assigned_erp_ids: Set[str] = set()
    assigned_gw_ids: Set[str] = set()
    
    for idx in sorted_indices:
        block = candidate_blocks[idx]
        gw_id = block["gw_id"]
        erp_ids = block["erp_ids"]
        prob = float(probs[idx])
        
        if gw_id in claimed_gw_ids or gw_id in assigned_gw_ids:
            continue
        
        conflict = False
        for e_id in erp_ids:
            if e_id in claimed_erp_ids or e_id in assigned_erp_ids:
                conflict = True
                break
        
        if conflict:
            continue
        
        assigned_gw_ids.add(gw_id)
        assigned_erp_ids.update(erp_ids)
        
        is_bulk = len(erp_ids) > 1
        for erp_rec in block["erp_rows"]:
            e_id = erp_rec.get("_pid", erp_rec.get("erp_entry_id", ""))
            erp_gross = float(erp_rec.get("_gross", erp_rec.get("gross_amount", 0.0)))
            assigned_edges.append({
                "erp_order_id": e_id,
                "gateway_payment_id": gw_id,
                "allocated_amount": erp_gross,
                "match_type": MATCH_TYPE_BULK if is_bulk else MATCH_TYPE_EXACT,
                "matching_stage": MATCH_STAGE_ML_CLUSTER,
                "confidence_score": round(prob, 4),
                "notes": f"ML cluster match (Score: {prob:.4f}, Batch: {len(erp_ids)})."
            })
    
    # Save edges
    if assigned_edges:
        _save_edges(
            db_path,
            TABLE_ERP_GW_PRED,
            assigned_edges,
            ["erp_order_id", "gateway_payment_id", "allocated_amount", "match_type", "matching_stage", "confidence_score", "notes"]
        )
    
    elapsed = time.time() - start_time
    print(f"[✔] ERP↔Gateway inference completed in {elapsed:.2f}s: {len(assigned_edges)} edges")
    return len(assigned_edges)


# Backward compatibility aliases
run_residual_ai_inference = run_gateway_bank_inference
run_erp_gw_ai_inference = run_erp_gateway_inference


def main():
    """Run both inference pipelines."""
    run_gateway_bank_inference()
    print()
    run_erp_gateway_inference()


if __name__ == "__main__":
    main()