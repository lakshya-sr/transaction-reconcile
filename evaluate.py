#!/usr/bin/env python3
"""
Phase Evaluator: Strictly measures accuracy of graph edges using exact ID matching.
"""

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DB_PATH, DATA_DIR
from src.database import get_connection

def calculate_metrics(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def main():
    truth_csv = DATA_DIR / "ground_truth.csv"
    if not truth_csv.exists():
        print(f"[!] Ground truth CSV not found at {truth_csv}. Run pipeline first.")
        return

    df_truth = pd.read_csv(truth_csv)
    
    # Extract Ground Truth ID Pairs
    dt_eg = df_truth.dropna(subset=['erp_id', 'gw_id'])
    true_erp_gw = set(zip(dt_eg['erp_id'], dt_eg['gw_id']))
    
    dt_gb = df_truth.dropna(subset=['gw_id', 'bank_id'])
    true_gw_bnk = set(zip(dt_gb['gw_id'], dt_gb['bank_id']))

    # Load Predicted Edges from SQLite Database
    conn = get_connection(DB_PATH)
    try:
        df_eg_pred = pd.read_sql_query("SELECT erp_order_id, gateway_payment_id FROM erp_to_gateway_edges", conn)
        df_gb_pred = pd.read_sql_query("SELECT gateway_payment_id, bank_entry_id FROM gateway_to_bank_edges", conn)
    finally:
        conn.close()

    pred_erp_gw = set(zip(df_eg_pred['erp_order_id'], df_eg_pred['gateway_payment_id']))
    pred_gw_bnk = set(zip(df_gb_pred['gateway_payment_id'], df_gb_pred['bank_entry_id']))

    metrics_table = []
    layers = [
        ("Layer 1: ERP ↔ Gateway", true_erp_gw, pred_erp_gw),
        ("Layer 2: Gateway ↔ Bank", true_gw_bnk, pred_gw_bnk)
    ]
    
    for layer_name, true_set, pred_set in layers:
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        p, r, f1 = calculate_metrics(tp, fp, fn)
        
        metrics_table.append([
            layer_name, len(true_set), len(pred_set),
            tp, fp, fn,
            f"{p:.1%}", f"{r:.1%}", f"{f1:.1%}"
        ])

    print("=" * 100)
    print("  STRICT ID GRAPH ACCURACY EVALUATOR")
    print("=" * 100)
    headers = ["Reconciliation Graph Layer", "True Edges", "Pred Edges", "TP", "FP", "FN", "Precision", "Recall", "F1 Score"]
    print(tabulate(metrics_table, headers=headers, tablefmt="fancy_grid"))
    print("=" * 100)

if __name__ == "__main__":
    main()