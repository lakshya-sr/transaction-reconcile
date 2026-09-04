#!/usr/bin/env python3
"""
Phase Evaluator: Strictly measures accuracy of graph edges using exact ID matching.
"""

from pathlib import Path
from itertools import zip_longest
import pandas as pd
from tabulate import tabulate

from src.core.config import (
    DB_PATH,
    TABLE_ERP_GW_TRUE,
    TABLE_GW_BANK_TRUE,
    TABLE_ERP_GW_PRED,
    TABLE_GW_BANK_PRED,
    GROUND_TRUTH_ERP_GW_PATH,
    GROUND_TRUTH_GW_BANK_PATH,
)
from src.core.database import get_connection


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    conn = get_connection(DB_PATH)
    try:
        # Load Ground Truth Edges
        df_eg_true = pd.read_sql_query(f"SELECT erp_id, gw_id FROM {TABLE_ERP_GW_TRUE}", conn)
        df_gb_true = pd.read_sql_query(f"SELECT gw_id, bank_id FROM {TABLE_GW_BANK_TRUE}", conn)

        # Load Predicted Edges with stage metadata
        df_eg_pred = pd.read_sql_query(f"SELECT erp_order_id, gateway_payment_id, matching_stage FROM {TABLE_ERP_GW_PRED}", conn)
        df_gb_pred = pd.read_sql_query(f"SELECT gateway_payment_id, bank_entry_id, matching_stage FROM {TABLE_GW_BANK_PRED}", conn)
    except Exception as e:
        # Fallback to CSV files if DB tables are empty
        if GROUND_TRUTH_ERP_GW_PATH.exists() and GROUND_TRUTH_GW_BANK_PATH.exists():
            df_eg_true = pd.read_csv(GROUND_TRUTH_ERP_GW_PATH)[["erp_id", "gw_id"]]
            df_gb_true = pd.read_csv(GROUND_TRUTH_GW_BANK_PATH)[["gw_id", "bank_id"]]
            df_eg_pred = pd.DataFrame(columns=["erp_order_id", "gateway_payment_id", "matching_stage"])
            df_gb_pred = pd.DataFrame(columns=["gateway_payment_id", "bank_entry_id", "matching_stage"])
        else:
            print(f"[!] Ground truth not found. Please run the simulation and db setup first. Error: {e}")
            return
    finally:
        conn.close()

    true_erp_gw = set(zip(df_eg_true['erp_id'], df_eg_true['gw_id']))
    true_gw_bnk = set(zip(df_gb_true['gw_id'], df_gb_true['bank_id']))

    pred_erp_gw = set(zip(df_eg_pred['erp_order_id'], df_eg_pred['gateway_payment_id']))
    pred_gw_bnk = set(zip(df_gb_pred['gateway_payment_id'], df_gb_pred['bank_entry_id']))

    metrics_table = []
    fp_rows = []
    layers = [
        ("Layer 1: ERP ↔ Gateway", true_erp_gw, pred_erp_gw,
         {tuple(row[["erp_order_id", "gateway_payment_id"]].tolist()): row["matching_stage"] for _, row in df_eg_pred.iterrows()}),
        ("Layer 2: Gateway ↔ Bank", true_gw_bnk, pred_gw_bnk,
         {tuple(row[["gateway_payment_id", "bank_entry_id"]].tolist()): row["matching_stage"] for _, row in df_gb_pred.iterrows()})
    ]

    for layer_name, true_set, pred_set, stage_lookup in layers:
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        p, r, f1 = calculate_metrics(tp, fp, fn)

        metrics_table.append([
            layer_name, len(true_set), len(pred_set),
            tp, fp, fn,
            f"{p:.1%}", f"{r:.1%}", f"{f1:.1%}"
        ])

        for left_id, right_id in sorted(pred_set - true_set):
            fp_rows.append({
                "Layer": layer_name,
                "Left": left_id,
                "Right": right_id,
                "Stage": stage_lookup.get((left_id, right_id), "Unknown")
            })

    print("=" * 100)
    print("  STRICT ID GRAPH ACCURACY EVALUATOR")
    print("=" * 100)
    headers = ["Reconciliation Graph Layer", "True Edges", "Pred Edges", "TP", "FP", "FN", "Precision", "Recall", "F1 Score"]
    print(tabulate(metrics_table, headers=headers, tablefmt="fancy_grid"))
    print("=" * 100)

    if fp_rows:
        print("FALSE POSITIVE EDGE DETAILS (with production stage)")
        fp_df = pd.DataFrame(fp_rows)
        print(tabulate(
            fp_df[["Layer", "Left", "Right", "Stage"]].values.tolist(),
            headers=["Layer", "Left", "Right", "Stage"],
            tablefmt="fancy_grid",
        ))
        print("=" * 100)


if __name__ == "__main__":
    main()

