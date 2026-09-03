#!/usr/bin/env python3
"""
Dataset Builder for ERP<->Gateway Residual XGBoost Cluster Matching.

Generates labeled cluster training samples (Positives & Hard Negatives)
across multiple simulation runs using ERP-GW candidate blocking and feature aggregation.

Adapted from dataset_builder.py (GW<->Bank). Key differences:
- Uses df_eg (ERP<->GW ground truth): columns erp_id, gw_id, erp_gw_amount
- Positives: true ERP clusters for each orphaned GW record
- Negatives: candidate blocks from ERPGWCandidateBlockGenerator that don't match truth
- Saves to data/raw/erp_gw_train_features.csv
"""

from collections import defaultdict
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.erp_gw_block_generator import ERPGWCandidateBlockGenerator
from src.ai.erp_gw_features import FEATURE_COLUMNS, extract_cluster_features
from src.simulation.generate_data import run_continuous_simulation

DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "erp_gw_train_features.csv"


def generate_training_samples(num_simulation_seeds: int = 4, days: int = 3) -> pd.DataFrame:
    all_positive_rows = []
    all_negative_rows = []
    block_gen = ERPGWCandidateBlockGenerator(max_delay_days=5, max_amount_diff_pct=0.25)

    for seed in range(200, 200 + num_simulation_seeds):
        df_erp, df_gw, df_bank, df_eg, df_gb = run_continuous_simulation(days=days, seed=seed)

        erp_dict = {row["erp_entry_id"]: row.to_dict() for _, row in df_erp.iterrows()}
        gw_dict = {row["payment_id"]: row.to_dict() for _, row in df_gw.iterrows()}

        # Build true ground truth mapping: gw_id -> frozenset(erp_ids)
        true_erps_by_gw = defaultdict(set)
        for _, row in df_eg.iterrows():
            erp_id = str(row["erp_id"])
            gw_id = str(row["gw_id"])
            true_erps_by_gw[gw_id].add(erp_id)

        # 1. Positives (Label = 1)
        for gw_id, true_erp_ids in true_erps_by_gw.items():
            if gw_id not in gw_dict:
                continue
            erp_rows = [erp_dict[e_id] for e_id in true_erp_ids if e_id in erp_dict]
            if len(erp_rows) == len(true_erp_ids):
                feats = extract_cluster_features(erp_rows, gw_dict[gw_id])
                feats["label"] = 1
                feats["gw_id"] = gw_id
                feats["cluster_erp_ids"] = ",".join(sorted(true_erp_ids))
                all_positive_rows.append(feats)

        # 2. Hard Negatives (Label = 0 via candidate block generator)
        unmatched_erps = [r.to_dict() for _, r in df_erp.iterrows()]
        unmatched_gws = [r.to_dict() for _, r in df_gw.iterrows()]
        candidate_blocks = block_gen.generate_blocks(unmatched_erps, unmatched_gws)

        for block in candidate_blocks:
            g_id = block["gw_id"]
            cand_erp_set = set(block["erp_ids"])
            true_erp_set = true_erps_by_gw.get(g_id, set())

            # If the candidate block does NOT exactly match the true ground truth, it's a negative
            if cand_erp_set != true_erp_set:
                feats = extract_cluster_features(block["erp_rows"], block["gw_row"])
                feats["label"] = 0
                feats["gw_id"] = g_id
                feats["cluster_erp_ids"] = ",".join(sorted(block["erp_ids"]))
                all_negative_rows.append(feats)

    df_pos = pd.DataFrame(all_positive_rows)
    df_neg = pd.DataFrame(all_negative_rows)

    if df_pos.empty:
        raise ValueError("No positive cluster samples generated. Check simulation data.")

    # Balance negatives: 3:1 ratio
    target_neg = min(len(df_neg), max(1, 3 * len(df_pos)))
    if len(df_neg) > target_neg:
        df_neg = df_neg.sample(n=target_neg, random_state=42).reset_index(drop=True)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False)

    print(f"[✔] ERP<->GW training dataset saved to: {OUTPUT_PATH}")
    print(f"    - Total samples: {len(df_all)}")
    print(f"    - Positive clusters (1): {len(df_pos)}")
    print(f"    - Hard Negative clusters (0): {len(df_neg)}")
    return df_all


def main():
    generate_training_samples()


if __name__ == "__main__":
    main()
