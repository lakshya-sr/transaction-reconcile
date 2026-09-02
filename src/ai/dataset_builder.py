#!/usr/bin/env python3
"""
Dataset Builder for Residual XGBoost Cluster Matching.

Generates labeled cluster training samples (Positives & Hard Negatives)
across multiple simulation runs using candidate blocking and feature aggregation.
"""

from collections import defaultdict
from pathlib import Path
import random
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.block_generator import CandidateBlockGenerator
from src.ai.features import FEATURE_COLUMNS, extract_cluster_features
from src.simulation.generate_data import run_continuous_simulation

DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "train_features.csv"


def generate_training_samples(num_simulation_seeds: int = 4, days: int = 3) -> pd.DataFrame:
    all_positive_rows = []
    all_negative_rows = []
    block_gen = CandidateBlockGenerator(max_delay_days=5, max_amount_diff_pct=0.25)

    for seed in range(100, 100 + num_simulation_seeds):
        df_erp, df_gw, df_bank, df_eg, df_gb = run_continuous_simulation(days=days, seed=seed)

        gw_dict = {row["payment_id"]: row.to_dict() for _, row in df_gw.iterrows()}
        bank_dict = {row["bank_entry_id"]: row.to_dict() for _, row in df_bank.iterrows()}

        # Build true ground truth mapping: bank_id -> frozenset(gw_ids)
        true_gws_by_bank = defaultdict(set)
        for _, row in df_gb.iterrows():
            gw_id = str(row["gw_id"])
            bank_id = str(row["bank_id"])
            true_gws_by_bank[bank_id].add(gw_id)

        # 1. Positives (Label = 1)
        for bank_id, true_gw_ids in true_gws_by_bank.items():
            if bank_id not in bank_dict:
                continue
            gw_rows = [gw_dict[g_id] for g_id in true_gw_ids if g_id in gw_dict]
            if len(gw_rows) == len(true_gw_ids):
                feats = extract_cluster_features(gw_rows, bank_dict[bank_id])
                feats["label"] = 1
                feats["bank_id"] = bank_id
                feats["cluster_gw_ids"] = ",".join(sorted(true_gw_ids))
                all_positive_rows.append(feats)

        # 2. Hard Negatives (Label = 0 via candidate block generator)
        unmatched_gws = [r.to_dict() for _, r in df_gw.iterrows()]
        unmatched_banks = [r.to_dict() for _, r in df_bank.iterrows()]
        candidate_blocks = block_gen.generate_blocks(unmatched_gws, unmatched_banks)

        for block in candidate_blocks:
            b_id = block["bank_id"]
            cand_gw_set = set(block["gw_ids"])
            true_gw_set = true_gws_by_bank.get(b_id, set())

            # If the candidate block does NOT exactly match the true ground truth set, it is a negative
            if cand_gw_set != true_gw_set:
                feats = extract_cluster_features(block["gw_rows"], block["bank_row"])
                feats["label"] = 0
                feats["bank_id"] = b_id
                feats["cluster_gw_ids"] = ",".join(sorted(block["gw_ids"]))
                all_negative_rows.append(feats)

    df_pos = pd.DataFrame(all_positive_rows)
    df_neg = pd.DataFrame(all_negative_rows)

    if df_pos.empty:
        raise ValueError("No positive cluster samples generated.")

    # Balance negatives: ratio 3:1
    target_neg = min(len(df_neg), max(1, 3 * len(df_pos)))
    if len(df_neg) > target_neg:
        df_neg = df_neg.sample(n=target_neg, random_state=42).reset_index(drop=True)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False)

    print(f"[✔] Cluster training dataset saved to: {OUTPUT_PATH}")
    print(f"    - Total samples: {len(df_all)}")
    print(f"    - Positive clusters (1): {len(df_pos)}")
    print(f"    - Hard Negative clusters (0): {len(df_neg)}")
    return df_all


def main():
    generate_training_samples()


if __name__ == "__main__":
    main()
