#!/usr/bin/env python3
"""
Dataset Builder for both GW↔Bank and ERP↔GW XGBoost models.

Generates labeled training samples with positives and hard negatives
across multiple simulation runs.
"""

from collections import defaultdict
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai.candidate_generation import (
    GatewayBankCandidateGenerator,
    ERPGatewayCandidateGenerator,
)
from src.ai.features import (
    extract_gateway_bank_features,
    extract_erp_gateway_features,
)
from src.simulation.generate_data import run_continuous_simulation

DATA_DIR = ROOT_DIR / "data" / "raw"
GATEWAY_BANK_DATASET_PATH = DATA_DIR / "train_features.csv"
ERP_GATEWAY_DATASET_PATH = DATA_DIR / "erp_gw_train_features.csv"


def _balance_dataset(df_pos: pd.DataFrame, df_neg: pd.DataFrame, neg_ratio: int = 3) -> pd.DataFrame:
    """Balance positive and negative samples with given ratio."""
    if df_pos.empty:
        raise ValueError("No positive samples generated.")
    
    target_neg = min(len(df_neg), max(1, neg_ratio * len(df_pos)))
    if len(df_neg) > target_neg:
        df_neg = df_neg.sample(n=target_neg, random_state=42).reset_index(drop=True)
    
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return df_all


def generate_gateway_bank_dataset(
    num_simulation_seeds: int = 4,
    days: int = 3,
) -> pd.DataFrame:
    """
    Generate training dataset for GW↔Bank matching.
    
    Args:
        num_simulation_seeds: Number of simulation runs with different seeds.
        days: Number of days to simulate per run.
    
    Returns:
        DataFrame with labeled training samples.
    """
    all_positive_rows = []
    all_negative_rows = []
    block_gen = GatewayBankCandidateGenerator(max_delay_days=5, max_amount_diff_pct=0.25)

    for seed in range(100, 100 + num_simulation_seeds):
        print(f"[*] Simulating seed {seed}...")
        df_erp, df_gw, df_bank, df_eg, df_gb = run_continuous_simulation(days=days, seed=seed)

        gw_dict = {row["payment_id"]: row.to_dict() for _, row in df_gw.iterrows()}
        bank_dict = {row["bank_entry_id"]: row.to_dict() for _, row in df_bank.iterrows()}

        # Build ground truth mapping: bank_id -> set(gw_ids)
        true_gws_by_bank = defaultdict(set)
        for _, row in df_gb.iterrows():
            gw_id = str(row["gw_id"])
            bank_id = str(row["bank_id"])
            true_gws_by_bank[bank_id].add(gw_id)

        # 1. Generate positives (Label = 1)
        for bank_id, true_gw_ids in true_gws_by_bank.items():
            if bank_id not in bank_dict:
                continue
            gw_rows = [gw_dict[g_id] for g_id in true_gw_ids if g_id in gw_dict]
            if len(gw_rows) == len(true_gw_ids):
                feats = extract_gateway_bank_features(gw_rows, bank_dict[bank_id])
                feats["label"] = 1
                feats["bank_id"] = bank_id
                feats["cluster_gw_ids"] = ",".join(sorted(true_gw_ids))
                all_positive_rows.append(feats)

        # 2. Generate hard negatives (Label = 0)
        unmatched_gws = [r.to_dict() for _, r in df_gw.iterrows()]
        unmatched_banks = [r.to_dict() for _, r in df_bank.iterrows()]
        candidate_blocks = block_gen.generate(unmatched_gws, unmatched_banks)

        for block in candidate_blocks:
            b_id = block["bank_id"]
            cand_gw_set = set(block["gw_ids"])
            true_gw_set = true_gws_by_bank.get(b_id, set())

            # If candidate doesn't match ground truth, it's a negative
            if cand_gw_set != true_gw_set:
                feats = extract_gateway_bank_features(block["gw_rows"], block["bank_row"])
                feats["label"] = 0
                feats["bank_id"] = b_id
                feats["cluster_gw_ids"] = ",".join(sorted(block["gw_ids"]))
                all_negative_rows.append(feats)

    # Balance and combine
    df_pos = pd.DataFrame(all_positive_rows)
    df_neg = pd.DataFrame(all_negative_rows)
    df_all = _balance_dataset(df_pos, df_neg, neg_ratio=3)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(GATEWAY_BANK_DATASET_PATH, index=False)

    print(f"\n[✔] Gateway↔Bank dataset saved: {GATEWAY_BANK_DATASET_PATH}")
    print(f"    - Total samples: {len(df_all)}")
    print(f"    - Positives (1): {len(df_pos)}")
    print(f"    - Negatives (0): {len(df_neg)}")
    return df_all


def generate_erp_gateway_dataset(
    num_simulation_seeds: int = 4,
    days: int = 3,
) -> pd.DataFrame:
    """
    Generate training dataset for ERP↔GW matching.
    
    Args:
        num_simulation_seeds: Number of simulation runs with different seeds.
        days: Number of days to simulate per run.
    
    Returns:
        DataFrame with labeled training samples.
    """
    all_positive_rows = []
    all_negative_rows = []
    block_gen = ERPGatewayCandidateGenerator(max_delay_days=5, max_amount_diff_pct=0.25)

    for seed in range(200, 200 + num_simulation_seeds):
        print(f"[*] Simulating seed {seed}...")
        df_erp, df_gw, df_bank, df_eg, df_gb = run_continuous_simulation(days=days, seed=seed)

        erp_dict = {row["erp_entry_id"]: row.to_dict() for _, row in df_erp.iterrows()}
        gw_dict = {row["payment_id"]: row.to_dict() for _, row in df_gw.iterrows()}

        # Build ground truth mapping: gw_id -> set(erp_ids)
        true_erps_by_gw = defaultdict(set)
        for _, row in df_eg.iterrows():
            erp_id = str(row["erp_id"])
            gw_id = str(row["gw_id"])
            true_erps_by_gw[gw_id].add(erp_id)

        # 1. Generate positives (Label = 1)
        for gw_id, true_erp_ids in true_erps_by_gw.items():
            if gw_id not in gw_dict:
                continue
            erp_rows = [erp_dict[e_id] for e_id in true_erp_ids if e_id in erp_dict]
            if len(erp_rows) == len(true_erp_ids):
                feats = extract_erp_gateway_features(erp_rows, gw_dict[gw_id])
                feats["label"] = 1
                feats["gw_id"] = gw_id
                feats["cluster_erp_ids"] = ",".join(sorted(true_erp_ids))
                all_positive_rows.append(feats)

        # 2. Generate hard negatives (Label = 0)
        unmatched_erps = [r.to_dict() for _, r in df_erp.iterrows()]
        unmatched_gws = [r.to_dict() for _, r in df_gw.iterrows()]
        candidate_blocks = block_gen.generate(unmatched_erps, unmatched_gws)

        for block in candidate_blocks:
            g_id = block["gw_id"]
            cand_erp_set = set(block["erp_ids"])
            true_erp_set = true_erps_by_gw.get(g_id, set())

            # If candidate doesn't match ground truth, it's a negative
            if cand_erp_set != true_erp_set:
                feats = extract_erp_gateway_features(block["erp_rows"], block["gw_row"])
                feats["label"] = 0
                feats["gw_id"] = g_id
                feats["cluster_erp_ids"] = ",".join(sorted(block["erp_ids"]))
                all_negative_rows.append(feats)

    # Balance and combine
    df_pos = pd.DataFrame(all_positive_rows)
    df_neg = pd.DataFrame(all_negative_rows)
    df_all = _balance_dataset(df_pos, df_neg, neg_ratio=3)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(ERP_GATEWAY_DATASET_PATH, index=False)

    print(f"\n[✔] ERP↔Gateway dataset saved: {ERP_GATEWAY_DATASET_PATH}")
    print(f"    - Total samples: {len(df_all)}")
    print(f"    - Positives (1): {len(df_pos)}")
    print(f"    - Negatives (0): {len(df_neg)}")
    return df_all


def main():
    """Generate both datasets."""
    print("=" * 60)
    print("  GENERATING GATEWAY↔BANK TRAINING DATASET")
    print("=" * 60)
    generate_gateway_bank_dataset()
    
    print("\n" + "=" * 60)
    print("  GENERATING ERP↔GATEWAY TRAINING DATASET")
    print("=" * 60)
    generate_erp_gateway_dataset()


if __name__ == "__main__":
    main()