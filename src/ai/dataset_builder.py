#!/usr/bin/env python3
"""Build residual Gateway↔Bank training data for the Hard-Residual XGBoost model."""

import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import DB_PATH
from src.core.database import fetch_table_df

DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "train_features.csv"

FEATURE_COLUMNS = ["amount_diff", "time_delta_hours", "utr_fuzz_ratio"]


def _safe_ratio(left_value, right_value) -> float:
    left_text = "" if pd.isna(left_value) else str(left_value)
    right_text = "" if pd.isna(right_value) else str(right_value)
    return float(fuzz.ratio(left_text, right_text))


def build_residual_dataset(db_path: Path = DB_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    gw_df = fetch_table_df("gateway_settlements", db_path)
    bank_df = fetch_table_df("bank_statement", db_path)
    true_df = fetch_table_df("gw_bank_true", db_path)
    pred_df = fetch_table_df("gw_bank_pred", db_path)

    if gw_df.empty or bank_df.empty:
        raise ValueError("Gateway or bank tables are empty; run the deterministic baseline first.")

    predicted_gateway_ids = set(pred_df["gateway_payment_id"].astype(str).dropna()) if not pred_df.empty else set()
    predicted_bank_ids = set(pred_df["bank_entry_id"].astype(str).dropna()) if not pred_df.empty else set()

    orphan_gw = gw_df.loc[~gw_df["payment_id"].astype(str).isin(predicted_gateway_ids), ["payment_id", "net_settled", "settled_at", "bank_utr"]].copy()
    orphan_bank = bank_df.loc[~bank_df["bank_entry_id"].astype(str).isin(predicted_bank_ids), ["bank_entry_id", "credit_amount", "value_date", "remittance_info"]].copy()

    if orphan_gw.empty or orphan_bank.empty:
        raise ValueError("No orphaned Gateway/Bank records remain after filtering resolved matches.")

    orphan_gw = orphan_gw.rename(columns={"payment_id": "gateway_payment_id"})
    orphan_bank = orphan_bank.rename(columns={"bank_entry_id": "bank_entry_id"})

    true_pairs = true_df[["gw_id", "bank_id"]].rename(columns={"gw_id": "gateway_payment_id", "bank_id": "bank_entry_id"}).copy()
    true_pair_keys = set(zip(true_pairs["gateway_payment_id"].astype(str), true_pairs["bank_entry_id"].astype(str)))

    positive = (
        true_pairs.merge(orphan_gw, on="gateway_payment_id", how="inner")
        .merge(orphan_bank, on="bank_entry_id", how="inner")
        .copy()
    )

    if positive.empty:
        raise ValueError("No positive residual pairs were found in the ground-truth data for this run.")

    positive["gw_settled_at"] = pd.to_datetime(positive["settled_at"], errors="coerce")
    positive["bank_value_date"] = pd.to_datetime(positive["value_date"], errors="coerce")
    positive["amount_diff"] = (positive["net_settled"] - positive["credit_amount"]).abs()
    positive["time_delta_hours"] = (positive["bank_value_date"] - positive["gw_settled_at"]).dt.total_seconds() / 3600.0
    positive["utr_fuzz_ratio"] = positive.apply(
        lambda row: _safe_ratio(row.get("bank_utr"), row.get("remittance_info")),
        axis=1,
    )
    positive["label"] = 1

    gw_candidates = orphan_gw.copy()
    bank_candidates = orphan_bank.copy()
    negative = gw_candidates.merge(bank_candidates, how="cross").copy()
    negative["gw_settled_at"] = pd.to_datetime(negative["settled_at"], errors="coerce")
    negative["bank_value_date"] = pd.to_datetime(negative["value_date"], errors="coerce")
    negative = negative[
        negative["gw_settled_at"].notna()
        & negative["bank_value_date"].notna()
        & (negative["bank_value_date"] >= negative["gw_settled_at"])
        & (negative["bank_value_date"] <= negative["gw_settled_at"] + pd.Timedelta(days=3))
    ].copy()

    negative["pair_key"] = list(zip(negative["gateway_payment_id"].astype(str), negative["bank_entry_id"].astype(str)))
    negative = negative[~negative["pair_key"].isin(true_pair_keys)].copy()

    if negative.empty:
        raise ValueError("No valid negative residual candidates remain after filtering by the 3-day window.")

    negative["amount_diff"] = (negative["net_settled"] - negative["credit_amount"]).abs()
    negative["time_delta_hours"] = (negative["bank_value_date"] - negative["gw_settled_at"]).dt.total_seconds() / 3600.0
    negative["utr_fuzz_ratio"] = negative.apply(
        lambda row: _safe_ratio(row.get("bank_utr"), row.get("remittance_info")),
        axis=1,
    )
    negative["label"] = 0

    target_negatives = max(3 * len(positive), 1)
    if len(negative) > target_negatives:
        negative = negative.sample(n=target_negatives, random_state=42).reset_index(drop=True)

    dataset = pd.concat([positive, negative], ignore_index=True, sort=False)
    dataset = dataset[["gateway_payment_id", "bank_entry_id", *FEATURE_COLUMNS, "label"]].copy()
    dataset[FEATURE_COLUMNS] = dataset[FEATURE_COLUMNS].astype(float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"[✔] Residual dataset saved to: {output_path}")
    print(f"    - Positive samples: {len(positive)}")
    print(f"    - Negative samples: {len(negative)}")
    return dataset


def main():
    build_residual_dataset()


if __name__ == "__main__":
    main()
