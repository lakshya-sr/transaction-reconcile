#!/usr/bin/env python3
"""Appends AI residual Gateway↔Bank matches to the database without clearing deterministic results."""

import json
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import DB_PATH
from src.core.database import fetch_table_df, save_graph_edges

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank.json"
THRESHOLD_PATH = ROOT_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank_threshold.json"
FEATURE_COLUMNS = ["amount_diff", "time_delta_hours", "utr_fuzz_ratio"]
AMOUNT_MATCH_TOLERANCE = 0.01


def _enforce_amount_cluster_match(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected

    gw_lookup = fetch_table_df("gateway_settlements", DB_PATH)[["payment_id", "net_settled", "settled_at", "bank_utr"]].rename(columns={"payment_id": "gateway_payment_id"})
    bank_lookup = fetch_table_df("bank_statement", DB_PATH)[["bank_entry_id", "credit_amount", "value_date", "remittance_info"]].copy()

    working = selected.rename(columns={"net_settled": "gw_amount", "credit_amount": "bank_amount"})
    validated = working.merge(gw_lookup, on="gateway_payment_id", how="left", suffixes=("", "_gw")).merge(
        bank_lookup, on="bank_entry_id", how="left", suffixes=("", "_bank")
    )
    validated = validated[validated["net_settled"].notna() & validated["credit_amount"].notna()].copy()

    bank_totals = validated.groupby("bank_entry_id", as_index=False).agg(
        bank_total=("gw_amount", "sum"),
        bank_expected=("credit_amount", "first"),
    )
    valid_bank_ids = bank_totals.loc[
        (bank_totals["bank_total"] - bank_totals["bank_expected"]).abs() <= AMOUNT_MATCH_TOLERANCE,
        "bank_entry_id",
    ]
    validated = validated[validated["bank_entry_id"].isin(valid_bank_ids)]

    gw_totals = validated.groupby("gateway_payment_id", as_index=False).agg(
        gw_total=("bank_amount", "sum"),
        gw_expected=("net_settled", "first"),
    )
    valid_gw_ids = gw_totals.loc[
        (gw_totals["gw_total"] - gw_totals["gw_expected"]).abs() <= AMOUNT_MATCH_TOLERANCE,
        "gateway_payment_id",
    ]
    validated = validated[validated["gateway_payment_id"].isin(valid_gw_ids)]

    return validated[["gateway_payment_id", "bank_entry_id", "gw_amount", "bank_amount", "bank_utr", "remittance_info", "probability"]].rename(
        columns={"gw_amount": "net_settled", "bank_amount": "credit_amount"}
    ).copy()


def _candidate_pairs_for_inference() -> pd.DataFrame:
    gw_df = fetch_table_df("gateway_settlements", DB_PATH)
    bank_df = fetch_table_df("bank_statement", DB_PATH)
    pred_df = fetch_table_df("gw_bank_pred", DB_PATH)

    if gw_df.empty or bank_df.empty:
        return pd.DataFrame(columns=["gateway_payment_id", "bank_entry_id", "amount_diff", "time_delta_hours", "utr_fuzz_ratio"])

    predicted_gateway_ids = set(pred_df["gateway_payment_id"].astype(str).dropna()) if not pred_df.empty else set()
    predicted_bank_ids = set(pred_df["bank_entry_id"].astype(str).dropna()) if not pred_df.empty else set()

    orphan_gw = gw_df.loc[~gw_df["payment_id"].astype(str).isin(predicted_gateway_ids), ["payment_id", "net_settled", "settled_at", "bank_utr"]].copy()
    orphan_bank = bank_df.loc[~bank_df["bank_entry_id"].astype(str).isin(predicted_bank_ids), ["bank_entry_id", "credit_amount", "value_date", "remittance_info"]].copy()

    if orphan_gw.empty or orphan_bank.empty:
        return pd.DataFrame(columns=["gateway_payment_id", "bank_entry_id", "amount_diff", "time_delta_hours", "utr_fuzz_ratio"])

    orphan_gw = orphan_gw.rename(columns={"payment_id": "gateway_payment_id"})
    candidates = orphan_gw.merge(orphan_bank, how="cross").copy()
    candidates["gw_settled_at"] = pd.to_datetime(candidates["settled_at"], errors="coerce")
    candidates["bank_value_date"] = pd.to_datetime(candidates["value_date"], errors="coerce")
    candidates = candidates[
        candidates["gw_settled_at"].notna()
        & candidates["bank_value_date"].notna()
        & (candidates["bank_value_date"] >= candidates["gw_settled_at"])
        & (candidates["bank_value_date"] <= candidates["gw_settled_at"] + pd.Timedelta(days=3))
    ].copy()

    if candidates.empty:
        return pd.DataFrame(columns=["gateway_payment_id", "bank_entry_id", "amount_diff", "time_delta_hours", "utr_fuzz_ratio"])

    candidates["amount_diff"] = (candidates["net_settled"] - candidates["credit_amount"]).abs()
    candidates["time_delta_hours"] = (candidates["bank_value_date"] - candidates["gw_settled_at"]).dt.total_seconds() / 3600.0
    candidates["utr_fuzz_ratio"] = candidates.apply(
        lambda row: 0.0 if pd.isna(row.get("bank_utr")) and pd.isna(row.get("remittance_info")) else (100.0 if str(row.get("bank_utr", "")) == str(row.get("remittance_info", "")) else 0.0),
        axis=1,
    )

    return candidates[["gateway_payment_id", "bank_entry_id", "amount_diff", "time_delta_hours", "utr_fuzz_ratio"]].copy()


def _ratio_score(left_value, right_value) -> float:
    left_text = "" if pd.isna(left_value) else str(left_value)
    right_text = "" if pd.isna(right_value) else str(right_value)
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(left_text, right_text))
    except Exception:
        return 0.0


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold metadata not found: {THRESHOLD_PATH}")

    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"]
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    candidates = _candidate_pairs_for_inference()
    if candidates.empty:
        print("[✔] No inference candidates remain after deterministic filtering.")
        return

    gw_lookup = fetch_table_df("gateway_settlements", DB_PATH)[["payment_id", "net_settled", "settled_at", "bank_utr"]].rename(columns={"payment_id": "gateway_payment_id"})
    bank_lookup = fetch_table_df("bank_statement", DB_PATH)[["bank_entry_id", "credit_amount", "value_date", "remittance_info"]].copy()
    merged = candidates.merge(gw_lookup, on="gateway_payment_id", how="left").merge(bank_lookup, on="bank_entry_id", how="left")
    merged["amount_diff"] = (merged["net_settled"] - merged["credit_amount"]).abs()
    merged["time_delta_hours"] = (pd.to_datetime(merged["value_date"], errors="coerce") - pd.to_datetime(merged["settled_at"], errors="coerce")).dt.total_seconds() / 3600.0
    merged["utr_fuzz_ratio"] = merged.apply(
        lambda row: _ratio_score(row.get("bank_utr"), row.get("remittance_info")),
        axis=1,
    )

    feature_frame = merged[[*FEATURE_COLUMNS]].copy()
    probabilities = model.predict_proba(feature_frame)[:, 1]
    selected = merged.loc[probabilities >= float(threshold), ["gateway_payment_id", "bank_entry_id", "net_settled", "credit_amount", "bank_utr", "remittance_info"]].copy()
    selected["probability"] = probabilities[probabilities >= float(threshold)]

    selected = _enforce_amount_cluster_match(selected)
    if selected.empty:
        print("[✔] No AI predictions met the optimized threshold and amount-consistency check.")
        return

    ai_edges = []
    for _, row in selected.iterrows():
        ai_edges.append({
            "gateway_payment_id": row["gateway_payment_id"],
            "bank_entry_id": row["bank_entry_id"],
            "allocated_amount": float(row["net_settled"] if pd.notna(row["net_settled"]) else row["credit_amount"]),
            "match_type": "AI_PREDICTION",
            "matching_stage": "XGBoost_Pass_1",
            "confidence_score": float(row["probability"]),
            "notes": "AI residual prediction via XGBoost",
        })

    save_graph_edges([], ai_edges, DB_PATH)
    print(f"[✔] Saved {len(ai_edges)} AI residual Gateway↔Bank edges to the database.")


if __name__ == "__main__":
    main()
