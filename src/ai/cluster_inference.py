#!/usr/bin/env python3
"""Post-fuzzy cluster scoring layer for residual Gateway↔Bank matches.

This pass is intentionally conservative: it only scores small, exact-sum
clusters in the residual pool after deterministic and fuzzy layers have
exhausted the easy matches. The model is applied to cluster-level evidence
(features aggregated per candidate cluster) rather than a naive pairwise scan.
"""

import itertools
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from rapidfuzz import fuzz
from xgboost import XGBClassifier

from src.core.config import (
    DB_PATH,
    MATCH_TYPE_CLUSTER,
    STAGE_CLUSTER_EXPAND_XGB,
    STAGE_CLUSTER_SEED_XGB,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank.json"
THRESHOLD_PATH = ROOT_DIR / "src" / "ai" / "artifacts" / "xgb_gw_bank_threshold.json"


def _token_similarity(left_value, right_value) -> float:
    left_text = "" if pd.isna(left_value) else str(left_value)
    right_text = "" if pd.isna(right_value) else str(right_value)
    left_norm = "".join(ch for ch in left_text.upper() if ch.isalnum())
    right_norm = "".join(ch for ch in right_text.upper() if ch.isalnum())
    if not left_norm or not right_norm:
        return 0.0
    return fuzz.ratio(left_norm, right_norm) / 100.0


def _candidate_clusters(engine) -> List[dict]:
    unmatched_gws = [
        row for _, row in engine.df_gateway.iterrows()
        if row["payment_id"] not in engine.matched_gw_payments
    ]
    unmatched_banks = [
        row for _, row in engine.df_bank.iterrows()
        if row["bank_entry_id"] not in engine.matched_bank_entries
    ]

    if not unmatched_gws or not unmatched_banks:
        return []

    candidates: List[dict] = []
    for bank_row in unmatched_banks:
        bank_id = bank_row["bank_entry_id"]
        bank_amt = float(bank_row["credit_amount"])
        if pd.isna(bank_row.get("value_date")):
            continue
        bank_dt = pd.to_datetime(bank_row["value_date"], errors="coerce")
        if pd.isna(bank_dt):
            continue

        valid_gws = []
        for gw_row in unmatched_gws:
            gw_id = gw_row["payment_id"]
            if gw_id in engine.matched_gw_payments:
                continue
            if pd.isna(gw_row.get("settled_at")):
                continue
            gw_dt = pd.to_datetime(gw_row["settled_at"], errors="coerce")
            if pd.isna(gw_dt):
                continue
            gw_amt = float(gw_row["net_settled"])
            delta_days = abs((gw_dt - bank_dt).total_seconds() / 86400.0)
            if delta_days > 7:
                continue
            if abs(gw_amt - bank_amt) > max(10.0, 0.15 * bank_amt):
                continue
            valid_gws.append(gw_row)

        if not valid_gws:
            continue

        max_size = min(4, len(valid_gws))
        for subset_size in range(1, max_size + 1):
            for combo in itertools.combinations(valid_gws, subset_size):
                subset_total = sum(float(item["net_settled"]) for item in combo)
                amount_gap = abs(subset_total - bank_amt)
                if amount_gap > max(2.0, 0.03 * bank_amt):
                    continue

                bank_tokens = set()
                remittance = str(bank_row.get("remittance_info") or "")
                bank_tokens.add(remittance.upper())
                for token in [
                    part.strip() for part in remittance.replace("_", " ").replace("-", " ").split() if part.strip()
                ]:
                    bank_tokens.add(token.upper())

                gw_tokens = []
                for gw_row in combo:
                    gw_utr = str(gw_row.get("bank_utr") or "")
                    gw_invoices = []
                    if isinstance(gw_row.get("invoices"), str):
                        gw_invoices = [entry.strip() for entry in gw_row["invoices"].replace("[", "").replace("]", "").replace("\"", "").split(",") if entry.strip()]
                    if gw_utr:
                        gw_tokens.append(gw_utr)
                    gw_tokens.extend(gw_invoices)

                similarity = 0.0
                if gw_tokens:
                    similarity = max(_token_similarity(token, remittance) for token in gw_tokens)
                time_delta_hours = abs(
                    (pd.to_datetime(combo[0]["settled_at"], errors="coerce") - bank_dt).total_seconds() / 3600.0
                ) if combo else 0.0
                cluster_feature = {
                    "amount_diff": amount_gap,
                    "time_delta_hours": time_delta_hours,
                    "utr_fuzz_ratio": similarity,
                    "cluster_size": len(combo),
                    "bank_amount": bank_amt,
                    "cluster_total": subset_total,
                }
                candidates.append({
                    "bank_id": bank_id,
                    "bank_amount": bank_amt,
                    "bank_date": bank_dt,
                    "subset": combo,
                    "cluster_total": subset_total,
                    "amount_gap": amount_gap,
                    "feature": cluster_feature,
                })

    return candidates


def _score_cluster(model: XGBClassifier, cluster_feature: dict) -> float:
    item = pd.DataFrame([
        {
            "amount_diff": float(cluster_feature["amount_diff"]),
            "time_delta_hours": float(cluster_feature["time_delta_hours"]),
            "utr_fuzz_ratio": float(cluster_feature["utr_fuzz_ratio"]),
        }
    ])
    return float(model.predict_proba(item)[0, 1])


def apply_cluster_xgb_after_fuzzy(engine) -> None:
    """Run a conservative cluster-level XGBoost pass on the residual unmatched pool."""
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        return

    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")).get("threshold", 0.85)
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    candidate_clusters = _candidate_clusters(engine)
    if not candidate_clusters:
        return

    scored_clusters = []
    for cluster in candidate_clusters:
        prob = _score_cluster(model, cluster["feature"])
        if prob < float(threshold):
            continue
        if abs(cluster["cluster_total"] - cluster["bank_amount"]) > 1e-3:
            continue
        scored_clusters.append((prob, cluster))

    if not scored_clusters:
        return

    scored_clusters.sort(key=lambda x: x[0], reverse=True)
    claimed_bank_ids = set()
    claimed_gw_ids = set()
    for prob, cluster in scored_clusters:
        bank_id = cluster["bank_id"]
        subset = cluster["subset"]
        subset_ids = {item["payment_id"] for item in subset}
        if bank_id in claimed_bank_ids or subset_ids & claimed_gw_ids:
            continue

        claimed_bank_ids.add(bank_id)
        claimed_gw_ids |= subset_ids
        for gw_row in subset:
            gw_id = gw_row["payment_id"]
            if gw_id in engine.matched_gw_payments:
                continue
            engine.matched_gw_payments.add(gw_id)
            engine.matched_bank_entries.add(bank_id)
            stage = STAGE_CLUSTER_SEED_XGB if len(subset) <= 1 else STAGE_CLUSTER_EXPAND_XGB
            engine._add_gw_bank_link(
                gw_id,
                bank_id,
                gw_row.get("bank_utr"),
                MATCH_TYPE_CLUSTER,
                stage,
                float(prob),
                f"Cluster-XGB residual match ({len(subset)}-item cluster, prob={prob:.3f}).",
            )
