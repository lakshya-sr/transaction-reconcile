#!/usr/bin/env python3
"""Separate fuzzy-matching layer for residual reconciliation cases.

This layer is intentionally more permissive than the deterministic engine, but
still guarded by a precise uniqueness and evidence threshold so it does not
explode the false-positive rate.
"""

import ast
import re
from typing import List, Optional

import pandas as pd
from rapidfuzz import fuzz

from src.core.config import (
    MATCH_TYPE_FUZZY,
    STAGE_FUZZY_ERP_GW,
    STAGE_FUZZY_GW_BANK,
)


def parse_invoices(val) -> List[str]:
    if pd.isna(val) or not val:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('['):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except (ValueError, SyntaxError):
                pass
        return [val.replace('"', '').replace("'", "")]
    return []


def normalize_ref(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def invoice_candidates_from_text(text: str) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"INV[-_]?([A-Z0-9]{4,})|ORD[-_]?([A-Z0-9]{4,})", str(text).upper())
    results = []
    for pair in matches:
        for part in pair:
            if part:
                results.append(part)
    return results


def score_amount_similarity(gw_amount: float, bank_amount: float) -> float:
    if gw_amount <= 0 or bank_amount <= 0:
        return 0.0
    diff = abs(gw_amount - bank_amount)
    tolerance = max(2.0, 0.05 * max(gw_amount, bank_amount))
    if diff <= 0.01:
        return 1.0
    return max(0.0, 1.0 - (diff / tolerance))


def score_date_similarity(gw_dt, bank_dt) -> float:
    if gw_dt is None or bank_dt is None:
        return 0.0
    delta_days = abs((pd.to_datetime(gw_dt) - pd.to_datetime(bank_dt)).total_seconds() / 86400.0)
    if delta_days <= 1:
        return 1.0
    if delta_days <= 3:
        return 0.7
    if delta_days <= 5:
        return 0.4
    return 0.0


def strong_token_similarity(gw_row, bank_row) -> float:
    remittance = str(bank_row.get("remittance_info", "") or "").upper()
    gw_utr = normalize_ref(gw_row.get("bank_utr"))
    invoice_tokens = [normalize_ref(v) for v in parse_invoices(gw_row.get("invoices"))]
    bank_tokens = set(invoice_candidates_from_text(remittance))
    bank_tokens |= {normalize_ref(x) for x in re.findall(r"UTR[A-Z0-9]+", remittance)}

    token_score = 0.0
    if gw_utr:
        if gw_utr in remittance:
            token_score = max(token_score, 1.0)
        elif len(gw_utr) >= 8 and (gw_utr[-8:] in remittance or gw_utr[:8] in remittance):
            token_score = max(token_score, 0.85)
        elif any(gw_utr in token for token in bank_tokens):
            token_score = max(token_score, 0.75)
        else:
            token_score = max(token_score, fuzz.ratio(gw_utr, normalize_ref(remittance)) / 100.0)

    invoice_score = 0.0
    for inv in invoice_tokens:
        if not inv:
            continue
        if inv in remittance:
            invoice_score = max(invoice_score, 1.0)
        elif any(inv in token for token in bank_tokens):
            invoice_score = max(invoice_score, 0.8)
        else:
            best = 0.0
            for token in bank_tokens:
                if not token:
                    continue
                best = max(best, fuzz.ratio(inv, token) / 100.0)
            invoice_score = max(invoice_score, best)

    return max(token_score, invoice_score)


def apply_fuzzy_layer(engine) -> None:
    """Run fuzzy residual matching on unresolved Gateway↔Bank and ERP↔Gateway edges."""
    apply_fuzzy_gateway_to_bank(engine)
    apply_fuzzy_erp_to_gateway(engine)


def apply_fuzzy_gateway_to_bank(engine) -> None:
    unmatched_gws = [
        row for _, row in engine.df_gateway.iterrows()
        if row["payment_id"] not in engine.matched_gw_payments
    ]
    unmatched_banks = [
        row for _, row in engine.df_bank.iterrows()
        if row["bank_entry_id"] not in engine.matched_bank_entries
    ]

    if not unmatched_gws or not unmatched_banks:
        return

    candidate_subsets = []
    for bank_row in unmatched_banks:
        bank_id = bank_row["bank_entry_id"]
        bank_amt = float(bank_row["credit_amount"])
        bank_dt = pd.to_datetime(bank_row["value_date"])

        valid_candidates = []
        for gw_row in unmatched_gws:
            gw_id = gw_row["payment_id"]
            if gw_id in engine.matched_gw_payments:
                continue
            gw_amt = float(gw_row["net_settled"])
            gw_dt = pd.to_datetime(gw_row["settled_at"])
            if abs((gw_dt - bank_dt).total_seconds()) > 7 * 86400:
                continue
            if abs(gw_amt - bank_amt) > max(10.0, 0.15 * bank_amt):
                continue
            valid_candidates.append(gw_row)

        if not valid_candidates:
            continue

        best_subset = None
        best_score = -1.0
        for subset_size in range(1, min(4, len(valid_candidates)) + 1):
            for combo in __import__('itertools').combinations(valid_candidates, subset_size):
                subset_total = sum(float(item["net_settled"]) for item in combo)
                amt_gap = abs(subset_total - bank_amt)
                if amt_gap > max(5.0, 0.03 * bank_amt):
                    continue
                token_scores = [strong_token_similarity(item, bank_row) for item in combo]
                date_scores = [score_date_similarity(pd.to_datetime(item["settled_at"]), bank_dt) for item in combo]
                amount_score = score_amount_similarity(subset_total, bank_amt)
                cluster_token = max(token_scores) if token_scores else 0.0
                cluster_date = sum(date_scores) / len(date_scores) if date_scores else 0.0
                cluster_score = 0.55 * amount_score + 0.30 * cluster_token + 0.15 * cluster_date
                if cluster_score >= 0.74 and (amount_score >= 0.55 or cluster_token >= 0.60):
                    if best_subset is None or cluster_score > best_score:
                        best_subset = combo
                        best_score = cluster_score

        if best_subset is not None:
            candidate_subsets.append((best_score, bank_id, best_subset))

    # Keep only the strongest unique batch assignment for each bank and ensure unique
    # gateway usage across accepted fuzzy clusters.
    claimed_gw_ids = set()
    claimed_bank_ids = set()
    for score, bank_id, subset in sorted(candidate_subsets, key=lambda x: x[0], reverse=True):
        subset_ids = {item["payment_id"] for item in subset}
        if bank_id in claimed_bank_ids or subset_ids & claimed_gw_ids:
            continue
        if len(subset_ids) > 1 and len(subset_ids) > 4:
            continue
        claimed_bank_ids.add(bank_id)
        claimed_gw_ids |= subset_ids
        for gw_row in subset:
            gw_id = gw_row["payment_id"]
            if gw_id in engine.matched_gw_payments:
                continue
            engine.matched_gw_payments.add(gw_id)
            engine.matched_bank_entries.add(bank_id)
            engine._add_gw_bank_link(
                gw_id,
                bank_id,
                gw_row.get("bank_utr"),
                MATCH_TYPE_FUZZY,
                STAGE_FUZZY_GW_BANK,
                float(score),
                f"Fuzzy residual subset clustering: sum-delta={abs(sum(float(item['net_settled']) for item in subset) - float(engine.df_bank[engine.df_bank['bank_entry_id'] == bank_id].iloc[0]['credit_amount'])):.2f}",
            )


def apply_fuzzy_erp_to_gateway(engine) -> None:
    unmatched_gws = [row for _, row in engine.df_gateway.iterrows() if row["payment_id"] not in engine.gw_to_erp_links]
    unmatched_erps = [row for _, row in engine.df_erp.iterrows() if row["erp_entry_id"] not in engine.matched_erp_entries]

    if not unmatched_gws or not unmatched_erps:
        return

    for gw_row in unmatched_gws:
        gw_id = gw_row["payment_id"]
        if gw_id in engine.gw_to_erp_links:
            continue

        gw_gross = float(gw_row["gross_amount"])
        invoices = parse_invoices(gw_row.get("invoices"))
        remittance_ctx = ""
        linked_bank = engine.gw_bank_links.get(gw_id)
        if linked_bank:
            bank_rows = engine.df_bank[engine.df_bank["bank_entry_id"].isin(linked_bank["bank_ids"])]
            remittance_ctx = " ".join(bank_rows["remittance_info"].fillna("").astype(str).tolist())

        best_pair = None
        for erp_row in unmatched_erps:
            erp_id = erp_row["erp_entry_id"]
            if erp_id in engine.matched_erp_entries:
                continue
            erp_gross = float(erp_row["gross_amount"])
            invoice_num = str(erp_row["invoice_number"] or "").upper()
            amount_score = score_amount_similarity(gw_gross, erp_gross)
            invoice_score = 0.0
            if invoice_num:
                invoice_score = max(
                    fuzz.ratio(invoice_num, " ".join([normalize_ref(i) for i in invoices])) / 100.0,
                    fuzz.ratio(invoice_num.replace("INV-", ""), normalize_ref(remittance_ctx)) / 100.0,
                )
            if not invoices and not remittance_ctx:
                invoice_score = 0.0
            score = 0.7 * amount_score + 0.3 * invoice_score
            if score >= 0.72 and (amount_score >= 0.60 or invoice_score >= 0.60):
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, erp_id)

        if best_pair is not None:
            score, erp_id = best_pair
            engine.matched_erp_entries.add(erp_id)
            engine.matched_gw_payments.add(gw_id)
            engine.gw_to_erp_links.setdefault(gw_id, []).append(erp_id)
            engine.erp_gw_stage_map.setdefault(gw_id, {})[erp_id] = STAGE_FUZZY_ERP_GW
            if gw_id in engine.gw_bank_links:
                engine.gw_bank_links[gw_id]["match_type"] = MATCH_TYPE_FUZZY
                engine.gw_bank_links[gw_id]["matching_stage"] = STAGE_FUZZY_GW_BANK
                engine.gw_bank_links[gw_id]["score"] = max(float(engine.gw_bank_links[gw_id].get("score", 0.0)), float(score))
                engine.gw_bank_links[gw_id]["note"] = f"Fuzzy residual ERP↔GW + bank evidence (score={score:.2f})."
            else:
                engine.gw_bank_links[gw_id] = {
                    "bank_ids": [],
                    "utr": gw_row.get("bank_utr"),
                    "match_type": MATCH_TYPE_FUZZY,
                    "matching_stage": STAGE_FUZZY_GW_BANK,
                    "score": float(score),
                    "note": "Fuzzy residual ERP↔GW match.",
                }
