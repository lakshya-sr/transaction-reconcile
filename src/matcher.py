"""
Phase 3: Exact Matching Engine Module.

Implements deterministic 1:1 exact matching across enterprise schemas:
1. `erp_ledger` (ERPNext/BenchRec)
2. `gateway_settlements` (Razorpay Payload)
3. `bank_statement` (ISO 20022 CAMT.053)

Persists high-confidence matches to `reconciliation_results` and provides
structured diagnostics for downstream LLM resolution.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tabulate import tabulate

from src.config import (
    DB_PATH,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_TDS,
    TABLE_BANK,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_RESULTS,
)
from src.database import get_connection


def extract_invoice_number(remittance_info: str) -> Optional[str]:
    """Extract invoice number (e.g. 'INV-10016') from CAMT.053 remittance narrative."""
    if not remittance_info or not isinstance(remittance_info, str):
        return None
    match = re.search(r"(INV[-_/]?(?:2026[-_/])?\d{4,6}|ORD[-_]?\d{4,6})", remittance_info, re.IGNORECASE)
    if match:
        raw = match.group(1).upper().replace("/", "-").replace("_", "-")
        # Standardize prefix to INV-XXXXX
        if raw.startswith("ORD"):
            raw = "INV-" + raw.split("-")[-1]
        elif not raw.startswith("INV-"):
            digits = re.search(r"\d{4,6}", raw)
            if digits:
                raw = f"INV-{digits.group(0)}"
        return raw
    return None


def extract_utr_number(remittance_info: str) -> Optional[str]:
    """Extract bank UTR (e.g. 'UTR342160733754') from CAMT.053 remittance narrative."""
    if not remittance_info or not isinstance(remittance_info, str):
        return None
    match = re.search(r"(UTR\d{12})", remittance_info, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_settlement_id(remittance_info: str) -> Optional[str]:
    """Extract settlement ID (e.g. 'setl_12345') from CAMT.053 remittance narrative."""
    if not remittance_info or not isinstance(remittance_info, str):
        return None
    match = re.search(r"(setl_[a-zA-Z0-9]+)", remittance_info, re.IGNORECASE)
    return match.group(1) if match else None


def fetch_unmatched_records(db_path: Path = DB_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch active unreconciled records across all three sources."""
    conn = get_connection(db_path)
    
    try:
        # ERP records not in reconciliation_results
        query_erp = f"""
            SELECT erp_entry_id, customer_account_id, invoice_number, gross_amount, tds_expected, currency, entry_date, status, allocation_key
            FROM {TABLE_ERP}
            WHERE erp_entry_id NOT IN (
                SELECT erp_order_id FROM {TABLE_RESULTS} WHERE erp_order_id IS NOT NULL
            );
        """
        df_erp = pd.read_sql_query(query_erp, conn)

        # Gateway settlements not in reconciliation_results
        query_gateway = f"""
            SELECT payment_id, settlement_id, gateway_status, gross_amount, fee_deducted, tax_on_fee, net_settled, amount_reversed, settled_at, bank_utr
            FROM {TABLE_GATEWAY}
            WHERE payment_id NOT IN (
                SELECT gateway_payment_id FROM {TABLE_RESULTS} WHERE gateway_payment_id IS NOT NULL
            );
        """
        df_gateway = pd.read_sql_query(query_gateway, conn)

        # Bank statements not in reconciliation_results
        query_bank = f"""
            SELECT bank_entry_id, value_date, transaction_type, credit_amount, debit_amount, running_balance, remittance_info, reversal_indicator
            FROM {TABLE_BANK}
            WHERE bank_entry_id NOT IN (
                SELECT bank_utr FROM {TABLE_RESULTS} WHERE bank_utr IS NOT NULL
            );
        """
        df_bank = pd.read_sql_query(query_bank, conn)

    finally:
        conn.close()

    return df_erp, df_gateway, df_bank


def run_exact_matching(db_path: Path = DB_PATH) -> Dict[str, any]:
    """
    Execute deterministic 1:1 exact matching algorithm and persist matches.
    """
    df_erp, df_gateway, df_bank = fetch_unmatched_records(db_path)

    total_erp = len(df_erp)
    total_gateway = len(df_gateway)
    total_bank = len(df_bank)

    # 1. Parse Bank Records
    df_bank = df_bank.copy()
    df_bank["extracted_invoice"] = df_bank["remittance_info"].apply(extract_invoice_number)
    df_bank["extracted_utr"] = df_bank["remittance_info"].apply(extract_utr_number)
    df_bank["extracted_setl"] = df_bank["remittance_info"].apply(extract_settlement_id)

    # 2. Index ERP records by invoice_number and erp_entry_id
    erp_by_invoice: Dict[str, Dict] = {row["invoice_number"]: row.to_dict() for _, row in df_erp.iterrows()}
    erp_by_entry_id: Dict[str, Dict] = {row["erp_entry_id"]: row.to_dict() for _, row in df_erp.iterrows()}

    # 3. Index Gateway records by bank_utr and settlement_id
    # Note: In our dataset, each transaction has a unique bank_utr linking gateway to bank
    gw_by_utr: Dict[str, List[Dict]] = {}
    gw_by_setl: Dict[str, List[Dict]] = {}
    for _, row in df_gateway.iterrows():
        gw_dict = row.to_dict()
        if row["bank_utr"]:
            gw_by_utr.setdefault(row["bank_utr"], []).append(gw_dict)
        if row["settlement_id"]:
            gw_by_setl.setdefault(row["settlement_id"], []).append(gw_dict)

    matches_to_insert: List[Dict] = []
    matched_erp_entries = set()
    matched_gw_payments = set()
    matched_bank_entries = set()

    # 4. Iterate over Bank Statement Credits
    for _, bank_row in df_bank.iterrows():
        bank_entry_id = bank_row["bank_entry_id"]
        remittance = bank_row["remittance_info"]
        credit_amt = round(float(bank_row["credit_amount"]), 2)
        extracted_inv = bank_row["extracted_invoice"]
        extracted_utr = bank_row["extracted_utr"]
        extracted_setl = bank_row["extracted_setl"]

        # Find matching gateway record
        candidate_gw_records = []
        if extracted_utr and extracted_utr in gw_by_utr:
            candidate_gw_records = gw_by_utr[extracted_utr]
        elif extracted_setl and extracted_setl in gw_by_setl:
            candidate_gw_records = gw_by_setl[extracted_setl]

        if not candidate_gw_records:
            continue

        for gw_entry in candidate_gw_records:
            gw_payment_id = gw_entry["payment_id"]
            gw_net_settled = round(float(gw_entry["net_settled"]), 2)
            gw_utr = gw_entry["bank_utr"]

            if gw_payment_id in matched_gw_payments:
                continue

            # Verify amount match: Bank Credit == Gateway Net Settled
            if abs(credit_amt - gw_net_settled) < 0.001:
                # Find matching ERP invoice
                if extracted_inv and extracted_inv in erp_by_invoice:
                    erp_entry = erp_by_invoice[extracted_inv]
                    erp_id = erp_entry["erp_entry_id"]
                    tds_expected = float(erp_entry.get("tds_expected", 0.0))

                    if erp_id in matched_erp_entries:
                        continue

                    # Exact 1:1 Match (when no TDS exception)
                    if tds_expected == 0.0:
                        matched_erp_entries.add(erp_id)
                        matched_gw_payments.add(gw_payment_id)
                        matched_bank_entries.add(bank_entry_id)

                        fee_str = (
                            f" [MDR Fee: ₹{gw_entry['fee_deducted']:.2f}, GST: ₹{gw_entry['tax_on_fee']:.2f}]"
                            if gw_entry['fee_deducted'] > 0 else ""
                        )
                        notes = (
                            f"Exact 1:1 Match: Invoice '{extracted_inv}' / Entry '{erp_id}' verified. "
                            f"Bank Credit ₹{credit_amt:.2f} matches Gateway Net Settled ₹{gw_net_settled:.2f}{fee_str}. "
                            f"Value Date: {bank_row['value_date']}."
                        )

                        matches_to_insert.append({
                            "erp_order_id": erp_id,
                            "gateway_payment_id": gw_payment_id,
                            "bank_utr": gw_utr,
                            "match_type": MATCH_TYPE_EXACT,
                            "confidence_score": 1.00,
                            "notes": notes,
                        })
                        break

    # 5. Persist matches into SQLite table
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        for match in matches_to_insert:
            cursor.execute(f"""
                INSERT INTO {TABLE_RESULTS}
                (erp_order_id, gateway_payment_id, bank_utr, match_type, confidence_score, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                match["erp_order_id"],
                match["gateway_payment_id"],
                match["bank_utr"],
                match["match_type"],
                match["confidence_score"],
                match["notes"],
            ))
        conn.commit()
    finally:
        conn.close()

    # 6. Unmatched Records Diagnosis
    unmatched_erp = df_erp[~df_erp["erp_entry_id"].isin(matched_erp_entries)].copy()
    unmatched_gateway = df_gateway[~df_gateway["payment_id"].isin(matched_gw_payments)].copy()
    unmatched_bank = df_bank[~df_bank["bank_entry_id"].isin(matched_bank_entries)].copy()

    def diagnose_bank_record(row):
        rem = row.get("remittance_info", "")
        inv = row.get("extracted_invoice")
        utr = row.get("extracted_utr")
        if not inv and not utr:
            return "Obscured CAMT.053 Narration (Requires AI/LLM Semantic Extraction)"
        if utr and utr in gw_by_utr:
            gw = gw_by_utr[utr][0]
            if inv and inv in erp_by_invoice:
                erp = erp_by_invoice[inv]
                if erp.get("tds_expected", 0) > 0:
                    return f"TDS Withholding Exception (TDS Expected: ₹{erp['tds_expected']:.2f})"
            return f"Amount Discrepancy: Bank credited ₹{row['credit_amount']:.2f} vs Gateway Net ₹{gw['net_settled']:.2f}"
        return "Gateway Settlement / UTR Missing (Orphaned Bank Credit)"

    unmatched_bank["diagnosis"] = unmatched_bank.apply(diagnose_bank_record, axis=1)

    return {
        "total_erp": total_erp,
        "total_gateway": total_gateway,
        "total_bank": total_bank,
        "exact_matches_count": len(matches_to_insert),
        "unmatched_erp_count": len(unmatched_erp),
        "unmatched_gateway_count": len(unmatched_gateway),
        "unmatched_bank_count": len(unmatched_bank),
        "matches": matches_to_insert,
        "unmatched_erp": unmatched_erp,
        "unmatched_gateway": unmatched_gateway,
        "unmatched_bank": unmatched_bank,
    }
