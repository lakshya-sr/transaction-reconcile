"""
Phase 1: Enterprise Synthetic Data Generation Module.

Generates 100 linked multi-source financial transaction records across:
1. ERP Ledger (JSON) - BenchRec/ERPNext standard:
   - erp_entry_id, customer_account_id, invoice_number, gross_amount, tds_expected, currency, entry_date, status, allocation_key
2. Gateway Settlements (JSON) - Razorpay Payload standard:
   - payment_id, settlement_id, gateway_status, gross_amount, fee_deducted, tax_on_fee (18% GST), net_settled, amount_reversed, settled_at, bank_utr
3. Bank Statement (CSV) - ISO 20022 CAMT.053 standard:
   - bank_entry_id, value_date, transaction_type, credit_amount, debit_amount, running_balance, remittance_info, reversal_indicator

Edge Cases & Realistic Banking Discrepancies:
- 2% Razorpay MDR fee deducted on 30% of settlements with 18% GST on fee.
- Value date delayed by 1-2 days compared to ERP booking date for 50% of records.
- Messy CAMT.053 remittance_info strings with mixed banking protocols (NEFT/RTGS/IMPS/UPI/CMS).
- Bank credit_amount exactly aligns with gateway net_settled (gross - fee - tax_on_fee).
- TDS withholding, obscured narrations, and fee variances for LLM agent resolution.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from faker import Faker

from src.config import (
    BANK_STATEMENT_PATH,
    DATA_DIR,
    ERP_LEDGER_PATH,
    GATEWAY_PAYOUTS_PATH,
    GATEWAY_SETTLEMENTS_PATH,
)

fake = Faker("en_IN")


def generate_remittance_info(
    invoice_number: str,
    settlement_id: str,
    bank_utr: str,
    pattern_type: str = "standard",
) -> str:
    """
    Generate realistic ISO 20022 CAMT.053 remittance narrative strings.
    """
    if pattern_type == "obscured":
        templates = [
            f"NEFT-BULK-CR-SETTLE-{random.randint(100000, 999999)}/NODAL-POOL",
            f"CMS-DIRECT-PAYOUT-REF{random.randint(1000, 9999)}-UNTAGGED-CORP",
            f"UPI/RPAY/SETTLEMENT/BATCH-{random.randint(100, 999)}/INTERBANK",
            f"IMPS-P2A-RAZORPAY-NODAL-TRANSFER-{random.randint(1000, 9999)}",
            f"ACH-CR-RPAY-BULK-SETTLEMENT-{random.randint(10000, 99999)}-HDFC",
        ]
        return random.choice(templates)

    templates = [
        f"NEFT-RAZORPAY-{invoice_number}-{bank_utr}-UPI",
        f"UPI/RPAY/{settlement_id}/{invoice_number}/DIRECT",
        f"IMPS-P2A-{invoice_number}-SETTLEMENT-{settlement_id}",
        f"CMS/RAZORPAYPAYMENT/{invoice_number}/{bank_utr}/NODAL",
        f"ACH-CR-RAZORPAYPAYMENT-{invoice_number}-{settlement_id}",
        f"NODAL/SETTLE/{invoice_number}/RAZORPAY_X/{bank_utr}",
        f"RTGS-RPAY-SETTLE-{invoice_number}-{bank_utr}-CORP",
        f"UPI-MERCHANT-SETTLE-{invoice_number}/{settlement_id}",
    ]
    return random.choice(templates)


def generate_dataset(
    total_transactions: int = 100,
    seed: int = 42,
    output_dir: Path = DATA_DIR,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate 100 linked synthetic transaction records across ERP, Gateway, and Bank sources.

    Args:
        total_transactions: Total records to generate (default 100).
        seed: Random seed for deterministic reproducibility.
        output_dir: Destination directory for files.

    Returns:
        Tuple containing (erp_records, gateway_records, bank_records).
    """
    random.seed(seed)
    Faker.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_date = datetime(2026, 8, 1, 9, 30, 0)
    running_balance = 10_000_000.00  # Initial bank balance ₹1,00,00,000.00

    erp_records: List[Dict] = []
    gateway_records: List[Dict] = []
    bank_records: List[Dict] = []

    # Distribution across 100 transactions:
    # - 1 to 75 (75 records): Exact 1:1 Clean 3-Way Matches (some with 2% MDR + 18% GST fee)
    # - 76 to 80 (5 records): B2B TDS Withholding Transactions (2% Section 194C TDS in ERP)
    # - 81 to 90 (10 records): Obscured CAMT.053 Remittance Narratives (Requires AI/LLM matching)
    # - 91 to 95 (5 records): Amount / Fee Variance Anomalies (disputed bank charges)
    # - 96 to 100 (5 records): Orphaned Settlements / Cross-reference Mismatches

    for i in range(1, total_transactions + 1):
        erp_entry_id = f"ERP-2026-{10000 + i}"
        invoice_number = f"INV-{10000 + i}"
        customer_account_id = f"CUST-ACC-{random.randint(1000, 9999)}"
        
        # Gross amount between ₹2,000.00 and ₹95,000.00
        gross_amount = round(random.uniform(2000.0, 95000.0), 2)
        
        # Transaction timestamp spread across August 2026
        days_offset = (i * 7) % 22
        hours_offset = (i * 3) % 12
        minutes_offset = (i * 11) % 60
        tx_date = base_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        entry_date_str = tx_date.strftime("%Y-%m-%d %H:%M:%S")

        # TDS expected (for B2B transactions: records 76-80 have 2% TDS)
        if 76 <= i <= 80:
            tds_expected = round(gross_amount * 0.02, 2)
            status = "Partially Paid"
        else:
            tds_expected = 0.00
            status = "Paid"

        allocation_key = f"{customer_account_id}|{invoice_number}|{gross_amount:.2f}"

        # 1. ERP Ledger Record
        erp_record = {
            "erp_entry_id": erp_entry_id,
            "customer_account_id": customer_account_id,
            "invoice_number": invoice_number,
            "gross_amount": gross_amount,
            "tds_expected": tds_expected,
            "currency": "INR",
            "entry_date": entry_date_str,
            "status": status,
            "allocation_key": allocation_key,
        }
        erp_records.append(erp_record)

        # 2. Gateway Settlement Record
        # Induce 2% MDR fee on 30% of records (indices where i % 10 in [1, 4, 7])
        has_fee = (i % 10 in [1, 4, 7])
        if has_fee:
            fee_deducted = round(gross_amount * 0.02, 2)
            tax_on_fee = round(fee_deducted * 0.18, 2)  # 18% GST on MDR
        else:
            fee_deducted = 0.00
            tax_on_fee = 0.00

        amount_reversed = 0.00
        net_settled = round(gross_amount - (fee_deducted + tax_on_fee) - amount_reversed, 2)
        payment_id = f"pay_{fake.bothify(text='??????????????')}"
        settlement_id = f"setl_{fake.bothify(text='??????????????')}"
        bank_utr = f"UTR{fake.bothify(text='############')}"

        # Anomaly case: Records 96-100 have orphaned references
        if i >= 96:
            gw_bank_utr = f"UTR999999{fake.bothify(text='######')}"  # Divergent UTR
            gw_settlement_id = f"setl_ORPHAN_{i}"
        else:
            gw_bank_utr = bank_utr
            gw_settlement_id = settlement_id

        gateway_record = {
            "payment_id": payment_id,
            "settlement_id": gw_settlement_id,
            "gateway_status": "captured",
            "gross_amount": gross_amount,
            "fee_deducted": fee_deducted,
            "tax_on_fee": tax_on_fee,
            "net_settled": net_settled,
            "amount_reversed": amount_reversed,
            "settled_at": entry_date_str,
            "bank_utr": gw_bank_utr,
        }
        gateway_records.append(gateway_record)

        # 3. Bank Statement Record (ISO 20022 CAMT.053)
        # Edge Case 1: Delay Value Date by 1-2 days for 50% of the records
        is_delayed = (i % 2 == 0)
        if is_delayed:
            delay_days = 1 if (i % 4 == 0) else 2
            val_date = (tx_date + timedelta(days=delay_days)).date()
        else:
            val_date = tx_date.date()
        value_date_str = val_date.strftime("%Y-%m-%d")

        # Edge Case 2: Messy CAMT.053 narrative string
        if 81 <= i <= 90:
            remittance_info = generate_remittance_info(invoice_number, settlement_id, bank_utr, pattern_type="obscured")
        elif i >= 96:
            # Records 96-100: Bank statement references its own bank_utr and orphan settlement
            remittance_info = f"NEFT-BANK-DIRECT-{bank_utr}-EXTPAY"
        else:
            remittance_info = generate_remittance_info(invoice_number, settlement_id, bank_utr, pattern_type="standard")

        # Edge Case 3: Credit Amount matches Net Settled
        if 91 <= i <= 95:
            # Discrepancy anomaly: unexpected ₹50.00 bank processing variance
            credit_amount = round(net_settled - 50.00, 2)
        else:
            credit_amount = net_settled

        debit_amount = 0.00
        running_balance = round(running_balance + credit_amount - debit_amount, 2)
        bank_entry_id = f"CAMT-{val_date.strftime('%Y%m%d')}-{10000 + i}"

        bank_record = {
            "bank_entry_id": bank_entry_id,
            "value_date": value_date_str,
            "transaction_type": "CRDT",
            "credit_amount": credit_amount,
            "debit_amount": debit_amount,
            "running_balance": running_balance,
            "remittance_info": remittance_info,
            "reversal_indicator": False,
        }
        bank_records.append(bank_record)

    # Shuffle bank statements to reflect out-of-order CAMT file processing
    random.shuffle(bank_records)

    # Save to disk
    save_datasets(erp_records, gateway_records, bank_records, output_dir)

    return erp_records, gateway_records, bank_records


def save_datasets(
    erp_records: List[Dict],
    gateway_records: List[Dict],
    bank_records: List[Dict],
    output_dir: Path = DATA_DIR,
) -> None:
    """Save generated datasets to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    erp_file = output_dir / "erp_ledger.json"
    gateway_file = output_dir / "gateway_payouts.json"
    gateway_settlements_file = output_dir / "gateway_settlements.json"
    bank_file = output_dir / "bank_statement.csv"

    # 1. erp_ledger.json
    with open(erp_file, "w", encoding="utf-8") as f:
        json.dump(erp_records, f, indent=2)

    # 2. gateway_payouts.json & gateway_settlements.json
    with open(gateway_file, "w", encoding="utf-8") as f:
        json.dump(gateway_records, f, indent=2)
    with open(gateway_settlements_file, "w", encoding="utf-8") as f:
        json.dump(gateway_records, f, indent=2)

    # 3. bank_statement.csv
    if bank_records:
        fieldnames = [
            "bank_entry_id",
            "value_date",
            "transaction_type",
            "credit_amount",
            "debit_amount",
            "running_balance",
            "remittance_info",
            "reversal_indicator",
        ]
        with open(bank_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bank_records)


if __name__ == "__main__":
    erp, gw, bank = generate_dataset(total_transactions=100)
    print(f"Generated {len(erp)} ERP, {len(gw)} Gateway, {len(bank)} Bank records.")
