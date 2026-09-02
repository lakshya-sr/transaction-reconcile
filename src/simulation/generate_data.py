#!/usr/bin/env python3
"""
Phase 1: Multi-Source Continuous Simulation & Data Generation with Noise Injection.
"""

import json
import random
import sys
import uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import (
    DATA_DIR,
    ERP_LEDGER_PATH,
    GATEWAY_SETTLEMENTS_PATH,
    GATEWAY_PAYOUTS_PATH,
    BANK_STATEMENT_PATH,
    GROUND_TRUTH_ERP_GW_PATH,
    GROUND_TRUTH_ERP_GW_JSON_PATH,
    GROUND_TRUTH_GW_BANK_PATH,
    GROUND_TRUTH_GW_BANK_JSON_PATH,
)

# ERP generation constants (Mutated: 5x higher amount ranges & higher density)
ERP_MAX_ORDERS_PER_TICK = 8
ERP_CUSTOMER_ACCOUNT_ID_MIN = 5000
ERP_CUSTOMER_ACCOUNT_ID_MAX = 89999
ERP_AMOUNT_MIN = 25.0
ERP_AMOUNT_MAX = 2500.0

# Gateway matching/simulation constants (Mutated: altered MDR fee 1.8%, altered GST 12%, higher noise)
GATEWAY_BATCH_MAX_SIZE = 14
GATEWAY_SCENARIO_OPTIONS = ["1:1", "N:1", "1:N"]
GATEWAY_SCENARIO_WEIGHTS = [0.5, 0.25, 0.25]
GATEWAY_NO_INVOICES_PROBABILITY = 0.25
GATEWAY_SPLIT_DIVISOR = 2
GATEWAY_SETTLEMENT_DELAY_SECONDS_MIN = 10
GATEWAY_SETTLEMENT_DELAY_SECONDS_MAX = 600
GATEWAY_FEE_RATE = 0.018
GATEWAY_TAX_RATE = 0.12

# Bank settlement constants (Mutated: higher corruption 35%, 85/15 reserve split, larger batches)
BANK_INITIAL_RUNNING_BALANCE = 50_000_000.00
BANK_INSTANT_SETTLEMENT_PROBABILITY = 0.35
BANK_BATCH_SCENARIO_OPTIONS = ["N:1", "1:N"]
BANK_BATCH_SCENARIO_WEIGHTS = [0.75, 0.25]
BANK_BATCH_MIN_SIZE = 3
BANK_BATCH_MAX_SIZE = 6
BANK_N_TO_1_MIN_BATCH_GAP = 3
BANK_RESERVE_SPLIT_RATIO = 0.85
BANK_BATCH_SETTLEMENT_DELAY_DAYS = 5
BANK_LATE_SETTLEMENT_PROBABILITY = 0.25
BANK_LATE_SETTLEMENT_DELAY_DAYS_MIN = 1
BANK_LATE_SETTLEMENT_DELAY_DAYS_MAX = 3
BANK_INSTANT_SETTLEMENT_DELAY_MINUTES_MIN = 2
BANK_INSTANT_SETTLEMENT_DELAY_MINUTES_MAX = 30
BANK_CORRUPTION_PROBABILITY = 0.35
BANK_CORRUPTION_TYPES = ["strip_prefix", "truncate_utr", "missing_invoice"]
BANK_UTR_MODULUS = 10**12
BANK_BATCH_SETTLEMENT_BASE_HOUR = 9
BANK_ZERO_AMOUNT = 0.00

# Simulation timing constants
SIMULATION_START_YEAR = 2024
SIMULATION_START_MONTH = 1
SIMULATION_START_DAY = 1
SIMULATION_DEFAULT_DAYS = 3
SIMULATION_GROUND_TRUTH_DAYS = 5
SIMULATION_PIPELINE_HOUR_INCREMENT = 1
BANK_SETTLEMENT_PROCESSING_HOUR = 23


class GroundTruthRegistry:
    def __init__(self):
        self.erp_gw_edges = []
        self.gw_bank_edges = []

    def log_erp_to_gw(self, erp_id: str, gw_id: str, allocated_amount: float):
        self.erp_gw_edges.append(
            {"erp_id": erp_id, "gw_id": gw_id, "erp_gw_amount": round(float(allocated_amount), 2)}
        )

    def log_gw_to_bank(self, gw_id: str, bank_id: str, allocated_amount: float):
        self.gw_bank_edges.append(
            {"gw_id": gw_id, "bank_id": bank_id, "gw_bank_amount": round(float(allocated_amount), 2)}
        )

    def get_erp_gw_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.erp_gw_edges) if self.erp_gw_edges else pd.DataFrame(columns=["erp_id", "gw_id", "erp_gw_amount"])

    def get_gw_bank_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.gw_bank_edges) if self.gw_bank_edges else pd.DataFrame(columns=["gw_id", "bank_id", "gw_bank_amount"])


class ERPAgent:
    def __init__(self):
        self.ledger = []
        self.outbox = deque()

    def tick(self, current_time: datetime):
        num_orders = random.randint(0, ERP_MAX_ORDERS_PER_TICK)
        for _ in range(num_orders):
            erp_uid = uuid.uuid4().hex[:8]
            erp_id = f"ERP-{erp_uid}"
            invoice_number = f"INV-{erp_uid.upper()}"
            customer_account_id = f"CUST-ACC-{random.randint(ERP_CUSTOMER_ACCOUNT_ID_MIN, ERP_CUSTOMER_ACCOUNT_ID_MAX)}"
            erp_amount = round(random.uniform(ERP_AMOUNT_MIN, ERP_AMOUNT_MAX), 2)
            entry_date_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            order = {
                "erp_entry_id": erp_id,
                "customer_account_id": customer_account_id,
                "invoice_number": invoice_number,
                "gross_amount": erp_amount,
                "tds_expected": BANK_ZERO_AMOUNT,
                "currency": "INR",
                "entry_date": entry_date_str,
                "status": "Paid",
                "allocation_key": f"{customer_account_id}|{invoice_number}|{erp_amount:.2f}",
                "erp_id": erp_id,
                "erp_amount": erp_amount,
                "erp_date": current_time,
            }
            self.ledger.append(order)
            self.outbox.append(order)


class GatewayAgent:
    def __init__(self, registry: GroundTruthRegistry):
        self.ledger = []
        self.inbox = deque()
        self.outbox = deque()
        self.registry = registry
        self.fee_rate = GATEWAY_FEE_RATE

    def tick(self, current_time: datetime, force_all: bool = False):
        batch_size = len(self.inbox) if force_all else min(len(self.inbox), random.randint(0, GATEWAY_BATCH_MAX_SIZE))
        erp_batch = [self.inbox.popleft() for _ in range(batch_size)]

        i = 0
        while i < len(erp_batch):
            scenario = random.choices(GATEWAY_SCENARIO_OPTIONS, weights=GATEWAY_SCENARIO_WEIGHTS)[0]

            if scenario == "N:1" and i < len(erp_batch) - 1:
                o1, o2 = erp_batch[i], erp_batch[i + 1]
                total_amt = round(o1["erp_amount"] + o2["erp_amount"], 2)
                gw_record = self._create_record(total_amt, current_time, invoices=[o1["invoice_number"], o2["invoice_number"]])
                self.registry.log_erp_to_gw(o1["erp_id"], gw_record["gw_id"], o1["erp_amount"])
                self.registry.log_erp_to_gw(o2["erp_id"], gw_record["gw_id"], o2["erp_amount"])
                i += 2

            elif scenario == "1:N":
                o1 = erp_batch[i]
                split_amt = round(o1["erp_amount"] / GATEWAY_SPLIT_DIVISOR, 2)
                rem_amt = round(o1["erp_amount"] - split_amt, 2)
                gw_rec1 = self._create_record(split_amt, current_time, invoices=[o1["invoice_number"]])
                gw_rec2 = self._create_record(rem_amt, current_time, invoices=[o1["invoice_number"]])
                self.registry.log_erp_to_gw(o1["erp_id"], gw_rec1["gw_id"], split_amt)
                self.registry.log_erp_to_gw(o1["erp_id"], gw_rec2["gw_id"], rem_amt)
                i += 1

            else:
                o1 = erp_batch[i]
                gw_record = self._create_record(o1["erp_amount"], current_time, invoices=[o1["invoice_number"]])
                self.registry.log_erp_to_gw(o1["erp_id"], gw_record["gw_id"], o1["erp_amount"])
                i += 1

    def _create_record(self, amount: float, date: datetime, invoices: Optional[List[str]] = None) -> Dict:
        gw_date = date + timedelta(seconds=random.randint(GATEWAY_SETTLEMENT_DELAY_SECONDS_MIN, GATEWAY_SETTLEMENT_DELAY_SECONDS_MAX))
        fee = round(amount * self.fee_rate, 2)
        tax_on_fee = round(fee * GATEWAY_TAX_RATE, 2)
        net_settled = round(amount - (fee + tax_on_fee), 2)

        gw_id = f"GW-{uuid.uuid4().hex[:8]}"
        settlement_id = f"setl_{uuid.uuid4().hex[:12]}"
        bank_utr = f"UTR{uuid.uuid4().int % BANK_UTR_MODULUS:012d}"

        if random.random() < GATEWAY_NO_INVOICES_PROBABILITY:
            invoices = []

        record = {
            "payment_id": gw_id,
            "settlement_id": settlement_id,
            "gateway_status": "captured",
            "gross_amount": amount,
            "fee_deducted": fee,
            "tax_on_fee": tax_on_fee,
            "net_settled": net_settled,
            "amount_reversed": 0.00,
            "settled_at": gw_date.strftime("%Y-%m-%d %H:%M:%S"),
            "bank_utr": bank_utr,
            "gw_id": gw_id,
            "gw_gross": amount,
            "gw_fee": fee,
            "gw_net": net_settled,
            "gw_date": gw_date,
            "invoices": invoices or [],
        }
        self.ledger.append(record)
        self.outbox.append(record)
        return record


class BankAgent:
    def __init__(self, registry: GroundTruthRegistry):
        self.ledger = []
        self.inbox = deque()
        self.batch_queue = []
        self.registry = registry
        self.running_balance = BANK_INITIAL_RUNNING_BALANCE

    def tick(self, current_time: datetime):
        while self.inbox:
            tx = self.inbox.popleft()
            if random.random() < BANK_INSTANT_SETTLEMENT_PROBABILITY:
                self._process_instant_settlement(tx, current_time)
            else:
                self.batch_queue.append(tx)

        if current_time.hour == BANK_SETTLEMENT_PROCESSING_HOUR and self.batch_queue:
            self._process_batch_settlements(current_time)

    def flush(self, current_time: datetime):
        while self.inbox:
            tx = self.inbox.popleft()
            if random.random() < BANK_INSTANT_SETTLEMENT_PROBABILITY:
                self._process_instant_settlement(tx, current_time)
            else:
                self.batch_queue.append(tx)

        if self.batch_queue:
            self._process_batch_settlements(current_time)

    def _process_instant_settlement(self, tx: Dict, current_time: datetime):
        bank_id = f"BNK-{uuid.uuid4().hex[:8]}-INST"
        bank_date = current_time + timedelta(minutes=random.randint(BANK_INSTANT_SETTLEMENT_DELAY_MINUTES_MIN, BANK_INSTANT_SETTLEMENT_DELAY_MINUTES_MAX))
        amount = tx["gw_net"]

        self._record_bank_entry(bank_id, amount, bank_date, source_txns=[tx], is_instant=True)
        self.registry.log_gw_to_bank(tx["gw_id"], bank_id, amount)

    def _process_batch_settlements(self, current_time: datetime):
        daily_txns = list(self.batch_queue)
        self.batch_queue.clear()

        i = 0
        while i < len(daily_txns):
            scenario = random.choices(BANK_BATCH_SCENARIO_OPTIONS, weights=BANK_BATCH_SCENARIO_WEIGHTS)[0]

            if scenario == "N:1" and i < len(daily_txns) - BANK_N_TO_1_MIN_BATCH_GAP:
                batch_len = random.randint(BANK_BATCH_MIN_SIZE, BANK_BATCH_MAX_SIZE)
                batch = daily_txns[i : i + batch_len]
                total_net = round(sum(t["gw_net"] for t in batch), 2)
                bank_id = f"BNK-{uuid.uuid4().hex[:8]}"
                bank_date = self._get_batch_settlement_date(current_time)

                self._record_bank_entry(bank_id, total_net, bank_date, source_txns=batch)
                for t in batch:
                    self.registry.log_gw_to_bank(t["gw_id"], bank_id, t["gw_net"])
                i += len(batch)

            else:
                tx = daily_txns[i]
                upfront = round(tx["gw_net"] * BANK_RESERVE_SPLIT_RATIO, 2)
                reserve = round(tx["gw_net"] - upfront, 2)

                bank_id_main = f"BNK-{uuid.uuid4().hex[:8]}-MAIN"
                bank_id_rsv = f"BNK-{uuid.uuid4().hex[:8]}-RSV"

                bank_date_main = self._get_batch_settlement_date(current_time)
                bank_date_rsv = current_time + timedelta(days=BANK_BATCH_SETTLEMENT_DELAY_DAYS)

                self._record_bank_entry(bank_id_main, upfront, bank_date_main, source_txns=[tx])
                self._record_bank_entry(bank_id_rsv, reserve, bank_date_rsv, source_txns=[tx])

                self.registry.log_gw_to_bank(tx["gw_id"], bank_id_main, upfront)
                self.registry.log_gw_to_bank(tx["gw_id"], bank_id_rsv, reserve)
                i += 1

    def _get_batch_settlement_date(self, current_time: datetime) -> datetime:
        base_date = current_time + timedelta(hours=BANK_BATCH_SETTLEMENT_BASE_HOUR)
        if random.random() < BANK_LATE_SETTLEMENT_PROBABILITY:
            base_date += timedelta(days=random.randint(BANK_LATE_SETTLEMENT_DELAY_DAYS_MIN, BANK_LATE_SETTLEMENT_DELAY_DAYS_MAX))
        return base_date

    def _record_bank_entry(self, bank_id: str, amount: float, bank_date: datetime, source_txns: Optional[List[Dict]] = None, is_instant: bool = False):
        self.running_balance = round(self.running_balance + amount, 2)
        value_date_str = bank_date.strftime("%Y-%m-%d %H:%M:%S") if is_instant else bank_date.strftime("%Y-%m-%d")

        if source_txns:
            utr = source_txns[0].get("bank_utr", f"UTR{uuid.uuid4().int % BANK_UTR_MODULUS:012d}")
            inv_list = []
            for t in source_txns:
                inv_list.extend(t.get("invoices", []))
            inv_str = inv_list[0] if inv_list else "INV-GEN"

            if random.random() < BANK_CORRUPTION_PROBABILITY:
                corruption_type = random.choice(BANK_CORRUPTION_TYPES)
                if corruption_type == "strip_prefix":
                    inv_str = inv_str.replace("INV-", "")
                elif corruption_type == "truncate_utr":
                    utr = utr[:-3]
                elif corruption_type == "missing_invoice":
                    inv_str = ""

            prefix = "IMPS-INSTANT" if is_instant else "NEFT-RAZORPAY"
            remittance_info = f"{prefix}-{inv_str}-{utr}-NODAL/{bank_id}".replace("--", "-")
        else:
            remittance_info = f"NEFT-RAZORPAY-DIRECT-SETTLEMENT-{bank_id}"

        self.ledger.append({
            "bank_entry_id": bank_id,
            "value_date": value_date_str,
            "transaction_type": "CRDT",
            "credit_amount": amount,
            "debit_amount": 0.00,
            "running_balance": self.running_balance,
            "remittance_info": remittance_info,
            "reversal_indicator": False,
            "bank_id": bank_id,
            "bank_amount": amount,
            "bank_date": bank_date,
        })


def run_continuous_simulation(days: int = SIMULATION_DEFAULT_DAYS, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if seed is not None:
        random.seed(seed)

    registry = GroundTruthRegistry()
    erp = ERPAgent()
    gw = GatewayAgent(registry)
    bank = BankAgent(registry)

    start_time = datetime(SIMULATION_START_YEAR, SIMULATION_START_MONTH, SIMULATION_START_DAY, 0, 0, 0)
    end_time = start_time + timedelta(days=days)
    current_time = start_time

    while current_time < end_time:
        erp.tick(current_time)
        while erp.outbox:
            gw.inbox.append(erp.outbox.popleft())
        gw.tick(current_time)
        while gw.outbox:
            bank.inbox.append(gw.outbox.popleft())
        bank.tick(current_time)
        current_time += timedelta(hours=SIMULATION_PIPELINE_HOUR_INCREMENT)

    while erp.outbox:
        gw.inbox.append(erp.outbox.popleft())

    while gw.inbox:
        gw.tick(current_time, force_all=True)
        while gw.outbox:
            bank.inbox.append(gw.outbox.popleft())
        current_time += timedelta(hours=SIMULATION_PIPELINE_HOUR_INCREMENT)

    while bank.inbox or bank.batch_queue:
        bank.flush(current_time)
        current_time += timedelta(hours=SIMULATION_PIPELINE_HOUR_INCREMENT)

    return (
        pd.DataFrame(erp.ledger),
        pd.DataFrame(gw.ledger),
        pd.DataFrame(bank.ledger),
        registry.get_erp_gw_df(),
        registry.get_gw_bank_df(),
    )


def save_datasets(
    df_erp: pd.DataFrame,
    df_gw: pd.DataFrame,
    df_bank: pd.DataFrame,
    df_erp_gw_true: pd.DataFrame,
    df_gw_bank_true: pd.DataFrame,
    output_dir: Path = DATA_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    erp_cols = ["erp_entry_id", "customer_account_id", "invoice_number", "gross_amount", "tds_expected", "currency", "entry_date", "status", "allocation_key"]
    df_erp_clean = df_erp[erp_cols] if all(c in df_erp.columns for c in erp_cols) else df_erp
    with open(output_dir / "erp_ledger.json", "w", encoding="utf-8") as f:
        json.dump(df_erp_clean.to_dict(orient="records"), f, indent=2)

    gw_cols = ["payment_id", "settlement_id", "gateway_status", "gross_amount", "fee_deducted", "tax_on_fee", "net_settled", "amount_reversed", "settled_at", "bank_utr", "invoices"]
    df_gw_clean = df_gw[gw_cols] if all(c in df_gw.columns for c in gw_cols) else df_gw
    with open(output_dir / "gateway_settlements.json", "w", encoding="utf-8") as f:
        json.dump(df_gw_clean.to_dict(orient="records"), f, indent=2)
    with open(output_dir / "gateway_payouts.json", "w", encoding="utf-8") as f:
        json.dump(df_gw_clean.to_dict(orient="records"), f, indent=2)

    bank_cols = ["bank_entry_id", "value_date", "transaction_type", "credit_amount", "debit_amount", "running_balance", "remittance_info", "reversal_indicator"]
    df_bank_clean = df_bank[bank_cols] if all(c in df_bank.columns for c in bank_cols) else df_bank
    df_bank_clean.to_csv(output_dir / "bank_statement.csv", index=False)

    if not df_erp_gw_true.empty:
        with open(output_dir / "ground_truth_erp_gw.json", "w", encoding="utf-8") as f:
            json.dump(df_erp_gw_true.to_dict(orient="records"), f, indent=2)
        df_erp_gw_true.to_csv(output_dir / "ground_truth_erp_gw.csv", index=False)

    if not df_gw_bank_true.empty:
        with open(output_dir / "ground_truth_gw_bank.json", "w", encoding="utf-8") as f:
            json.dump(df_gw_bank_true.to_dict(orient="records"), f, indent=2)
        df_gw_bank_true.to_csv(output_dir / "ground_truth_gw_bank.csv", index=False)


def main():
    df_erp, df_gw, df_bank, df_eg_true, df_gb_true = run_continuous_simulation(days=SIMULATION_GROUND_TRUTH_DAYS, seed=None)
    save_datasets(df_erp, df_gw, df_bank, df_eg_true, df_gb_true, output_dir=DATA_DIR)
    print(f"[✔] Generated {len(df_erp)} ERP, {len(df_gw)} GW, {len(df_bank)} Bank records.")
    print(f"    - Ground Truth ERP <-> GW Edges: {len(df_eg_true)}")
    print(f"    - Ground Truth GW <-> Bank Edges: {len(df_gb_true)}")

if __name__ == "__main__":
    main()