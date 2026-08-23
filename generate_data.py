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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ERP_LEDGER_PATH = DATA_DIR / "erp_ledger.json"
GATEWAY_SETTLEMENTS_PATH = DATA_DIR / "gateway_settlements.json"
GATEWAY_PAYOUTS_PATH = DATA_DIR / "gateway_payouts.json"
BANK_STATEMENT_PATH = DATA_DIR / "bank_statement.csv"

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

    def get_ground_truth_df(self) -> pd.DataFrame:
        df_eg = pd.DataFrame(self.erp_gw_edges)
        df_gb = pd.DataFrame(self.gw_bank_edges)
        if not df_eg.empty and not df_gb.empty:
            return pd.merge(df_eg, df_gb, on="gw_id", how="outer")
        elif not df_eg.empty:
            return df_eg
        elif not df_gb.empty:
            return df_gb
        return pd.DataFrame()


class ERPAgent:
    def __init__(self):
        self.ledger = []
        self.outbox = deque()

    def tick(self, current_time: datetime):
        num_orders = random.randint(0, 5)
        for _ in range(num_orders):
            erp_uid = uuid.uuid4().hex[:8]
            erp_id = f"ERP-{erp_uid}"
            invoice_number = f"INV-{erp_uid.upper()}"
            customer_account_id = f"CUST-ACC-{random.randint(1000, 9999)}"
            erp_amount = round(random.uniform(10.0, 500.0), 2)
            entry_date_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            order = {
                "erp_entry_id": erp_id,
                "customer_account_id": customer_account_id,
                "invoice_number": invoice_number,
                "gross_amount": erp_amount,
                "tds_expected": 0.00,
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
        self.fee_rate = 0.025

    def tick(self, current_time: datetime):
        batch_size = min(len(self.inbox), random.randint(0, 10))
        erp_batch = [self.inbox.popleft() for _ in range(batch_size)]

        i = 0
        while i < len(erp_batch):
            scenario = random.choices(["1:1", "N:1", "1:N"], weights=[0.6, 0.2, 0.2])[0]

            if scenario == "N:1" and i < len(erp_batch) - 1:
                o1, o2 = erp_batch[i], erp_batch[i + 1]
                total_amt = o1["erp_amount"] + o2["erp_amount"]
                gw_record = self._create_record(total_amt, current_time, invoices=[o1["invoice_number"], o2["invoice_number"]])
                self.registry.log_erp_to_gw(o1["erp_id"], gw_record["gw_id"], o1["erp_amount"])
                self.registry.log_erp_to_gw(o2["erp_id"], gw_record["gw_id"], o2["erp_amount"])
                i += 2

            elif scenario == "1:N":
                o1 = erp_batch[i]
                split_amt = round(o1["erp_amount"] / 2, 2)
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
        if random.random() < 0.05:
            amount = round(amount + random.choice([-0.02, -0.01, 0.01, 0.02]), 2)
            
        gw_date = date + timedelta(seconds=random.randint(2, 300))
        fee = round(amount * self.fee_rate, 2)
        tax_on_fee = round(fee * 0.18, 2)
        net_settled = round(amount - (fee + tax_on_fee), 2)
        
        gw_id = f"GW-{uuid.uuid4().hex[:8]}"
        settlement_id = f"setl_{uuid.uuid4().hex[:12]}"
        bank_utr = f"UTR{uuid.uuid4().int % 10**12:012d}"
        
        if random.random() < 0.10:
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
        self.running_balance = 10_000_000.00

    def tick(self, current_time: datetime):
        while self.inbox:
            tx = self.inbox.popleft()
            if random.random() < 0.20:
                self._process_instant_settlement(tx, current_time)
            else:
                self.batch_queue.append(tx)

        if current_time.hour == 23 and self.batch_queue:
            self._process_batch_settlements(current_time)

    def _process_instant_settlement(self, tx: Dict, current_time: datetime):
        bank_id = f"BNK-{uuid.uuid4().hex[:8]}-INST"
        bank_date = current_time + timedelta(minutes=random.randint(1, 15))
        amount = tx["gw_net"]

        self._record_bank_entry(bank_id, amount, bank_date, source_txns=[tx], is_instant=True)
        self.registry.log_gw_to_bank(tx["gw_id"], bank_id, amount)

    def _process_batch_settlements(self, current_time: datetime):
        daily_txns = list(self.batch_queue)
        self.batch_queue.clear()

        i = 0
        while i < len(daily_txns):
            scenario = random.choices(["N:1", "1:N"], weights=[0.8, 0.2])[0]

            if scenario == "N:1" and i < len(daily_txns) - 4:
                batch_len = random.randint(2, 5)
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
                upfront = round(tx["gw_net"] * 0.9, 2)
                reserve = round(tx["gw_net"] - upfront, 2)

                bank_id_main = f"BNK-{uuid.uuid4().hex[:8]}-MAIN"
                bank_id_rsv = f"BNK-{uuid.uuid4().hex[:8]}-RSV"
                
                bank_date_main = self._get_batch_settlement_date(current_time)
                bank_date_rsv = current_time + timedelta(days=7)

                self._record_bank_entry(bank_id_main, upfront, bank_date_main, source_txns=[tx])
                self._record_bank_entry(bank_id_rsv, reserve, bank_date_rsv, source_txns=[tx])

                self.registry.log_gw_to_bank(tx["gw_id"], bank_id_main, upfront)
                self.registry.log_gw_to_bank(tx["gw_id"], bank_id_rsv, reserve)
                i += 1

    def _get_batch_settlement_date(self, current_time: datetime) -> datetime:
        base_date = current_time + timedelta(hours=8)
        if random.random() < 0.15:
            base_date += timedelta(days=random.randint(1, 2))
        return base_date

    def _record_bank_entry(self, bank_id: str, amount: float, bank_date: datetime, source_txns: Optional[List[Dict]] = None, is_instant: bool = False):
        if not is_instant and random.random() < 0.05:
            amount = round(amount - random.choice([5.00, 10.00, 2.50]), 2)
            
        self.running_balance = round(self.running_balance + amount, 2)
        value_date_str = bank_date.strftime("%Y-%m-%d %H:%M:%S") if is_instant else bank_date.strftime("%Y-%m-%d")

        if source_txns:
            utr = source_txns[0].get("bank_utr", f"UTR{uuid.uuid4().int % 10**12:012d}")
            inv_list = []
            for t in source_txns:
                inv_list.extend(t.get("invoices", []))
            inv_str = inv_list[0] if inv_list else "INV-GEN"
            
            if random.random() < 0.20:
                corruption_type = random.choice(["strip_prefix", "truncate_utr", "missing_invoice"])
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


def run_continuous_simulation(days: int = 3, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if seed is not None:
        random.seed(seed)

    registry = GroundTruthRegistry()
    erp = ERPAgent()
    gw = GatewayAgent(registry)
    bank = BankAgent(registry)

    start_time = datetime(2024, 1, 1, 0, 0, 0)
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
        current_time += timedelta(hours=1)

    return pd.DataFrame(erp.ledger), pd.DataFrame(gw.ledger), pd.DataFrame(bank.ledger), registry.get_ground_truth_df()


def save_datasets(df_erp: pd.DataFrame, df_gw: pd.DataFrame, df_bank: pd.DataFrame, df_truth: pd.DataFrame, output_dir: Path = DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    erp_cols = ["erp_entry_id", "customer_account_id", "invoice_number", "gross_amount", "tds_expected", "currency", "entry_date", "status", "allocation_key"]
    df_erp_clean = df_erp[erp_cols] if all(c in df_erp.columns for c in erp_cols) else df_erp
    with open(output_dir / "erp_ledger.json", "w", encoding="utf-8") as f:
        json.dump(df_erp_clean.to_dict(orient="records"), f, indent=2)

    gw_cols = [
        "payment_id", "settlement_id", "gateway_status", 
        "gross_amount", "fee_deducted", "tax_on_fee", 
        "net_settled", "amount_reversed", "settled_at", 
        "bank_utr", "invoices"  
    ]
    df_gw_clean = df_gw[gw_cols] if all(c in df_gw.columns for c in gw_cols) else df_gw
    with open(output_dir / "gateway_settlements.json", "w", encoding="utf-8") as f:
        json.dump(df_gw_clean.to_dict(orient="records"), f, indent=2)
    with open(output_dir / "gateway_payouts.json", "w", encoding="utf-8") as f:
        json.dump(df_gw_clean.to_dict(orient="records"), f, indent=2)

    bank_cols = ["bank_entry_id", "value_date", "transaction_type", "credit_amount", "debit_amount", "running_balance", "remittance_info", "reversal_indicator"]
    df_bank_clean = df_bank[bank_cols] if all(c in df_bank.columns for c in bank_cols) else df_bank
    df_bank_clean.to_csv(output_dir / "bank_statement.csv", index=False)

    if not df_truth.empty:
        with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(truth_records := df_truth.to_dict(orient="records"), f, indent=2)
        df_truth.to_csv(output_dir / "ground_truth.csv", index=False)


def main():
    df_erp, df_gw, df_bank, df_truth = run_continuous_simulation(days=5, seed=None)
    save_datasets(df_erp, df_gw, df_bank, df_truth, output_dir=DATA_DIR)
    print(f"[✔] Generated {len(df_erp)} ERP, {len(df_gw)} GW, {len(df_bank)} Bank records.")

if __name__ == "__main__":
    main()