#!/usr/bin/env python3
"""
Diagnostic Visualizer: Ground Truth Graph Visualizer.

Visualizes the ground truth graph directly from `erp_gw_true` and `gw_bank_true`
along with all records from `erp_ledger`, `gateway_settlements`, and `bank_statement`.
"""

from pathlib import Path
from src.core.config import DB_PATH, ALL_DATA_GRAPH_PATH
from src.reporting.visualizer import generate_graph_visualization


def main():
    print("=" * 80)
    print("  GROUND TRUTH GRAPH VISUALIZER (erp_gw_true & gw_bank_true)")
    print("=" * 80)
    print(f"[*] Extracting ground truth edges from database: {DB_PATH.name}")

    output_path = generate_graph_visualization(
        db_path=DB_PATH,
        output_file=ALL_DATA_GRAPH_PATH,
        use_ground_truth=True,
        include_unmatched=True,
        heading_title="Ground Truth Multi-Source Reconciliation Graph",
    )

    print(f"[✔] Ground truth visualization saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

