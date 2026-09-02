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
    print("  COMBINED GROUND TRUTH + PREDICTION VISUALIZER")
    print("=" * 80)
    print(f"[*] Overlaying predicted edges on the full ground-truth graph: {DB_PATH.name}")

    output_path = generate_graph_visualization(
        db_path=DB_PATH,
        output_file=ALL_DATA_GRAPH_PATH,
        use_ground_truth=False,
        include_unmatched=True,
        heading_title="Ground Truth + Predicted Reconciliation Graph",
    )

    print(f"[✔] Combined visualization saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

