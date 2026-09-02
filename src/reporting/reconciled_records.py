#!/usr/bin/env python3
"""
Graph Visualizer: Reconciled Records Pre-Computed Grid Network.

Visualizes only matched reconciliation clusters (ERP -> GW -> Bank).
"""

from pathlib import Path
from src.core.config import DB_PATH, RECONCILIATION_GRAPH_PATH
from src.reporting.visualizer import generate_graph_visualization


def main():
    print("=" * 80)
    print("  RECONCILIATION GRAPH VISUALIZER (EXACT/RECONCILED GRID)")
    print("=" * 80)
    print(f"[*] Extracting matched edges from database: {DB_PATH.name}")

    output_path = generate_graph_visualization(
        db_path=DB_PATH,
        output_file=RECONCILIATION_GRAPH_PATH,
        include_unmatched=False,
        heading_title="Multi-Source Reconciliation Graph (Reconciled Clusters Only)",
    )

    print(f"[✔] Visualization saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

