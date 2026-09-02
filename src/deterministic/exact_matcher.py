#!/usr/bin/env python3
"""
Phase 3 Entrypoint: Exact & Fuzzy Matching Engine.
"""

from src.deterministic.matcher import run_exact_matching
from src.core.config import DB_PATH


def main(deterministic_only: bool = False):
    stats = run_exact_matching(DB_PATH, deterministic_only=deterministic_only)
    # print(f"[✔] Matching Completed: {stats['matches_count']} graph edges established.")
    print(f"    - ERP <-> Gateway Edges : {len(stats['erp_gw_edges'])}")
    print(f"    - Gateway <-> Bank Edges: {len(stats['gw_bank_edges'])}")


if __name__ == "__main__":
    main()
