#!/usr/bin/env python3
"""
Phase 3 Entrypoint: Exact & Fuzzy Matching Engine.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.matcher import run_exact_matching
from src.config import DB_PATH


def main():
    stats = run_exact_matching(DB_PATH)
    print(f"[✔] Matching Completed: {stats['matches_count']} graph edges established.")


if __name__ == "__main__":
    main()