#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_run_sro.py -- run the G1 SRO export (paper equations 1-4).

Thin wrapper around ``src.sro.export``: reads the cleaned per-curve series and
the curated table, writes X.npy / y.csv / meta.json / gate_report.json.

Usage:
  python scripts/02_run_sro.py --series data/series \
      --curated data/series-new_curated.csv --out results/sro_g1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sro import export


def main():
    ap = argparse.ArgumentParser(description="G1 sliding-window sequence reorganization")
    ap.add_argument("--series", required=True, help="cleaned per-curve series directory")
    ap.add_argument("--curated", required=True, help="series-new_curated.csv")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n", type=int, default=10, dest="n_step")
    args = ap.parse_args()
    _, _, summary = export(args.series, args.curated, args.out,
                           L=args.L, K=args.K, n_step=args.n_step)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
