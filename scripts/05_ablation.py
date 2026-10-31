#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_ablation.py -- Phase C ablations (Supplementary Tables S12–S14).

Thin wrapper around ``src.ablation.run``: feature-selection top-20 mask,
K = 4 shorter subsequences and non-overlapping n = 80 subsequences, each
re-exported by the SRO pipeline and re-trained under the main configuration
(3 seeds, main split); deltas are reported against the main configuration.

Usage:
  python scripts/05_ablation.py --sro results/sro_g1 --series data/series \
      --curated data/series-new_curated.csv --results results/phaseB_results.json \
      --featselect results/sro_g1/featselect_report.json \
      --out results/phaseC_ablation.json --work results/ablation_variants
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ablation import run


def main():
    ap = argparse.ArgumentParser(description="Phase C ablations")
    ap.add_argument("--sro", required=True, help="main SRO output directory")
    ap.add_argument("--series", required=True, help="cleaned per-curve series directory")
    ap.add_argument("--curated", required=True, help="series-new_curated.csv")
    ap.add_argument("--results", required=True, help="training results JSON (03 output)")
    ap.add_argument("--featselect", required=True, help="featselect_report.json")
    ap.add_argument("--out", required=True, help="output ablation JSON")
    ap.add_argument("--work", required=True, help="directory for variant re-exports")
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()
    run(args.sro, args.series, args.curated, args.results, args.featselect,
        args.out, args.work, hidden=args.hidden)


if __name__ == "__main__":
    main()
