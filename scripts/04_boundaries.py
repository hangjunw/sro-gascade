#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_boundaries.py -- grade-boundary schemes and sensitivity (main-text Fig. 6; Supplementary Table S7).

Thin wrapper around ``src.boundaries.run``: fits four three-level boundary
schemes on training curve counts, re-trains the main model (3 seeds) and
re-scores the identical test predictions under every scheme plus +-10%/+/-5%
threshold shifts.  With ``--from-preds`` the main-model predictions are read
from a file written by ``scripts/03_train.py --dump-preds`` instead of being
re-trained, so that this study and the main-split table score the same model.

Usage:
  python scripts/04_boundaries.py --sro results/sro_g1 \
      --out results/phaseC_boundaries.json [--hidden 128] \
      [--from-preds results/main_model_test_predictions.json]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from boundaries import run


def main():
    ap = argparse.ArgumentParser(description="Grade boundary schemes and sensitivity")
    ap.add_argument("--sro", required=True, help="SRO output directory")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--hidden", type=int, default=128, help="BiLSTM hidden size")
    ap.add_argument("--from-preds", default="",
                    help="JSON of main-model test-curve count predictions (written by "
                         "scripts/03_train.py --dump-preds); when given, the model is NOT "
                         "re-trained and the same predictions are re-scored under every scheme")
    args = ap.parse_args()
    run(args.sro, args.out, hidden=args.hidden, from_preds=args.from_preds)


if __name__ == "__main__":
    main()
