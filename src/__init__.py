"""sro-gascade -- code for the paper
"From presence-absence to termite counts and re-calibratable infestation
grades: leakage-safe evaluation of enclosed-chamber CO2-CH4 curves".

Modules:
    tgsd         raw TGSD workbook loader (data1 / data2 batches)
    sro          sliding-window sequence reorganization (paper Eq. 1-4)
    models       GASCADE (DSG + BiLSTM + CORAL) and baselines
    metrics      regression / ordinal-grading metrics
    evaluate     leakage-safe evaluation machinery (curve-level pooling)
    featselect   train-only feature selection (mRMR / LASSO / permutation)
    boundaries   grade-boundary schemes and sensitivity
    ablation     Phase-C ablations (feature set, K, overlap)
"""

import os
import sys

# Allow "python -m src.<module>" from the repository root: the modules in this
# package use flat intra-package imports (from metrics import ...), which
# requires this directory itself on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__version__ = "1.0.0"
