"""
metrics.py -- regression and ordinal-grading metrics used throughout the paper.

Regression is scored on the log1p scale (primary calibration scale) and, after
expm1 back-transform, on the raw count scale.  Grading is scored with macro-F1
(primary), quadratic weighted kappa (QWK, Cohen 1968), ordinal accuracy and
off-by-one accuracy.  All functions are pure NumPy/scikit-learn.
"""

import numpy as np


def r2_score(yt, yp):
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def reg_metrics(yt_log, yp_log):
    """Dual-caliber regression metrics (both inputs in log1p space)."""
    yt_c = np.expm1(yt_log)
    yp_c = np.clip(np.expm1(yp_log), 0.0, None)
    return {
        "mae_log1p": float(np.mean(np.abs(yt_log - yp_log))),
        "rmse_log1p": float(np.sqrt(np.mean((yt_log - yp_log) ** 2))),
        "r2_log1p": r2_score(yt_log, yp_log),
        "mae_count": float(np.mean(np.abs(yt_c - yp_c))),
        "rmse_count": float(np.sqrt(np.mean((yt_c - yp_c) ** 2))),
        "r2_count": r2_score(yt_c, yp_c),
    }


def grade_metrics(yt, yp, n_levels):
    """Ordinal grading metrics (labels 0..n_levels-1)."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    yt = np.asarray(yt, int)
    yp = np.asarray(yp, int)
    labels = list(range(n_levels))
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic", labels=labels)),
        "ordacc": float(accuracy_score(yt, yp)),
        "off_by_one_acc": float(np.mean(np.abs(yt - yp) <= 1)),
    }


def level_of(counts, b):
    """Ordered level of counts under boundaries b (L = sum(count > b_k))."""
    counts = np.asarray(counts, dtype=float)
    out = np.zeros(len(counts), dtype=int)
    for bk in b:
        out += (counts > bk).astype(int)
    return out
