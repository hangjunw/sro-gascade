"""
evaluate.py -- leakage-safe evaluation machinery shared by the experiment
drivers.

Curve-level pooling (the mean of a curve's subsequence predictions) is the
primary caliber; subsequence-level outputs are diagnostics only.  Evaluation
units are whole curves (main split) or whole device batches (LOBO), never
single subsequences.  Grade boundaries are always fitted on training-fold
counts only.
"""

import numpy as np

from metrics import grade_metrics, level_of, reg_metrics


def build_index(y):
    """Curve-level index arrays over the subsequence table."""
    curves = y.curve_id.drop_duplicates().tolist()
    cid2i = {c: i for i, c in enumerate(curves)}
    cidx = y.curve_id.map(cid2i).values.astype(int)
    M = np.bincount(cidx, minlength=len(curves)).astype(float)
    curve_count = np.array([y.label_count.values[cidx == i][0] for i in range(len(curves))], float)
    curve_log = np.log1p(curve_count)
    curve_split = np.array([y.main_split.values[cidx == i][0] for i in range(len(curves))])
    curve_fold = np.array([y.lobo_fold.values[cidx == i][0] for i in range(len(curves))])
    return curves, cid2i, cidx, M, curve_count, curve_log, curve_split, curve_fold


def _pool_to_curves(vals_sub, m, cidx, M, curves_in, n_curves):
    """Mean-pool subsequence values to their curves (restricted to curves_in)."""
    sums = np.bincount(cidx[m], weights=vals_sub, minlength=n_curves)
    return (sums / M)[curves_in]


def evaluate_split(yreg_pred_log_sub=None, prob_coral_sub=None, prob_cls_sub=None, *,
                   mask_sub, cidx, M, curves, curve_count, curve_log,
                   b3, b2, n_curves, include_subseq_diag=False):
    """Score one held-out set at the curve level (primary) plus diagnostics."""
    m = mask_sub
    curves_in = np.unique(cidx[m])
    cc = curve_count[curves_in]
    cl = curve_log[curves_in]
    yt3 = level_of(cc, b3)
    yt2 = (cc > b2[0]).astype(int)
    out = {"n_test_curves": int(len(curves_in)),
           "test_curve_ids": [curves[i] for i in curves_in]}

    if yreg_pred_log_sub is not None:
        yp_curve = _pool_to_curves(yreg_pred_log_sub[m], m, cidx, M, curves_in, n_curves)
        out["reg_curve"] = reg_metrics(cl, yp_curve)          # curve level = primary
        if include_subseq_diag:
            out["reg_subseq_diag"] = reg_metrics(curve_log[cidx[m]], yreg_pred_log_sub[m])
        yp3 = level_of(np.expm1(np.clip(yp_curve, 0, None)), b3)
        out["grade3_regthr"] = grade_metrics(yt3, yp3, 3)
        yp2 = (np.expm1(np.clip(yp_curve, 0, None)) > b2[0]).astype(int)
        out["grade2_regthr"] = grade_metrics(yt2, yp2, 2)

    if prob_coral_sub is not None and len(prob_coral_sub) > 0:
        pc = np.stack([_pool_to_curves(prob_coral_sub[m, k], m, cidx, M, curves_in, n_curves)
                       for k in range(prob_coral_sub.shape[1])], axis=1)
        out["grade3_coral"] = grade_metrics(yt3, (pc > 0.5).sum(axis=1), 3)

    if prob_cls_sub is not None and len(prob_cls_sub) > 0:
        nc = prob_cls_sub.shape[1]
        pc = np.stack([_pool_to_curves(prob_cls_sub[m, k], m, cidx, M, curves_in, n_curves)
                       for k in range(nc)], axis=1)
        out["grade3_softmax"] = grade_metrics(yt3, pc.argmax(axis=1), 3)

    return out


def pooled_curve_arrays(yreg_pred_log_sub, prob_coral_sub, prob_cls_sub,
                        mask_sub, cidx, M, curves, curve_count, curve_log, n_curves):
    """Per-curve prediction arrays of one held-out set (for LOBO pooling)."""
    m = mask_sub
    curves_in = np.unique(cidx[m])
    d = {"curve_ids": [curves[i] for i in curves_in],
         "true_count": curve_count[curves_in].tolist()}
    if yreg_pred_log_sub is not None:
        d["pred_log1p"] = _pool_to_curves(yreg_pred_log_sub[m], m, cidx, M,
                                          curves_in, n_curves).tolist()
    if prob_coral_sub is not None and len(prob_coral_sub) > 0:
        d["coral_probs"] = np.stack(
            [_pool_to_curves(prob_coral_sub[m, k], m, cidx, M, curves_in, n_curves)
             for k in range(prob_coral_sub.shape[1])], axis=1).round(4).tolist()
    if prob_cls_sub is not None and len(prob_cls_sub) > 0:
        d["cls_probs"] = np.stack(
            [_pool_to_curves(prob_cls_sub[m, k], m, cidx, M, curves_in, n_curves)
             for k in range(prob_cls_sub.shape[1])], axis=1).round(4).tolist()
    return d


def boundaries_from_train_counts(curve_count, is_train_bool):
    """Three-level quantile boundaries + two-level median, train folds only."""
    counts = curve_count[is_train_bool]
    return (float(np.percentile(counts, 100.0 / 3.0)),
            float(np.percentile(counts, 200.0 / 3.0)),
            float(np.median(counts)))
