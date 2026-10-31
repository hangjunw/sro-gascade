"""
featselect.py -- train-only feature selection over the 56 group-point features.

Protocol (Supplementary S4): candidates are the 56 features, mean-aggregated
over the K = 8 group points of each subsequence.  Three methods run strictly on
main-split training subsequences:

  1. mRMR (hand-written greedy MID): relevance = 0.5*(normalized F-test +
     normalized mutual information), redundancy = mean |Pearson rho| with the
     already-selected set; top-20 candidates.
  2. LASSO on the log1p target (train-only standardization), lambda chosen by
     curve-grouped GroupKFold(5); non-zero coefficients selected, truncated to
     [10, 25] by |coef|.
  3. Permutation importance (confirmation ranking only, never fed back):
     RF(300) on an internal curve-grouped 85/15 split of the training side,
     n_repeats = 20.

Stability across the eight LOBO folds re-runs methods 1-2 on each fold's
training side only (the held-out batch never enters any fit).  Output:
``featselect_report.json`` with the selected sets and Jaccard stability.

"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TOP_M = 20               # mRMR candidate count
K_MIN, K_MAX = 10, 25    # LASSO selected-set size constraint
SEED = 42
MI_MAX_ROWS = 10000      # MI estimation subsample cap (fixed seed, reproducible)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def mi_and_f(Xm, ym, seed=SEED):
    """F-test + mutual information (subsampled), both normalized."""
    if len(ym) > MI_MAX_ROWS:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ym), MI_MAX_ROWS, replace=False)
        Xs, ys = Xm[idx], ym[idx]
    else:
        Xs, ys = Xm, ym
    F, _ = f_regression(Xs, ys)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    mi = mutual_info_regression(Xs, ys, random_state=seed)
    mi = np.nan_to_num(mi, nan=0.0)
    f_n = F / F.max() if F.max() > 0 else np.zeros_like(F)
    mi_n = mi / mi.max() if mi.max() > 0 else np.zeros_like(mi)
    return 0.5 * (f_n + mi_n), {"F": F, "MI": mi}


def abs_corr(Xm):
    C = np.corrcoef(Xm, rowvar=False)
    return np.nan_to_num(np.abs(C), nan=0.0)


def mrmr_select(Xm, ym, topk=TOP_M):
    """Greedy mRMR (MID): score = relevance - mean|rho| with selected."""
    rel, raw = mi_and_f(Xm, ym)
    R = abs_corr(Xm)
    cand = list(range(Xm.shape[1]))
    selected, scores = [], []
    red = np.zeros(Xm.shape[1])
    for _ in range(min(topk, len(cand))):
        best, best_s = None, -np.inf
        for j in cand:
            s = rel[j] - (red[j] / len(selected) if selected else 0.0)
            if s > best_s:
                best, best_s = j, s
        selected.append(best)
        scores.append(float(best_s))
        cand.remove(best)
        red = red + R[:, best]
    return selected, raw


def lasso_select(Xm, ym, groups, cand_idx, feat_names):
    """LASSO with curve-grouped CV over a fixed grid; non-zero set truncated."""
    Xc = Xm[:, cand_idx]
    sc = StandardScaler().fit(Xc)               # mu/sigma from train only
    Xz = sc.transform(Xc)
    alphas = np.logspace(-3, 1, 12)
    if len(ym) > 6000:                          # CV on a fixed-seed subsample
        rng = np.random.RandomState(SEED)
        si = rng.choice(len(ym), 6000, replace=False)
    else:
        si = np.arange(len(ym))
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(Xz[si], ym[si], groups[si]))
    mse = np.zeros(len(alphas))
    for (ti, vi) in splits:
        m = Lasso(alpha=alphas[0], max_iter=2000, selection="random", random_state=SEED)
        for ai, a in enumerate(alphas):
            m.alpha = a
            m.fit(Xz[si][ti], ym[si][ti])
            mse[ai] += np.mean((m.predict(Xz[si][vi]) - ym[si][vi]) ** 2)
    best_a = float(alphas[int(np.argmin(mse))])
    las = Lasso(alpha=best_a, max_iter=5000, selection="random",
                random_state=SEED).fit(Xz, ym)
    coef = np.abs(las.coef_)
    nz = [i for i in np.argsort(-coef)[:K_MAX] if coef[i] > 0]
    if len(nz) < K_MIN:
        nz = list(np.argsort(-coef)[:K_MIN])
    return [cand_idx[i] for i in nz], best_a, \
           {feat_names[cand_idx[i]]: float(las.coef_[i]) for i in nz}


def perm_importance_selected(Xm, ym, groups, sel_idx, feat_names):
    """Permutation importance on an internal curve-grouped 85/15 split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_i, ca_i = next(gss.split(Xm[:, sel_idx], ym, groups))
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(Xm[np.ix_(tr_i, sel_idx)], ym[tr_i])
    pi = permutation_importance(rf, Xm[np.ix_(ca_i, sel_idx)], ym[ca_i],
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    order = np.argsort(-pi.importances_mean)
    return {feat_names[sel_idx[i]]: float(pi.importances_mean[i]) for i in order}, \
           [sel_idx[i] for i in order[:TOP_M]]


def run(sro_dir, out_json, log=print):
    t0 = time.time()
    sro_dir, out_json = Path(sro_dir), Path(out_json)
    meta = json.loads((sro_dir / "meta.json").read_text(encoding="utf-8"))
    feat_names = meta["feature_names"]
    X = np.load(sro_dir / "X.npy")                       # (T, 8, 56) float32
    y = pd.read_csv(sro_dir / "y.csv")
    assert X.shape[0] == len(y) and X.shape[2] == 56
    Xagg = X.mean(axis=1).astype(np.float64)             # group-point mean pooling
    t_target = y["label_log1p"].to_numpy()
    curve_ids = y["curve_id"].to_numpy()
    is_train = (y["main_split"] == "train").to_numpy()
    log(f"[load] X{X.shape} -> pooled {Xagg.shape}; train subsequences "
        f"{is_train.sum()} (val/test never enter any fit)")

    Xtr, ytr, ctr = Xagg[is_train], t_target[is_train], curve_ids[is_train]

    log("=== (1) mRMR (main train) ===")
    mrmr_idx, raw_main = mrmr_select(Xtr, ytr)
    log("top-20: " + ", ".join(feat_names[i] for i in mrmr_idx))

    log("=== (2) LASSO (main train, GroupKFold by curve) ===")
    lasso_idx, alpha, lasso_coef = lasso_select(Xtr, ytr, ctr, mrmr_idx, feat_names)
    log(f"alpha={alpha:.2e}, selected {len(lasso_idx)} dims: "
        + ", ".join(feat_names[i] for i in lasso_idx))

    log("=== (3) permutation importance (main train, internal 85/15 by curve) ===")
    perm_ranked, perm_top20 = perm_importance_selected(Xtr, ytr, ctr, lasso_idx, feat_names)
    log("top-10: " + ", ".join(f"{k}({v:.4f})" for k, v in list(perm_ranked.items())[:10]))

    inter = sorted(set(mrmr_idx) & set(lasso_idx) & set(perm_top20))
    union = sorted(set(mrmr_idx) | set(lasso_idx) | set(perm_top20))
    log(f"=== three-method intersection {len(inter)} dims; union {len(union)} dims")

    log("=== LOBO stability (per fold: train side only) ===")
    lobo_sets = {}
    for fold in sorted(y["lobo_fold"].unique()):
        m = (y["lobo_fold"] != fold).to_numpy()
        Xf, yf, cf = Xagg[m], t_target[m], curve_ids[m]
        idx20, _ = mrmr_select(Xf, yf)
        sel, _, _ = lasso_select(Xf, yf, cf, idx20, feat_names)
        lobo_sets[fold] = sel
        log(f"  {fold}: |set|={len(sel)}  " + ", ".join(feat_names[i] for i in sel[:6]) + " ...")

    folds = list(lobo_sets)
    pair_j = [jaccard(lobo_sets[a], lobo_sets[b]) for i, a in enumerate(folds)
              for b in folds[i + 1:]]
    vs_main = [jaccard(lobo_sets[f], lasso_idx) for f in folds]
    j_mean, j_min = float(np.mean(pair_j)), float(np.min(pair_j))
    log(f"Jaccard: pairwise mean {j_mean:.3f} (min {j_min:.3f}; >=0.6 = stable); "
        f"vs main-train set mean {np.mean(vs_main):.3f}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "src/featselect.py",
        "leakage_discipline": ("all fits use main_split=='train' only; LOBO stability uses "
                               "each fold's train side; permutation importance is computed on "
                               "an internal curve-grouped 85/15 validation side; val/test "
                               "subsequences never enter any fit."),
        "feature_repr": "56 group-point features, mean-aggregated over the K=8 group points",
        "mrmr": {"top20": [feat_names[i] for i in mrmr_idx],
                 "relevance": "0.5*(F_test_norm + MI_norm)",
                 "redundancy": "mean |Pearson corr| with selected (MID greedy)"},
        "lasso": {"alpha": float(alpha), "n_selected": len(lasso_idx),
                  "selected": [feat_names[i] for i in lasso_idx],
                  "coefs": lasso_coef,
                  "cv": "GroupKFold(5) by curve_id, log1p target, standardized"},
        "permutation_importance": {"ranking": perm_ranked,
                                   "top20": [feat_names[i] for i in perm_top20],
                                   "note": "confirmation ranking only, not fed back"},
        "three_method_intersection": [feat_names[i] for i in inter],
        "three_method_union_size": len(union),
        "lobo_stability": {"per_fold_sets": {f: [feat_names[i] for i in s]
                                             for f, s in lobo_sets.items()},
                           "pairwise_jaccard_mean": j_mean,
                           "pairwise_jaccard_min": j_min,
                           "jaccard_vs_main_mean": float(np.mean(vs_main)),
                           "stable_threshold": 0.6,
                           "stable": bool(j_mean >= 0.6)},
        "top10_by_lasso": [feat_names[i] for i in lasso_idx[:10]],
        "wall_sec": time.time() - t0,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[write] {out_json}  ({time.time() - t0:.1f}s)")
    return report


def main():
    ap = argparse.ArgumentParser(description="Train-only feature selection over 56 features")
    ap.add_argument("--sro", required=True, help="SRO output directory (X.npy, y.csv, meta.json)")
    ap.add_argument("--out", required=True, help="output featselect_report.json path")
    args = ap.parse_args()
    run(args.sro, args.out)


if __name__ == "__main__":
    main()
