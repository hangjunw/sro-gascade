"""
boundaries.py -- infestation-grade boundary schemes and their sensitivity.

Four three-level boundary schemes are compared (all fitted on main-split
TRAINING curve counts only; test curves only ever have thresholds applied):

  1. ``quantile`` -- training-count terciles q33/q67 (main scheme);
  2. ``jenks``    -- 1-D Fisher-Jenks natural breaks (exact DP, K = 3);
  3. ``gmm``      -- 1-D Gaussian mixture (K = 3), boundaries at adjacent
                     component density crossings; a BIC table over
                     K in {2, 3, 4} is reported for order selection;
  4. ``manual``   -- fixed artificial thresholds <=100 / 101-499 / >=500
                     (the standards-style comparator).

The main model is re-trained under the identical configuration (3 seeds) and
its test predictions are re-scored under every scheme without retraining.
Alternatively, ``--from-preds`` re-uses a prediction file written by
``scripts/03_train.py --dump-preds``, so that this study and the main-split
table score the *same* model instance instead of two separate ones.
A boundary-sensitivity analysis then shifts both thresholds of each scheme
simultaneously by -10%, -5%, +5%, +10% and reports the maximum absolute
metric change in percentage points.

"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import build_index, _pool_to_curves
from metrics import grade_metrics, level_of
from models import (BiLSTMNet, K, LR, MAX_EPOCHS, N_FEAT, PATIENCE,
                    fit_transform_scaler, seed_all, train_torch)

warnings.filterwarnings("ignore")

SEEDS = [2024, 1, 2]
MANUAL_BOUNDS = (100.0, 499.0)             # <=100 / 101-499 / >=500
SHIFTS = [-0.10, -0.05, 0.05, 0.10]
METRICS = ["macro_f1", "qwk", "ordacc", "off_by_one_acc"]


# ------------------------------------------------------------------ schemes
def fisher_jenks_bounds(values, k=3):
    """1-D Fisher-Jenks natural breaks (exact DP, within-class SS).

    Returns ``(b1, b2, total_within_ss)`` where b1/b2 are the upper bounds of
    class 1 / class 2 (inclusive), matching the ``level_of`` ``>`` rule.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    S1 = np.concatenate([[0.0], np.cumsum(v)])
    S2 = np.concatenate([[0.0], np.cumsum(v ** 2)])

    def wcv(i, j):                          # within-class SS for v[i:j]
        m = j - i
        s1, s2 = S1[j] - S1[i], S2[j] - S2[i]
        return s2 - s1 * s1 / m

    INF = float("inf")
    D = np.full((k + 1, n + 1), INF)
    B = np.zeros((k + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            for i in range(kk - 1, j):
                cost = D[kk - 1, i] + wcv(i, j)
                if cost < D[kk, j]:
                    D[kk, j] = cost
                    B[kk, j] = i
    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = B[kk, j]
        bounds.append(float(v[i - 1]))
        j = i
    bounds = sorted(bounds)
    return bounds[0], bounds[1], float(D[k, n])


def gmm_bounds(values, k):
    """1-D GMM with fixed component count; boundaries at adjacent density
    crossings (numeric root finding; midpoint fallback when no crossing)."""
    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture

    x = np.asarray(values, dtype=float).reshape(-1, 1)
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=10, random_state=2024).fit(x)
    order = np.argsort(gm.means_.ravel())
    mus = gm.means_.ravel()[order]
    sds = np.sqrt(gm.covariances_.ravel())[order]
    wts = gm.weights_.ravel()[order]

    def crossing(i, j):
        """Solve w_i N(x|mu_i,sd_i) = w_j N(x|mu_j,sd_j) within (mu_i, mu_j)."""
        m1, s1, w1 = mus[i], sds[i], wts[i]
        m2, s2, w2 = mus[j], sds[j], wts[j]

        def f(xv):
            return np.log(w1 * norm.pdf(xv, m1, s1) + 1e-300) - \
                   np.log(w2 * norm.pdf(xv, m2, s2) + 1e-300)

        grid = np.linspace(m1, m2, 801)
        fv = f(grid)
        idx = np.where(np.diff(np.sign(fv - 0.0)) != 0)[0]
        if len(idx) == 0:
            return float((m1 + m2) / 2.0), False
        i0 = idx[0]
        lo, hi = grid[i0], grid[i0 + 1]
        for _ in range(60):                 # bisection refinement
            mid = (lo + hi) / 2.0
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return float((lo + hi) / 2.0), True

    bounds, exact_flags = [], []
    for i in range(k - 1):
        b, exact = crossing(i, i + 1)
        bounds.append(b)
        exact_flags.append(exact)
    return bounds, float(gm.bic(x)), exact_flags


def gmm_bic_table(values, k_range=(2, 3, 4)):
    """BIC order-selection table over K in {2, 3, 4}."""
    from sklearn.mixture import GaussianMixture

    x = np.asarray(values, dtype=float).reshape(-1, 1)
    fits = {}
    for kk in k_range:
        gm = GaussianMixture(n_components=kk, covariance_type="full",
                             n_init=10, random_state=2024).fit(x)
        fits[kk] = float(gm.bic(x))
    return fits, int(min(fits, key=lambda kk: fits[kk]))


def scheme_metrics(true_counts, pred_counts_by_seed, bounds):
    """All metrics of one boundary scheme (3-seed mean + per-seed values)."""
    n_levels = len(bounds) + 1
    yt = level_of(true_counts, bounds)
    dist = {f"L{k + 1}": int((yt == k).sum()) for k in range(n_levels)}
    per_seed = {}
    for seed, pc in pred_counts_by_seed.items():
        per_seed[str(seed)] = grade_metrics(yt, level_of(pc, bounds), n_levels)
    mets = list(next(iter(per_seed.values())).keys())
    mean = {m: float(np.mean([per_seed[str(sd)][m] for sd in pred_counts_by_seed]))
            for m in mets}
    return {"n_levels": n_levels, "test_class_dist": dist,
            "per_seed": per_seed, "mean": mean}


# ------------------------------------------------------------------ driver
def run(sro_dir, out_json, hidden=128, log=print, from_preds=""):
    t0 = time.time()
    sro_dir, out_json = Path(sro_dir), Path(out_json)

    X = np.load(sro_dir / "X.npy").astype(np.float32)
    y = pd.read_csv(sro_dir / "y.csv", encoding="utf-8-sig")
    gate = json.loads((sro_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert X.shape == (len(y), K, N_FEAT)

    curves, _cid2i, cidx, M, curve_count, _curve_log, curve_split, _curve_fold = build_index(y)
    n_curves = len(curves)
    split_arr = y.main_split.values
    tr_m, va_m, te_m = split_arr == "train", split_arr == "val", split_arr == "test"
    tr_curve_mask = curve_split == "train"
    te_curve_mask = curve_split == "test"
    train_counts = curve_count[tr_curve_mask]
    test_true_counts = curve_count[te_curve_mask]
    test_curve_ids = [curves[i] for i in range(n_curves) if te_curve_mask[i]]

    # ---- four boundary schemes, all fitted on the 62 training curves
    log("========== fitting four boundary schemes (train curves) ==========")
    q1 = float(np.quantile(train_counts, 1 / 3))
    q2 = float(np.quantile(train_counts, 2 / 3))
    assert abs(q1 - gate["boundaries"]["q33"]) < 0.5 and \
           abs(q2 - gate["boundaries"]["q67"]) < 0.5, "quantiles disagree with gate_report"
    schemes = {"quantile": {"name": "count terciles q33/q67 (main scheme)", "bounds": [q1, q2]}}
    j1, j2, jwcv = fisher_jenks_bounds(train_counts, k=3)
    schemes["jenks"] = {"name": "Jenks natural breaks (Fisher-Jenks DP, K=3)",
                        "bounds": [j1, j2], "total_within_ss": jwcv}
    bic_fits, bic_best_k = gmm_bic_table(train_counts)
    g3b, g3bic, g3exact = gmm_bounds(train_counts, k=3)
    schemes["gmm"] = {"name": "1-D GMM (K=3 density crossings)",
                      "bounds": g3b, "bic_K3": g3bic, "crossing_exact": g3exact,
                      "bic_by_K": {str(k): v for k, v in bic_fits.items()},
                      "bic_selected_K": bic_best_k}
    schemes["manual"] = {"name": "fixed artificial <=100 / 101-499 / >=500",
                         "bounds": list(MANUAL_BOUNDS)}
    for s in schemes.values():
        log(f"  {s['name']}: bounds={np.round(s['bounds'], 2).tolist()}")
    gmm4_aux = None
    if bic_best_k == 4:                     # auxiliary four-level report
        g4b, g4bic, g4exact = gmm_bounds(train_counts, k=4)
        gmm4_aux = {"name": "1-D GMM (BIC-selected K=4, auxiliary)",
                    "bounds": g4b, "bic_K4": g4bic, "crossing_exact": g4exact}

    # ---- main-model test predictions: either re-use a dumped prediction file
    #      (--from-preds, so that this study and the main-split table score the
    #      SAME model instance) or re-train the main model (3 seeds).
    per_seed_pred_counts = {}
    if from_preds:
        fp = json.loads(Path(from_preds).read_text(encoding="utf-8"))
        test_true_counts = np.asarray(fp["test_true_counts"], dtype=float)
        test_curve_ids = list(fp["test_curve_ids"])
        raw = {int(s) if str(s).isdigit() else s: np.asarray(v, dtype=float)
               for s, v in fp["per_seed_pred_counts"].items()}
        use_seeds = [sd for sd in SEEDS if sd in raw] or sorted(raw)
        per_seed_pred_counts = {sd: raw[sd] for sd in use_seeds}
        log(f"[from-preds] loaded test-curve count predictions for {len(use_seeds)} seed(s) "
            f"from {from_preds}; model training skipped")
    else:
        import torch
        from models import make_tensors

        log(f"========== re-training BiLSTM+DSG main model (hidden={hidden}, seeds={SEEDS}) ==========")
        Xs_flat = fit_transform_scaler(X, np.where(tr_m)[0])
        Xs_t = torch.tensor(Xs_flat.reshape(-1, K, N_FEAT))
        lvl_all = level_of(y.label_count.values, (q1, q2))
        yreg_t, yw_t, ycoral_t, ycls_t = make_tensors(y.assign(label_level3=lvl_all), lvl_all)
        tr_idx, va_idx = np.where(tr_m)[0], np.where(va_m)[0]

        use_seeds = list(SEEDS)
        for seed in use_seeds:
            seed_all(seed)
            model = BiLSTMNet(hidden=hidden, use_dsg=True)
            model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             tr_idx, va_idx, mode="dual", nat_on=False,
                                             seed=seed, tag=f"bnd_s{seed}", log=log)
            regs = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(Xs_t), 4096):
                    r, _, _, _ = model(Xs_t[i:i + 4096])
                    regs.append(r.numpy())
            reg_s = np.concatenate(regs)
            del model
            sums = np.bincount(cidx[te_m], weights=reg_s[te_m], minlength=n_curves)
            pooled = (sums / M)[np.where(te_curve_mask)[0]]
            per_seed_pred_counts[seed] = np.clip(np.expm1(pooled), 0.0, None)

    yt3 = level_of(test_true_counts, (q1, q2))
    for seed in use_seeds:
        gm = grade_metrics(yt3, level_of(per_seed_pred_counts[seed], (q1, q2)), 3)
        log(f"  [sanity seed{seed}] quantile macro_f1={gm['macro_f1']:.4f} "
            f"qwk={gm['qwk']:.4f}")

    # ---- scheme comparison table
    log("========== four-scheme test comparison ==========")
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "src/boundaries.py",
        "model": (f"BiLSTM+DSG+CORAL (hidden={hidden}, {len(use_seeds)} seed(s), main split)"
                  + ("; predictions reused from " + str(from_preds) if from_preds
                     else "; re-trained)")),
        "fitting_rule": "all schemes fitted on main-train curve counts only; test only applies thresholds",
        "test_true_counts": test_true_counts.tolist(),
        "test_curve_ids": test_curve_ids,
        "schemes": {},
        "sensitivity": {},
    }
    all_schemes = dict(schemes)
    if gmm4_aux:
        all_schemes["gmm_k4_bic_aux"] = gmm4_aux
    for name, s in all_schemes.items():
        b = tuple(s["bounds"])
        m = scheme_metrics(test_true_counts, per_seed_pred_counts, b)
        entry = {"name": s["name"], "bounds": list(b), **m}
        for extra in ("total_within_ss", "bic_K3", "bic_K4", "bic_by_K",
                      "bic_selected_K", "crossing_exact"):
            if extra in s:
                entry[extra] = s[extra]
        out["schemes"][name] = entry
        log(f"  {name}: bounds={np.round(b, 2).tolist()} dist={m['test_class_dist']} "
            f"macroF1={m['mean']['macro_f1']:.4f} QWK={m['mean']['qwk']:.4f}")

    # ---- +/-10% (and +/-5%) boundary sensitivity: same predictions, new thresholds
    log("========== boundary sensitivity ==========")
    for name, s in out["schemes"].items():
        b0 = np.array(s["bounds"], dtype=float)
        base = s["per_seed"]
        n_levels = len(b0) + 1
        shifts_res = {}
        max_chg = {m: 0.0 for m in METRICS}
        for sh in SHIFTS:
            b = tuple(b0 * (1.0 + sh))
            per_seed = {}
            for seed in use_seeds:
                per_seed[str(seed)] = grade_metrics(level_of(test_true_counts, b),
                                                    level_of(per_seed_pred_counts[seed], b),
                                                    n_levels)
            mean = {m: float(np.mean([per_seed[str(sd)][m] for sd in use_seeds])) for m in METRICS}
            mean_base = {m: float(np.mean([base[str(sd)][m] for sd in use_seeds])) for m in METRICS}
            dpp = {m: abs(mean[m] - mean_base[m]) * 100.0 for m in METRICS}
            for m in METRICS:
                max_chg[m] = max(max_chg[m], dpp[m])
            shifts_res[f"{int(sh * 100):+d}%"] = {
                "bounds": list(b), "mean": mean,
                "abs_change_pp": {m: round(v, 2) for m, v in dpp.items()}}
        out["sensitivity"][name] = {
            "shifts": shifts_res,
            "max_abs_change_pp": {m: round(v, 2) for m, v in max_chg.items()},
            "note": "max over the 4 shifts (+/-10%, +/-5%) of |mean - baseline| in pp",
        }
        log(f"  {name}: max d pp={out['sensitivity'][name]['max_abs_change_pp']}")
    out["sensitivity_protocol"] = {
        "shifts": [f"{int(s * 100):+d}%" for s in SHIFTS],
        "rule": "both thresholds scaled by (1+s) together; same test predictions, no retraining",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[write] {out_json}  ({time.time() - t0:.1f}s)")
    return out


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
