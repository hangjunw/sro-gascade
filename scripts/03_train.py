#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train.py -- model training and leakage-safe evaluation (paper Table 2, Fig. 4-5).

Three stages, all on the G1 SRO export (``results/sro_g1``):

  main  -- the fixed 62/15/12 curve-level main split, 3 seeds (2024, 1, 2):
           Ridge, random forest, XGBoost (small grid on validation), 1D-CNN,
           BiLSTM+DSG+CORAL (GASCADE, the main model), BiLSTM without DSG
           (ablation off), BiLSTM+DSG with a softmax-CE head, and GASCADE with
           the NAT-lite loss weighting on.  The BiLSTM hidden size (64/128) is
           selected on the validation curves.  Regression is scored on the
           log1p scale (primary) and the count scale; grading on both the
           ordinal-head route (i) and the boundary route (ii).
  lobo  -- eight leave-one-batch-out folds (one per evaluation unit; batches
           11+12 jointly): GASCADE (seed 2024) and XGBoost per fold, each with
           an in-fold 90/10 curve split for early stopping, plus pooled
           across-fold metrics.
  leak  -- the leakage-control contrast: the identical dataset, export and
           gated-BiLSTM configuration, but subsequences split at random
           80/10/10 at the window level (3 seeds).  Compared with the
           curve-grouped arm at the subsequence-level caliber.

Protocol rules enforced throughout: (1) all subsequences of one curve stay in
one split; (2) the repeat-test pair group11plus12 is one fold; (3) features,
boundaries and normalization statistics are fitted on training folds only;
(4) curve-level mean pooling is the primary caliber.

Usage:
  python scripts/03_train.py --sro results/sro_g1 --out results/phaseB_results.json \
      --stages main,lobo,leak [--smoke] [--dump-preds results/main_model_test_predictions.json]
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from evaluate import (build_index, evaluate_split, pooled_curve_arrays,
                      boundaries_from_train_counts, _pool_to_curves)
from metrics import grade_metrics, level_of, reg_metrics
from models import (BATCH, CNN1D, BiLSTMCESoftmax, BiLSTMNet, K, LR, MAX_EPOCHS,
                    NAT_WEIGHTS, N_FEAT, N_JOBS, PATIENCE, SEEDS, fit_transform_scaler,
                    make_tensors, make_xgb_like, predict_torch, run_boost,
                    run_ridge, run_rf, seed_all, train_torch)

warnings.filterwarnings("ignore")

import torch  # noqa: E402

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)


def log(msg):
    logger.info(msg)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_results(res, path):
    res["last_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=_json_default)
    os.replace(tmp, path)


def record_run(res, key, payload, path):
    res["runs"][key] = payload
    save_results(res, path)
    log(f"[checkpoint] {key} written (wall={payload.get('wall_sec', 0):.0f}s)")


def summarize(res):
    """Aggregate main-split runs (mean +- sd over seeds) and the leakage contrast."""
    def agg(model_filter, metric_path):
        vals = []
        for k, v in res["runs"].items():
            if not (k.startswith(f"main|{model_filter}|seed")
                    or (model_filter == "ridge" and k == "main|ridge")):
                continue
            cur = v.get("test", {})
            for p in metric_path.split("."):
                cur = cur.get(p, None) if isinstance(cur, dict) else None
            if cur is not None and cur == cur:
                vals.append(float(cur))
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4), "n": len(vals),
                "values": [round(v, 4) for v in vals]}

    spec = {
        "ridge": ["reg_curve.mae_count", "reg_curve.mae_log1p", "reg_curve.r2_count",
                  "grade3_regthr.macro_f1", "grade3_regthr.qwk", "grade2_regthr.macro_f1"],
        "rf": ["reg_curve.mae_count", "reg_curve.mae_log1p", "reg_curve.r2_count",
               "grade3_softmax.macro_f1", "grade3_softmax.qwk", "grade3_regthr.macro_f1"],
        "boost": ["reg_curve.mae_count", "reg_curve.mae_log1p", "reg_curve.r2_count",
                  "grade3_softmax.macro_f1", "grade3_softmax.qwk", "grade3_regthr.macro_f1",
                  "grade2_regthr.macro_f1"],
        "cnn": ["reg_curve.mae_count", "reg_curve.mae_log1p", "reg_curve.r2_count",
                "grade3_coral.macro_f1", "grade3_coral.qwk", "grade2_regthr.macro_f1"],
        "bilstm_dsg": ["reg_curve.mae_count", "reg_curve.mae_log1p", "reg_curve.r2_count",
                       "grade3_coral.macro_f1", "grade3_coral.qwk", "grade3_regthr.macro_f1",
                       "grade2_regthr.macro_f1", "reg_subseq_diag.mae_log1p",
                       "reg_subseq_diag.mae_count"],
        "bilstm_nodsg": ["reg_curve.mae_count", "reg_curve.mae_log1p", "reg_curve.r2_count",
                         "grade3_coral.macro_f1", "grade3_coral.qwk"],
        "bilstm_ce": ["grade3_softmax.macro_f1", "grade3_softmax.qwk", "grade3_softmax.ordacc"],
        "bilstm_dsg_natlite": ["reg_curve.mae_count", "reg_curve.mae_log1p",
                               "grade3_coral.macro_f1", "grade3_coral.qwk"],
    }
    res.setdefault("aggregate", {})["main_split_test"] = {
        m: {p: agg(m, p) for p in paths} for m, paths in spec.items()}

    leak = [v for k, v in res["runs"].items() if k.startswith("leak|winrand|seed")]
    grp = [v["test"] for k, v in res["runs"].items()
           if k.startswith("main|bilstm_dsg|seed") and "reg_subseq_diag" in v.get("test", {})]
    if leak and grp:
        lm = [v["test_window_level"]["reg"]["mae_log1p"] for v in leak]
        lm_c = [v["test_window_level"]["reg"]["mae_count"] for v in leak]
        gm = [v["reg_subseq_diag"]["mae_log1p"] for v in grp]
        gm_c = [v["reg_subseq_diag"]["mae_count"] for v in grp]
        res["aggregate"]["leakage_contrast"] = {
            "note": ("same subsequence-level test caliber; MAE difference = curve-grouped "
                     "(this paper) - window-random; a positive value is the leakage-driven "
                     "inflation (window-random looks better = more optimistic)"),
            "winrand_mae_log1p": {"mean": round(float(np.mean(lm)), 4),
                                  "std": round(float(np.std(lm)), 4), "n": len(lm)},
            "grouped_mae_log1p": {"mean": round(float(np.mean(gm)), 4),
                                  "std": round(float(np.std(gm)), 4), "n": len(gm)},
            "inflation_mae_log1p": round(float(np.mean(gm) - np.mean(lm)), 4),
            "winrand_mae_count": {"mean": round(float(np.mean(lm_c)), 4),
                                  "std": round(float(np.std(lm_c)), 4)},
            "grouped_mae_count": {"mean": round(float(np.mean(gm_c)), 4),
                                  "std": round(float(np.std(gm_c)), 4)},
            "inflation_mae_count": round(float(np.mean(gm_c) - np.mean(lm_c)), 4),
            "winrand_qwk": round(float(np.mean(
                [v["test_window_level"]["grade3_coral"]["qwk"] for v in leak])), 4),
            "grouped_qwk_subseq": round(float(np.mean(
                [v["grade3_coral"]["qwk"] for v in grp if "grade3_coral" in v])), 4),
        }
        log(f"[leak] inflation (log1p MAE difference) = "
            f"{res['aggregate']['leakage_contrast']['inflation_mae_log1p']}")


def main():
    ap = argparse.ArgumentParser(description="Model training and leakage-safe evaluation")
    ap.add_argument("--sro", required=True, help="SRO output directory (X.npy, y.csv, ...)")
    ap.add_argument("--out", required=True, help="output results JSON")
    ap.add_argument("--stages", default="main,lobo,leak")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: 3 epochs, single seed, first two LOBO folds")
    ap.add_argument("--dump-preds", default="",
                    help="write the main model's (bilstm_dsg) test-curve count "
                         "predictions for all seeds to this JSON; consumable by "
                         "scripts/04_boundaries.py --from-preds")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    epochs = 3 if args.smoke else MAX_EPOCHS
    seeds = [2024] if args.smoke else SEEDS
    dump = {}                      # main-model test predictions for --dump-preds

    t_start = time.time()
    log(f"=== 03_train.py stages={sorted(stages)} smoke={args.smoke} ===")

    X = np.load(os.path.join(args.sro, "X.npy")).astype(np.float32)
    y = pd.read_csv(os.path.join(args.sro, "y.csv"), encoding="utf-8-sig")
    gate = json.loads(open(os.path.join(args.sro, "gate_report.json"),
                           encoding="utf-8").read())
    assert X.shape == (len(y), K, N_FEAT)
    assert not np.isnan(X).any() and not np.isinf(X).any()
    log(f"[data] X={X.shape}, y={len(y)} subsequences, "
        f"curves={y.curve_id.nunique()}")

    curves, cid2i, cidx, M, curve_count, curve_log, curve_split, curve_fold = build_index(y)
    n_curves = len(curves)
    split_arr = y.main_split.values
    tr_m, va_m, te_m = split_arr == "train", split_arr == "val", split_arr == "test"

    # grade boundaries: gate-report values (quantiles of training curve counts)
    q1g, q2g = gate["boundaries"]["q33"], gate["boundaries"]["q67"]
    q1c, q2c, med_main = boundaries_from_train_counts(curve_count, curve_split == "train")
    ok = abs(q1c - q1g) < 0.5 and abs(q2c - q2g) < 0.5
    log(f"[boundary] main 3-level gate=({q1g:.3f},{q2g:.3f}) recomputed=({q1c:.3f},{q2c:.3f}) "
        f"{'consistent' if ok else 'INCONSISTENT -> using gate values'}")
    b3_main = (q1g, q2g)
    y["label_level3"] = level_of(y.label_count.values, b3_main)

    if os.path.exists(args.out):
        res = json.loads(open(args.out, encoding="utf-8").read())
    else:
        res = {"config": {}, "selection": {}, "runs": {}, "aggregate": {},
               "timing": {}, "notes": []}
    res["config"] = {
        "seeds": seeds, "epochs": epochs, "batch": BATCH, "lr": LR,
        "patience": PATIENCE, "nat_weights": NAT_WEIGHTS,
        "boundaries_main": {"q33": q1g, "q67": q2g, "median_2lvl": med_main},
        "n_curves": n_curves, "n_subseq": int(len(y)),
        "splits_subseq": {s: int((split_arr == s).sum()) for s in ("train", "val", "test")},
        "splits_curves": {s: int((curve_split == s).sum()) for s in ("train", "val", "test")},
    }
    res.setdefault("notes", []).append(
        "NAT-lite: control weight equals exact=1.0 (0 counts are exact values)")
    save_results(res, args.out)

    # train-only standardization of the flattened 8x56 features
    Xs_flat = fit_transform_scaler(X, np.where(tr_m)[0])
    Xs_t = torch.tensor(Xs_flat.reshape(-1, K, N_FEAT))
    yreg_t, yw_t, ycoral_t, ycls_t = make_tensors(y, y.label_level3.values)
    nat_on = False                        # Phase B main results = unweighted

    COMMON = dict(mask_sub=None, cidx=cidx, M=M, curves=curves, curve_count=curve_count,
                  curve_log=curve_log, b3=b3_main, b2=(med_main, med_main),
                  n_curves=n_curves)

    # ================================================== stage: main
    if "main" in stages:
        log("---------- stage main: fixed 62/15/12 main split ----------")

        # (1) Ridge
        key = "main|ridge"
        if key not in res["runs"]:
            t0 = time.time()
            yp, alpha = run_ridge(Xs_flat, y, tr_m, te_m)
            sub = np.zeros(len(y)); sub[te_m] = yp
            payload = {"model": "Ridge (pooled 56-dim)", "alpha": alpha,
                       "test": evaluate_split(yreg_pred_log_sub=sub,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)

        # (2) Random forest
        for seed in seeds:
            key = f"main|rf|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            yp = run_rf(Xs_flat, y, tr_m, te_m, seed, "reg")
            pc = run_rf(Xs_flat, y, tr_m, te_m, seed, "cls")
            sub = np.zeros(len(y)); sub[te_m] = yp
            pcs = np.zeros((len(y), 3)); pcs[te_m] = pc
            payload = {"model": "RandomForest (500, flattened 448-dim)", "seed": seed,
                       "test": evaluate_split(yreg_pred_log_sub=sub, prob_cls_sub=pcs,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)

        # (3) XGBoost (small grid on validation -> 3 seeds)
        xgb_kind, XGBR, XGBC = make_xgb_like()
        if not any("boosting backend" in n for n in res["notes"]):
            res["notes"].append(f"boosting backend = {xgb_kind}")
        best_p = res.get("selection", {}).get("xgb_params")
        if xgb_kind == "xgboost" and best_p is None and not args.smoke:
            best_p, best_v = None, float("inf")
            for d in (3, 5):
                for n in (400, 800):
                    gp = dict(max_depth=d, n_estimators=n, learning_rate=0.05)
                    p = dict(gp, subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                             random_state=2024, n_jobs=N_JOBS, verbosity=0)
                    r = XGBR(**p)
                    r.fit(Xs_flat[tr_m], y.label_log1p.values[tr_m])
                    v = float(np.mean(np.abs(r.predict(Xs_flat[va_m])
                                             - y.label_log1p.values[va_m])))
                    log(f"[xgb-grid] {gp} -> val MAE(log1p)={v:.4f}")
                    if v < best_v:
                        best_v, best_p = v, gp
            res.setdefault("selection", {})["xgb_params"] = best_p
            save_results(res, args.out)
        if best_p is None:
            best_p = {"max_depth": 5, "n_estimators": 600, "learning_rate": 0.05}
        log(f"[selection] boosting backend={xgb_kind} params={best_p}")

        for seed in seeds:
            key = f"main|boost|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            yp, pc = run_boost(xgb_kind, XGBR, XGBC, Xs_flat, y, tr_m, te_m, seed, best_p)
            sub = np.zeros(len(y)); sub[te_m] = yp
            pcs = np.zeros((len(y), 3)); pcs[te_m] = pc
            payload = {"model": f"XGBoost ({xgb_kind})", "seed": seed, "params": best_p,
                       "test": evaluate_split(yreg_pred_log_sub=sub, prob_cls_sub=pcs,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)

        # BiLSTM hidden-size grid on validation curves (64/128)
        if "bilstm_hidden" not in res.get("selection", {}):
            grid_val = {}
            best_h, best_v = None, float("inf")
            for h in (64, 128):
                seed_all(2024)
                model = BiLSTMNet(hidden=h, use_dsg=True)
                _, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             np.where(tr_m)[0], np.where(va_m)[0],
                                             mode="dual", nat_on=nat_on, epochs=epochs,
                                             seed=2024, tag=f"grid_h{h}", log=log)
                grid_val[str(h)] = round(vl, 4)
                log(f"[hidden-grid] h={h} val_loss={vl:.4f} best_ep={bep} ({wt:.0f}s)")
                if vl < best_v:
                    best_v, best_h = vl, h
            res.setdefault("selection", {})["bilstm_hidden"] = int(best_h)
            res["selection"]["bilstm_hidden_grid_val"] = grid_val
            save_results(res, args.out)
        H = res["selection"]["bilstm_hidden"]
        log(f"[selection] BiLSTM hidden={H}")

        # (4) main model: BiLSTM+DSG+CORAL
        for seed in seeds:
            key = f"main|bilstm_dsg|seed{seed}"
            if key in res["runs"] and not args.dump_preds:
                continue
            t0 = time.time()
            seed_all(seed)
            model = BiLSTMNet(hidden=H, use_dsg=True)
            model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             np.where(tr_m)[0], np.where(va_m)[0],
                                             mode="dual", nat_on=nat_on, epochs=epochs,
                                             seed=seed, tag=f"dsg_s{seed}", log=log)
            reg_s, pc_s, _ = predict_torch(model, Xs_t, np.arange(len(y)))
            payload = {"model": "BiLSTM+DSG+CORAL (main model)", "seed": seed, "hidden": H,
                       "best_epoch": bep, "val_loss": vl,
                       "test": evaluate_split(yreg_pred_log_sub=reg_s, prob_coral_sub=pc_s,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)
            if args.dump_preds:
                # curve-level test predictions of THIS model instance, so that the
                # boundary study can score the same instance (no second model).
                curves_in = np.unique(cidx[te_m])
                yp_curve = _pool_to_curves(reg_s[te_m], te_m, cidx, M, curves_in, n_curves)
                pred_counts = np.clip(np.expm1(yp_curve), 0.0, None)
                dump.setdefault("per_seed_pred_counts", {})[str(seed)] = pred_counts.tolist()
                if "test_curve_ids" not in dump:
                    dump["test_curve_ids"] = [curves[i] for i in curves_in]
                    dump["test_true_counts"] = curve_count[curves_in].tolist()
            del model

        # (4') BiLSTM without DSG (ablation off)
        for seed in seeds:
            key = f"main|bilstm_nodsg|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            seed_all(seed)
            model = BiLSTMNet(hidden=H, use_dsg=False)
            model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             np.where(tr_m)[0], np.where(va_m)[0],
                                             mode="dual", nat_on=nat_on, epochs=epochs,
                                             seed=seed, tag=f"nodsg_s{seed}", log=log)
            reg_s, pc_s, _ = predict_torch(model, Xs_t, np.arange(len(y)))
            payload = {"model": "BiLSTM no-DSG (ablation off)", "seed": seed, "hidden": H,
                       "best_epoch": bep, "val_loss": vl,
                       "test": evaluate_split(yreg_pred_log_sub=reg_s, prob_coral_sub=pc_s,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)
            del model

        # (4'') softmax-CE baseline (same backbone, CE head)
        for seed in seeds:
            key = f"main|bilstm_ce|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            seed_all(seed)
            model = BiLSTMCESoftmax(hidden=H)
            model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             np.where(tr_m)[0], np.where(va_m)[0],
                                             mode="ce", epochs=epochs, seed=seed,
                                             tag=f"ce_s{seed}", log=log)
            _, _, ps_s = predict_torch(model, Xs_t, np.arange(len(y)), mode="ce")
            pcs = np.zeros((len(y), 3)); pcs[te_m] = ps_s[te_m]
            payload = {"model": "BiLSTM+DSG softmax (CE baseline)", "seed": seed,
                       "hidden": H, "best_epoch": bep, "val_loss": vl,
                       "test": evaluate_split(prob_cls_sub=pcs,
                                              **{**COMMON, "mask_sub": te_m})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)
            del model

        # (5) 1D-CNN
        for seed in seeds:
            key = f"main|cnn|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            seed_all(seed)
            model = CNN1D()
            model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             np.where(tr_m)[0], np.where(va_m)[0],
                                             mode="dual", epochs=epochs, seed=seed,
                                             tag=f"cnn_s{seed}", log=log)
            reg_s, pc_s, _ = predict_torch(model, Xs_t, np.arange(len(y)))
            payload = {"model": "1D-CNN", "seed": seed, "best_epoch": bep, "val_loss": vl,
                       "test": evaluate_split(yreg_pred_log_sub=reg_s, prob_coral_sub=pc_s,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)
            del model

        # (6) NAT-lite ON (ablation switch record)
        for seed in seeds:
            key = f"main|bilstm_dsg_natlite|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            seed_all(seed)
            model = BiLSTMNet(hidden=H, use_dsg=True)
            model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                             np.where(tr_m)[0], np.where(va_m)[0],
                                             mode="dual", nat_on=True, epochs=epochs,
                                             seed=seed, tag=f"nat_s{seed}", log=log)
            reg_s, pc_s, _ = predict_torch(model, Xs_t, np.arange(len(y)))
            payload = {"model": "BiLSTM+DSG+NAT-lite (switch ON)", "seed": seed,
                       "hidden": H, "best_epoch": bep, "val_loss": vl,
                       "test": evaluate_split(yreg_pred_log_sub=reg_s, prob_coral_sub=pc_s,
                                              **{**COMMON, "mask_sub": te_m,
                                                 "include_subseq_diag": True})}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)
            del model

    # ================================================== stage: lobo
    if "lobo" in stages:
        log("---------- stage lobo: 8 leave-one-batch-out folds ----------")
        folds = sorted(y.lobo_fold.unique())
        if args.smoke:
            folds = folds[:2]
        H = res.get("selection", {}).get("bilstm_hidden", 128)
        xgb_kind, XGBR, XGBC = make_xgb_like()
        res["runs"]["lobo|_protocol"] = {
            "note": ("main tables use the main split; LOBO = pooled + per-fold diagnostics "
                     "(sparse caveat noted). Only GASCADE (seed 2024) and XGBoost (strongest "
                     "tabular baseline) are run per fold to save compute; other models are "
                     "skipped."),
            "skipped_models": ["ridge", "rf", "cnn", "bilstm_ce", "bilstm_nodsg",
                               "bilstm_dsg_natlite"],
            "folds": list(folds),
        }
        save_results(res, args.out)

        for fold in folds:
            te_f = (y.lobo_fold.values == fold)
            tr_f = ~te_f
            # in-fold 90/10 curve split (early stopping)
            ftr_curves = np.unique(cidx[tr_f])
            perm = np.random.RandomState(2024).permutation(len(ftr_curves))
            n_va = max(1, int(round(0.1 * len(ftr_curves))))
            va_curves = set(ftr_curves[perm[:n_va]].tolist())
            is_tr = np.isin(cidx, ftr_curves) & ~np.isin(cidx, list(va_curves)) & tr_f
            is_va = np.isin(cidx, list(va_curves)) & tr_f

            # in-fold boundaries (fitted on in-fold training curves only)
            is_ftr_curve = np.array([curve_fold[i] != fold for i in range(n_curves)])
            q1f, q2f, medf = boundaries_from_train_counts(curve_count, is_ftr_curve)
            lvl_f = level_of(y.label_count.values, (q1f, q2f))
            y_lf = y.assign(label_level3=lvl_f)

            Xs_flat_f = fit_transform_scaler(X, np.where(is_tr)[0])
            Xs_t_f = torch.tensor(Xs_flat_f.reshape(-1, K, N_FEAT))
            yreg_tf, yw_tf, ycoral_tf, ycls_tf = make_tensors(y_lf, lvl_f)
            log(f"[LOBO {fold}] subseq train={int(is_tr.sum())} val={int(is_va.sum())} "
                f"test={int(te_f.sum())} | in-fold boundaries q33={q1f:.1f} "
                f"q67={q2f:.1f} med={medf:.1f}")

            # --- GASCADE
            key = f"lobo|bilstm_dsg|{fold}"
            if key not in res["runs"]:
                t0 = time.time()
                seed_all(2024)
                model = BiLSTMNet(hidden=H, use_dsg=True)
                model, vl, bep, wt = train_torch(model, Xs_t_f, yreg_tf, yw_tf, ycoral_tf,
                                                 ycls_tf, np.where(is_tr)[0],
                                                 np.where(is_va)[0], mode="dual",
                                                 nat_on=False, epochs=epochs, seed=2024,
                                                 tag=f"loboDSG_{fold}", log=log)
                reg_s, pc_s, _ = predict_torch(model, Xs_t_f, np.where(te_f)[0])
                reg_full = np.zeros(len(y)); reg_full[te_f] = reg_s
                pc_full = np.zeros((len(y), 2)); pc_full[te_f] = pc_s
                payload = {"model": "BiLSTM+DSG+CORAL", "seed": 2024, "fold": fold,
                           "best_epoch": bep, "val_loss": vl,
                           "test": evaluate_split(yreg_pred_log_sub=reg_full,
                                                  prob_coral_sub=pc_full,
                                                  **{**COMMON, "mask_sub": te_f,
                                                     "b3": (q1f, q2f),
                                                     "b2": (medf, medf)}),
                           "pooled_curves": pooled_curve_arrays(
                               reg_full, pc_full, None, te_f, cidx, M, curves,
                               curve_count, curve_log, n_curves)}
                payload["wall_sec"] = time.time() - t0
                record_run(res, key, payload, args.out)
                del model

            # --- XGBoost
            key = f"lobo|boost|{fold}"
            if key not in res["runs"]:
                t0 = time.time()
                yp, pc = run_boost(xgb_kind, XGBR, XGBC, Xs_flat_f, y_lf, is_tr, te_f,
                                   2024, res.get("selection", {}).get("xgb_params"))
                sub = np.zeros(len(y)); sub[te_f] = yp
                pcs = np.zeros((len(y), 3)); pcs[te_f] = pc
                payload = {"model": f"XGBoost ({xgb_kind})", "seed": 2024, "fold": fold,
                           "test": evaluate_split(yreg_pred_log_sub=sub, prob_cls_sub=pcs,
                                                  **{**COMMON, "mask_sub": te_f,
                                                     "b3": (q1f, q2f),
                                                     "b2": (medf, medf)}),
                           "pooled_curves": pooled_curve_arrays(
                               sub, None, pcs, te_f, cidx, M, curves, curve_count,
                               curve_log, n_curves)}
                payload["wall_sec"] = time.time() - t0
                record_run(res, key, payload, args.out)

            del Xs_t_f, Xs_flat_f

        # ---- pooled + per-fold aggregation
        for mdl in ("bilstm_dsg", "boost"):
            fold_keys = [k for k in res["runs"] if k.startswith(f"lobo|{mdl}|")]
            if not fold_keys:
                continue
            per_fold = {res["runs"][k]["fold"]: res["runs"][k]["test"] for k in fold_keys}
            pred_log, true_cnt = [], []
            pc_or = {}
            for k in fold_keys:
                pcarr = res["runs"][k]["pooled_curves"]
                if "pred_log1p" in pcarr:
                    pred_log += pcarr["pred_log1p"]
                    true_cnt += pcarr["true_count"]
                for cid, pr in zip(pcarr["curve_ids"],
                                   pcarr.get("coral_probs") or pcarr.get("cls_probs") or []):
                    pc_or[cid] = pr
            pooled = {}
            if pred_log:
                pooled["reg_curve_pooled"] = reg_metrics(np.log1p(np.array(true_cnt)),
                                                         np.array(pred_log))
                yp_c = np.expm1(np.clip(np.array(pred_log), 0, None))
                yt3p = level_of(np.array(true_cnt), b3_main)
                pooled["grade3_regthr_pooled_mainbnd"] = grade_metrics(
                    yt3p, level_of(yp_c, b3_main), 3)
            if pc_or:
                cids = [curves[i] for i in range(n_curves)]
                P = np.array([pc_or[c] for c in cids if c in pc_or])
                T = np.array([curve_count[i] for i in range(n_curves) if curves[i] in pc_or])
                yt3p = level_of(T, b3_main)
                is_coral = "coral_probs" in res["runs"][fold_keys[0]]["pooled_curves"]
                if is_coral:
                    pooled["grade3_coral_pooled_mainbnd"] = grade_metrics(
                        yt3p, (P > 0.5).sum(axis=1), 3)
                else:
                    pooled["grade3_softmax_pooled_mainbnd"] = grade_metrics(
                        yt3p, P.argmax(axis=1), 3)
            res.setdefault("aggregate", {}).setdefault("lobo", {})[mdl] = {
                "pooled": pooled,
                "per_fold": per_fold,
                "caveat": ("pooled grading uses main-split boundaries as the label caliber; "
                           "per-fold diagnostics use in-fold boundaries; LOBO test folds are "
                           "class-sparse (gate FAIL), so per-fold metrics are diagnostic only"),
            }
            save_results(res, args.out)
            log(f"[LOBO pooled] {mdl}: "
                f"{json.dumps({k: v for k, v in pooled.items()}, default=str)[:300]}")

    # ================================================== stage: leak
    if "leak" in stages:
        log("---------- stage leak: window-random vs curve-grouped ----------")
        H = res.get("selection", {}).get("bilstm_hidden", 128)
        for seed in seeds:
            key = f"leak|winrand|seed{seed}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(y))
            n_tr = int(0.8 * len(y)); n_va = int(0.1 * len(y))
            i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
            Xs_flat_w = fit_transform_scaler(X, i_tr)
            Xs_t_w = torch.tensor(Xs_flat_w.reshape(-1, K, N_FEAT))
            yreg_w, yw_w, ycoral_w, ycls_w = make_tensors(y, y.label_level3.values)
            seed_all(seed)
            model = BiLSTMNet(hidden=H, use_dsg=True)
            model, vl, bep, wt = train_torch(model, Xs_t_w, yreg_w, yw_w, ycoral_w,
                                             ycls_w, i_tr, i_va, mode="dual",
                                             nat_on=False, epochs=epochs, seed=seed,
                                             tag=f"leak_s{seed}", log=log)
            reg_s, pc_s, _ = predict_torch(model, Xs_t_w, i_te)
            rm = reg_metrics(y.label_log1p.values[i_te], reg_s)
            yt3 = y.label_level3.values[i_te]
            yp3 = (pc_s > 0.5).sum(axis=1)
            payload = {"model": "BiLSTM+DSG window-random 80/10/10 (leaky side)",
                       "seed": seed, "best_epoch": bep,
                       "test_window_level": {"reg": rm,
                                             "grade3_coral": grade_metrics(yt3, yp3, 3)}}
            payload["wall_sec"] = time.time() - t0
            record_run(res, key, payload, args.out)
            del model, Xs_t_w, Xs_flat_w

    # ================================================== summary
    log("---------- summary ----------")
    summarize(res)
    res["timing"] = {"total_sec": time.time() - t_start,
                     "total_min": round((time.time() - t_start) / 60, 1)}
    save_results(res, args.out)
    if args.dump_preds:
        with open(args.dump_preds, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=1)
        log(f"[dump-preds] wrote {args.dump_preds} "
            f"(seeds: {sorted(dump.get('per_seed_pred_counts', {}))})")
    log(f"=== done in {(time.time() - t_start) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
