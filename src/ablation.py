"""
ablation.py -- Phase C ablations (Supplementary Tables S12–S14).

Three new ablation arms, each 3 seeds x main split x BiLSTM+DSG (hidden 128),
LOBO skipped, boundaries = the exporting dataset's training-curve count
terciles (same rule as the main scheme):

  1. ``featselect_on_top20`` -- keep only the three-method union top-20 of the
     56 group-point features; the other 36 columns are zeroed BEFORE
     standardization (hence standardized to the mean and invisible to the
     model).  The DSG 4x14 channel-major structure is unchanged.
  2. ``K4_L10_n10`` -- shorter subsequences (K = 4 group points, 40 min),
     re-exported by the same SRO pipeline with L = 10, n = 10.
  3. ``nonoverlap_K8_n80`` -- non-overlapping subsequences (step n = 80 = K*L,
     ~36 subsequences per curve), re-exported with K = 8, n = 80.

Existing Phase-B arms (DSG on/off, NAT-lite on/off, CORAL vs softmax-CE) are
aggregated read-only from the training results JSON produced by
``scripts/03_train.py``; deltas are reported against the main configuration.

"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import build_index
from metrics import grade_metrics, level_of, reg_metrics
from models import (K as K_MAIN, N_FEAT, SEEDS, fit_transform_scaler,
                    make_tensors, seed_all, train_torch)
import sro

warnings.filterwarnings("ignore")


def _agg_runs(pb, prefix, paths):
    """Read-only aggregation of existing runs in the training results JSON."""
    out = {}
    for p in paths:
        vals = []
        for k, v in pb.get("runs", {}).items():
            if not k.startswith(prefix):
                continue
            cur = v.get("test", {})
            for q in p.split("."):
                cur = cur.get(q, None) if isinstance(cur, dict) else None
            if cur is not None and cur == cur:
                vals.append(float(cur))
        out[p] = ({"mean": round(float(np.mean(vals)), 4),
                   "std": round(float(np.std(vals)), 4), "n": len(vals)}
                  if vals else None)
    return out


def run_ablation(tag, X, y, hidden, res, log=print):
    """Standard main-split train/eval: curve-level regression + grade3 (route ii)."""
    curves, cid2i, cidx, M, curve_count, curve_log, curve_split, curve_fold = build_index(y)
    tr_c = curve_split == "train"
    q1 = float(np.quantile(curve_count[tr_c], 1 / 3))
    q2 = float(np.quantile(curve_count[tr_c], 2 / 3))
    b3 = (q1, q2)

    split_arr = y.main_split.values
    tr_m, va_m, te_m = split_arr == "train", split_arr == "val", split_arr == "test"
    Xs_flat = fit_transform_scaler(X, np.where(tr_m)[0])

    import torch
    Xs_t = torch.tensor(Xs_flat.reshape(-1, X.shape[1], N_FEAT))
    lvl_all = level_of(y.label_count.values, b3)
    yreg_t, yw_t, ycoral_t, ycls_t = make_tensors(y.assign(label_level3=lvl_all), lvl_all)

    te_curve_mask = curve_split == "test"
    yt3 = level_of(curve_count[te_curve_mask], b3)
    curves_in_test = np.unique(cidx[te_m])

    per_seed = {}
    for seed in SEEDS:
        from models import BiLSTMNet
        seed_all(seed)
        model = BiLSTMNet(hidden=hidden, use_dsg=True)
        model, vl, bep, wt = train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                                         np.where(tr_m)[0], np.where(va_m)[0],
                                         mode="dual", nat_on=False, seed=seed,
                                         tag=f"{tag}_s{seed}", log=log)
        regs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(Xs_t), 4096):
                r, _, _, _ = model(Xs_t[i:i + 4096])
                regs.append(r.numpy())
        reg_s = np.concatenate(regs)
        del model
        sums = np.bincount(cidx[te_m], weights=reg_s[te_m], minlength=len(curves))
        yp_curve = (sums / M)[curves_in_test]
        rm = reg_metrics(curve_log[curves_in_test], yp_curve)
        yp3 = level_of(np.expm1(np.clip(yp_curve, 0, None)), b3)
        gm = grade_metrics(yt3, yp3, 3)
        per_seed[str(seed)] = {"reg_curve": rm, "grade3_regthr": gm}

    agg = {}
    for grp, keys in (("reg_curve", ["mae_log1p", "r2_log1p", "mae_count"]),
                      ("grade3_regthr", ["macro_f1", "qwk", "ordacc"])):
        for k in keys:
            vals = [per_seed[str(s)][grp][k] for s in SEEDS]
            agg[f"{grp}.{k}"] = {"mean": round(float(np.mean(vals)), 4),
                                 "std": round(float(np.std(vals)), 4)}
    entry = {
        "tag": tag, "hidden": hidden, "n_subseq_total": int(len(y)),
        "n_train_subseq": int(tr_m.sum()), "n_test_curves": int(len(curves_in_test)),
        "boundaries_q33_q67": list(b3),
        "per_seed": per_seed, "aggregate": agg,
    }
    res["new_ablations"][tag] = entry
    log(f"[ablation {tag}] agg={json.dumps(agg)}")
    return entry


def run(sro_dir, series_dir, curated_csv, results_json, featselect_json,
        out_json, work_dir, hidden=128, log=print):
    t0 = time.time()
    sro_dir, work_dir = Path(sro_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    res = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "src/ablation.py",
        "protocol": ("new arms = 3 seeds x main split x BiLSTM+DSG (hidden 128), LOBO skipped; "
                     "boundaries = each dataset's training-curve count terciles"),
        "new_ablations": {},
    }

    # ---- read-only aggregation of existing Phase-B arms
    pb_path = Path(results_json)
    if pb_path.exists():
        pb = json.loads(pb_path.read_text(encoding="utf-8"))
        H = int(pb.get("selection", {}).get("bilstm_hidden", hidden))
        res["phaseB_existing"] = {
            "dsg_on": _agg_runs(pb, "main|bilstm_dsg|seed",
                                ["reg_curve.mae_log1p", "reg_curve.mae_count",
                                 "reg_curve.r2_log1p", "grade3_regthr.macro_f1",
                                 "grade3_regthr.qwk"]),
            "dsg_off": _agg_runs(pb, "main|bilstm_nodsg|seed",
                                 ["reg_curve.mae_log1p", "reg_curve.mae_count",
                                  "reg_curve.r2_log1p", "grade3_regthr.macro_f1",
                                  "grade3_regthr.qwk"]),
            "natlite_on": _agg_runs(pb, "main|bilstm_dsg_natlite|seed",
                                    ["reg_curve.mae_log1p", "reg_curve.mae_count",
                                     "grade3_regthr.macro_f1", "grade3_regthr.qwk"]),
            "softmax_ce": _agg_runs(pb, "main|bilstm_ce|seed",
                                    ["grade3_softmax.macro_f1", "grade3_softmax.qwk"]),
            "note": ("CORAL rows report grade3_regthr (DSG on/off and NAT-lite on all use the "
                     "CORAL head); softmax_ce is the same backbone with a CE head"),
        }
        main_base = res["phaseB_existing"]["dsg_on"]
    else:
        log(f"[warn] {pb_path} not found; Phase-B aggregation skipped")
        main_base = None

    # ---- (1) feature selection on: three-method union top-20
    log("========== (1) feature-selection ablation (top-20 union mask) ==========")
    fs = json.loads(Path(featselect_json).read_text(encoding="utf-8"))
    meta = json.loads((sro_dir / "meta.json").read_text(encoding="utf-8"))
    union = sorted(set(fs["mrmr"]["top20"]) | set(fs["lasso"]["selected"]) |
                   set(fs["permutation_importance"]["top20"]))
    name2idx = {n: i for i, n in enumerate(meta["feature_names"])}
    sel_idx = np.array([name2idx[n] for n in union], dtype=int)
    assert len(sel_idx) == 20, f"union should be 20 features, got {len(sel_idx)}"
    log(f"[featselect] union {len(sel_idx)} features, indices={sel_idx.tolist()}")

    X = np.load(sro_dir / "X.npy").astype(np.float32)
    y = pd.read_csv(sro_dir / "y.csv", encoding="utf-8-sig")
    Xk = X.copy()
    mask56 = np.zeros(N_FEAT, dtype=bool)
    mask56[sel_idx] = True
    Xk[:, :, ~mask56] = 0.0        # zero before standardization -> mean after
    run_ablation("featselect_on_top20", Xk, y, hidden, res, log)
    res["new_ablations"]["featselect_on_top20"]["selected_features"] = union
    res["new_ablations"]["featselect_on_top20"]["method"] = (
        "keep only the top-20 union columns of the 56 features; zero the other 36 "
        "before standardization (they become the mean afterwards); the DSG 4x14 "
        "channel-major structure is unchanged "
        "(feature_index = channel*14 + stat)")
    del Xk

    # ---- (2) K = 4 variant
    log("========== (2) K=4 re-export + re-train ==========")
    k4_dir = work_dir / "sro_K4"
    if (k4_dir / "X.npy").exists():
        X4 = np.load(k4_dir / "X.npy").astype(np.float32)
        y4 = pd.read_csv(k4_dir / "y.csv", encoding="utf-8-sig")
        log(f"[K4] reuse existing export {X4.shape}")
    else:
        X4, y4, _ = sro.export(series_dir, curated_csv, k4_dir, L=10, K=4, n_step=10,
                               with_gate=False, log=log)
    run_ablation("K4_L10_n10", X4, y4, hidden, res, log)
    res["new_ablations"]["K4_L10_n10"]["method"] = (
        "subsequence length K=4 (main config K=8), L=10, n=10 unchanged; same SRO "
        "pipeline re-export")
    del X4, y4

    # ---- (3) non-overlapping n = 80 variant
    log("========== (3) non-overlapping n=80 re-export + re-train ==========")
    no_dir = work_dir / "sro_nonoverlap"
    if (no_dir / "X.npy").exists():
        Xn = np.load(no_dir / "X.npy").astype(np.float32)
        yn = pd.read_csv(no_dir / "y.csv", encoding="utf-8-sig")
        log(f"[nonoverlap] reuse existing export {Xn.shape}")
    else:
        Xn, yn, _ = sro.export(series_dir, curated_csv, no_dir, L=10, K=8, n_step=80,
                               with_gate=False, log=log)
    run_ablation("nonoverlap_K8_n80", Xn, yn, hidden, res, log)
    res["new_ablations"]["nonoverlap_K8_n80"]["method"] = (
        "step n=80=K*L (zero overlap), ~N/80 ~ 36 subsequences per curve; same SRO "
        "pipeline re-export")
    del Xn, yn

    # ---- deltas vs the main configuration
    log("========== deltas vs main configuration ==========")
    deltas = {}
    for tag, e in res["new_ablations"].items():
        d = {}
        for p in ("reg_curve.mae_log1p", "reg_curve.r2_log1p",
                  "grade3_regthr.macro_f1", "grade3_regthr.qwk"):
            if main_base and main_base.get(p) and e["aggregate"].get(p):
                d[p] = round(e["aggregate"][p]["mean"] - main_base[p]["mean"], 4)
        deltas[tag] = d
        log(f"  {tag}: n_subseq={e['n_subseq_total']} delta={d}")
    res["deltas_vs_main"] = deltas
    res["sample_size_comparison"] = {
        "main_G1_K8_n10": 27318,
        "K4_L10_n10": res["new_ablations"]["K4_L10_n10"]["n_subseq_total"],
        "nonoverlap_K8_n80": res["new_ablations"]["nonoverlap_K8_n80"]["n_subseq_total"],
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[write] {out_json}  ({time.time() - t0:.1f}s)")
    return res


def main():
    ap = argparse.ArgumentParser(description="Phase C ablations")
    ap.add_argument("--sro", required=True, help="main SRO output directory")
    ap.add_argument("--series", required=True, help="cleaned per-curve series directory")
    ap.add_argument("--curated", required=True, help="series-new_curated.csv")
    ap.add_argument("--results", required=True, help="training results JSON (03 output)")
    ap.add_argument("--featselect", required=True, help="featselect_report.json")
    ap.add_argument("--out", required=True, help="output ablation JSON")
    ap.add_argument("--work", required=True, help="directory for variant re-exports")
    ap.add_argument("--hidden", type=int, default=128, help="BiLSTM hidden size")
    args = ap.parse_args()
    run(args.sro, args.series, args.curated, args.results, args.featselect,
        args.out, args.work, hidden=args.hidden)


if __name__ == "__main__":
    main()
