"""
sro.py -- Sliding-Window Sequence Reorganization (SRO).

Implements main-text equations (1)-(4) of the paper:

  (1) slice each curve into consecutive non-overlapping L-sample windows
      (trailing partial window discarded);
  (2) map every window, per channel, onto D = 14 deterministic statistics
      (one 56-dim group point: C = 4 channels x D = 14 statistics,
      channel-major layout feature_index = channel_idx * 14 + stat_idx);
  (3) pack K consecutive group points into one short subsequence with step n
      (in raw samples);
  (4) M = floor((N - K*L)/n) + 1 subsequences per curve of N raw samples.

Main configuration G1: L = 10, K = 8, n = 10  ->  one subsequence spans 80 min,
adjacent subsequences share 7 of their 8 group points, and the curve-level
label is broadcast to all of the curve's subsequences.  Ablation variants
(K = 4 shorter subsequences; n = 80 non-overlapping) are exported by the same
function with different parameters.

Input : a directory of cleaned per-curve series CSVs (one per curve, columns
        ``t_min,co2_ppm,dco2_ppm,temp_c,hum_rh,ch4_ppm``; see
        ``scripts/01_build_dataset.py``) plus the curated table
        ``series-new_curated.csv`` (labels, label tiers, main split).
Output: ``X.npy``   float32 (T, K, 56)
        ``y.csv``   subsequence index (leakage-audit columns included)
        ``meta.json``  configuration, feature list, per-curve M, self-checks
        ``gate_report.json``  class-balance gate for the three-level scheme
                            (written for the main configuration only)

Everything here is deterministic (no fitted parameters); feature selection,
boundaries and normalization are train-only obligations of downstream stages.

"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

L_MAIN, K_MAIN, N_STEP_MAIN = 10, 8, 10          # G1 main configuration

CHANNELS = ["co2_ppm", "ch4_ppm", "temp_c", "hum_rh"]   # channel axis order
STATS = [                                                # statistic axis order
    "mean", "std", "min", "max", "range", "median", "q10", "q90",
    "slope", "rms", "diff_mean", "diff_std", "lag1_autocorr", "kurtosis",
]
FEATURE_NAMES = [f"{c}__{s}" for c in CHANNELS for s in STATS]   # 56, channel-major
assert len(FEATURE_NAMES) == 56


def group_features(blk):
    """Window statistic map phi: (G, L, C) -> (G, C, 14).  Deterministic."""
    t = np.arange(blk.shape[1], dtype=np.float64) - (blk.shape[1] - 1) / 2.0
    mean = blk.mean(axis=1)                                # (G, C)
    std = blk.std(axis=1, ddof=0)                          # population std
    mn, mx = blk.min(axis=1), blk.max(axis=1)
    rng = mx - mn
    med = np.median(blk, axis=1)
    q10 = np.quantile(blk, 0.10, axis=1)
    q90 = np.quantile(blk, 0.90, axis=1)
    slope = np.einsum("glc,l->gc", blk - mean[:, None, :], t) / (t ** 2).sum()
    rms = np.sqrt((blk ** 2).mean(axis=1))
    d = np.diff(blk, axis=1)                               # (G, L-1, C)
    d_mean, d_std = d.mean(axis=1), d.std(axis=1, ddof=0)
    # lag-1 autocorrelation (Pearson); constant window (zero denominator) -> 0
    a = blk[:, :-1, :] - mean[:, None, :]
    b = blk[:, 1:, :] - mean[:, None, :]
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    acf = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    # excess kurtosis (Fisher); zero std -> 0
    m4 = ((blk - mean[:, None, :]) ** 4).mean(axis=1)
    kurt = np.where(std > 0, m4 / np.where(std > 0, std, 1.0) ** 4 - 3.0, 0.0)
    feats = [mean, std, mn, mx, rng, med, q10, q90, slope, rms,
             d_mean, d_std, acf, kurt]                     # each (G, C), STATS order
    return np.transpose(np.stack(feats, axis=1), (0, 2, 1))     # (G, C, 14)


def m_formula(N, K, L, n_step):
    """Eq. (4): number of subsequences of an N-sample curve."""
    return (N - K * L) // n_step + 1


def label_type_of(row):
    """Label tier rule (Supplementary S1): nominal / control / rounded / exact."""
    if int(row["is_control"]) == 1:
        return "control"
    if str(row["source"]) == "data1":
        return "nominal"
    return "rounded" if int(row["label"]) % 10 == 0 else "exact"


def lobo_fold_of(group):
    """LOBO fold name: one fold per evaluation unit (device group)."""
    return f"LOBO_{group}"


def export(series_dir, curated_csv, out_dir, L=L_MAIN, K=K_MAIN, n_step=N_STEP_MAIN,
           with_gate=True, log=print):
    """Run the SRO export; returns (X, y, summary dict).

    Curves are processed only when their cleaned series file exists; missing
    ones are reported (the released sample covers a subset of the 89 curves).
    """
    assert n_step % L == 0, "step n must be a multiple of window length L"
    t0 = time.time()
    series_dir, out_dir = Path(series_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curated = pd.read_csv(curated_csv)
    curated["label_type"] = curated.apply(label_type_of, axis=1)
    tier_counts = curated["label_type"].value_counts().to_dict()

    available = []
    for sid in curated["sample_id"]:
        if (series_dir / f"{sid}.csv").exists():
            available.append(sid)
        else:
            log(f"[skip] series file missing for {sid}")
    sub = curated[curated["sample_id"].isin(available)].reset_index(drop=True)
    log(f"[curated] {len(curated)} curves total, {len(sub)} with series files")

    assert sub["split"].isin(["train", "val", "test"]).all()
    assert sub["group"].notna().all()

    # repeat-pair rule: batches 11 and 12 share one evaluation unit and fold
    g11 = sub[sub["batch"] == "\u7ec411"]["sample_id"].tolist()
    g12 = sub[sub["batch"] == "\u7ec412"]["sample_id"].tolist()
    if g11 and g12:
        both = sub["sample_id"].isin(g11 + g12)
        assert sub.loc[both, "group"].nunique() == 1, \
            "batches 11 and 12 must share one evaluation unit (device leakage)"
        assert sub.loc[both, "split"].nunique() == 1, \
            "batches 11 and 12 must share one main split (device leakage)"
        log(f"[leakage audit] repeat pair group11plus12: {int(both.sum())} curves, unit="
            f"{sub.loc[both, 'group'].iloc[0]}")

    x_parts, rows = [], []
    subseq_id, per_curve_M, m_spot = 0, {}, []

    for r in sub.itertuples():
        sid = r.sample_id
        df = pd.read_csv(series_dir / f"{sid}.csv", usecols=CHANNELS)
        N = len(df)
        arr = df.to_numpy(dtype=np.float64)
        assert np.isfinite(arr).all(), f"{sid}: series contains NaN/Inf"
        assert N >= K * L, f"{sid}: {N} rows < K*L = {K * L}"

        n_groups = N // L                       # complete groups; tail discarded
        F = group_features(arr[: n_groups * L].reshape(n_groups, L, len(CHANNELS)))
        flat = F.reshape(n_groups, 56).astype(np.float32)

        g_stride = n_step // L                   # group offset between subsequences
        M = m_formula(N, K, L, n_step)
        assert M >= 1 and (M - 1) * g_stride + K <= n_groups, \
            f"{sid}: M formula contradicts group count"

        sw = np.lib.stride_tricks.sliding_window_view(flat, K, axis=0)
        sw = np.moveaxis(sw, -1, 1)                              # (n_groups-K+1, K, 56)
        Xc = np.ascontiguousarray(sw[:M])
        assert Xc.shape == (M, K, 56)

        if len(m_spot) < 3:                  # manual spot check of Eq. (4)
            cnt, s = 0, 0
            while s + K * L <= N:
                cnt += 1
                s += n_step
            m_spot.append({"curve_id": sid, "N_rows": int(N), "M_formula": int(M),
                           "M_manual_loop": int(cnt)})
            assert cnt == M, f"{sid}: manual M={cnt} != formula {M}"

        label = int(r.label)
        row_common = {
            "curve_id": sid, "group_id": r.group, "label_count": label,
            "label_log1p": float(np.log1p(label)), "label_type": r.label_type,
            "main_split": r.split, "lobo_fold": lobo_fold_of(r.group),
            "is_control": int(r.is_control), "n_minutes": int(r.n_minutes),
            "n_rows_actual": int(N), "M": int(M),
        }
        rows.append(pd.DataFrame({
            "subseq_id": np.arange(subseq_id, subseq_id + M),
            "start_idx": (np.arange(M) * n_step).astype(np.int64),
            **{k: np.repeat(v, M) if not isinstance(v, str) else [v] * M
               for k, v in row_common.items()},
        }))
        x_parts.append(Xc)
        per_curve_M[sid] = int(M)
        subseq_id += M

    X = np.concatenate(x_parts, axis=0)                   # (T, K, 56) float32
    y = pd.concat(rows, ignore_index=True)
    del x_parts

    # ---------------------------------------------------------- self-checks
    assert X.dtype == np.float32
    assert X.shape == (len(y), K, 56)
    assert len(y) == sum(per_curve_M.values())
    assert np.isfinite(X).all()
    assert y["subseq_id"].is_unique and y["subseq_id"].is_monotonic_increasing
    per_curve = y.groupby("curve_id")[["main_split", "lobo_fold"]].nunique()
    assert (per_curve == 1).all().all(), "a curve spans several folds -- leakage!"
    gr = y.groupby("curve_id")["subseq_id"].agg(["min", "max", "count"])
    assert ((gr["max"] - gr["min"] + 1) == gr["count"]).all()
    log(f"[self-check] OK: X{X.shape}, T={len(y)}, sum(M)={sum(per_curve_M.values())}")
    log(f"[manual check] {json.dumps(m_spot, ensure_ascii=False)}")

    np.save(out_dir / "X.npy", X)
    y.to_csv(out_dir / "y.csv", index=False, encoding="utf-8-sig")

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "src/sro.py",
        "config": {"L": L, "K": K, "n_step": n_step,
                   "M_formula": "M = floor((N - K*L)/n) + 1, N = actual rows",
                   "span_min": K * L},
        "channels_axis_order": CHANNELS,
        "stats_axis_order": STATS,
        "feature_layout": "channel-major: feature_index = channel_idx*14 + stat_idx",
        "feature_names": FEATURE_NAMES,
        "n_features": 56,
        "X_shape": list(X.shape),
        "X_dtype": "float32",
        "per_curve_M": per_curve_M,
        "per_curve_N_rows": {k: int(v) for k, v in
                             y.groupby("curve_id")["n_rows_actual"].first().items()},
        "total_subseqs": int(len(y)),
        "M_min": int(min(per_curve_M.values())),
        "M_max": int(max(per_curve_M.values())),
        "label_type_rule": ("nominal=source data1; control=is_control==1; "
                            "data2 non-control: label%10==0 -> rounded else exact"),
        "label_tier_counts_curve": {str(k): int(v) for k, v in tier_counts.items()},
        "stat_notes": {"std": "population (ddof=0)",
                       "kurtosis": "excess (Fisher), std=0 -> 0",
                       "lag1_autocorr": "Pearson lag-1; constant window -> 0"},
        "leakage_asserts_passed": ["no NaN/Inf", "shape consistent",
                                   "M formula (full + 3 manual)", "one split/fold per curve",
                                   "repeat pair group11plus12 together", "sum(M) = total"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    log(f"[write] X.npy / y.csv / meta.json -> {out_dir}  ({time.time() - t0:.1f}s)")

    summary = {"n_curves": int(len(sub)), "n_subseq": int(len(y)),
               "config": {"L": L, "K": K, "n_step": n_step},
               "wall_sec": time.time() - t0}

    # ------------------------------------------------ class-balance gate (main config)
    if with_gate and (L, K, n_step) == (L_MAIN, K_MAIN, N_STEP_MAIN):
        train_counts = sub.loc[sub["split"] == "train", "label"].to_numpy(dtype=float)
        q33, q67 = np.quantile(train_counts, [1 / 3, 2 / 3])

        def level_of(c):
            return 1 if c <= q33 else (2 if c <= q67 else 3)

        sub_g = sub.assign(grade_level=sub["label"].apply(level_of))
        lv_counts = sub_g["grade_level"].value_counts().sort_index().to_dict()
        curve_M = y.groupby("curve_id")["M"].max()

        def evaluate(curve_ids):
            s = sub_g[sub_g["sample_id"].isin(curve_ids)]
            cc, cs = {}, {}
            for lv in (1, 2, 3):
                ids_lv = s.loc[s["grade_level"] == lv, "sample_id"]
                cc[lv] = int(len(ids_lv))
                cs[lv] = int(curve_M.reindex(ids_lv).fillna(0).sum())
            total_s = sum(cs.values())
            viol = []
            for lv in (1, 2, 3):
                if total_s and cs[lv] / total_s < 0.15:
                    viol.append(f"L{lv} subsequence share {cs[lv] / total_s:.1%} < 15%")
                if cc[lv] < 4:
                    viol.append(f"L{lv} curve count {cc[lv]} < 4")
            return {"n_curves": int(len(s)), "curve_counts": cc,
                    "subseq_counts": cs, "violations": viol}

        gate = {
            "scheme": "main scheme: training-fold count terciles (33%/67%) define 3 levels",
            "boundaries": {"q33": float(q33), "q67": float(q67),
                           "rule": "L1 <= q33, L2 in (q33, q67], L3 > q67 (zero controls in L1)"},
            "level_counts_curve": {str(k): int(v) for k, v in lv_counts.items()},
            "pass_rule": "any evaluated subset: a level with <15% subsequences or <4 curves fails",
            "subsets": {},
        }
        for sp in ("train", "val", "test"):
            gate["subsets"][f"main_{sp}"] = evaluate(
                sub_g.loc[sub_g["split"] == sp, "sample_id"])
        for fold in sorted(sub_g["group"].unique()):
            gate["subsets"][f"{lobo_fold_of(fold)}__test"] = evaluate(
                sub_g.loc[sub_g["group"] == fold, "sample_id"])
            gate["subsets"][f"{lobo_fold_of(fold)}__train"] = evaluate(
                sub_g.loc[sub_g["group"] != fold, "sample_id"])
        viol = {k: v["violations"] for k, v in gate["subsets"].items() if v["violations"]}
        gate["n_violating_subsets"] = len(viol)
        gate["decision"] = "PASS" if not viol else "FAIL"
        gate["violations_detail"] = viol
        (out_dir / "gate_report.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[gate] class-balance decision: {gate['decision']} "
            f"({len(viol)} violating subsets; boundaries q33={q33:.2f} q67={q67:.2f})")
        summary["gate"] = gate["decision"]

    return X, y, summary


def main():
    ap = argparse.ArgumentParser(description="Sliding-window sequence reorganization (SRO)")
    ap.add_argument("--series", required=True,
                    help="directory of cleaned per-curve series CSVs")
    ap.add_argument("--curated", required=True, help="series-new_curated.csv")
    ap.add_argument("--out", required=True, help="output directory (X.npy, y.csv, meta.json)")
    ap.add_argument("--L", type=int, default=L_MAIN, help="window length in raw samples")
    ap.add_argument("--K", type=int, default=K_MAIN, help="group points per subsequence")
    ap.add_argument("--n", type=int, default=N_STEP_MAIN, dest="n_step",
                    help="subsequence step in raw samples")
    args = ap.parse_args()
    _, _, summary = export(args.series, args.curated, args.out,
                           L=args.L, K=args.K, n_step=args.n_step)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
