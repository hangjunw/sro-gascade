#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_build_dataset.py -- raw TGSD workbooks -> cleaned per-curve series + summary table.

For every raw workbook found under the TGSD root this script

  1. reads the four channels (CO2, CH4, temperature, relative humidity) with
     the header mapping of ``src/tgsd.py`` (Chinese and English headers);
  2. interpolates internal gaps (deterministic, no smoothing) and writes the
     cleaned per-curve series ``<sample_id>.csv`` with columns
     ``t_min,co2_ppm,dco2_ppm,temp_c,hum_rh,ch4_ppm`` (``dco2_ppm`` is
     baseline-subtracted with the mean of the first 10 minutes);
  3. computes the deterministic summary columns of the curated table: signal
     extrema, net dCO2 at 6/12/24/48 h (from an 11-point moving average, the
     QC convention), Spearman trend, SNR and clipping fraction;
  4. de-duplicates ``data1`` within each workbook: sheets whose CO2 series
     agree to within 1% relative mean absolute difference are one recording
     cluster and only the first sheet (cluster representative) is kept;
  5. optionally restricts the export to the curves that survived quality
     control (``--curated``, matched by ``sample_id``) and verifies that the
     transcribed batch labels agree with the curated table.

The QC verdicts themselves (verdict / quality_score / weight columns of the
curated table) come from the paper's internal QC pipeline and are shipped
as-is in ``series-new_curated.csv``; this script reproduces everything that
is a deterministic function of the raw workbooks.

Usage:
  python scripts/01_build_dataset.py --data data/TGSD_sample \
      --out data/series --summary data/series/dataset_summary.csv \
      [--curated data/series-new_curated.csv]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import tgsd  # noqa: E402

RISK_EDGES = [150, 300, 480]                      # 4-level risk scheme (curated)
# The four grade names are emitted in English (I_verylow / II_low / III_medium /
# IV_high) so the released series-new_curated.csv is fully English; re-running
# this script reproduces it exactly (see data/README.md, "Reproduction note").
RISK_NAMES = ["I_verylow", "II_low", "III_medium", "IV_high"]


def clean_interp(v):
    """Gap interpolation (deterministic; identical to the paper's series export)."""
    v = np.asarray(v, float)
    idx = np.where(np.isfinite(v))[0]
    if len(idx) == 0:
        return np.zeros(len(v))
    return np.interp(np.arange(len(v)), idx, v[idx])


def smooth(v, w=11):
    """Centred moving average (QC convention for fixed-time increments)."""
    if len(v) < w:
        return v
    k = np.ones(w) / w
    return np.convolve(v, k, mode="same")


def rankdata_ties(a):
    """Ranks with average ranks for ties (the paper's QC convention)."""
    a = np.asarray(a, float)
    r = np.empty(len(a), float)
    order = np.argsort(a, kind="mergesort")
    r[order] = np.arange(1, len(a) + 1)
    import collections
    d = collections.defaultdict(list)
    for i, v in enumerate(a):
        d[v].append(i)
    for v, idx in d.items():
        if len(idx) > 1:
            avg = np.mean([r[i] for i in idx])
            for i in idx:
                r[i] = avg
    return r


def spearman_ties(x, y):
    """Spearman rho with tie-aware ranks (matches the paper's QC pipeline)."""
    from statutil import pearson
    return pearson(rankdata_ties(x), rankdata_ties(y))


def qc_features(co2, ch4, temp, hum):
    """Deterministic QC-style features of one curve (s3_signal conventions)."""
    v = smooth(clean_interp(co2), 11)
    n = len(v)
    base = float(v[:10].mean())
    r = {"baseline_smooth": base, "co2_max_smooth": float(v.max())}
    for h in (6, 12, 24, 48):
        i = h * 60
        r[f"dco2_{h}h"] = float(v[i] - base) if i < n else None
    tt = np.arange(n, dtype=float)
    r["spearman_t"] = float(spearman_ties(tt, clean_interp(co2))[0])
    noise = float(np.std(np.diff(clean_interp(co2))))
    dmax = float(clean_interp(co2).max() - base)
    r["snr"] = dmax / noise if noise > 0 else None
    cv = clean_interp(co2)
    top = cv[cv >= cv.max() * 0.995]
    if len(top) > 5:
        vals, cnts = np.unique(np.round(top, 2), return_counts=True)
        r["clip_frac"] = float(cnts.max() / len(cv))
    else:
        r["clip_frac"] = 0.0
    c = clean_interp(ch4)
    r["ch4_max"] = float(np.nanmax(c))
    tp, hm = clean_interp(temp), clean_interp(hum)
    r["temp_mean"] = float(np.nanmean(tp))
    r["hum_mean"] = float(np.nanmean(hm))
    return r


def dedup_data1(records):
    """Cluster near-identical sheets within each data1 workbook (rel. MAD < 1%).

    Returns the subset of records that are cluster representatives
    (deterministic: the alphabetically first sheet of each cluster).
    """
    byfile = {}
    for s in records:
        if s["source"] == "data1":
            byfile.setdefault(s["file"], []).append(s)
    reps = []
    for f in sorted(byfile, key=lambda x: int(x)):
        lst = sorted(byfile[f], key=lambda x: x["sheet"])
        n = min(len(s["co2"]) for s in lst)
        X = np.vstack([s["co2"][:n] for s in lst])
        clusters = []
        for i, s in enumerate(lst):
            placed = False
            for cl in clusters:
                j = cl[0]
                d = np.abs(X[i] - X[j])
                rel = np.nanmean(d) / max(1e-9, np.nanmean(np.abs(X[j])))
                if rel < 0.01:
                    cl.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        for cl in clusters:
            reps.append(lst[cl[0]])
        print(f"[dedup] data1/{f}.xlsx: {len(lst)} sheets -> {len(clusters)} "
              f"independent curve clusters")
    return reps


def main():
    ap = argparse.ArgumentParser(description="Raw TGSD -> cleaned series + summary")
    ap.add_argument("--data", required=True,
                    help="TGSD root (contains data1/ and data2/ batch folders)")
    ap.add_argument("--out", required=True, help="output series directory")
    ap.add_argument("--summary", required=True, help="output summary CSV path")
    ap.add_argument("--curated", help="optional curated CSV: restrict to QC-surviving "
                                      "curves and verify labels")
    args = ap.parse_args()

    data1_dir = os.path.join(args.data, "data1")
    data2_dir = os.path.join(args.data, "data2")
    os.makedirs(args.out, exist_ok=True)

    records = list(tgsd.iter_raw(data1_dir, data2_dir))
    print(f"[load] {len(records)} raw sheet records from {args.data}")
    reps = dedup_data1(records)
    keep = reps + [s for s in records if s["source"] == "data2"]
    print(f"[dedup] {len(keep)} curves after data1 de-duplication")

    curated_ids = None
    if args.curated:
        import pandas as pd
        curated = pd.read_csv(args.curated)
        curated_ids = set(curated["sample_id"])
        print(f"[curated] restricting export to {len(curated_ids)} QC-surviving curves")

    rows = []
    for s in keep:
        if s["source"] == "data1":
            sid = f"D1_{s['file']}_{s['sheet']}"
            group = "data1"
        else:
            sid = f"D2_{s['batch']}_{s['file']}"
            group = tgsd.group_unit_of(s["source"], s["batch"])
        if curated_ids is not None and sid not in curated_ids:
            continue
        co2, ch4 = clean_interp(s["co2"]), clean_interp(s["ch4"])
        tp, hm = clean_interp(s["temp"]), clean_interp(s["hum"])
        base = float(np.mean(co2[:10]))
        dco2 = co2 - base
        n = len(co2)
        if n < 600:
            print(f"[skip] {sid}: only {n} minutes")
            continue
        q = qc_features(s["co2"], s["ch4"], s["temp"], s["hum"])
        label = s["label"]
        lab_s = s.get("label_surv")
        lab_s = lab_s if lab_s is not None and np.isfinite(lab_s) else None
        rl = None if label is None or not np.isfinite(label) else \
            int(np.searchsorted(RISK_EDGES, label, side="right"))
        rec = dict(
            sample_id=sid, source=s["source"], batch=s["batch"], file=s["file"],
            sheet=s["sheet"], group=group,
            label=(None if label is None or not np.isfinite(label) else int(label)),
            label_surv=(None if lab_s is None else int(lab_s)),
            risk_level=rl, risk_name=(None if rl is None else RISK_NAMES[rl]),
            n_minutes=int(n), duration_h=round(n / 60, 2),
            baseline_co2=round(base, 1), co2_max=round(float(co2.max()), 1),
            dco2_max=round(float(dco2.max()), 1), dco2_end=round(float(dco2[-1]), 1),
            dco2_6h=(None if q["dco2_6h"] is None else round(q["dco2_6h"], 1)),
            dco2_12h=(None if q["dco2_12h"] is None else round(q["dco2_12h"], 1)),
            dco2_24h=(None if q["dco2_24h"] is None else round(q["dco2_24h"], 1)),
            dco2_48h=(None if q["dco2_48h"] is None else round(q["dco2_48h"], 1)),
            spearman_t=round(q["spearman_t"], 4),
            snr=(None if q["snr"] is None else round(q["snr"], 2)),
            clip_frac=round(q["clip_frac"], 5),
            ch4_max=round(q["ch4_max"], 1),
            temp_mean=round(q["temp_mean"], 2), hum_mean=round(q["hum_mean"], 2),
            is_control=(1 if (label is not None and np.isfinite(label) and label == 0) else 0),
        )
        np.savetxt(os.path.join(args.out, sid + ".csv"),
                   np.vstack([np.arange(n, dtype=float), co2, dco2, tp, hm, ch4]).T,
                   delimiter=",", header="t_min,co2_ppm,dco2_ppm,temp_c,hum_rh,ch4_ppm",
                   comments="", fmt="%.3f")
        rows.append(rec)

    cols = list(rows[0].keys())
    with open(args.summary, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- label verification against the curated table (when provided)
    if args.curated:
        import pandas as pd
        curated = pd.read_csv(args.curated)
        cur_map = dict(zip(curated["sample_id"], curated["label"]))
        bad = [(r["sample_id"], r["label"], cur_map.get(r["sample_id"]))
               for r in rows if cur_map.get(r["sample_id"]) != r["label"]]
        if bad:
            print("[WARN] label mismatches vs curated (QC-excluded curves are expected "
                  "to differ only in presence, not value):")
            for b in bad:
                print("   ", b)
        else:
            print("[check] all exported labels agree with the curated table")

    print(f"[done] {len(rows)} series -> {args.out}")
    print(f"[done] summary table -> {args.summary}")


if __name__ == "__main__":
    main()
