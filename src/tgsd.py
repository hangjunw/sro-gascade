"""
tgsd.py -- Termite Gas Sensing Dataset (TGSD) raw-file loader.

TGSD is the dataset behind the paper "From presence-absence to termite counts
and re-calibratable infestation grades: leakage-safe evaluation of
enclosed-chamber CO2-CH4 curves".  It consists of 89 recording curves from a
7.5-L sealed-chamber, four-channel gas-monitoring platform (CO2, CH4,
temperature, relative humidity; 1-min resolution, nominal 48 h):

  * ``data1`` -- 10 workbooks (one per nominal count level 50...900), each
    holding several sheets (one sheet = one chamber recording).  Counts are
    nominal release numbers; batch provenance was lost, so these 45 curves
    (after within-file de-duplication) are training-only.
  * ``data2`` -- device batches ``group4`` ... ``group12`` (six chambers each,
    ``data1..data6``, one of which is a zero-termite control).  Batch 11 and
    batch 12 were recorded on the same physical device and form one
    repeat-test pair (``group11plus12``) that must never be split.

Each batch directory ships a ``batch_log.txt`` (batch log) recording the number
of termites released per chamber; the transcribed table below is the single
source of truth for labels and is checked against
``series-new_curated.csv`` in ``scripts/01_build_dataset.py``.

This module is I/O only: it reads the raw ``.xlsx`` workbooks and yields
uniform per-curve records.  All paths are parameters with repository-relative
defaults, so the loader runs on any subset of TGSD (e.g. the released sample).
"""

import glob
import os
import re

import numpy as np

from xlsxutil import read_xlsx, to_float

# ------------------------------------------------------------------ labels
# Termites released per chamber, transcribed from each batch's batch_log.txt
# (file stem ``dataN`` -> released count).  Batches 11 and 12 share one device.
BATCH_LABELS = {
    "group4":  {"data1": 225, "data2": 534, "data3": 575, "data4": 450, "data5": 483, "data6": 0},
    "group5":  {"data1": 260, "data2": 600, "data3": 620, "data4": 640, "data5": 130, "data6": 0},
    "group6":  {"data1": 360, "data2": 260, "data3": 270, "data4": 280, "data5": 310, "data6": 0},
    "group7":  {"data1": 105, "data2": 207, "data3": 324, "data4": 412, "data5": 500, "data6": 0},
    "group9":  {"data1": 367, "data2": 369, "data3": 461, "data4": 343, "data5": 424, "data6": 0},
    "group10": {"data1": 190, "data2": 280, "data3": 320, "data4": 440, "data5": 480, "data6": 0},
    "group11": {"data1": 0, "data2": 278, "data3": 339, "data4": 376, "data5": 393, "data6": 497},
    "group12": {"data1": 0, "data2": 281, "data3": 344, "data4": 365, "data5": 393, "data6": 499},
}

# Surviving counts re-counted after the 48-h run (batch 11 only; used for the
# survival audit, not as the regression target).
BATCH_SURVIVAL = {"group11": {"data1": 0, "data2": 225, "data3": 291,
                           "data4": 296, "data5": 308, "data6": 376}}

# Per-chamber mortality / activity remarks quoted from the batch logs.
BATCH_NOTES = {}

STD_COLS = ["co2", "temp", "hum", "ch4"]

# Evaluation units (== LOBO folds): the un-batched pool, one fold per device
# batch, and the merged repeat-test pair group11plus12.
GROUP_UNITS = ["data1", "group4", "group5", "group6", "group7", "group9", "group10", "group11plus12"]


def map_cols(hdr):
    """Map a raw header row to standard channel indices.

    Returns ``(channel->column index, time column index)``.  Handles the
    English headers emitted by the acquisition/export tooling
    (co2_ppm / temp_C / humidity_RH / ch4_ppm / time / cal_status).
    """
    m, ti = {}, None
    for i, h in enumerate(hdr):
        s = str(h).strip().lower()
        if "co" in s and "2" in s:                 # CO2 / co2_ppm
            m.setdefault("co2", i)
        elif "temp" in s:                          # temp_C
            m.setdefault("temp", i)
        elif "hum" in s:                           # humidity_RH
            m.setdefault("hum", i)
        elif "ch4" in s:                           # CH4 / ch4_ppm
            m.setdefault("ch4", i)
        elif "time" in s or "date" in s:           # time
            ti = i
    return m, ti


def load_data2_file(path):
    """Read one ``data2`` workbook.

    Returns a dict with per-channel float arrays (``co2/temp/hum/ch4``),
    timestamp strings, the calibration-status column (if present) and sheet
    metadata -- or ``None`` when no usable sheet is found.
    """
    names, data = read_xlsx(path)
    best = None
    for sn in names:
        rows = data.get(sn) or []
        if not rows:
            continue
        hdr = rows[0]
        m, ti = map_cols(hdr)
        if len(m) >= 4:                                   # header row present
            body = rows[1:]
            if len(body) == 0:
                continue                                  # header-only sheet
            best = (sn, hdr, m, ti, body, True)
            break
        # headerless sheet: first row is data, standard order co2,temp,hum,ch4
        if all(to_float(x) is not None for x in hdr[:4]):
            m2 = {"co2": 0, "temp": 1, "hum": 2, "ch4": 3}
            best = (sn, None, m2, 4 if len(hdr) > 4 else None, rows, False)
            break
    if best is None:
        return None
    sn, hdr, m, ti, body, has_hdr = best
    rec = {c: np.array([to_float(r[m[c]]) if m[c] < len(r) else np.nan
                        for r in body], dtype=float) for c in m}
    rec["tstr"] = [r[ti].strip() if (ti is not None and ti < len(r)) else ""
                   for r in body]
    cal = None
    if has_hdr and hdr:
        for i, h in enumerate(hdr):
            if "cal" in str(h).lower():                    # calibration status
                cal = [r[i] if i < len(r) else "" for r in body]
    rec["cal"] = cal
    rec["sheet"] = sn
    rec["has_header"] = has_hdr
    rec["header"] = hdr
    return rec


def parse_t2(ts):
    """'MM-DD HH:MM' -> minutes since start of year (NaN when unparsable)."""
    out = np.full(len(ts), np.nan)
    for i, s in enumerate(ts):
        if not s:
            continue
        s = s.replace("\u2011", "-").replace("\uff1a", ":")
        m = re.match(r"^\s*(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", s)
        if not m:
            continue
        mo, d, h, mi = map(int, m.groups())
        out[i] = ((mo * 31 + d) * 24 + h) * 60 + mi
    return out


def load_data1_file(path):
    """Read one ``data1`` workbook (headerless sheets; label = file stem)."""
    names, data = read_xlsx(path)
    label = int(os.path.splitext(os.path.basename(path))[0])
    out = []
    for sn in names:
        rows = data.get(sn) or []
        if len(rows) < 2:
            continue
        hdr = [str(x).strip().lower() for x in rows[0]]
        body = rows[1:]
        m = {}
        for c in ["co2", "temp", "hum", "ch4", "lable"]:
            if c in hdr:
                m[c] = hdr.index(c)
        if "co2" not in m:
            m = {"co2": 0, "temp": 1, "hum": 2, "ch4": 3, "lable": 4}
        rec = {c: np.array([to_float(r[m[c]]) if m[c] < len(r) else np.nan
                            for r in body], float) for c in m}
        rec["tstr"] = [""] * len(body)
        rec["sheet"] = sn
        rec["label_file"] = label
        out.append(rec)
    return out


def iter_raw(data1_dir, data2_dir):
    """Yield uniform per-curve records over every raw TGSD file found.

    Each record carries the four channels plus ``source``, ``batch``,
    ``file`` (workbook stem), ``sheet``, ``label`` (released count),
    ``label_surv`` (post-run survival where logged) and ``note``.
    """
    f1 = sorted(glob.glob(os.path.join(data1_dir, "*.xlsx")),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    for p in f1:
        for r in load_data1_file(p):
            r.update(source="data1", batch="data1",
                     file=os.path.splitext(os.path.basename(p))[0],
                     label=float(r["label_file"]))
            yield r
    if not os.path.isdir(data2_dir):
        return
    groups = sorted([d for d in os.listdir(data2_dir) if d.startswith("group")],
                    key=lambda x: int(x.replace("group", "")))
    for g in groups:
        gd = os.path.join(data2_dir, g)
        for fn in sorted(os.listdir(gd)):
            if not fn.lower().endswith(".xlsx"):
                continue
            key = os.path.splitext(fn)[0].replace(" ", "")   # 'data 3' -> 'data3'
            rec = load_data2_file(os.path.join(gd, fn))
            if rec is None:
                rec = {c: np.array([]) for c in STD_COLS}
                rec.update(tstr=[], cal=None, sheet="-",
                           has_header=False, header=None)
            lab = BATCH_LABELS.get(g, {}).get(key, None)
            rec.update(source="data2", batch=g, file=key, label=lab,
                       label_surv=(BATCH_SURVIVAL.get(g) or {}).get(key, None),
                       note=(BATCH_NOTES.get(g) or {}).get(
                           int(key.replace("data", ""))
                           if key.replace("data", "").isdigit() else -1, ""))
            yield rec


def group_unit_of(source, batch):
    """Evaluation unit of a curve (un-batched pool vs merged repeat pair)."""
    if source == "data1":
        return "data1"
    return "group11plus12" if batch in ("group11", "group12") else batch


def stats(v):
    """Descriptive statistics of one channel (NaN-aware)."""
    v = np.asarray(v, float)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return dict(n=0, miss=1.0, mn=np.nan, mx=np.nan, mean=np.nan,
                    std=np.nan, med=np.nan, p01=np.nan, p99=np.nan)
    return dict(n=v.size, miss=float(np.isnan(v).mean()),
                mn=float(ok.min()), mx=float(ok.max()), mean=float(ok.mean()),
                std=float(ok.std(ddof=1)) if ok.size > 1 else 0.0,
                med=float(np.median(ok)),
                p01=float(np.percentile(ok, 1)), p99=float(np.percentile(ok, 99)))
