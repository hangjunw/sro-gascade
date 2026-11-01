# TGSD — Termite Gas Sensing Dataset

This directory ships two objects of the paper *"From presence–absence to
termite counts and re-calibratable infestation grades: leakage-safe evaluation
of enclosed-chamber CO₂–CH₄ curves"*:

1. **`TGSD_sample/`** — a sample of the raw dataset (`.xlsx` workbooks + batch logs);
2. **`series-new_curated.csv`** — the curated table of all 89 recording curves.

Everything else the pipeline needs (cleaned per-curve series, SRO tensors,
results JSONs, figures) is *derived* from these two objects by the scripts in
`../scripts/` and is therefore not shipped.

---

## 1. The platform and the campaign

Each recording unit is a **7.5-L sealed chamber** instrumented with a
non-dispersive-infrared CO₂ sensor, a metal-oxide CH₄ sensor and a
temperature/relative-humidity probe — four channels at **1-min resolution**,
nominal **48 h** per run (realized 2,783–5,754 min, median 49.5 h). Termites at
controlled densities (50–900 individuals per chamber) were sealed in with a
uniform wood feed; one chamber per batch is a zero-termite control. Two
species were used across the campaign: *Coptotermes formosanus* (batches 4–7,
9–10 and the un-batched pool) and *Reticulitermes flaviceps* (batches 11+12).
The species assignment is stated as in the paper (Section 2.1); the batch logs
name the species of batches 11 and 12 by a vernacular name, and the two are
held in separate evaluation units by design, so no claim in the paper is
species-resolved.

The `risk_level` / `risk_name` columns carry the **source four-level
engineering grading** of the curated table (edges 150 / 300 / 480; the grade
names are kept in the source Chinese wording). This is a different thing from
the three-level grading schemes fitted and compared in the paper, which are
derived from the predicted counts and are re-fitted per deployment.

### Raw layout (`TGSD_sample/`)

```
TGSD_sample/
├── data1/                      # un-batched pool: 10 workbooks, one per nominal
│   ├── 50.xlsx                 #   count level (50, 100, ... 900). Each workbook
│   └── 900.xlsx                #   holds several sheets (one per chamber); sheets
│                               #   that are near-duplicates of each other are one
│                               #   recording cluster (see 01_build_dataset.py).
└── data2/                      # device batches group4 ... group12 (six chambers each;
    ├── group4/                 #   group12 contributes five retained curves)
    │   ├── data1.xlsx ... data6.xlsx  
    │   └── batch_log.txt        
    └── group12/
       └── data1.xlsx ... data6.xlsx
```

Workbook conventions (handled automatically by `src/tgsd.py`): `data2`
workbooks carry an English header row (`co2_ppm / temp_C / humidity_RH /
ch4_ppm / time`).
`data1` workbooks have the standard channel order
`co2, temp, hum, ch4, label`. The released **sample** contains 2 
`data1` workbooks and the batches group4 and group12 (21 QC-surviving curves);
the complete dataset (89 curves) is available from the corresponding author on
reasonable request.

### Batches 11 and 12 (repeat pair)

Batches 11 and 12 were recorded on the **same physical device** (a re-test of
the same six chambers). They therefore form one evaluation unit `group11plus12` and
must never be split across training and test folds — a device-level leakage
that single-batch studies cannot see. The fold assignment in the curated table
enforces this.

---

## 2. `series-new_curated.csv` — data dictionary

89 rows (one per recording curve) × 35 columns.

| Column | Meaning |
|---|---|
| `sample_id` | Curve identifier: `D1_<file>_<sheet>` (un-batched) or `D2_<batch>_<file>` |
| `source` | `data1` (un-batched pool) or `data2` (device batch) |
| `batch` | Raw batch: `data1` or `group4` … `group12` |
| `file`, `sheet` | Workbook stem / sheet name |
| `group` | **Evaluation unit** (= LOBO fold): `data1`, `group4` … `group10`, `group11plus12` |
| `label` | Labeled termite count (regression target; 0 = control) |
| `label_surv` | Post-run surviving count where re-counted (batch 11 only) |
| `risk_level`, `risk_name` | 4-level risk scheme (edges 150/300/480). `risk_name` is a **Chinese column kept in the source wording**; the script constant `RISK_NAMES` in `scripts/01_build_dataset.py` holds the same Chinese names, with the English glosses `I_very low` / `II_low` / `III_medium` / `IV_high` given in the comment beside it |
| `n_minutes`, `duration_h` | Realized series length (minutes / hours) |
| `baseline_co2` | CO₂ baseline = mean of first 10 min (ppm) |
| `co2_max`, `dco2_max`, `dco2_end` | Peak CO₂; max / final net ΔCO₂ vs baseline (ppm) |
| `dco2_12h`, `dco2_24h`, `dco2_48h` | Net ΔCO₂ at fixed horizons (ppm; 11-point moving-average convention) |
| `spearman_t` | Spearman ρ of CO₂ vs time (monotonicity diagnostic) |
| `snr` | Response amplitude / high-frequency noise |
| `clip_frac` | Fraction of samples pinned at full scale (clipping diagnostic) |
| `ch4_max`, `temp_mean`, `hum_mean` | CH₄ peak; mean temperature / humidity |
| `quality_score`, `verdict` | QC composite score and keep/exclude verdict (`kept_core` = kept as core curve, `kept_blank_control` = kept as zero-termite control; the source wording is kept) |
| `is_control` | 1 = zero-termite blank |
| `weight` | QC-derived sample weight |
| `split` | Main split: `train` (62) / `val` (15) / `test` (12) curves |
| `biomass_g` | Released termite biomass (g), where logged |
| `repeat_of`, `repeat_pair_id` | Repeat-test pairing of batches 11 and 12 |
| `species_note`, `device_id` | Species provenance note (as stated in the paper, Section 2.1); device identifier |

### Label tiers (Supplementary S1)

| Tier | n | Rule |
|---|---|---|
| nominal | 45 | Un-batched pool: nominal release number; counts accurate but batch provenance lost — training only |
| exact | 22 | Per-chamber individual count (label not divisible by 10) |
| rounded | 15 | Per-chamber count rounded to ten |
| control | 7 | Zero-termite blanks |

The tier is not stored as a column: it is derived from `source`, `is_control`
and `label` by the rule in `src/sro.py:label_type_of`, and is used to weight
the NAT-lite ablation loss (exact and control 1.0, rounded 0.8, nominal 0.5).

**Terminology.** The paper's Fig. 2a legend and the table above use
*exact* / *rounded*; the prose of Supplementary Note S1 uses *as-loaded* /
*decadal* for the same two tiers. The two vocabularies are synonymous:
as-loaded = exact, decadal = rounded.

The 8 evaluation units: un-batched pool (45, train only), batches 4 (test), 5
(train), 6 (val), 7 (train), 9 (train), 10 (test), and the repeat pair 11+12
(val).

### Reproduction note

`scripts/01_build_dataset.py` recomputes the deterministic columns of this
table from the raw workbooks and verifies label agreement. The shipped table's
QC-diagnostic columns (`spearman_t`, `snr`, `clip_frac`) follow the internal QC
pipeline's rounding conventions and may differ from a full-precision
recomputation in the last digits; all modelling-relevant columns (labels,
tiers, splits, extrema, ΔCO₂ horizons) reproduce exactly.
