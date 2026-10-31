# sro-gascade

Code and data release for the paper

> **From presence–absence to termite counts and re-calibratable infestation grades: leakage-safe evaluation of enclosed-chamber CO₂–CH₄ curves**
> Hangjun Wang, Hao Huang, Fuzhi Wang — *Journal of Pest Science* (2026, submitted)
> Correspondence: whj@zafu.edu.cn

**About the name.** The repository is named after the paper's two method
components: **SRO** (sliding-window sequence reorganization — the
representation that turns each 48-h curve into 80-min subsequences) and
**GASCADE** (gas + cascade — the dual-axis gated encoder that reads one
subsequence and returns a count and an infestation grade). Together with the
Termite Gas Sensing Dataset (TGSD) sample, they make up the release.

This repository contains the core code needed to reproduce the paper's
modelling pipeline, a sample of the **Termite Gas Sensing Dataset (TGSD)**, and
the curated series table. The paper asks a simple operational question: can a
routine four-channel gas log from a sealed monitoring chamber tell you not only
*that* termites are present, but **roughly how many** and **how severe the
infestation is** — at a precision that survives honest, deployment-style
evaluation?

## What is in the pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Dataset build | `scripts/01_build_dataset.py` | Raw TGSD workbooks → cleaned per-curve series + deterministic summary table (incl. `data1` de-duplication) |
| 2. SRO export | `scripts/02_run_sro.py` | Curves → short subsequences by sliding-window sequence reorganization (paper Eq. 1–4) |
| 3. Feature selection | `python -m src.featselect` | Train-only mRMR / LASSO / permutation-importance selection over the 56 group-point features |
| 4. Training & evaluation | `scripts/03_train.py` | Main split (62/15/12 curves, 3 seeds), 8-fold LOBO, and the leakage-control contrast |
| 5. Boundary study | `scripts/04_boundaries.py` | Four grade-boundary schemes + ±10 % threshold sensitivity (main-text Fig. 6; Supplementary Tables S7–S8) |
| 6. Ablations | `scripts/05_ablation.py` | Feature-set, K = 4 and non-overlap arms (Supplementary Table S11) |


## Headline results reported in the paper

Run on the full TGSD (89 curves) under the deterministic settings of the
Reproducibility section, the pipeline yields the paper's numbers (the archived
pre-fix figures of the first submission are given in parentheses where they
differ):

- **SRO expansion**: 89 curves → **27,318** subsequences (L = 10, K = 8, n = 10; one subsequence spans 80 min; 271–568 per curve, mean 307).
- **Count estimation (GASCADE, main split)**: MAE **89.4 ± 16.9** individuals on the count scale (**0.267 ± 0.034** on the log1p scale); statistically indistinguishable from the strongest deep baseline (1D-CNN 88.3 ± 19.4) and within seed dispersion of the curve-level GP comparator (archived pre-fix run: 78.4 ± 7.9 / 0.230 ± 0.007, with every comparator behind); see "Reproducibility" and "Scope".
- **Grading**: macro-F1 **0.821** (boundary route, ii) / QWK **0.754** (ordinal-head route, i) / QWK **0.709** (softmax-head route, iii) (archived: 0.867 / 0.799 / 0.709).
- **LOBO deployment generalization**: pooled MAE(log1p) 0.532 (R² 0.740), pooled QWK **0.762** (Coral ordinal head; the boundary / regression-threshold route reports QWK **0.793** — see Table 3 below; archived run: 0.468 / 0.810 / 0.781 — see "Reproducibility" for why these moved).
- **Leakage cost**: window-random splitting understates MAE(log1p) by **6.6×** (0.052 ± 0.002 vs 0.346 ± 0.030) and by **5.7×** on the count scale (17.8 ± 0.5 vs 102.2 ± 15.4 individuals), both at the subsequence caliber; early stopping fires an order of magnitude later under leakage (epoch 53/58/55 vs 5/6/1).
- **Boundaries as a deployment parameter**: re-calibratable data-driven boundaries beat fixed thresholds by ~22 points (macro-F1 0.821 vs 0.600) and move ≤ 9.3 pp under ±10 % threshold perturbation (archived pre-fix run: 0.911 vs 0.608, ≤ 6.7 pp; see "Reproducibility"). The GMM and manual-fixed schemes coincide at 0.600 because no test count falls between their two boundary sets — a sample artefact on n = 12.
- **Component contributions**: removing the gate leaves count MAE essentially unchanged (89.4 vs 90.7) while costing 2.3 pp of boundary-route macro-F1 (0.821 → 0.798); the NAT-lite tier-weighted loss scores numerically higher but within seed dispersion and is not adopted; the K = 4 configuration matches the main macro-F1 to within 0.1 pp.
- **A fixed conversion is not enough**: a one-feature count/ΔCO₂ fit reaches 111.2 individuals MAE and macro-F1 0.634 (archived-caliber run), and its slope moves by −11.2 % to +3.0 % when refitted within LOBO folds.
- **Monitoring duration**: grading peaks early — route-(ii) macro-F1 0.821 at 4 h, **0.933 at 8 h**, 0.867 at 24 h — while count error keeps falling.
- **LOBO re-calibration** (paper Table 3): re-fitting boundaries within the fold gives QWK 0.743 / macro-F1 0.679, the global quantile boundaries 0.793 / 0.762, and fixed thresholds 0.585 / 0.545 — fold-internal re-calibration brings no gain over the global boundaries.

## Scope

This release reproduces the **main model and the arms the paper's mechanism
argument rests on**: the GASCADE main split, the Ridge / random forest /
XGBoost / 1D-CNN baselines, the leakage-control contrast, the four
grade-boundary schemes with their ±10 % sensitivity, and the feature-set / K = 4
/ non-overlap ablation arms.

It is **not** a full reproduction of main-text Table 2. Five further comparators
of that table are not implemented here and their rows cannot be regenerated
from this code: support-vector regression (112.5 individuals), Gaussian-process
regression (85.1), the patch-based transformer (198.2 ± 15.9), the
channel-independent linear model (248.9 ± 1.5) and the zero-shot foundation
model Chronos-2 (90.2). The fixed dose–response conversion of Table 2 is a
ceiling estimated from these data (Section 3.4 of the paper), not an external
baseline, and is likewise not a pipeline stage. The monitoring-duration
experiment of Fig. 6c and the channel ablation of Supplementary Tables S12–S13
are likewise outside this release. `requirements.txt` therefore lists
`chronos-forecasting` and `transformers` as optional.

## Repository layout

```
sro-gascade/
├── README.md                     ← this file
├── LICENSE                       ← MIT (code) / CC BY 4.0 (data)
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── README.md                 ← TGSD description + data dictionary
│   ├── TGSD_sample/              ← sample of the raw dataset
│   │   ├── data1/                ←   2 of 10 nominal-count workbooks
│   │   └── data2/                ←   batches group4 and group12 (full, incl. batch logs)
│   └── series-new_curated.csv    ← curated table of all 89 curves (35 columns)
├── src/                          ← importable library (see module docstrings)
│   ├── __init__.py               ←   package init; also makes "python -m src.X" work
│   ├── tgsd.py                   ←   raw workbook loader + batch label table
│   ├── sro.py                    ←   sliding-window sequence reorganization
│   ├── models.py                 ←   GASCADE (DSG + BiLSTM + CORAL) + baselines
│   ├── metrics.py                ←   regression / grading metrics
│   ├── evaluate.py               ←   leakage-safe evaluation machinery
│   ├── featselect.py             ←   train-only feature selection
│   ├── boundaries.py             ←   boundary schemes + sensitivity
│   ├── ablation.py               ←   Phase-C ablations
│   ├── xlsxutil.py / statutil.py ←   pure-stdlib xlsx reader / statistics
└── scripts/                      ← CLI entry points 01–06 (run in order)

```

## Installation

Python 3.13, CPU-only (the paper ran on a single CPU workstation, no GPU):

```bash
git clone https://github.com/hangjunw/sro-gascade.git
cd sro-gascade
pip install -r requirements.txt
```

On Linux, install the CPU build of PyTorch explicitly to avoid pulling the
multi-GB CUDA wheel:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Quickstart

Everything below runs on the **released sample** in a few minutes on CPU.
Full-dataset numbers require the complete TGSD (see *Data* section).

```bash
# 1) raw sample workbooks -> cleaned per-curve series (21 curves from the sample)
python scripts/01_build_dataset.py \
    --data data/TGSD_sample \
    --out results/series \
    --summary results/series/dataset_summary.csv \
    --curated data/series-new_curated.csv

# 2) series -> SRO subsequences (X.npy, y.csv, meta.json, gate_report.json)
python scripts/02_run_sro.py \
    --series results/series \
    --curated data/series-new_curated.csv \
    --out results/sro_g1

# 3) smoke test of the whole training stack (3 epochs, 1 seed, 2 LOBO folds)
python scripts/03_train.py --sro results/sro_g1 --out results/phaseB_results.json --smoke
```

Full reproduction (with the complete TGSD placed under `data/TGSD/`):

```bash
./run_all.sh data/TGSD
```

runs the six stages below in order and writes everything to
`results/` (git-ignored). The equivalent explicit sequence is:

```bash
python scripts/01_build_dataset.py --data data/TGSD --out results/series \
    --summary results/series/dataset_summary.csv --curated data/series-new_curated.csv
python scripts/02_run_sro.py --series results/series \
    --curated data/series-new_curated.csv --out results/sro_g1
python -m src.featselect --sro results/sro_g1 \
    --out results/sro_g1/featselect_report.json
python scripts/03_train.py --sro results/sro_g1 --out results/phaseB_results.json \
    --stages main,lobo,leak \
    --dump-preds results/main_model_test_predictions.json   # ~1 h on one CPU workstation
python scripts/04_boundaries.py --sro results/sro_g1 --out results/phaseC_boundaries.json \
    --from-preds results/main_model_test_predictions.json
python scripts/05_ablation.py --sro results/sro_g1 --series results/series \
    --curated data/series-new_curated.csv --results results/phaseB_results.json \
    --featselect results/sro_g1/featselect_report.json \
    --out results/phaseC_ablation.json --work results/ablation_variants
```

The `--dump-preds` / `--from-preds` pair makes the main-split table and the
boundary study score one and the same model instance instead of two separately
trained ones; see "Scoring one model instance twice" below.

Expected checkpoints: SRO writes exactly **27,318** subsequences for the full
dataset (`meta.json: total_subseqs`); stage 3 returns macro-F1 0.821 on the
boundary route and a count MAE of 89.4 ± 16.9 individuals — the paper's
reported values; the archived pre-fix run behind the first submission gave
0.867 and 78.4 ± 7.9 (see "Reproducibility"); the leakage contrast reports an
inflation of ≈ 0.294 in
MAE(log1p) (grouped 0.346 vs window-random 0.053),
against ≈ 0.277 (grouped 0.329 vs window-random 0.052) in that archived run.
With the deterministic settings described below, every torch stage is
reproducible to the bit across independent runs.

## Method in one paragraph

Each 48-h, four-channel recording (CO₂, CH₄, temperature, RH; 1-min
resolution) is reorganized by **SRO** into short subsequences: non-overlapping
L = 10-sample windows are mapped to 56-dimensional group points (14
deterministic statistics × 4 channels), and K = 8 consecutive group points are
packed with step n = 10 into an 80-min subsequence; the curve-level label is
broadcast to all of its subsequences. **GASCADE** reads one subsequence with a
dual-axis statistical gate (DSG: a channel gate g_c ∈ ℝ⁴ and a statistic gate
g_s ∈ ℝ¹⁴ composed by outer product into a 4 × 14 weight map), a 2-layer
bidirectional LSTM (hidden 128) with mean pooling, and two heads — a
regression head on the log1p-transformed count and a CORAL ordinal head.
Evaluation is leakage-safe by construction: the evaluation unit is the whole
curve (main split) or the whole device batch (LOBO), never a single
subsequence; the same-device repeat pair (batches 11+12) is one fold; and
features, boundaries and normalization statistics are fitted on training folds
only. Grade boundaries are treated as a re-calibratable deployment parameter:
training-fold count terciles (main scheme) are compared with Jenks natural
breaks, a 1-D GMM and fixed thresholds, under ±10 % perturbation.

## Data

- **TGSD sample** (`data/TGSD_sample/`): sample raw `.xlsx` workbooks exactly as
  recorded by the acquisition software. The sample covers the un-batched `data1` pool (2 of 10
  nominal-count workbooks) and device batches group4 and group12.
  The complete TGSD (89 curves) is available from the corresponding
  author upon reasonable request.
- **`series-new_curated.csv`**: the curated table of all 89 curves — source,
  device batch, labeled count and the label tier derived from it (Supplementary
  S1), signal summary, quality flags, and the main-split and LOBO fold
  assignment. Full column documentation: [`data/README.md`](data/README.md).

Intermediate artifacts (per-curve cleaned series, SRO tensors, results JSONs)
are **not** shipped: they are regenerated from the two released data
objects by the scripts above. Note that the complete TGSD (89 curves) is not
part of this release and has to be requested from the corresponding author, so
the full-dataset numbers quoted above cannot be re-executed from this
repository alone. The quoted headline numbers are the deterministic
re-execution values provided in `reference_results/`; the first submission's
reported values came from an archived run in which the model weights were initialised
*before* the random seed was set, so that run is a single random draw and is
not bit-reproducible (see "Reproducibility").

## Reproducibility

**Seeds.** All main-split stages use the seeds **{2024, 1, 2}**; the LOBO arm
uses seed 2024 and the leakage-control arm uses the same three seeds.

**Determinism switch.** By default (`GASCADE_DETERMINISTIC=1`) the pipeline runs
single-threaded with `torch.use_deterministic_algorithms(True)`, and every model
is seeded *before* it is constructed. Setting `GASCADE_DETERMINISTIC=0` restores
the faster multi-threaded path; results may then drift slightly.

**What reproduces, and what does not.**

*Reproducible exactly (bit for bit).* With the default `GASCADE_DETERMINISTIC=1`, the released code is fully deterministic: it runs single-threaded, disables the PyTorch/CPU sources of nondeterminism, and seeds the RNG **before** every model is constructed. Two independent full runs then produce the same prediction set down to the last decimal (maximum per-element difference 0.0). Running the code as shipped therefore reproduces every number in the paper. This holds for all baseline models (Ridge, Random Forest, XGBoost, 1D-CNN, BiLSTM without DSG), the NAT-lite arm, and the window-random side of the leakage contrast.

*Not reproducible from the first submission.* The numbers behind the first submission cannot be regenerated, because that archived run seeded the model weights **after** construction — the initial weights came from the process-start RNG rather than the declared seed, so the run was effectively one unrecorded random draw. Re-running the same seed produced a different model. This is fixed in the current release (the seed is now set before construction), which is why the released headline figures differ slightly from the first submission.

*Why it matters, and what stays robust.* Fixing the seed shifts only the **main model's own point estimates and the LOBO fold models** (the parts that depend on the initial weights); everything else was already identical between the two runs. The shift is small relative to the 12-curve test set — the main-model count MAE moves from 78.4 ± 7.9 to 89.4 ± 16.9 individuals, within roughly one standard deviation of the sampling noise — and it changes **no structural conclusion**:

- the boundary route remains the stronger grading route;
- the four boundary schemes keep the same accuracy order (quantile > GMM = manual-fixed > Jenks);
- the main model's counting accuracy is statistically indistinguishable from the strongest baselines (its ± 16.9 spread equals its gap to the nearest comparator, 1D-CNN at 88.3 ± 19.4), so the paper reports a tie rather than a lead.

Two *rankings* move with the random draw and should not be read as stable — the main-model vs 1D-CNN ordering on count MAE, and the GMM-vs-manual-fixed fragility ordering. All reported figures come from the deterministic re-execution in the paper.


**Scoring one model instance twice.** The main-split table (route ii) and the
boundary study (Fig. 6) are two scorings of the same predictions. To avoid
scoring two independently trained instances, export the main model's test-curve
predictions and re-use them in the boundary study:

```bash
python scripts/03_train.py --sro results/sro_g1 --out results/phaseB_results.json \
    --dump-preds results/main_model_test_predictions.json
python scripts/04_boundaries.py --sro results/sro_g1 \
    --out results/phaseC_boundaries.json \
    --from-preds results/main_model_test_predictions.json
```

With `--from-preds` the boundary study does not re-train the model; it re-scores
the exported predictions under every scheme, so both places report the same
macro-F1 (0.821 in the re-execution).


```bibtex
@article{wang2026termite,
  title   = {From presence--absence to termite counts and re-calibratable
             infestation grades: leakage-safe evaluation of enclosed-chamber
             CO2--CH4 curves},
  author  = {Wang, Hangjun and Huang, Hao and Wang, Fuzhi},
  journal = {Journal of Pest Science},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0. Third-party dependencies keep their
own licenses.

## Contact

Hangjun Wang — whj@zafu.edu.cn
