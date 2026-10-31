"""
models.py -- GASCADE (the paper's main model), its ablations and the tabular /
convolutional baselines, plus the shared training and evaluation machinery.

GASCADE (gas + cascade) is a dual-axis gated encoder for one 80-min SRO
subsequence (K = 8 group points x 56 statistics):

  * DSG, the dual-axis statistical gate (main-text Eq. 5-6): a channel gate
    g_c in R^4 and a statistic gate g_s in R^14, each a squeeze-excitation
    projection of the pooled input, composed by outer product into a 4x14
    weight map G = g_c (x) g_s that re-weights every group point element-wise;
  * a 2-layer bidirectional LSTM (hidden 128, dropout 0.3) with mean pooling
    over time (Eq. 7);
  * two heads: a regression head on the log1p-transformed count and a CORAL
    ordinal head (Cao et al. 2020) with two rank-consistent binary tasks and
    monotone biases (Eq. 8).

Training minimizes, with equal weight, the count MSE (log1p) and the mean
binary cross-entropy of the two CORAL tasks; Adam (lr 1e-3, batch 256, <= 60
epochs) with curve-level validation early stopping (patience 8).  An optional
NAT-lite loss weighting (exact/control 1.0, rounded 0.8, nominal 0.5) is
implemented for the ablation switch.

Baselines share splits, features and early-stopping budgets: Ridge, random
forest (500 trees), XGBoost (falling back to scikit-learn
HistGradientBoosting when XGBoost is unavailable), a 1D-CNN, and a
softmax-cross-entropy variant of the same BiLSTM+DSG backbone.
"""

import os

import numpy as np

# ------------------------------------------------------------------ constants
N_CH, N_STAT, N_FEAT, K = 4, 14, 56, 8
BATCH, LR, MAX_EPOCHS, PATIENCE = 256, 1e-3, 60, 8
SEEDS = [2024, 1, 2]                       # main-split seeds (paper: 2024, 1, 2)
NAT_WEIGHTS = {"exact": 1.0, "control": 1.0, "rounded": 0.8, "nominal": 0.5}

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Determinism (see README, "Reproducibility"): by default the pipeline runs
# single-threaded with PyTorch deterministic algorithms, so that seeding before
# model construction fully determines the result. Set GASCADE_DETERMINISTIC=0
# to restore the faster multi-threaded path.
DETERMINISTIC = os.environ.get("GASCADE_DETERMINISTIC", "1").strip().lower() not in (
    "0", "false", "no")
if DETERMINISTIC:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
else:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

# The tree ensembles (random forest, XGBoost) reduce in parallel, so their
# floating-point summation order -- and hence the last bits of their outputs --
# depends on the thread count.  Pin them to one thread under the deterministic
# setting so that the tabular baselines are reproducible as well.
N_JOBS = 1 if DETERMINISTIC else -1


def seed_all(seed):
    """Seed the global torch and numpy RNGs.

    IMPORTANT: call this *before* constructing a model.  PyTorch draws a
    module's initial weights at construction time (nn.Linear.reset_parameters
    and friends), so seeding only inside the training loop leaves the weights
    at the mercy of the RNG state of the process start -- the same seed then
    yields a different model on every run.  Seeding here and again inside
    ``train_torch`` (which additionally fixes the batch order) makes a run
    reproducible to the bit.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))


# ------------------------------------------------------------------ torch nets
class DSG(nn.Module):
    """Dual-axis statistical gate: channel gate g_c in R^4 x statistic gate
    g_s in R^14, composed element-wise into G = g_c (x) g_s (4x14)."""

    def __init__(self, n_ch=N_CH, n_stat=N_STAT):
        super().__init__()
        self.n_ch, self.n_stat = n_ch, n_stat
        self.ch_mlp = nn.Sequential(nn.Linear(n_ch, 8), nn.ReLU(), nn.Linear(8, n_ch))
        self.st_mlp = nn.Sequential(nn.Linear(n_stat, 8), nn.ReLU(), nn.Linear(8, n_stat))

    def forward(self, x):                       # x: (B, T, 56)
        B, T, _ = x.shape
        x4 = x.view(B, T, self.n_ch, self.n_stat)
        c = x4.mean(dim=(1, 3))                 # (B, 4)
        s = x4.mean(dim=(1, 2))                 # (B, 14)
        g_c = torch.sigmoid(self.ch_mlp(c)).view(B, 1, self.n_ch, 1)
        g_s = torch.sigmoid(self.st_mlp(s)).view(B, 1, 1, self.n_stat)
        G = g_c * g_s                           # (B, 1, 4, 14)
        return (x4 * G).view(B, T, -1), G


class CoralHead(nn.Module):
    """CORAL ordinal head (Cao et al. 2020): shared weight + K-1 monotone biases."""

    def __init__(self, in_dim, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.bias = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, h):
        return self.fc(h).squeeze(-1), self.bias          # a: (B,), bias: (K-1,)


class BiLSTMNet(nn.Module):
    """Main model: DSG (optional) -> 2-layer BiLSTM -> mean pooling ->
    regression head + CORAL head."""

    def __init__(self, hidden=128, use_dsg=True):
        super().__init__()
        self.dsg = DSG() if use_dsg else None
        self.lstm = nn.LSTM(N_FEAT, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.reg_head = nn.Linear(2 * hidden, 1)
        self.coral = CoralHead(2 * hidden)

    def forward(self, x):
        G = None
        if self.dsg is not None:
            x, G = self.dsg(x)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        reg = self.reg_head(pooled).squeeze(-1)
        a, bias = self.coral(pooled)
        return reg, a, bias, G


class BiLSTMCESoftmax(nn.Module):
    """Softmax baseline: same backbone (incl. DSG) with a CE classification head."""

    def __init__(self, hidden=128, n_classes=3):
        super().__init__()
        self.dsg = DSG()
        self.lstm = nn.LSTM(N_FEAT, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.cls_head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        x, _ = self.dsg(x)
        out, _ = self.lstm(x)
        return self.cls_head(out.mean(dim=1))


class CNN1D(nn.Module):
    """1D-CNN baseline: 3x Conv1d(56->64->128->128, k=3) + GAP + heads."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_FEAT, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.reg_head = nn.Linear(128, 1)
        self.coral = CoralHead(128)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2))        # (B, 128, T)
        pooled = z.mean(dim=2)                  # GAP over T
        reg = self.reg_head(pooled).squeeze(-1)
        a, bias = self.coral(pooled)
        return reg, a, bias, None


# ------------------------------------------------------------------ tensors
def make_tensors(y, lvl):
    """Target tensors: log1p count, NAT-lite weights, CORAL tasks, class ids."""
    yreg = torch.tensor(y.label_log1p.values, dtype=torch.float32)
    w = y.label_type.map(NAT_WEIGHTS).astype(float).values
    yw = torch.tensor(w, dtype=torch.float32)
    ycoral = torch.tensor(np.stack([(np.asarray(lvl) > k).astype(np.float32)
                                    for k in range(2)], axis=1))
    ycls = torch.tensor(np.asarray(lvl), dtype=torch.long)
    return yreg, yw, ycoral, ycls


def _loss_from_outputs(out, yreg_b, yw_b, ycoral_b, ycls_b, mode, nat_on):
    if mode == "ce":
        return F.cross_entropy(out, ycls_b)
    reg, a, bias, _ = out
    w = yw_b if nat_on else torch.ones_like(yw_b)
    mse = (w * (reg - yreg_b) ** 2).sum() / w.sum()
    logits = a.unsqueeze(1) - bias.unsqueeze(0)          # (B, K-1)
    bce = F.binary_cross_entropy_with_logits(logits, ycoral_b, reduction="none").mean(dim=1)
    bce = (w * bce).sum() / w.sum()
    return mse + bce


def train_torch(model, Xs_t, yreg_t, yw_t, ycoral_t, ycls_t,
                tr_idx, va_idx, mode="dual", nat_on=False,
                epochs=MAX_EPOCHS, patience=PATIENCE, seed=0, tag="", log=print):
    """Train one torch model with curve-level validation early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, bad, best_ep = float("inf"), None, 0, -1
    import time
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), BATCH):
            b = torch.tensor(perm[i:i + BATCH], dtype=torch.long)
            out = model(Xs_t[b])
            loss = _loss_from_outputs(out, yreg_t[b], yw_t[b], ycoral_t[b], ycls_t[b],
                                      mode, nat_on)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, nb = 0.0, 0
            for i in range(0, len(va_idx), BATCH):
                b = torch.tensor(va_idx[i:i + BATCH], dtype=torch.long)
                out = model(Xs_t[b])
                vl += float(_loss_from_outputs(out, yreg_t[b], yw_t[b], ycoral_t[b],
                                               ycls_t[b], mode, nat_on)) * len(b)
                nb += len(b)
            vl /= max(nb, 1)
        if vl < best_val - 1e-5:
            best_val, bad, best_ep = vl, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"    [{tag}] ep{ep:02d} val_loss={vl:.4f} best={best_val:.4f}@ep{best_ep} "
                f"({time.time() - t0:.0f}s)")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_ep, time.time() - t0


@torch.no_grad()
def predict_torch(model, Xs_t, idx, mode="dual", batch=4096):
    """Subsequence-level predictions: count (log1p) + CORAL / softmax probs."""
    model.eval()
    regs, probs_c, probs_s = [], [], []
    for i in range(0, len(idx), batch):
        b = torch.tensor(np.asarray(idx[i:i + batch]), dtype=torch.long)
        out = model(Xs_t[b])
        if mode == "ce":
            probs_s.append(F.softmax(out, dim=1).cpu().numpy())
        else:
            reg, a, bias, _ = out
            regs.append(reg.cpu().numpy())
            probs_c.append(torch.sigmoid(a.unsqueeze(1) - bias.unsqueeze(0)).cpu().numpy())
    if mode == "ce":
        return None, None, (np.vstack(probs_s) if probs_s else np.zeros((0, 3)))
    return (np.concatenate(regs) if regs else np.array([]),
            np.vstack(probs_c) if probs_c else np.zeros((0, 2)),
            None)


# ------------------------------------------------------------------ scaler
def fit_transform_scaler(X, fit_idx):
    """Standardize the flattened 8x56 features; statistics from fit rows only."""
    from sklearn.preprocessing import StandardScaler

    Xflat = X.reshape(len(X), -1)
    sc = StandardScaler().fit(Xflat[fit_idx])
    return sc.transform(Xflat).astype(np.float32)


# ------------------------------------------------------------------ baselines
def run_ridge(Xs_flat, y, tr_m, te_m):
    from sklearn.linear_model import RidgeCV

    Xagg = Xs_flat.reshape(len(Xs_flat), K, N_FEAT).mean(axis=1)   # pooled 56-dim
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(Xagg[tr_m], y.label_log1p.values[tr_m])
    return model.predict(Xagg[te_m]), float(model.alpha_)


def _proba_full3(clf, Xte):
    pc = np.zeros((len(Xte), 3))
    for j, c in enumerate(clf.classes_):
        pc[:, int(c)] = clf.predict_proba(Xte)[:, j]
    return pc


def _proba_full3_enc(clf, Xte, uniq):
    """Probabilities of a classifier fitted on *encoded* labels, mapped back onto
    the original three-column layout (column index = original label value).

    Tree boosters require contiguous class labels.  On small samples the
    three-level label can miss a level (e.g. degenerate boundary quantiles), so
    the label is encoded for fitting and the probabilities are mapped back here.
    On the full dataset all three levels occur and the mapping is the identity.
    """
    pc = np.zeros((len(Xte), 3))
    proba = clf.predict_proba(Xte)
    for j, c in enumerate(clf.classes_):
        pc[:, int(uniq[int(c)])] = proba[:, j]
    return pc


def run_rf(Xs_flat, y, tr_m, te_m, seed, task="reg"):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if task == "reg":
        m = RandomForestRegressor(n_estimators=500, n_jobs=N_JOBS, random_state=seed)
        m.fit(Xs_flat[tr_m], y.label_log1p.values[tr_m])
        return m.predict(Xs_flat[te_m])
    m = RandomForestClassifier(n_estimators=500, n_jobs=N_JOBS, random_state=seed)
    m.fit(Xs_flat[tr_m], y.label_level3.values[tr_m])
    return _proba_full3(m, Xs_flat[te_m])


def make_xgb_like():
    try:
        import xgboost  # noqa: F401
        from xgboost import XGBClassifier, XGBRegressor
        return "xgboost", XGBRegressor, XGBClassifier
    except Exception:
        return "HistGradientBoosting(fallback)", None, None


def run_boost(xgb_kind, XGBR, XGBC, Xs_flat, y, tr_m, te_m, seed, params=None):
    # Tree boosters need contiguous class labels: encode the three-level label
    # for fitting and map the probabilities back to the original label columns.
    cls = np.asarray(y.label_level3.values)
    uniq = np.unique(cls)
    enc = np.searchsorted(uniq, cls)
    if xgb_kind == "xgboost":
        p = dict(params or {})
        p.setdefault("max_depth", 5)
        p.setdefault("n_estimators", 600)
        p.setdefault("learning_rate", 0.05)
        base = dict(subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                    random_state=seed, n_jobs=N_JOBS, verbosity=0)
        r = XGBR(**{**base, **p})
        r.fit(Xs_flat[tr_m], y.label_log1p.values[tr_m])
        yp = r.predict(Xs_flat[te_m])
        c = XGBC(objective="multi:softprob", num_class=int(len(uniq)), **{**base, **p})
        c.fit(Xs_flat[tr_m], enc[tr_m])
        return yp, _proba_full3_enc(c, Xs_flat[te_m], uniq)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    r = HistGradientBoostingRegressor(random_state=seed)
    r.fit(Xs_flat[tr_m], y.label_log1p.values[tr_m])
    yp = r.predict(Xs_flat[te_m])
    c = HistGradientBoostingClassifier(random_state=seed)
    c.fit(Xs_flat[tr_m], enc[tr_m])
    return yp, _proba_full3_enc(c, Xs_flat[te_m], uniq)
