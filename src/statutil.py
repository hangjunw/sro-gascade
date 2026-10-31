"""Statistical helpers (no scipy dependency)."""
import math
import numpy as np


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if abs(r) >= 1:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def rankdata(a):
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(n, float)
    r[order] = np.arange(1, n + 1)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            r[order[i:j + 1]] = avg
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def linfit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def partial_corr(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 5:
        return np.nan
    Z = np.vstack([z, np.ones_like(z)]).T
    def resid(v):
        c, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ c
    return pearson(resid(x), resid(y))[0]
