"""Shared statistical utilities for the analyzer (bootstrap CIs, ICC, correlations).

Centralises the uncertainty-quantification helpers used by the experiment
modules (ensemble ablation, verbosity bias, positional bias) and the paper
tables.  All randomness is seeded so results are reproducible across runs:
passing the same data and seed always yields the same confidence interval.

Functions return plain Python floats / dataclasses (no pandas dependency) so
they can be serialised to JSON directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Estimate:
    """A point estimate with a bootstrap confidence interval."""
    point: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


_NAN = float("nan")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    data: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for an arbitrary statistic of a 1-D sample.

    Resamples ``data`` with replacement ``n_boot`` times, recomputes
    ``statistic`` on each resample, and returns the observed point estimate
    plus the (1-ci)/2 and 1-(1-ci)/2 percentiles of the bootstrap distribution.
    Returns NaN bounds when the sample has fewer than 2 elements.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return Estimate(_NAN, _NAN, _NAN, 0)
    point = float(statistic(arr))
    if n < 2:
        return Estimate(point, point, point, n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(arr[row]) for row in idx], dtype=float)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return Estimate(point, _NAN, _NAN, n)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1.0 - alpha)))
    return Estimate(point, lo, hi, n)


def mean_ci(values: Sequence[float], n_boot: int = 1000, ci: float = 0.95,
            seed: int = 0) -> Estimate:
    """Bootstrap CI for the mean of ``values``."""
    return bootstrap_ci(np.mean, values, n_boot=n_boot, ci=ci, seed=seed)


# ---------------------------------------------------------------------------
# Paired-sample correlations with bootstrap CIs
# ---------------------------------------------------------------------------

def _paired_clean(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = ~(np.isnan(xa) | np.isnan(ya))
    return xa[mask], ya[mask]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return _NAN
    rho, _ = spearmanr(x, y)
    return float(rho)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return _NAN
    return float(np.corrcoef(x, y)[0, 1])


def cluster_correlation_ci(
    x: Sequence[float],
    y: Sequence[float],
    clusters: Sequence,
    method: str = "spearman",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Cluster (block) bootstrap CI for a correlation when observations are nested.

    When each item (e.g. a benchmark datapoint) contributes several non-independent
    observations (e.g. one response per student), a naive bootstrap over individual
    rows understates uncertainty. This resamples whole CLUSTERS with replacement,
    preserving within-cluster dependence and yielding honest, item-level CIs.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    cl = np.asarray(list(clusters))
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya, cl = xa[mask], ya[mask], cl[mask]
    fn = _spearman if method == "spearman" else _pearson
    if xa.size < 2:
        return Estimate(_NAN, _NAN, _NAN, int(xa.size))
    point = fn(xa, ya)
    uniq = np.unique(cl)
    # index lists per cluster
    idx_by_cluster = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        chosen = rng.integers(0, uniq.size, size=uniq.size)
        rows = np.concatenate([idx_by_cluster[uniq[c]] for c in chosen])
        boot.append(fn(xa[rows], ya[rows]))
    boot = np.array(boot, dtype=float)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return Estimate(point, _NAN, _NAN, int(uniq.size))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1.0 - alpha)))
    return Estimate(point, lo, hi, int(uniq.size))


def correlation_ci(
    x: Sequence[float],
    y: Sequence[float],
    method: str = "spearman",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Bootstrap CI for Spearman rho or Pearson r between paired samples.

    Resampling is paired (the same resampled indices select from both x and y)
    so the dependency structure is preserved.
    """
    xa, ya = _paired_clean(x, y)
    n = xa.size
    fn = _spearman if method == "spearman" else _pearson
    if n < 2:
        return Estimate(_NAN, _NAN, _NAN, n)
    point = fn(xa, ya)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([fn(xa[row], ya[row]) for row in idx], dtype=float)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return Estimate(point, _NAN, _NAN, n)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1.0 - alpha)))
    return Estimate(point, lo, hi, n)


# ---------------------------------------------------------------------------
# Intraclass correlation (inter-rater reliability)
# ---------------------------------------------------------------------------

@dataclass
class ICCResult:
    """ICC(3,1) and ICC(3,k) for a subjects x raters score matrix."""
    icc_3_1: float      # single-rater reliability
    icc_3_k: float      # average-of-k-raters reliability
    n_subjects: int
    n_raters: int

    def as_dict(self) -> dict:
        return asdict(self)


def icc(matrix: Sequence[Sequence[float]]) -> ICCResult:
    """Two-way mixed, consistency ICC(3,1) and ICC(3,k).

    ``matrix`` is shaped (n_subjects, n_raters): each row is one item scored by
    every rater, each column is one rater.  Rows containing any NaN are dropped
    (complete-case analysis), since a balanced design is required for the ANOVA
    decomposition.  This is the appropriate reliability statistic for a fixed
    panel of judges scoring a common set of items, and unlike pairwise Cohen's
    kappa it extends naturally to more than two raters and does not collapse
    when the marginal score distribution is highly skewed.

    Formulae (Shrout & Fleiss 1979):
        ICC(3,1) = (MSR - MSE) / (MSR + (k-1) * MSE)
        ICC(3,k) = (MSR - MSE) / MSR
    where MSR is the between-subjects mean square and MSE the residual mean
    square from a two-way ANOVA without the rater main effect.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("ICC input must be 2-D (subjects x raters)")
    # complete-case: drop rows with any NaN
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return ICCResult(_NAN, _NAN, n, k)

    grand_mean = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)

    # Sums of squares
    ss_total = ((m - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_err = (n - 1) * (k - 1)
    if df_err <= 0:
        return ICCResult(_NAN, _NAN, n, k)

    msr = ss_rows / df_rows
    mse = ss_err / df_err

    denom_1 = msr + (k - 1) * mse
    icc_3_1 = (msr - mse) / denom_1 if denom_1 != 0 else _NAN
    icc_3_k = (msr - mse) / msr if msr != 0 else _NAN
    return ICCResult(float(icc_3_1), float(icc_3_k), n, k)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control.  Returns a reject-null mask per p-value.

    Preserves input order.  NaN p-values are never rejected.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    reject = np.zeros(n, dtype=bool)
    valid_idx = np.where(~np.isnan(p))[0]
    if valid_idx.size == 0:
        return reject.tolist()
    order = valid_idx[np.argsort(p[valid_idx])]
    m = valid_idx.size
    thresh_idx = -1
    for rank, i in enumerate(order, start=1):
        if p[i] <= (rank / m) * alpha:
            thresh_idx = rank
    if thresh_idx > 0:
        for rank, i in enumerate(order, start=1):
            if rank <= thresh_idx:
                reject[i] = True
    return reject.tolist()
