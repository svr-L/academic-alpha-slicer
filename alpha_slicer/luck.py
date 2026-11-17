"""Luck gate for the core alpha.

A positive core with t >= 2 is evidence only if it was the one specification tried. In practice
a researcher tries several factor sets, state definitions, tail factors and strategy variants,
and reports the best. The gate asks whether the reported core beats what the best of that many
tries would show by luck alone.

* ``psr`` / ``deflated_sharpe`` - Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de
  Prado 2012, 2014) of the core return series, with skewness/kurtosis and the expected maximum
  Sharpe ratio of N null trials (False Strategy Theorem).
* ``effective_trials`` - effective number of independent trials from the correlation of the
  trials' core returns (Li & Ji 2005 eigenvalue count).
* ``spa_test`` - Hansen (2005) Superior Predictive Ability test of H0: no specification has a
  positive mean core return, with the Politis-Romano stationary bootstrap (robust to the
  autocorrelation of daily returns and to the correlation between specifications). The
  ``upper`` p-value is White's (2000) Reality Check.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_moments(x) -> Dict[str, float]:
    """Per-period Sharpe ratio, skewness, kurtosis (non-excess) and length of a return series."""
    x = pd.Series(x).dropna().astype(float).values
    n = len(x)
    if n < 30 or x.std(ddof=1) == 0:
        return {'sr': np.nan, 'skew': np.nan, 'kurt': np.nan, 'n': n}
    return {'sr': float(x.mean() / x.std(ddof=1)), 'skew': float(stats.skew(x)),
            'kurt': float(stats.kurtosis(x, fisher=False)), 'n': n}


def psr(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_star: float = 0.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_star) given an estimate sr from n returns.
    Returns (probability, z). All Sharpe ratios are per period (not annualized)."""
    if not np.isfinite(sr) or n < 3:
        return np.nan, np.nan
    denom = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12)
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z)), float(z)


def expected_max_sr(n_trials: float, sr_var: float) -> float:
    """Expected maximum Sharpe ratio of n_trials independent trials whose true SR is zero and
    whose SR estimates have variance sr_var (Bailey & Lopez de Prado 2014)."""
    if n_trials is None or n_trials <= 1 or not np.isfinite(sr_var) or sr_var <= 0:
        return 0.0
    n = float(n_trials)
    q1 = stats.norm.ppf(1.0 - 1.0 / n)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_var) * ((1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2))


def deflated_sharpe(x, n_trials: float = 1, sr_var: Optional[float] = None) -> Dict[str, float]:
    """Deflated Sharpe Ratio of a return series after n_trials tries.

    sr_var is the variance of the trials' SR estimates. Default: the null sampling variance
    1/(n-1), i.e. the max is computed as if every trial were luck, which is what a test of
    "the best is luck" needs. Pass the cross-trial variance to follow BLdP's empirical variant.
    """
    m = sharpe_moments(x)
    if not np.isfinite(m['sr']):
        return {**m, 'sr_star': np.nan, 'psr': np.nan, 'dsr': np.nan, 'dsr_z': np.nan, 'n_trials': n_trials}
    v = 1.0 / max(m['n'] - 1, 1) if sr_var is None else float(sr_var)
    sr_star = expected_max_sr(n_trials, v)
    p0, _ = psr(m['sr'], m['n'], m['skew'], m['kurt'], 0.0)
    d, z = psr(m['sr'], m['n'], m['skew'], m['kurt'], sr_star)
    return {**m, 'sr_star': sr_star, 'psr': p0, 'dsr': d, 'dsr_z': z, 'n_trials': n_trials}


def effective_trials(R) -> float:
    """Li & Ji (2005) effective number of independent trials from the trials' correlation matrix."""
    R = pd.DataFrame(R).dropna(how='any')
    k = R.shape[1]
    if k <= 1:
        return float(k)
    C = np.corrcoef(R.values, rowvar=False)
    lam = np.abs(np.linalg.eigvalsh(np.nan_to_num(C)))
    return float(np.sum((lam >= 1.0).astype(float) + (lam - np.floor(lam))))


def stationary_bootstrap_indices(T: int, B: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: B resampled index paths of length T, geometric
    blocks with mean length mean_block, wrapping around the end of the sample."""
    p = 1.0 / float(mean_block)
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    new_block = rng.random((B, T)) < p
    starts = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def spa_test(R, B: int = 2000, mean_block: float = 21, seed: int = 0) -> Dict[str, float]:
    """Hansen's SPA test of H0: max_k E[R_k] <= 0 for the columns of R (benchmark = 0).

    Returns consistent / lower / upper p-values (upper = White's Reality Check), the statistic,
    and the best column. Columns are specifications or strategy variants (daily core returns).
    """
    R = pd.DataFrame(R).dropna(how='any')
    X = R.values.astype(float)
    T, K = X.shape
    if T < 50 or K == 0:
        return {'p_consistent': np.nan, 'p_lower': np.nan, 'p_upper': np.nan, 'stat': np.nan,
                'best': None, 'n_specs': K, 'n_obs': T}
    rng = np.random.default_rng(seed)
    dbar = X.mean(axis=0)
    idx = stationary_bootstrap_indices(T, B, mean_block, rng)
    boot = np.stack([X[idx[:, :], k].mean(axis=1) for k in range(K)], axis=1)   # B x K bootstrap means
    omega = np.sqrt(T * boot.var(axis=0, ddof=1))
    omega = np.where(omega > 0, omega, np.nan)
    tstat = math.sqrt(T) * dbar / omega
    stat = float(max(np.nanmax(tstat), 0.0))
    thr = -np.sqrt((omega ** 2 / T) * 2.0 * math.log(math.log(T)))
    mu_c = np.where(dbar >= thr, dbar, 0.0)
    mu_l = np.maximum(dbar, 0.0)
    out = {}
    for name, g in (('consistent', mu_c), ('lower', mu_l), ('upper', dbar)):
        Z = boot - g      # Hansen's Z* = d* - g(dbar): mean 0 for competitive specs, dbar < 0 for poor ones
        Tb = np.maximum(np.nanmax(math.sqrt(T) * Z / omega, axis=1), 0.0)
        out[f'p_{name}'] = float(np.mean(Tb >= stat)) if stat > 0 else 1.0
    best = R.columns[int(np.nanargmax(tstat))]
    return {**out, 'stat': stat, 'best': best, 'n_specs': K, 'n_obs': T}


def luck_gate(core_returns, hac_t: float, n_trials: float = 1, dsr_min: float = 0.95,
              t_min: float = 2.0) -> Dict[str, float]:
    """Single-specification luck gate: positive core, HAC t >= t_min and DSR(n_trials) >= dsr_min."""
    d = deflated_sharpe(core_returns, n_trials)
    passed = bool(np.isfinite(d['dsr']) and d['sr'] > 0 and np.isfinite(hac_t) and hac_t >= t_min
                  and d['dsr'] >= dsr_min)
    return {**d, 'hac_t': hac_t, 'pass': passed}


def multi_core_gate(core_returns: pd.DataFrame, hac_t: pd.Series, n_trials_extra: float = 1,
                    alpha: float = 0.05, dsr_min: float = 0.95, t_min: float = 2.0,
                    B: int = 2000, mean_block: float = 21, seed: int = 0) -> Dict[str, object]:
    """Luck gate for the BEST of several specifications / variants (columns of core_returns).

    The best column passes only if all hold:
      * positive core and HAC t >= t_min (the single-specification rule),
      * Hansen SPA consistent p-value < alpha across all columns,
      * DSR >= dsr_min with N = effective number of columns x n_trials_extra (trials spent
        before this sweep, e.g. strategy variants).
    Also reports the naive rule (best column with t >= t_min) for comparison.
    """
    R = pd.DataFrame(core_returns).dropna(how='any')
    hac_t = pd.Series(hac_t).reindex(R.columns)
    best = hac_t.idxmax()
    spa = spa_test(R, B=B, mean_block=mean_block, seed=seed)
    m_eff = effective_trials(R)
    n_eff = max(m_eff, 1.0) * max(float(n_trials_extra), 1.0)
    d = deflated_sharpe(R[best], n_eff)
    core_best = float(R[best].mean())
    naive = bool(core_best > 0 and hac_t[best] >= t_min)
    passed = bool(naive and np.isfinite(spa['p_consistent']) and spa['p_consistent'] < alpha
                  and np.isfinite(d['dsr']) and d['dsr'] >= dsr_min)
    return {'best': best, 'best_hac_t': float(hac_t[best]), 'best_mean': core_best,
            'n_specs': R.shape[1], 'effective_trials': m_eff, 'n_trials_used': n_eff,
            'spa_p_consistent': spa['p_consistent'], 'spa_p_upper_reality_check': spa['p_upper'],
            'best_dsr': d['dsr'], 'best_dsr_z': d['dsr_z'], 'naive_pass': naive, 'pass': passed}
