from __future__ import annotations

"""Inference utilities for Academic Alpha Slicer (v0.4, corrected in v0.4b).

Two jobs:

1. Measure the core-alpha standard error under several dependence assumptions instead of
   treating one bandwidth as a universal answer.
2. Put uncertainty bands around the *decomposition itself* through a stationary bootstrap
   that jointly resamples the strategy and the prepared traded-factor design and refits every
   spanning regression / Shapley subset in each replication.

v0.4b: every row is computed on the TRADED-ONLY spanning regression. Its intercept equals the
core (the zero-priced controls are orthogonal to the constant and to the traded regressors),
but its residuals keep the variation of the non-traded payoff. In v0.4 the grid used the full
design, whose controls absorb that variation while their in-sample mean, part of the core's
estimation error, stays in the intercept, so no bandwidth could widen the interval. HAC rows use
Kiefer-Vogelsang fixed-b critical values.

The bootstrap conditions on the feature definitions chosen on the original sample. Non-traded
blocks have a zero price and hence zero mean contribution; the bootstrap covers the priced
traded blocks, the core and implementation drag.
"""

from itertools import permutations
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .regression import (NORMAL_CV, align_yx, fixed_b_critical_value, hac_ols, prewhitened_hac_ols,
                         size_adjusted_p)
from .luck import stationary_bootstrap_indices as _sb_paths


def automatic_hac_lags(n: int) -> int:
    """Newey-West / statsmodels-style automatic lag rule.

    floor(4 * (T/100)^(2/9)), with a minimum of one lag when T > 1.
    This is a generic bandwidth benchmark, not the persistence-safe choice by itself.
    """
    n = int(n)
    if n <= 1:
        return 0
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def stationary_bootstrap_indices(n: int, mean_block: float = 63.0,
                                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices of length ``n`` (one path).

    Blocks have geometric length with expectation ``mean_block`` and wrap around the sample.
    Thin wrapper on ``luck.stationary_bootstrap_indices``, which draws many paths at once.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    mean_block = float(mean_block)
    if not np.isfinite(mean_block) or mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    rng = np.random.default_rng() if rng is None else rng
    return _sb_paths(n, 1, mean_block, rng)[0]


def _ols_alpha_fast(y: np.ndarray, X: Optional[np.ndarray] = None) -> float:
    """Fast OLS intercept used inside the bootstrap (point estimate only)."""
    y = np.asarray(y, float).reshape(-1)
    if X is None or np.asarray(X).size == 0:
        return float(np.mean(y))
    X = np.asarray(X, float)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0])


def _shapley_fast(y: np.ndarray, arrays: Dict[str, np.ndarray], traded: Sequence[str],
                  idx: np.ndarray, annualization: int) -> Dict[str, float]:
    """Exact block-Shapley attribution for one bootstrap draw (2^k subset regressions)."""
    traded = list(traded)
    raw = float(np.mean(y) * annualization)
    cache = {(): raw}

    def alpha_after(subset: Iterable[str]) -> float:
        key = tuple(sorted(subset))
        if key not in cache:
            cols = [arrays[n][idx] for n in key if n in arrays and arrays[n].shape[1] > 0]
            X = np.column_stack(cols) if cols else None
            cache[key] = _ols_alpha_fast(y, X) * annualization
        return cache[key]

    contrib = {n: [] for n in traded}
    for perm in permutations(traded):
        S, a_prev = [], raw
        for name in perm:
            S2 = S + [name]
            a_new = alpha_after(S2)
            contrib[name].append(a_prev - a_new)
            S, a_prev = S2, a_new
    out = {f"premium_{n}": float(np.mean(contrib[n])) for n in traded}
    out["core_unexplained"] = float(alpha_after(traded))
    out["raw_net_mean"] = raw
    return out


def _bootstrap_pvalue(draws: np.ndarray) -> float:
    """Two-sided percentile p-value for H0: parameter = 0 (0 inside / outside the percentile interval).

    A diagnostic alongside the percentile interval, not a replacement for the luck gate.
    """
    x = np.asarray(draws, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    pneg = (1.0 + np.sum(x <= 0.0)) / (len(x) + 1.0)
    ppos = (1.0 + np.sum(x >= 0.0)) / (len(x) + 1.0)
    return float(min(1.0, 2.0 * min(pneg, ppos)))


def stationary_bootstrap_decomposition(
    y: pd.Series,
    blocks: Dict[str, pd.DataFrame],
    traded: Sequence[str],
    annualization: int = 252,
    reps: int = 500,
    mean_block: float = 63.0,
    ci: float = 0.95,
    seed: int = 1729,
    strategy_gross: Optional[pd.Series] = None,
    costs: Optional[pd.Series] = None,
    point_shapley: Optional[pd.DataFrame] = None,
    raw_point: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stationary-bootstrap inference for the additive alpha decomposition.

    Every replication jointly resamples the common estimation sample and refits all traded
    spanning regressions required by the exact block-Shapley decomposition. This keeps serial
    dependence inside geometric blocks and carries factor-loading and factor-mean estimation
    error into each slice. Only traded blocks enter the refits, so the variation of non-traded
    payoffs stays in the core's distribution, where it belongs.

    Returns
    -------
    inference : DataFrame
        Point estimate, bootstrap mean/bias/SE, percentile CI, sign probability and p-value.
    draws : DataFrame
        One row per bootstrap replication.
    """
    reps = int(reps)
    if reps <= 0:
        return pd.DataFrame(), pd.DataFrame()
    if not (0 < ci < 1):
        raise ValueError("ci must be in (0,1)")

    y = pd.Series(y).astype(float)
    n = len(y)
    if n < 20:
        raise ValueError("Need at least 20 observations for bootstrap inference")
    arrays = {k: pd.DataFrame(v).reindex(y.index).to_numpy(float) for k, v in blocks.items() if k in traded}
    g = None if strategy_gross is None else pd.Series(strategy_gross).reindex(y.index).astype(float).to_numpy()
    c = None if costs is None else pd.Series(costs).reindex(y.index).astype(float).to_numpy()

    rng = np.random.default_rng(seed)
    paths = _sb_paths(n, reps, float(mean_block), rng)
    yv = y.to_numpy(float)
    rows = []
    for b in range(reps):
        idx = paths[b]
        stats_b = _shapley_fast(yv[idx], arrays, traded, idx, annualization)
        if g is not None:
            gv = g[idx]
            stats_b["gross_mean"] = float(np.nanmean(gv) * annualization)
            stats_b["implementation_cost_drag"] = float(np.nanmean(gv - yv[idx]) * annualization)
        elif c is not None:
            cv = c[idx]
            stats_b["implementation_cost_drag"] = float(np.nanmean(cv) * annualization)
            stats_b["gross_mean"] = stats_b["raw_net_mean"] + stats_b["implementation_cost_drag"]
        rows.append(stats_b)
    draws = pd.DataFrame(rows)

    point = {"raw_net_mean": float(y.mean() * annualization) if raw_point is None else float(raw_point)}
    if point_shapley is not None:
        for name in traded:
            if name in point_shapley.index:
                point[f"premium_{name}"] = float(point_shapley.loc[name, "contribution_ann"])
        if "core_unexplained" in point_shapley.index:
            point["core_unexplained"] = float(point_shapley.loc["core_unexplained", "contribution_ann"])
    if "core_unexplained" not in point:
        point.update(_shapley_fast(yv, arrays, traded, np.arange(n), annualization))
    if g is not None:
        point["gross_mean"] = float(np.nanmean(g) * annualization)
        point["implementation_cost_drag"] = float(np.nanmean(g - yv) * annualization)
    elif c is not None:
        point["implementation_cost_drag"] = float(np.nanmean(c) * annualization)
        point["gross_mean"] = point["raw_net_mean"] + point["implementation_cost_drag"]

    alpha = (1.0 - ci) / 2.0
    out = []
    for col in draws.columns:
        x = draws[col].to_numpy(float)
        x = x[np.isfinite(x)]
        est = float(point.get(col, np.nan))
        if len(x):
            lo, hi = np.quantile(x, [alpha, 1 - alpha])
            se = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
            bootmean = float(np.mean(x))
            signprob = float(max(np.mean(x > 0), np.mean(x < 0)))
            p = _bootstrap_pvalue(x)
        else:
            lo = hi = se = bootmean = signprob = p = np.nan
        kind = "premium" if col.startswith("premium_") else ("core" if col == "core_unexplained" else "accounting")
        out.append({
            "metric": col, "kind": kind, "estimate_ann": est, "bootstrap_mean_ann": bootmean,
            "bootstrap_bias_ann": bootmean - est if np.isfinite(bootmean) and np.isfinite(est) else np.nan,
            "bootstrap_se_ann": se, "ci_low_ann": float(lo), "ci_high_ann": float(hi),
            "p_two_sided": p, "sign_probability": signprob, "n_boot": len(x),
            "mean_block": float(mean_block), "ci_level": float(ci),
        })
    inf = pd.DataFrame(out).set_index("metric")
    return inf, draws


def _hac_row(method, fit, ann, lags, cv, model, decision):
    a = float(fit["params"]["const"] * ann)
    se = float(np.sqrt(max(fit["cov"].loc["const", "const"], 0.0)) * ann)
    t = float(fit["tvalues"]["const"])
    return {"method": method, "alpha_ann": a, "se_ann": se, "t": t, "critical_value": cv,
            "p": size_adjusted_p(t, cv), "ci_low_ann": a - cv * se, "ci_high_ann": a + cv * se,
            "lags": lags, "dependence_model": model, "decision_family": decision}


def core_inference_grid(y: pd.Series, X: pd.DataFrame, annualization: int = 252,
                        primary_lags: int = 126, hac_lag_grid: Sequence[int] = (21, 63, 126),
                        include_auto: bool = True, include_prewhite: bool = True,
                        bootstrap_inference: Optional[pd.DataFrame] = None, fixed_b: bool = True,
                        X_full: Optional[pd.DataFrame] = None, v03_lags: int = 21) -> pd.DataFrame:
    """The core intercept under several dependence estimators, on the traded-only design ``X``.

    ``decision_family`` marks the rows that define ``inference_fragile``: the Bartlett HAC rows
    (fixed-b critical values) and the stationary bootstrap. Two rows are reported but excluded:
    VAR(1) score prewhitening, which the v0.4 stress test showed over-rejects when the mean
    shifts with persistent regimes, and the pre-v0.4b statistic (full design incl. zero-priced
    controls, HAC(``v03_lags``), normal critical value), kept for comparison.
    """
    y, X = align_yx(y, X)
    n = len(y)
    rows, seen = [], []
    for lag in [int(primary_lags)] + [int(x) for x in hac_lag_grid]:
        if lag in seen:
            continue
        seen.append(lag)
        cv = fixed_b_critical_value((lag + 1) / n) if fixed_b else NORMAL_CV
        rows.append(_hac_row(f"HAC({lag})", hac_ols(y, X, lags=lag), annualization, lag, cv,
                             "Bartlett HAC, traded-only design" + (", fixed-b" if fixed_b else ""), True))
    if include_auto:
        lag = automatic_hac_lags(n)
        if lag not in seen:
            cv = fixed_b_critical_value((lag + 1) / n) if fixed_b else NORMAL_CV
            rows.append(_hac_row("HAC(auto)", hac_ols(y, X, lags=lag), annualization, lag, cv,
                                 "automatic Bartlett HAC, traded-only design", True))
    if include_prewhite:
        fit = prewhitened_hac_ols(y, X, lags="auto")
        row = _hac_row("HAC(prewhite)", fit, annualization, fit.get("hac_lags", np.nan), NORMAL_CV,
                       f"VAR(1) score prewhitening; spectral radius={fit.get('prewhite_radius', np.nan):.3f}; "
                       "audit only", False)
        rows.append(row)
    if bootstrap_inference is not None and not bootstrap_inference.empty and "core_unexplained" in bootstrap_inference.index:
        b = bootstrap_inference.loc["core_unexplained"]
        se = float(b["bootstrap_se_ann"])
        a = float(b["estimate_ann"])
        rows.append({"method": "Stationary bootstrap", "alpha_ann": a, "se_ann": se,
                     "t": a / se if se > 0 else np.nan, "critical_value": NORMAL_CV,
                     "p": float(b["p_two_sided"]), "ci_low_ann": float(b["ci_low_ann"]),
                     "ci_high_ann": float(b["ci_high_ann"]), "lags": np.nan,
                     "dependence_model": f"geometric blocks; mean length={b['mean_block']:.0f}; percentile CI",
                     "decision_family": True})
    if X_full is not None:
        yf, Xf = align_yx(y, X_full)
        rows.append(_hac_row(f"v0.3 full design HAC({v03_lags})", hac_ols(yf, Xf, lags=v03_lags), annualization,
                             v03_lags, NORMAL_CV, "pre-v0.4b statistic: zero-priced controls in the regression; "
                             "comparison only", False))
    out = pd.DataFrame(rows).set_index("method")
    if len(out):
        out["significant_5pct"] = (out["ci_low_ann"] > 0) | (out["ci_high_ann"] < 0)
    return out


def inference_fragility_summary(core_inference: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """How much the alpha claim depends on the standard-error estimator (decision family only).

    Each row's decision threshold is ``threshold`` scaled by its critical value / 1.96, so a
    long-bandwidth HAC row needs a larger t. ``inference_fragile``: at least one decision-family
    row clears its threshold and at least one does not.
    """
    if core_inference is None or core_inference.empty:
        return pd.Series(dtype=object)
    ci = core_inference
    if "decision_family" in ci.columns:
        ci = ci[ci["decision_family"].astype(bool)]
    if ci.empty:
        return pd.Series(dtype=object)
    t = ci["t"].astype(float)
    at = t.abs()
    cvs = ci["critical_value"].astype(float) if "critical_value" in ci.columns else pd.Series(NORMAL_CV, index=ci.index)
    thr = float(threshold) * cvs / NORMAL_CV
    ratio = at / thr
    p = ci["p"].astype(float)
    ci_excludes_zero = (ci["ci_low_ann"] > 0) | (ci["ci_high_ann"] < 0)
    signs = np.sign(ci["alpha_ann"].astype(float))
    return pd.Series({
        "inference_methods": len(ci),
        "core_t_min_inference": float(t.min()),
        "core_t_max_inference": float(t.max()),
        "core_abs_t_min_inference": float(at.min()),
        "core_abs_t_max_inference": float(at.max()),
        "core_p_max_inference": float(p.max()),
        "core_ci_all_exclude_zero": bool(ci_excludes_zero.all()),
        "core_inference_sign_consistent": bool((signs == signs.iloc[0]).all()),
        "core_abs_t_all_above_threshold": bool((ratio >= 1.0).all()),
        "inference_fragile": bool((ratio.max() >= 1.0) and (ratio.min() < 1.0)),
    })
