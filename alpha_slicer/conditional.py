from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from .regression import hac_ols, linear_combination


def _is_binary(z: pd.Series) -> bool:
    vals = pd.unique(z.dropna())
    return len(vals) > 0 and set(np.round(vals, 12)).issubset({0.0, 1.0})


def conditional_alpha_table(y, X, states, lags=21, ann=252, min_regime_obs=63, crit=None):
    """Regime alphas with the state on the ALPHA side (Christopherson-Ferson-Glassman style).

    For each lagged state z (one regression per state):

        y_t = a0 + a1 * z_{t-1} + b'X_t + e_t

    where X is the TRADED design (traded blocks charged at their means, conditional betas F*z
    included). Since v0.4b the zero-priced non-traded controls are left out: they are orthogonal
    to the constant and the traded regressors but not to the state, so inside this regression
    they would be charged a regime-dependent price, and they would shrink the standard errors
    the same way they did for the core. A state dummy is time-varying alpha, not a risk premium,
    so it is reported here instead of being removed as a "premium".

    Binary states -> rows ``off`` (a0), ``on`` (a0+a1), ``on_minus_off`` (a1) and
    ``time_weighted`` (share-weighted average = unconditional alpha within this model).
    Continuous states -> ``alpha_at_state_mean`` and ``alpha_slope_per_unit``.
    Standard errors are Bartlett HAC with ``lags``; ``crit`` is the two-sided 5% critical value
    used for the p-values (fixed-b for long bandwidths; default 1.96).
    """
    y = pd.Series(y).astype(float)
    X = pd.DataFrame(X).reindex(y.index) if X is not None else pd.DataFrame(index=y.index)
    S = pd.DataFrame(states).reindex(y.index)
    rows = []
    cv = float(stats.norm.ppf(0.975)) if crit is None else float(crit)

    def _row(state, regime, est_se_t, n_days, share):
        est, se, t = est_se_t
        p = float(2 * stats.norm.sf(abs(t) * stats.norm.ppf(0.975) / cv)) if np.isfinite(t) else np.nan
        return {"state": state, "regime": regime, "alpha_ann": est * ann,
                "se_ann": se * ann if np.isfinite(se) else np.nan, "t": t, "p": p,
                "critical_value": cv, "n_days": n_days, "share_of_days": share}

    for c in S.columns:
        z = S[c].astype(float)
        name = f"ALPHA_STATE_{c}"
        if _is_binary(z):
            n_on = int((z == 1).sum()); n_off = int((z == 0).sum())
            if min(n_on, n_off) < min_regime_obs:
                for regime in ("off", "on", "on_minus_off", "time_weighted"):
                    rows.append({"state": c, "regime": regime, "alpha_ann": np.nan, "se_ann": np.nan,
                                 "t": np.nan, "p": np.nan, "n_days": n_on if regime == "on" else n_off,
                                 "share_of_days": np.nan})
                continue
            fit = hac_ols(y, pd.concat([X, z.rename(name)], axis=1), lags=lags)
            share = n_on / (n_on + n_off)
            rows.append(_row(c, "off", linear_combination(fit, {"const": 1.0}), n_off, 1 - share))
            rows.append(_row(c, "on", linear_combination(fit, {"const": 1.0, name: 1.0}), n_on, share))
            rows.append(_row(c, "on_minus_off", linear_combination(fit, {name: 1.0}), n_on + n_off, np.nan))
            rows.append(_row(c, "time_weighted", linear_combination(fit, {"const": 1.0, name: share}),
                             n_on + n_off, 1.0))
        else:
            zc = (z - z.mean()).rename(name)
            fit = hac_ols(y, pd.concat([X, zc], axis=1), lags=lags)
            n = int(z.notna().sum())
            rows.append(_row(c, "alpha_at_state_mean", linear_combination(fit, {"const": 1.0}), n, 1.0))
            rows.append(_row(c, "alpha_slope_per_unit", linear_combination(fit, {name: 1.0}), n, np.nan))
    if not rows:
        return pd.DataFrame(columns=["alpha_ann", "se_ann", "t", "p", "n_days", "share_of_days"])
    return pd.DataFrame(rows).set_index(["state", "regime"])
