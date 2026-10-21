from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


NORMAL_CV = float(stats.norm.ppf(0.975))


def fixed_b_critical_value(b: float) -> float:
    """Two-sided 5% critical value of a Bartlett-HAC t statistic under fixed-b asymptotics.

    Kiefer & Vogelsang (2005): cv(b) = 1.96 + 2.9694 b + 0.4160 b^2 - 0.5324 b^3, with
    b = bandwidth / T and bandwidth = lags + 1 for the Newey-West weights 1 - j/(lags+1).
    A simulation with iid data (T = 1,000, 20,000 draws) reproduces it within 0.02 for
    b = 0.02 to 0.2. With a long bandwidth the HAC variance estimate is noisy, so the normal
    1.96 over-rejects; this is the correction.
    """
    b = float(min(max(b, 0.0), 1.0))
    return 1.96 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3


def core_hac_settings(n: int, lags: int, fixed_b: bool = True):
    """Bandwidth actually used for n observations (capped at n/4) and its two-sided 5% critical value."""
    L = int(max(1, min(int(lags), max(1, int(n) // 4))))
    cv = fixed_b_critical_value((L + 1) / max(int(n), 1)) if fixed_b else NORMAL_CV
    return L, float(cv)


def size_adjusted_p(t: float, cv: float) -> float:
    """Two-sided p-value of t scaled so that |t| = cv maps to 5% (exact at 5%, approximate elsewhere)."""
    if not np.isfinite(t):
        return np.nan
    return float(2.0 * stats.norm.sf(abs(t) * NORMAL_CV / cv))


def align_yx(y: pd.Series, X: pd.DataFrame):
    y = pd.Series(y, name=getattr(y, "name", "strategy")).astype(float)
    X = pd.DataFrame(X).astype(float)
    d = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.Series(dtype=float), pd.DataFrame(columns=X.columns, dtype=float)
    return d.iloc[:, 0], d.iloc[:, 1:]


def hac_ols(y: pd.Series, X: pd.DataFrame, lags: int = 21):
    """OLS with Newey-West/HAC covariance. Always includes an intercept.

    Returns coefficients, HAC t/p-values and the HAC covariance matrix, all in the
    original (unscaled) units of X.
    """
    if X is None or pd.DataFrame(X).shape[1] == 0:
        X = pd.DataFrame(index=pd.Series(y).index)
    y, X = align_yx(y, X)
    if len(y) < max(20, X.shape[1] + 5):
        raise ValueError("Insufficient aligned observations for regression")
    # Scale, but do NOT center, regressors. This improves numerical conditioning
    # without changing the intercept or the beta-only residual mean.
    scale = X.std(ddof=0).replace(0, 1.0)
    Xs = X.div(scale, axis=1)
    if X.shape[1] == 0:
        Xc = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
    else:
        Xc = sm.add_constant(Xs, has_constant="add")
    fit = sm.OLS(y.values, Xc.values).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    names = list(Xc.columns)
    params_scaled = pd.Series(fit.params, index=names, name="coef")
    params = params_scaled.copy()
    unscale = pd.Series(1.0, index=names)
    for c in X.columns:
        params[c] = params_scaled[c] / scale[c]
        unscale[c] = 1.0 / scale[c]
    cov_scaled = pd.DataFrame(np.asarray(fit.cov_params()), index=names, columns=names)
    cov = cov_scaled * np.outer(unscale.values, unscale.values)
    tvals = pd.Series(fit.tvalues, index=names, name="t")
    pvals = pd.Series(fit.pvalues, index=names, name="p")
    resid = pd.Series(fit.resid, index=y.index, name="resid")
    return {
        "params": params,
        "tvalues": tvals,
        "pvalues": pvals,
        "cov": cov,
        "resid": resid,
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "n": int(fit.nobs),
        "condition_number": float(np.linalg.cond(Xc.values)),
    }


def linear_combination(fit, weights: dict):
    """Estimate, HAC standard error and t-stat of sum_j w_j * coef_j."""
    names = list(weights)
    w = np.array([weights[k] for k in names], float)
    est = float(w @ fit["params"][names].values)
    var = float(w @ fit["cov"].loc[names, names].values @ w)
    se = float(np.sqrt(var)) if var > 0 else np.nan
    return est, se, (est / se if se and np.isfinite(se) and se > 0 else np.nan)


def wald_test(fit, terms):
    """HAC Wald test that all coefficients in `terms` are zero (chi-square, asymptotic)."""
    terms = [t for t in terms if t in fit["params"].index]
    if not terms:
        return np.nan, np.nan, 0
    b = fit["params"][terms].values
    V = fit["cov"].loc[terms, terms].values
    W = float(b @ np.linalg.pinv(V) @ b)
    k = len(terms)
    return W, float(stats.chi2.sf(W, k)), k


def orthogonalize(Z: pd.DataFrame, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Residuals of every column of Z on [1, X] (in-sample OLS).

    The result has zero mean and zero sample covariance with every column of X. By
    Frisch-Waugh-Lovell, adding such columns to a regression on [1, X] changes neither the
    intercept nor the coefficients on X: they can explain variance but cannot move alpha.
    """
    Z = pd.DataFrame(Z).astype(float)
    if Z.shape[1] == 0:
        return Z.copy()
    if X is None or pd.DataFrame(X).shape[1] == 0:
        return Z - Z.mean()
    X = pd.DataFrame(X).reindex(Z.index).astype(float)
    A = np.column_stack([np.ones(len(Z)), X.values])
    B, *_ = np.linalg.lstsq(A, Z.values, rcond=None)
    return pd.DataFrame(Z.values - A @ B, index=Z.index, columns=Z.columns)


def slopes_only_residual(y: pd.Series, X: pd.DataFrame, lags: int = 21):
    """Fit with intercept, subtract slopes only. The residual mean equals the intercept.

    This mirrors the beta-only residual convention used in Tail-Aware: slopes are
    removed without mechanically demeaning the series.
    """
    y2, X2 = align_yx(y, X)
    fit = hac_ols(y2, X2, lags=lags)
    slopes = fit["params"].drop("const", errors="ignore")
    out = y2 - X2[slopes.index].mul(slopes, axis=1).sum(axis=1)
    out.name = getattr(y, "name", "strategy")
    return out, fit



def _bartlett_lrv(scores: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel long-run covariance of a zero-mean score process."""
    G = np.asarray(scores, float)
    if G.ndim == 1:
        G = G[:, None]
    n, k = G.shape
    if n < 2:
        return np.full((k, k), np.nan)
    G = G - G.mean(axis=0, keepdims=True)
    L = max(0, min(int(lags), n - 1))
    S = (G.T @ G) / n
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        Gamma = (G[ell:].T @ G[:-ell]) / n
        S += w * (Gamma + Gamma.T)
    return S


def prewhitened_hac_ols(y: pd.Series, X: pd.DataFrame, lags="auto"):
    """OLS with VAR(1)-prewhitened HAC covariance on the regression score process.

    This is an Andrews-Monahan-style dependence audit: the score vectors x_t * u_t are
    prewhitened by a VAR(1), a Bartlett HAC is computed on the innovations, and the long-run
    covariance is recoloured by (I-A)^(-1).  Coefficients are the same OLS point estimates as
    ``hac_ols``; only their covariance changes.

    It is intended as a robustness estimator, not as the frozen v0.3 decision rule.
    """
    if X is None or pd.DataFrame(X).shape[1] == 0:
        X = pd.DataFrame(index=pd.Series(y).index)
    y, X = align_yx(y, X)
    if len(y) < max(30, X.shape[1] + 10):
        raise ValueError("Insufficient aligned observations for prewhitened HAC regression")
    scale = X.std(ddof=0).replace(0, 1.0)
    Xs = X.div(scale, axis=1)
    if X.shape[1] == 0:
        Xc = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
    else:
        Xc = sm.add_constant(Xs, has_constant="add")
    Adata = Xc.to_numpy(float)
    yv = y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(Adata, yv, rcond=None)
    resid_v = yv - Adata @ beta
    scores = Adata * resid_v[:, None]

    # Fit g_t = A g_{t-1} + e_t without an intercept (scores are asymptotically zero mean).
    Z, Y = scores[:-1], scores[1:]
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)  # Y(row) = Z(row) @ B ; A(column)=B.T
    Avar = B.T
    vals = np.linalg.eigvals(Avar)
    radius = float(np.max(np.abs(vals))) if len(vals) else 0.0
    if radius >= 0.98:
        Avar = Avar * (0.98 / max(radius, 1e-12))
        radius = 0.98
    innov = Y - Z @ Avar.T
    if lags == "auto":
        L = max(1, int(np.floor(4.0 * (len(innov) / 100.0) ** (2.0 / 9.0))))
    else:
        L = int(lags)
    Se = _bartlett_lrv(innov, L)
    I = np.eye(Avar.shape[0])
    C = np.linalg.pinv(I - Avar)
    Sg = C @ Se @ C.T
    Q = (Adata.T @ Adata) / len(y)
    Qi = np.linalg.pinv(Q)
    cov_scaled = Qi @ Sg @ Qi / len(y)

    names = list(Xc.columns)
    params_scaled = pd.Series(beta, index=names, name="coef")
    params = params_scaled.copy()
    unscale = pd.Series(1.0, index=names)
    for c in X.columns:
        params[c] = params_scaled[c] / scale[c]
        unscale[c] = 1.0 / scale[c]
    cov = pd.DataFrame(cov_scaled * np.outer(unscale.values, unscale.values), index=names, columns=names)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = pd.Series(params.values / se, index=names, name="t")
    pvals = pd.Series(2.0 * stats.norm.sf(np.abs(tvals.values)), index=names, name="p")
    resid = pd.Series(resid_v, index=y.index, name="resid")
    ssr = float(np.sum(resid_v ** 2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ssr / tss if tss > 0 else np.nan
    n, k = len(y), Adata.shape[1]
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1) if np.isfinite(r2) else np.nan
    # Information criteria use the OLS Gaussian likelihood; inference estimator does not alter them.
    sigma2 = max(ssr / n, 1e-300)
    llf = -0.5 * n * (np.log(2 * np.pi) + 1 + np.log(sigma2))
    aic = -2 * llf + 2 * k
    bic = -2 * llf + np.log(n) * k
    return {
        "params": params,
        "tvalues": tvals,
        "pvalues": pvals,
        "cov": cov,
        "resid": resid,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "aic": float(aic),
        "bic": float(bic),
        "n": int(n),
        "condition_number": float(np.linalg.cond(Adata)),
        "hac_lags": int(L),
        "prewhite_radius": float(radius),
    }
