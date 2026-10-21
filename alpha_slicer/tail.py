from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .regression import hac_ols, orthogonalize


def max_drawdown(r):
    r = pd.Series(r).dropna().astype(float)
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min()) if len(w) else np.nan


def expected_shortfall(r, q=0.025):
    r = pd.Series(r).dropna().astype(float)
    if not len(r): return np.nan
    cut = r.quantile(q)
    return float(r[r <= cut].mean())


def drawdown_series(r):
    r = pd.Series(r).fillna(0).astype(float)
    w = (1 + r).cumprod()
    return w / w.cummax() - 1


def cdar_style_drawdown_beta(p, m, q=0.2):
    p = pd.Series(p).dropna(); m = pd.Series(m).reindex(p.index)
    d = pd.concat([drawdown_series(p-p.mean()).rename('p'),
                   drawdown_series(m-m.mean()).rename('m')], axis=1).dropna()
    if len(d) < 50: return np.nan
    s = d[d['m'] <= d['m'].quantile(q)]
    den = s['m'].mean()
    return float(s['p'].mean()/den) if len(s) >= 10 and den != 0 else np.nan


def coskewness(p, m):
    d = pd.concat([pd.Series(p).rename('p'), pd.Series(m).rename('m')], axis=1).dropna()
    if len(d)<50 or d.p.std()==0 or d.m.std()==0: return np.nan
    pc=d.p-d.p.mean(); mc=d.m-d.m.mean()
    return float((pc*mc**2).mean()/(d.p.std()*d.m.var()))


def quantile_beta(y, x, tau=0.10):
    d=pd.concat([pd.Series(y).rename('y'),pd.Series(x).rename('x')],axis=1).dropna()
    if len(d)<100 or d.x.std()==0: return np.nan
    X=sm.add_constant(d[['x']],has_constant='add')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit=sm.QuantReg(d.y,X).fit(q=tau, max_iter=4000)
        return float(fit.params['x'])
    except Exception:
        return np.nan


def tail_diagnostics(strategy, benchmark, q=0.10):
    d=pd.concat([pd.Series(strategy).rename('p'),pd.Series(benchmark).rename('m')],axis=1).dropna()
    if len(d)<50:
        return pd.Series(dtype=float)
    X=sm.add_constant(d[['m']],has_constant='add')
    beta=float(sm.OLS(d.p,X).fit().params['m'])
    threshold=d.m.quantile(q)
    ds=d[d.m<=threshold]
    downside_beta=float(sm.OLS(ds.p,sm.add_constant(ds[['m']],has_constant='add')).fit().params['m']) if len(ds)>=20 else np.nan
    return pd.Series({
        'market_beta': beta,
        'quantile_beta_q10': quantile_beta(d.p,d.m,tau=q),
        'downside_beta_q10': downside_beta,
        'downside_minus_plain': downside_beta-beta if np.isfinite(downside_beta) else np.nan,
        'skew': float(d.p.skew()),
        'coskewness': coskewness(d.p,d.m),
        'cdar_drawdown_beta': cdar_style_drawdown_beta(d.p,d.m),
        'expected_shortfall_2p5': expected_shortfall(d.p,0.025),
        'max_drawdown': max_drawdown(d.p),
    })


def left_right_es_ratio(r, q=0.025):
    """|ES of the lowest q| / ES of the highest q, on the de-meaned series (1 = symmetric)."""
    c = pd.Series(r).dropna().astype(float)
    c = c - c.mean()
    if len(c) < 200:
        return np.nan
    lo, hi = c.quantile(q), c.quantile(1 - q)
    es_left, es_right = c[c <= lo].mean(), c[c >= hi].mean()
    return float(-es_left / es_right) if es_right > 0 else np.nan


def tail_risk_flags(y, X_traded, market, core_returns, lags=21, tail_quantile=0.10,
                    t_threshold=2.0, es_q=0.025, es_ratio=1.5):
    """Unpriced tail-risk flags that hand a candidate core over to the referee's H2 gate.

    All three are measured AFTER the traded blocks, so a supplied traded tail factor that
    prices the risk switches them off:

    short_convexity_tm    Treynor-Mazuy: y on traded factors + (m^2 - mean). Fires when the
                          quadratic loading is significantly negative (short volatility).
    excess_left_tail_beta y on traded factors + (m*1{m<=q} - mean). Fires when the extra beta in
                          the market's left tail is significantly positive (short-put-like).
    left_tail_asymmetry   |ES_left| / ES_right of the core return (strategy net of traded
                          charges). Fires above ``es_ratio``: crash risk that need not be
                          linked to the market at all.
    """
    y = pd.Series(y).astype(float)
    m = pd.Series(market).reindex(y.index).astype(float)
    Xt = pd.DataFrame(X_traded).reindex(y.index) if X_traded is not None else pd.DataFrame(index=y.index)
    if Xt.shape[1] == 0 or not any(np.allclose(Xt[c].values, m.values) for c in Xt.columns):
        # make sure the auxiliary regressions are conditional on a linear market exposure
        Xt = pd.concat([Xt, m.rename('MKT_LINEAR_CONTROL')], axis=1)
    rows = []

    def _aux(x, name):
        # Orthogonalize to the traded regressors: the t-stat is unchanged (Frisch-Waugh-Lovell),
        # and a regressor fully spanned by a traded factor (e.g. m^2 when a traded variance swap
        # is supplied) is recognized as priced instead of producing a collinear regression.
        xo = orthogonalize(x.to_frame(name), Xt)[name]
        if not xo.std(ddof=0) > 1e-8 * max(x.std(ddof=0), 1e-300):
            return np.nan, np.nan
        fit = hac_ols(y, pd.concat([Xt, xo], axis=1), lags=lags)
        return float(fit['tvalues'][name]), float(fit['params'][name])

    t_tm, c_tm = _aux(m ** 2, 'AUX_MKT_SQ')
    rows.append({'flag': 'short_convexity_tm', 'statistic': t_tm, 'threshold': -t_threshold,
                 'rule': 't <= threshold', 'coef': c_tm,
                 'fired': bool(np.isfinite(t_tm) and t_tm <= -t_threshold)})
    q = float(m.quantile(tail_quantile))
    t_tb, c_tb = _aux(m * (m <= q).astype(float), 'AUX_MKT_LEFT_TAIL')
    rows.append({'flag': 'excess_left_tail_beta', 'statistic': t_tb, 'threshold': t_threshold,
                 'rule': 't >= threshold', 'coef': c_tb,
                 'fired': bool(np.isfinite(t_tb) and t_tb >= t_threshold)})
    ratio = left_right_es_ratio(core_returns, es_q)
    rows.append({'flag': 'left_tail_asymmetry', 'statistic': ratio, 'threshold': es_ratio,
                 'rule': 'ratio >= threshold', 'coef': np.nan,
                 'fired': bool(np.isfinite(ratio) and ratio >= es_ratio)})
    return pd.DataFrame(rows).set_index('flag')
