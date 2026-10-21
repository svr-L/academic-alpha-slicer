"""Traded proxies for nonlinear / tail premia.

The slicer can only remove a premium it can price. Non-traded regressors (m^2, min(m,0), tail
dummies) enter at a zero premium, so a strategy that sells convexity keeps its premium in the
core unless a TRADED convexity factor is supplied through ``tail_factors``. Helpers here build
such factors from public data.
"""
from __future__ import annotations
import pandas as pd


def variance_swap_pnl(market_returns, implied_vol, ann=252, lag=1, vol_in_percent=None):
    """Daily P&L of a LONG variance-swap proxy: m_t^2 - IV_{t-lag}^2 / ann.

    Pass it as a traded tail factor, e.g. ``tail_factors=pd.DataFrame({'VARSWAP': pnl})``.
    Its mean is the (typically negative) variance risk premium: a short-variance strategy loads
    negatively on it, so the premium is charged instead of being labelled alpha.

    implied_vol : annualized implied volatility of the same market, e.g. VIX. Values with a
                  median above 1 are treated as vol points and divided by 100 unless
                  ``vol_in_percent`` says otherwise.
    Approximation: daily realized-vs-implied variance with a one-day-old strike; no discounting
    and no mark-to-market of the remaining tenor.
    """
    m = pd.Series(market_returns).astype(float)
    iv = pd.Series(implied_vol).astype(float).reindex(m.index)
    if vol_in_percent is None:
        vol_in_percent = bool(iv.median() > 1.0)
    if vol_in_percent:
        iv = iv / 100.0
    strike = iv.shift(int(lag)) ** 2 / float(ann)
    return (m ** 2 - strike).rename('VARSWAP_LONG')


def excess_return(index_level, rf_daily=None):
    """Daily excess return of a total-return index level (e.g. a put-writing or buy-write index)."""
    r = pd.Series(index_level).astype(float).pct_change(fill_method=None)
    if rf_daily is not None:
        r = r - pd.Series(rf_daily).astype(float).reindex(r.index)
    return r
