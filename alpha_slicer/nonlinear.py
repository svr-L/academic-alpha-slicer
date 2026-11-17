from __future__ import annotations
import numpy as np
import pandas as pd
from .regression import hac_ols, orthogonalize


def nonlinear_order_table(strategy, factors, benchmark, market_col='MKT', max_order=3,
                          lags=21, ann=252):
    """Compare linear/quadratic/cubic market dependence with HAC inference.

    Market powers are non-traded, so they enter ORTHOGONALIZED to the factors (zero price):
    the order of the dependence changes R2/BIC and the power t-stats, but ``alpha_ann`` is the
    same at every order by construction. The raw mean of m^2 or m^3 is never charged as if it
    were a price. Selection is descriptive (lowest BIC); the table intentionally exposes all
    candidates rather than hiding specification search behind a single winner.
    """
    y=pd.Series(strategy).astype(float)
    F=pd.DataFrame(factors).astype(float).copy()
    m=F[market_col] if market_col in F.columns else pd.Series(benchmark).reindex(F.index)
    powers=pd.DataFrame({f'{market_col}_POW{p}':m**p for p in range(2,int(max_order)+1)},index=F.index)
    powers=orthogonalize(powers,F) if powers.shape[1] else powers
    rows=[]
    for order in range(1,int(max_order)+1):
        X=F.copy()
        for p in range(2,order+1):
            X[f'{market_col}_POW{p}']=powers[f'{market_col}_POW{p}']
        fit=hac_ols(y,X,lags=lags)
        rows.append({
            'order':order,
            'alpha_ann':float(fit['params']['const']*ann),
            'alpha_t':float(fit['tvalues']['const']),
            'r2':fit['r2'],'adj_r2':fit['adj_r2'],'bic':fit['bic'],
            'highest_power_t': float(fit['tvalues'].get(f'{market_col}_POW{order}',np.nan)) if order>1 else np.nan,
            'condition_number':fit['condition_number'],
        })
    out=pd.DataFrame(rows).set_index('order')
    out['selected_bic']=False
    if len(out): out.loc[out['bic'].idxmin(),'selected_bic']=True
    return out
