"""Academic-alpha attribution relative to a FIXED, ex-ante reference factor span.

Explaining the reference intercept does not make its components cease to be alpha
relative to that reference, and does not establish either skill or lack of skill.
"""
from dataclasses import dataclass
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd
from .regression import hac_ols, size_adjusted_p
from .inference import stationary_bootstrap_indices

@dataclass
class AcademicAlphaResult:
    shapley: pd.DataFrame
    structural: pd.DataFrame
    bridge: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: pd.Series
    bootstrap_inference: pd.DataFrame

def ols_alpha(y, X):
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    if X.size == 0:
        return float(y.mean())
    return float(np.linalg.lstsq(np.column_stack((np.ones(len(y)), X)), y, rcond=None)[0][0])

def group_shapley(y, baseline, additions, annualization):
    """Exact Shapley of alpha(reference)-alpha(reference+new blocks)."""
    n = len(y)
    keys = list(additions)
    cache = {}
    def alpha(S):
        key = tuple(sorted(S))
        if key not in cache:
            arr = [baseline] + [additions[k] for k in key]
            X = np.column_stack([a for a in arr if a.size]) if any(a.size for a in arr) else np.empty((n, 0))
            cache[key] = ols_alpha(y, X) * annualization
        return cache[key]
    a0 = alpha(())
    out, extreme = {}, {}
    k = len(keys)
    for b in keys:
        others = [v for v in keys if v != b]
        contrib, vals = 0.0, []
        for size in range(k):
            for S in combinations(others, size):
                marginal = alpha(S) - alpha((*S, b))
                contrib += factorial(size) * factorial(k-size-1) / factorial(k) * marginal
                vals.append(marginal)
        out[b], extreme[b] = contrib, (min(vals), max(vals))
    a1 = alpha(keys)
    assert np.isclose(a0, a1 + sum(out.values()), atol=1e-9)
    return a0, a1, out, extreme

def academic_alpha_decomposition(y, factors, blocks, traded, controls, reference_factors,
                                annualization=252, hac_lags=126, critical_value=1.96,
                                bootstrap_reps=0, bootstrap_mean_block=63.,
                                bootstrap_seed=1729, bootstrap_ci=.95):
    y = pd.Series(y).astype(float)
    F = pd.DataFrame(factors).reindex(y.index).astype(float)
    ref = tuple(reference_factors)
    if len(set(ref)) != len(ref):
        raise ValueError('Duplicate academic reference factors')
    if any(c not in F.columns for c in ref):
        raise ValueError(f'Academic reference {ref} must be included in factors')
    if ref and ('linear' not in traded or any(c not in blocks['linear'] for c in ref)):
        raise ValueError('Reference factor(s) must also belong to the traded linear span')
    base = F.loc[:, list(ref)]
    extras = {}
    if 'linear' in traded:
        other = blocks['linear'].drop(columns=list(ref), errors='ignore')
        if other.shape[1]: extras['additional_linear'] = other
    for name in ('conditional','tail'):
        if name in traded and name in blocks and blocks[name].shape[1]:
            extras[name] = blocks[name]
    if any(b not in ('linear','conditional','tail') for b in traded):
        raise ValueError('Unknown traded block in academic decomposition')
    Xall = pd.concat([base, *extras.values()], axis=1)
    if Xall.columns.duplicated().any():
        raise ValueError('Duplicate factor term names')
    yf, bf = y.to_numpy(), base.to_numpy()
    xf = {k: v.to_numpy() for k,v in extras.items()}
    a0,a1,phi,ranges = group_shapley(yf,bf,xf,annualization)
    fit0 = hac_ols(y,base,lags=hac_lags)
    fit1 = hac_ols(y,Xall,lags=hac_lags)
    assert np.isclose(a0,fit0['params']['const']*annualization,atol=1e-9)
    assert np.isclose(a1,fit1['params']['const']*annualization,atol=1e-9)
    srows = [{'component':b,'contribution_ann':v,
              'min_order_contribution_ann':ranges[b][0],
              'max_order_contribution_ann':ranges[b][1],
              'interpretation':'part of original academic alpha, explained by expanded traded span'}
             for b,v in phi.items()]
    srows += [{'component':'unexplained_after_expansion','contribution_ann':a1,
               'min_order_contribution_ann':a1,'max_order_contribution_ann':a1,
               'interpretation':'remaining intercept; neither automatically genuine alpha nor skill'}]
    shapley = pd.DataFrame(srows).set_index('component')
    shapley['share_of_reference_alpha'] = shapley.contribution_ann/a0 if abs(a0)>1e-12 else np.nan
    beta0 = fit0['params'].drop('const')
    beta1 = fit1['params'].drop('const')
    correction = float(((beta1.reindex(base.columns)-beta0)*base.mean()).sum()*annualization)
    rows = [{'component':'reference_beta_reallocation','contribution_ann':correction,
             'interpretation':'change in original benchmark betas, not a separate premium'}]
    for b,X in extras.items():
        v = float((beta1.reindex(X.columns)*X.mean()).sum()*annualization)
        rows.append({'component':b,'contribution_ann':v,
                     'interpretation':'full-model slope x mean of newly added traded payoff'})
    rows.append({'component':'unexplained_after_expansion','contribution_ann':a1,
                 'interpretation':'intercept of expanded traded model'})
    structural = pd.DataFrame(rows).set_index('component')
    assert np.isclose(structural.contribution_ann.sum(),a0,atol=1e-9)
    prev=a0
    history=[{'stage':'academic_reference','alpha_ann':a0,'explained_from_previous_ann':np.nan}]
    accum=[]
    for b in extras:
        accum.append(b)
        X=pd.concat([base,*[extras[z] for z in accum]],axis=1)
        now=ols_alpha(yf,X.to_numpy())*annualization
        history.append({'stage':'after_'+b,'alpha_ann':now,'explained_from_previous_ann':prev-now})
        prev=now
    bridge=pd.DataFrame(history).set_index('stage')
    diagnostics=pd.DataFrame([{'block':b,'term':t,'estimated_premium_ann':np.nan,
                               'economic_status':'unidentified; statistical exposure is not a zero economic premium'}
                              for b in controls for t in blocks[b].columns])
    if len(diagnostics): diagnostics=diagnostics.set_index(['block','term'])
    se=float(np.sqrt(max(0.,fit0['cov'].loc['const','const']))*annualization)
    tt=float(fit0['tvalues']['const'])
    summary=pd.Series({'academic_reference_factors':','.join(ref) if ref else '(none)',
                       'academic_reference_alpha_ann':a0,
                       'academic_reference_alpha_t':tt,
                       'academic_reference_alpha_se_ann':se,
                       'academic_reference_alpha_p_approx':size_adjusted_p(tt,critical_value),
                       'academic_reference_alpha_ci_low_ann':a0-critical_value*se,
                       'academic_reference_alpha_ci_high_ann':a0+critical_value*se,
                       'academic_expanded_alpha_ann':a1,
                       'academic_attributed_mechanisms_ann':a0-a1,
                       'academic_additivity_error_ann':float(shapley.contribution_ann.sum()-a0),
                       'skill_inference':'none: explaining alpha does not reject skill',
                       'nontraded_premia':'unidentified unless supplied as independently priced factors'})
    bootstrap=pd.DataFrame()
    if int(bootstrap_reps)>0:
        if not 0 < bootstrap_ci < 1: raise ValueError('bootstrap_ci must be between 0 and 1')
        rng=np.random.default_rng(bootstrap_seed)
        keys=['academic_reference_alpha_ann',*extras,'unexplained_after_expansion']
        vals=[]
        for _ in range(int(bootstrap_reps)):
            ix=stationary_bootstrap_indices(len(yf),mean_block=bootstrap_mean_block,rng=rng)
            ab,az,ps,_=group_shapley(yf[ix],bf[ix],{k:v[ix] for k,v in xf.items()},annualization)
            vals.append([ab,*[ps[k] for k in extras],az])
        draws=pd.DataFrame(vals,columns=keys)
        qlo,qhi=(1-bootstrap_ci)/2,1-(1-bootstrap_ci)/2
        pts={'academic_reference_alpha_ann':a0,**phi,'unexplained_after_expansion':a1}
        bootstrap=pd.DataFrame([{'component':k,'point_ann':pts[k],
                                  'ci_low_ann':draws[k].quantile(qlo),
                                  'ci_high_ann':draws[k].quantile(qhi),
                                  'bootstrap_p_positive':(draws[k]>0).mean(),
                                  'bootstrap_reps':int(bootstrap_reps)} for k in keys]).set_index('component')
    return AcademicAlphaResult(shapley,structural,bridge,diagnostics,summary,bootstrap)
