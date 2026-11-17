"""Frozen tests for the user's original academic-alpha question (v0.5.1)."""
import numpy as np
import pandas as pd
import pytest
from alpha_slicer import AlphaSlicer, SliceConfig, make_demo_case
from alpha_slicer.academic import group_shapley
from alpha_slicer.regression import hac_ols


def demo_result(ref=('MKT',), bootstrap_reps=0):
    d = make_demo_case(n=1250, seed=14)
    cfg = SliceConfig(min_obs=500, academic_reference_factors=ref,
                      academic_bootstrap_reps=bootstrap_reps)
    r = AlphaSlicer(cfg).fit(d['strategy_net'], d['factors'], d['benchmark'],
                             states=d['states'], tail_factors=d['tail_factors'])
    return d, r


def test_reference_alpha_is_original_intercept_not_net_mean():
    d, r = demo_result()
    a = r.academic_alpha
    y = d['strategy_net'].reindex(r.core_returns.index)
    X = d['factors'][['MKT']].reindex(r.core_returns.index)
    ref = hac_ols(y, X, lags=21)['params']['const'] * 252
    assert np.isclose(a.summary.academic_reference_alpha_ann, ref, atol=1e-10)
    assert abs(ref - r.summary.raw_net_mean_ann) > .001
    assert np.isclose(a.shapley.contribution_ann.sum(), ref, atol=1e-10)
    assert np.isclose(a.structural.contribution_ann.sum(), ref, atol=1e-10)
    assert np.isclose(a.summary.academic_expanded_alpha_ann, r.summary.core_unexplained_alpha_ann, atol=1e-10)
    assert np.isclose(a.bridge.iloc[0].alpha_ann, ref)


def test_ff3_is_an_explicit_valid_reference():
    d, r = demo_result(('MKT', 'SMB', 'HML'))
    a = r.academic_alpha
    assert set(a.shapley.index) == {'additional_linear', 'conditional', 'tail',
                                     'unexplained_after_expansion'}
    y = d['strategy_net'].reindex(r.core_returns.index)
    x = d['factors'].reindex(y.index)[['MKT','SMB','HML']]
    assert np.isclose(a.summary.academic_reference_alpha_ann,
                      hac_ols(y,x,lags=21)['params']['const']*252,atol=1e-10)


def test_beta_reallocation_is_not_silently_lost():
    rng=np.random.default_rng(12)
    n=1000
    x=rng.normal(.00035,.01,n)
    z=.75*x+rng.normal(.0009,.003,n)
    y=.0003+.3*x+1.4*z+rng.normal(0,.004,n)
    a0,a1,phi,_=group_shapley(y,x.reshape(-1,1),{'extra':z.reshape(-1,1)},252)
    mu=x.mean()
    b0=np.linalg.lstsq(np.column_stack((np.ones(n),x)),y,rcond=None)[0][1]
    b1=np.linalg.lstsq(np.column_stack((np.ones(n),x,z)),y,rcond=None)[0]
    beta_shift=(b1[1]-b0)*mu*252
    explanation=b1[2]*z.mean()*252
    assert abs(beta_shift) > .0001
    assert np.isclose(a0,a1+phi['extra'],atol=1e-12)
    assert np.isclose(a0,a1+beta_shift+explanation,atol=1e-12)
    assert not np.isclose(a0,a1+explanation,atol=1e-6)


def test_nontraded_not_mislabeled_zero_economic_premium():
    _,r=demo_result()
    t=r.academic_alpha.diagnostics
    assert len(t)>0
    assert t.estimated_premium_ann.isna().all()
    assert all('not a zero economic premium' in s for s in t.economic_status)
    assert all(b not in r.academic_alpha.shapley.index for b in ('nonlinear', 'tail_shape'))


def test_stationary_bootstrap_respects_reference_identity():
    _,r=demo_result(bootstrap_reps=25)
    tab=r.academic_alpha.bootstrap_inference
    assert 'academic_reference_alpha_ann' in tab.index
    assert 'conditional' in tab.index
    assert (tab.ci_high_ann >= tab.ci_low_ann).all()


def test_invalid_reference_fails_without_silent_fallback():
    d=make_demo_case(n=900,seed=15)
    with pytest.raises(ValueError,match='reference'):
        AlphaSlicer(SliceConfig(min_obs=500,academic_reference_factors=('DOES_NOT_EXIST',))).fit(
            d['strategy_net'],d['factors'],d['benchmark'],states=d['states'],tail_factors=d['tail_factors'])
