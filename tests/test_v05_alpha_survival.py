import numpy as np
import pandas as pd
import pytest

from alpha_slicer import AlphaSlicer, SliceConfig, make_demo_case
from alpha_slicer.survival import default_linear_spans, resolve_linear_spans


def _fit(seed=14, **cfg_kw):
    d = make_demo_case(n=2600, seed=seed)
    cfg = SliceConfig(min_obs=500, **cfg_kw)
    r = AlphaSlicer(cfg).fit(d['strategy_net'], d['factors'], d['benchmark'],
                             strategy_gross=d['strategy_gross'], tail_factors=d['tail_factors'],
                             states=d['states'])
    return d, r


def test_default_span_hierarchy_demo():
    d, r = _fit()
    stages = list(r.alpha_survival.index)
    assert stages[0] == 'RAW_NET'
    assert 'MKT' in stages
    assert 'FF3' in stages
    assert 'FF3+MOM' in stages
    assert '+CONDITIONAL' in stages
    assert stages[-1] == '+TRADED_TAIL'


def test_survival_final_alpha_equals_full_priced_core():
    _, r = _fit(seed=15)
    final = float(r.alpha_survival.iloc[-1]['alpha_ann'])
    assert abs(final - float(r.summary['core_unexplained_alpha_ann'])) < 1e-10
    assert abs(final - float(r.shapley.loc['core_unexplained', 'contribution_ann'])) < 1e-10


def test_survival_span_contains_only_priced_terms():
    _, r = _fit(seed=16)
    text = ' '.join(r.alpha_survival['priced_terms'].astype(str))
    assert 'MKT_POW2' not in text
    assert 'MKT_POW3' not in text
    assert 'MKT_DOWNSIDE' not in text
    assert 'MKT_TAIL_SQ' not in text


def test_custom_span_is_validated_and_reported():
    d = make_demo_case(n=1800, seed=17)
    cfg = SliceConfig(min_obs=500, survival_include_default_linear_spans=False, survival_custom_linear_spans=(('MKT+MOM', ('MKT','MOM')),))
    r = AlphaSlicer(cfg).fit(d['strategy_net'], d['factors'], d['benchmark'],
                             tail_factors=d['tail_factors'], states=d['states'])
    assert 'MKT+MOM' in r.alpha_survival.index

    bad = SliceConfig(min_obs=500, survival_include_default_linear_spans=False, survival_custom_linear_spans=(('BAD', ('MKT','NOT_A_FACTOR')),))
    with pytest.raises(ValueError):
        AlphaSlicer(bad).fit(d['strategy_net'], d['factors'], d['benchmark'],
                              tail_factors=d['tail_factors'], states=d['states'])


def test_survival_summary_is_descriptive_not_forced_monotone():
    _, r = _fit(seed=18)
    c = r.alpha_survival
    assert np.isfinite(c['alpha_ann']).all()
    assert 'erosion_from_prior_ann' in c.columns
    assert 'screen_pass' in c.columns
    assert r.summary['alpha_survival_stages'] == len(c)
    assert r.summary['alpha_survival_final_stage'] == c.index[-1]


def test_default_linear_spans_skip_unavailable_families():
    spans = default_linear_spans(['MKT','SMB','HML','MOM'])
    names = [x[0] for x in spans]
    assert names == ['MKT','FF3','FF3+MOM']
    spans2 = default_linear_spans(['MKT','SMB','HML','RMW','CMA','MOM'])
    names2 = [x[0] for x in spans2]
    assert names2 == ['MKT','FF3','FF5','FF5+MOM']


def test_resolve_spans_rejects_missing_columns():
    F = pd.DataFrame({'MKT':[0.0,0.1], 'SMB':[0.0,0.1]})
    with pytest.raises(ValueError):
        resolve_linear_spans(F, custom_spans=(('x', ('MKT','HML')),))
