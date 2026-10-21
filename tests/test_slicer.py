import numpy as np
import pandas as pd
import pytest

from alpha_slicer import AlphaSlicer, SliceConfig, make_demo_case, build_states
from alpha_slicer.montecarlo import case_regime_alpha, case_vrp_short_variance, case_actuarial_short_put

CFG = SliceConfig(min_obs=500)


def _fit_demo(n=1800, seed=1, **kw):
    d = make_demo_case(n=n, seed=seed)
    r = AlphaSlicer(CFG).fit(d['strategy_net'], d['factors'], d['benchmark'],
                             strategy_gross=d['strategy_gross'], tail_factors=d['tail_factors'],
                             states=d['states'], **kw)
    return d, r


def test_demo_runs_and_adds_up():
    _, r = _fit_demo(seed=1)
    raw = r.summary['raw_net_mean_ann']
    assert np.isfinite(raw)
    assert abs(raw - r.shapley['contribution_ann'].sum()) < 1e-10
    # one row per traded block (linear, conditional, tail) plus the raw mean
    assert list(r.waterfall.index) == ['raw_net_mean', 'after_linear', 'after_conditional', 'after_tail']
    assert r.summary['n_obs'] >= 500


def test_cost_drag_recovered():
    _, r = _fit_demo(seed=2)
    assert abs(r.summary['implementation_cost_drag_ann'] - 0.008) < 1e-10


def test_nonlinear_table_has_three_orders_and_constant_alpha():
    d = make_demo_case(n=1200, seed=3)
    r = AlphaSlicer(SliceConfig(min_obs=500, nonlinear_max_order=3)).fit(d['strategy_net'], d['factors'], d['benchmark'])
    t = r.nonlinear_order
    assert list(t.index) == [1, 2, 3]
    assert t['selected_bic'].sum() == 1
    # market powers carry a zero price: the order changes R2, never alpha
    assert np.allclose(t['alpha_ann'], t['alpha_ann'].iloc[0], atol=1e-10)


def test_known_truth_core_is_recovered_within_its_standard_error():
    d, r = _fit_demo(n=3500, seed=14)
    core, t = r.summary['core_unexplained_alpha_ann'], r.summary['core_alpha_t']
    se = abs(core / t)
    assert abs(core - d['truth']['core_alpha_net_ann']) < 2.5 * se


def test_nontraded_blocks_cannot_move_alpha():
    _, r = _fit_demo(seed=4)
    nm = r.nested_models
    traded_model = 'linear+conditional+tail'
    assert traded_model in nm.index
    after = nm.index[list(nm.index).index(traded_model):]
    assert np.allclose(nm.loc[after, 'alpha_ann'], nm.loc[traded_model, 'alpha_ann'], atol=1e-10)
    assert (r.shapley.loc[['nonlinear', 'tail_shape'], 'contribution_ann'] == 0).all()
    # orthogonalized controls: zero mean and zero covariance with every traded regressor
    Xt = pd.concat([r.feature_blocks[b] for b in ('linear', 'conditional', 'tail')], axis=1)
    for b in ('nonlinear', 'tail_shape'):
        Z = r.feature_blocks[b]
        assert np.allclose(Z.mean(), 0, atol=1e-12)
        assert np.abs(Xt.T.values @ Z.values).max() < 1e-8


def test_core_returns_mean_equals_core():
    _, r = _fit_demo(seed=5)
    assert abs(r.core_returns.mean() * 252 - r.summary['core_unexplained_alpha_ann']) < 1e-10


def test_supplied_states_are_lagged_one_day():
    d, r = _fit_demo(seed=6)
    expected = d['states']['HIGH_VOL'].shift(1).reindex(r.states.index)
    assert np.allclose(r.states['HIGH_VOL'].values, expected.values)


def test_built_states_are_lagged_one_day():
    d = make_demo_case(n=1500, seed=7)
    r = AlphaSlicer(CFG).fit(d['strategy_net'], d['factors'], d['benchmark'])
    built = build_states(d['benchmark']).shift(1).reindex(r.states.index)
    assert np.allclose(r.states.values, built.values)
    assert r.states.notna().all().all()


def test_state_dummies_are_not_in_the_premium_block():
    _, r = _fit_demo(seed=8)
    for X in r.feature_blocks.values():
        assert not any(c.startswith('STATE_') for c in X.columns)
    assert all('_X_' in c for c in r.feature_blocks['conditional'].columns)


def test_regime_alpha_table_is_not_degenerate():
    data = case_regime_alpha(11)
    r = AlphaSlicer(CFG).fit(**data['inputs'])
    ca = r.conditional_alpha
    on, off = ca.loc[('HIGH_VOL', 'on'), 'alpha_ann'], ca.loc[('HIGH_VOL', 'off'), 'alpha_ann']
    assert on - off > 0.02
    assert abs(ca.loc[('HIGH_VOL', 'on_minus_off'), 'alpha_ann'] - (on - off)) < 1e-12
    share = ca.loc[('HIGH_VOL', 'on'), 'share_of_days']
    assert abs(ca.loc[('HIGH_VOL', 'time_weighted'), 'alpha_ann'] - (share * on + (1 - share) * off)) < 1e-12


def test_traded_variance_swap_spans_the_quadratic_basis():
    data = case_vrp_short_variance(3, traded=True)
    r = AlphaSlicer(CFG).fit(**data['inputs'])
    assert 'MKT_POW2' in r.summary['dropped_columns']
    assert np.isnan(r.tail_flags.loc['short_convexity_tm', 'statistic'])
    assert r.summary['diagnostic_label'] == 'tail_dominated'


def test_short_put_is_flagged_and_not_certified():
    data = case_actuarial_short_put(5)
    r = AlphaSlicer(CFG).fit(**data['inputs'])
    assert r.summary['unpriced_tail_risk_flag']
    assert r.summary['diagnostic_label'] != 'core_unexplained_candidate'
    # the v0.1 convention charged raw means of non-traded regressors: a large artificial alpha
    assert r.summary['core_if_nontraded_charged_at_raw_mean_ann'] > 0.5


def test_nontraded_premia_what_if():
    _, r = _fit_demo(seed=9)
    term = r.nontraded.xs('tail_shape', level='block').index[0]
    coef = r.nontraded.loc[('tail_shape', term), 'coef']
    _, r2 = _fit_demo(seed=9, nontraded_premia={term: 0.05})
    expected = r.summary['core_unexplained_alpha_ann'] - coef * 0.05
    assert abs(r2.summary['core_after_assumed_nontraded_premia_ann'] - expected) < 1e-12
    with pytest.raises(ValueError):
        _fit_demo(seed=9, nontraded_premia={'NOT_A_TERM': 0.01})
