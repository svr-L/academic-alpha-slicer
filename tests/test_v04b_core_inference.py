"""v0.4b: the core's standard error comes from the traded-only spanning regression."""
import numpy as np
import pandas as pd

from alpha_slicer import AlphaSlicer, SliceConfig, fixed_b_critical_value
from alpha_slicer.conditional import conditional_alpha_table
from alpha_slicer.inference import inference_fragility_summary
from alpha_slicer.montecarlo import CASES, MC_CONFIG, evaluate_one
from alpha_slicer.regression import NORMAL_CV, hac_ols, size_adjusted_p

DEV_SEEDS = range(10)   # development seeds; the published evaluation uses 1000-1099


def test_fixed_b_critical_values():
    assert abs(fixed_b_critical_value(0.0) - 1.96) < 1e-12
    assert abs(fixed_b_critical_value(0.05) - 2.109) < 0.002
    grid = [fixed_b_critical_value(b) for b in np.linspace(0, 0.3, 31)]
    assert all(np.diff(grid) > 0)
    cv = fixed_b_critical_value(0.05)
    assert abs(size_adjusted_p(cv, cv) - 0.05) < 1e-12
    assert size_adjusted_p(1.96, cv) > 0.05


def test_core_se_comes_from_the_traded_only_regression():
    fn, kw = CASES['peso_short_put']
    data = fn(3, **kw)
    slicer = AlphaSlicer(MC_CONFIG)
    r = slicer.fit(**data['inputs'])
    s = r.summary
    inp = {k: v for k, v in data['inputs'].items() if k in ('strategy_net', 'factors', 'benchmark', 'states', 'tail_factors')}
    d = slicer.prepare(**inp)
    tr = hac_ols(d.y, d.X_traded, lags=int(s['core_hac_lags_used']))
    full = hac_ols(d.y, d.X_full, lags=MC_CONFIG.hac_lags)
    # same intercept (Frisch-Waugh-Lovell), different residuals
    assert abs(tr['params']['const'] - full['params']['const']) < 1e-12
    assert abs(s['core_unexplained_alpha_ann'] - r.shapley.loc['core_unexplained', 'contribution_ann']) < 1e-10
    se_tr = np.sqrt(tr['cov'].loc['const', 'const']) * 252
    assert abs(s['core_alpha_se_ann'] - se_tr) < 1e-12
    assert abs(s['core_alpha_t'] - tr['tvalues']['const']) < 1e-9
    # the zero-priced controls had been shrinking the standard error several-fold
    assert s['core_alpha_se_ann'] > 4 * s['core_alpha_se_ann_v03']
    assert abs(s['core_t_critical'] - 2.0 * s['core_ci_critical'] / NORMAL_CV) < 1e-12


def test_rare_crash_put_interval_covers_the_population_alpha():
    rows = [evaluate_one('peso_short_put', s, pop_truth=-0.0618) for s in DEV_SEEDS]
    covered = sum(bool(r['covered_population']) for r in rows)
    covered_v03 = sum(bool(r['covered_population_v03']) for r in rows)
    assert covered >= 7 and covered_v03 <= 4
    assert all(r['core_se'] > 4 * r['core_se_v03'] for r in rows)


def test_genuine_alpha_keeps_its_t_stat():
    for case in ('regime_alpha', 'timing_skill'):
        fn, kw = CASES[case]
        s = AlphaSlicer(MC_CONFIG).fit(**fn(11, **kw)['inputs']).summary
        assert s['diagnostic_label'].startswith('core_unexplained_candidate'), case
        assert s['core_alpha_t'] >= s['core_t_critical']


def test_fragility_ignores_audit_only_rows():
    base = dict(alpha_ann=0.05, se_ann=0.02, critical_value=1.96, p=0.01, ci_low_ann=0.01, ci_high_ann=0.09)
    ci = pd.DataFrame([
        {**base, 'method': 'HAC(126)', 't': 1.5, 'decision_family': True},
        {**base, 'method': 'HAC(21)', 't': 1.7, 'decision_family': True},
        {**base, 'method': 'HAC(prewhite)', 't': 5.0, 'decision_family': False},
        {**base, 'method': 'v0.3 full design HAC(21)', 't': 9.0, 'decision_family': False},
    ]).set_index('method')
    assert not inference_fragility_summary(ci)['inference_fragile']
    ci.loc['HAC(21)', 't'] = 2.5
    assert inference_fragility_summary(ci)['inference_fragile']


def test_regime_table_uses_the_traded_design():
    fn, kw = CASES['regime_alpha']
    data = fn(11, **kw)
    slicer = AlphaSlicer(MC_CONFIG)
    r = slicer.fit(**data['inputs'])
    inp = {k: v for k, v in data['inputs'].items() if k in ('strategy_net', 'factors', 'benchmark', 'states', 'tail_factors')}
    d = slicer.prepare(**inp)
    L = int(r.summary['core_hac_lags_used'])
    manual = conditional_alpha_table(d.y, d.X_traded, d.states, L, 252, MC_CONFIG.min_regime_obs)
    assert np.allclose(r.conditional_alpha['alpha_ann'], manual['alpha_ann'], equal_nan=True)
    assert np.allclose(r.conditional_alpha['se_ann'], manual['se_ann'], equal_nan=True)
