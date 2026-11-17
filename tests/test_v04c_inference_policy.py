import numpy as np
import pandas as pd
import pytest
from alpha_slicer.luck import deflated_sharpe

from alpha_slicer import AlphaSlicer, SliceConfig, make_demo_case, policy_presets, resolve_policy, evaluate_inference_profiles


def _cfg(profile='balanced', reps=120, precommitted=True):
    return SliceConfig(
        annualization=252, market_col='MKT', min_obs=500,
        block_order=('linear','conditional','tail','nonlinear','tail_shape'),
        inference_profile=profile, inference_profile_precommitted=precommitted,
        inference_bootstrap_reps=reps, inference_bootstrap_mean_block=63,
        inference_bootstrap_seed=12345,
    )


def test_policy_order_and_definitions():
    p = policy_presets(126)
    assert p['discovery'].hac_lags == 63 and not p['discovery'].fixed_b
    assert p['balanced'].hac_lags == 126 and p['balanced'].fixed_b
    assert p['certification'].require_bootstrap_agreement
    assert p['discovery'].conservativeness < p['balanced'].conservativeness < p['certification'].conservativeness
    c = resolve_policy('custom', 126, custom_lags=42, custom_fixed_b=False, custom_require_bootstrap=True)
    assert c.hac_lags == 42 and not c.fixed_b and c.require_bootstrap_agreement


def test_all_profiles_and_frontier_are_always_reported():
    d = make_demo_case(n=1800, seed=19)
    r = AlphaSlicer(_cfg('discovery', reps=80)).fit(
        d['strategy_net'], d['factors'], benchmark=d['benchmark'], strategy_gross=d['strategy_gross'],
        costs=d['costs'], states=d['states'], tail_factors=d['tail_factors'])
    assert {'discovery','balanced','certification'} <= set(r.profile_comparison.index)
    assert int(r.profile_comparison['primary'].sum()) == 1
    assert bool(r.profile_comparison.loc['discovery','primary'])
    assert len(r.inference_frontier) >= 7
    assert 0.0 <= float(r.summary['inference_frontier_robustness_score']) <= 1.0


def test_selected_profile_drives_candidate_and_audit_fields():
    d = make_demo_case(n=2200, seed=11)
    r = AlphaSlicer(_cfg('balanced', reps=100)).fit(
        d['strategy_net'], d['factors'], benchmark=d['benchmark'], strategy_gross=d['strategy_gross'],
        costs=d['costs'], states=d['states'], tail_factors=d['tail_factors'])
    assert r.summary['inference_profile'] == 'balanced'
    assert bool(r.summary['inference_profile_precommitted'])
    assert bool(r.summary['primary_profile_candidate_pass']) == bool(r.profile_comparison.loc['balanced','candidate_pass'])
    assert abs(float(r.summary['core_alpha_t']) - float(r.profile_comparison.loc['balanced','t'])) < 1e-10
    assert abs(float(r.summary['core_alpha_t_v04b_balanced']) - float(r.profile_comparison.loc['balanced','t'])) < 1e-10


def test_certification_requires_bootstrap_configured():
    d = make_demo_case(n=1000, seed=5)
    with pytest.raises(ValueError, match='bootstrap-required'):
        AlphaSlicer(_cfg('certification', reps=0)).fit(
            d['strategy_net'], d['factors'], benchmark=d['benchmark'], states=d['states'], tail_factors=d['tail_factors'])


def test_profile_comparison_can_reveal_power_robustness_tradeoff():
    # Same point estimate under every profile; only uncertainty standard changes.
    d = make_demo_case(n=1800, seed=31)
    r = AlphaSlicer(_cfg('balanced', reps=80)).fit(
        d['strategy_net'], d['factors'], benchmark=d['benchmark'], states=d['states'], tail_factors=d['tail_factors'])
    a = r.profile_comparison['alpha_ann'].to_numpy(float)
    assert np.max(np.abs(a-a[0])) < 1e-10
    assert r.profile_comparison.loc['certification','bootstrap_required']
    # Certification can never pass when balanced fails its HAC hurdle.
    if not bool(r.profile_comparison.loc['balanced','hac_pass']):
        assert not bool(r.profile_comparison.loc['certification','candidate_pass'])


def test_deterministic_power_boundary_discovery_passes_balanced_fails():
    rng = np.random.default_rng(3)
    n, rho = 2500, 0.5
    e = np.zeros(n)
    z = rng.normal(0, 0.008, n)
    for t in range(1, n):
        e[t] = rho * e[t-1] + z[t]
    idx = pd.bdate_range('2010-01-01', periods=n)
    y = pd.Series(e + 0.045/252, index=idx)
    luck = deflated_sharpe(y, 1)
    luck['dsr_min'] = 0.95
    tab = evaluate_inference_profiles(y, pd.DataFrame(index=idx), 252, 2.0, 126, None, luck, 'balanced')
    assert bool(tab.loc['discovery','candidate_pass'])
    assert not bool(tab.loc['balanced','candidate_pass'])
    assert tab.loc['discovery','t'] >= tab.loc['discovery','t_threshold']
    assert tab.loc['balanced','t'] < tab.loc['balanced','t_threshold']
