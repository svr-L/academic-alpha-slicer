import numpy as np
import pandas as pd

from alpha_slicer import AlphaSlicer, SliceConfig, make_demo_case
from alpha_slicer.inference import stationary_bootstrap_indices, stationary_bootstrap_decomposition
from alpha_slicer.regression import prewhitened_hac_ols


def test_stationary_bootstrap_preserves_local_blocks():
    rng = np.random.default_rng(123)
    n, L = 5000, 50
    idx = stationary_bootstrap_indices(n, mean_block=L, rng=rng)
    continuation = ((idx[1:] - idx[:-1]) % n == 1).mean()
    # Expected continuation probability is 1 - 1/L = 0.98.
    assert 0.965 < continuation < 0.992


def test_prewhitened_hac_matches_ar1_long_run_scale():
    rng = np.random.default_rng(4)
    n, rho, sigma = 12000, 0.85, 0.01
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    y = pd.Series(x)
    fit = prewhitened_hac_ols(y, pd.DataFrame(index=y.index), lags='auto')
    se = np.sqrt(fit['cov'].loc['const', 'const'])
    # For stationary AR(1), LRV = sigma_e^2 / (1-rho)^2.
    truth = sigma / (1 - rho) / np.sqrt(n)
    assert abs(se / truth - 1.0) < 0.20
    assert fit['prewhite_radius'] > 0.75


def test_bootstrap_decomposition_adds_up_in_every_draw():
    d = make_demo_case(n=1400, seed=9)
    cfg = SliceConfig(min_obs=500)
    base = AlphaSlicer(cfg).fit(d['strategy_net'], d['factors'], d['benchmark'],
                                strategy_gross=d['strategy_gross'], states=d['states'],
                                tail_factors=d['tail_factors'])
    inf, draws = stationary_bootstrap_decomposition(
        y=base.core_returns * 0 + d['strategy_net'].reindex(base.core_returns.index),
        blocks=base.feature_blocks,
        traded=['linear', 'conditional', 'tail'],
        annualization=252, reps=80, mean_block=42, seed=11,
        strategy_gross=d['strategy_gross'].reindex(base.core_returns.index),
        point_shapley=base.shapley,
        raw_point=base.summary['raw_net_mean_ann'],
    )
    lhs = draws['premium_linear'] + draws['premium_conditional'] + draws['premium_tail'] + draws['core_unexplained']
    assert np.max(np.abs(lhs - draws['raw_net_mean'])) < 1e-10
    assert 'core_unexplained' in inf.index
    assert inf.loc['core_unexplained', 'n_boot'] == 80


def test_fit_exposes_v04_inference_tables_and_known_truth_ci():
    d = make_demo_case(n=2200, seed=14)
    cfg = SliceConfig(min_obs=500, inference_bootstrap_reps=100,
                      inference_bootstrap_mean_block=63, inference_bootstrap_seed=99)
    r = AlphaSlicer(cfg).fit(d['strategy_net'], d['factors'], d['benchmark'],
                             strategy_gross=d['strategy_gross'], states=d['states'],
                             tail_factors=d['tail_factors'])
    assert {'HAC(21)', 'HAC(63)', 'HAC(126)', 'HAC(prewhite)', 'Stationary bootstrap'} <= set(r.core_inference.index)
    assert 'core_unexplained' in r.bootstrap_inference.index
    lo = r.bootstrap_inference.loc['core_unexplained', 'ci_low_ann']
    hi = r.bootstrap_inference.loc['core_unexplained', 'ci_high_ann']
    truth = float(d['truth']['core_alpha_net_ann'])
    assert lo < truth < hi
    assert abs(r.bootstrap_inference.loc['raw_net_mean', 'estimate_ann'] - r.summary['raw_net_mean_ann']) < 1e-12
