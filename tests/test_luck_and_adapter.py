import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha_slicer import AlphaSlicer, SliceConfig, spec_sweep, make_demo_case
from alpha_slicer.luck import psr, deflated_sharpe, expected_max_sr, spa_test, effective_trials, multi_core_gate
from alpha_slicer.montecarlo import case_regime_alpha

CFG = SliceConfig(min_obs=500)


def test_psr_and_dsr_basics():
    p1, _ = psr(0.05, 1000)
    p2, _ = psr(0.10, 1000)
    assert 0.5 < p1 < p2 < 1
    x = np.random.default_rng(0).normal(0.0005, 0.01, 2000)
    d1 = deflated_sharpe(x, n_trials=1)
    assert abs(d1['dsr'] - d1['psr']) < 1e-12          # one trial: DSR is the PSR against zero
    d100 = deflated_sharpe(x, n_trials=100)
    assert d100['dsr'] < d1['dsr'] and d100['sr_star'] > 0
    assert abs(expected_max_sr(10, 1.0) - 1.575) < 0.01


def test_effective_trials_bounds():
    rng = np.random.default_rng(1)
    assert abs(effective_trials(pd.DataFrame(rng.normal(size=(2000, 6)))) - 6) < 0.5
    common = rng.normal(size=(2000, 1))
    assert effective_trials(pd.DataFrame(common + 0.05 * rng.normal(size=(2000, 6)))) < 2.5


def test_spa_size_and_power():
    rng = np.random.default_rng(2)
    rej = 0
    for r in range(40):
        X = pd.DataFrame(rng.normal(0, 0.003, (1000, 5)))
        rej += spa_test(X, B=300, mean_block=10, seed=r)['p_consistent'] < 0.05
    assert rej <= 5
    X = pd.DataFrame(rng.normal(0, 0.003, (1000, 5)))
    X[2] += 0.003 * 5 / np.sqrt(1000)
    out = spa_test(X, B=500, seed=3)
    assert out['p_consistent'] < 0.05 and out['best'] == 2


def test_fit_applies_the_luck_gate():
    data = case_regime_alpha(11)
    r1 = AlphaSlicer(CFG).fit(**data['inputs'])
    s1 = r1.summary
    for k in ('core_sharpe_ann', 'core_psr', 'core_dsr', 'luck_gate_pass', 'luck_sr_hurdle_ann'):
        assert k in s1.index
    assert s1['luck_gate_pass'] and s1['diagnostic_label'].startswith('core_unexplained_candidate')
    # the same core after 10,000 declared tries is no longer a candidate
    r2 = AlphaSlicer(replace(CFG, luck_n_trials=10_000)).fit(**data['inputs'])
    assert not r2.summary['luck_gate_pass']
    assert not r2.summary['diagnostic_label'].startswith('core_unexplained_candidate')
    assert abs(r2.summary['core_unexplained_alpha_ann'] - s1['core_unexplained_alpha_ann']) < 1e-12


def test_multi_core_gate_on_best_of_k():
    rng = np.random.default_rng(4)
    R = pd.DataFrame(rng.normal(0, 0.003, (2500, 8)), columns=[f'v{i}' for i in range(8)])
    t = R.mean() / R.std() * np.sqrt(len(R))
    g = multi_core_gate(R, t, B=500, seed=1)
    assert g['n_specs'] == 8 and 6 < g['effective_trials'] <= 8.5
    assert g['pass'] == (g['naive_pass'] and g['spa_p_consistent'] < 0.05 and g['best_dsr'] >= 0.95)


def test_spec_sweep_default_grid():
    d = make_demo_case(n=2000, seed=5)
    sw = spec_sweep(d['strategy_net'], d['factors'], d['benchmark'], base_config=CFG, B=300,
                    states=d['states'], tail_factors=d['tail_factors'])
    assert set(sw.table.index) == {'linear_only', 'conditional_default', 'conditional_vol63', 'conditional_dd20',
                                   'no_tail_factors'}
    assert sw.table['is_best'].sum() == 1
    assert sw.core_returns.shape[1] == len(sw.table)
    assert sw.label in {'no_candidate', 'core_not_robust_to_specification_search', 'core_survives_luck_gate',
                        'core_survives_luck_gate_tail_flagged'}


REFEREE_CORE = os.environ.get('TAIL_AWARE_CORE')


@pytest.mark.skipif(not REFEREE_CORE or not os.path.exists(REFEREE_CORE),
                    reason='set TAIL_AWARE_CORE to the referee F0 notebook or core .py to run the parity test')
def test_tailaware_adapter_matches_the_referee_pipeline():
    from alpha_slicer.adapters import load_referee_core, run_tailaware_sp100_live
    core = load_referee_core(REFEREE_CORE)
    out = run_tailaware_sp100_live(core_path=REFEREE_CORE, start=core.CONFIG['start'], end=core.CONFIG['end'],
                                   data_source='shaped')
    cfg = dict(core.CONFIG, data_source='shaped')
    prices, ff = core.load_panel(cfg)
    rets, excess, ff = core.build_returns(prices, ff, cfg)
    sig = core.build_signals(core.build_ladder(excess, ff, cfg), cfg)[(cfg['verdict_level'], 'S1_resmom')]
    net = core.backtest(sig, rets, cfg)['net']
    common = out['strategy_net'].index
    assert np.allclose(out['strategy_net'].values, net.reindex(common).values)
    r = AlphaSlicer(CFG).fit(out['strategy_net'], out['factors'], out['benchmark'], strategy_gross=out['strategy_gross'])
    assert np.isfinite(r.summary['core_unexplained_alpha_ann'])
