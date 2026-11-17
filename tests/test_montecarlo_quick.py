"""Fast smoke version of the Monte Carlo suite (a few seeds, loose bounds).

The full suite with pre-registered targets runs with:
    python -m alpha_slicer.montecarlo --seeds 100 --jobs 2 --out outputs
"""
from alpha_slicer.montecarlo import run_cases

SEEDS = range(500, 506)   # outside the evaluation range used for the published targets


def test_quick_suite_behaves():
    D, _ = run_cases(['noise', 'actuarial_short_put', 'timing_skill', 'crash_premium_traded'],
                     SEEDS, jobs=1, references={'actuarial_short_put': {'slopes': None}})
    g = D.groupby('case')
    assert g.get_group('noise')['candidate'].sum() <= 1
    put = g.get_group('actuarial_short_put')
    assert put['candidate'].sum() == 0 and put['flagged'].mean() >= 5 / 6
    tim = g.get_group('timing_skill')
    assert tim['candidate'].all()
    assert 0.75 <= (tim['core'] / tim['truth']).median() <= 1.25
    crash = g.get_group('crash_premium_traded')
    assert crash['candidate'].sum() == 0 and (crash['label'] == 'tail_dominated').mean() >= 5 / 6
    assert D['addup_error'].max() < 1e-8 and D['core_returns_error'].max() < 1e-8
