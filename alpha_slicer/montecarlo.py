"""Known-truth Monte Carlo suite with pre-registered targets.

Every case is a data-generating process whose answer is known under the slicer's pricing
convention: alpha is the intercept on TRADED factors (each charged at its mean); non-traded
regressors are orthogonalized to them and carry a zero price; states are lagged.

Truths
------
* exact: noise, timing skill (in-sample timing term), regime alpha (8% / 0), the cases where the
  premium is supplied as a traded factor (0), and the demo (1%).
* conditional: for payoffs that are not traded (short variance, short put, crash and variance
  premia without a traded factor) the truth is the traded-model alpha of THIS sample computed by
  an oracle that knows the population betas: mean(y_clean) - b_pop' mean(X_traded), on the
  slicer's estimation sample, with b_pop from long noise-free paths. The estimator's error
  against it is beta-estimation error plus noise.
  The unconditional population value E[y] - b_pop'E[f] is reported alongside and, since v0.4b,
  has its own coverage metric (``coverage95_population``). It is the number an investor cares
  about: the conditional truth already contains the luck of the sample (how many crashes, which
  regimes), so covering it says nothing about that luck.

Pre-registration log
--------------------
* Metrics and thresholds in ``TARGETS`` were written before the evaluation seeds (0-99) were run.
* Changes after smoke tests on seeds 900-901 and 500-519 (both outside the evaluation range):
  1. Non-traded regressors were first only de-meaned; the short put sold at its average payoff
     came out at +5.5%/yr (t ~ 2.4): the zero-premium basis absorbed part of the put's market
     delta, so the delta escaped the market-premium charge, and the result depended on the
     chosen basis. The convention became orthogonalization (alpha = spanning-regression alpha
     on traded factors). The truth of payoffs sold at their actuarial value is therefore the
     traded-model alpha, no longer 0 (a put sold at its expected payoff is not paid for the
     market risk in its delta, so that truth is negative); fair_* cases became actuarial_*.
  2. Truths for non-traded payoffs became conditional (see above). With population-demeaned
     put payoffs, regime persistence moved the in-sample put P&L by +-20%/yr across seeds and
     HAC(21) t-stats reached +-17: that peso problem is now its own case (peso_short_put),
     and the actuarial cases keep the sample-demeaned payoffs of the original design.
  3. The demo's convexity leg became a traded variance swap (exact truth). The VRP and peso
     cases were added with their targets before the evaluation run. No threshold was changed.
* v0.3 (luck gate): the best-of-10 luck family and ``LUCK_TARGETS`` were added and fixed before
  that family was first run. ``candidate`` now counts the slicer's own candidate labels, which
  since v0.3 also require the single-specification luck gate (DSR >= 0.95, 1 declared trial);
  the v0.2 thresholds are unchanged.
* v0.4b (new decision statistic, 2026-10-19). The v0.4 review showed that the core's standard
  error came from the full design, whose zero-priced controls (m^2, m^3, tail terms) absorb the
  non-traded payoff's daily variation while its in-sample mean stays in the intercept. On the
  rare-crash put (development seeds 0-99) the estimated s.e. was 1.9%/yr against an actual
  error of 16%/yr: 15% coverage of the population alpha, and HAC(126) on the same design changed
  nothing. Taking the s.e. from the traded-only regression (same intercept) gave 79% coverage
  with HAC(21) and 91% with HAC(126). Changes, decided on those development seeds:
  1. The core t, its interval and the luck-gate threshold use the traded-only regression,
     Bartlett HAC(126) and a Kiefer-Vogelsang fixed-b critical value; the "t >= 2" rule is
     scaled by cv(b)/1.96. The waterfall and the regime-alpha table use the same bandwidth;
     the regime table drops the zero-priced controls. ``covered`` uses cv(b) instead of 1.96 and
     ``diff_sig_rate`` uses the same scaled threshold.
  2. New metrics: ``coverage95_population`` and, on the same fits, the pre-v0.4b statistic
     (``*_v03`` columns) for comparison.
  3. ``TARGETS_V04B`` was written before the evaluation run; all v0.2/v0.3 targets are kept
     unchanged. The evaluation moved to fresh seeds 1000-1099, run once; seeds 0-99 had been
     seen while diagnosing the problem.

Run:  python -m alpha_slicer.montecarlo --seeds 100 --seed-start 1000 --jobs 2 --out outputs
"""
from __future__ import annotations
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import integrate, stats

from .config import SliceConfig
from .decomposition import AlphaSlicer
from .luck import multi_core_gate
from .demo import make_demo_case
from .features import build_states

ANN = 252
DEFAULT_N = 3000
MC_CONFIG = SliceConfig(min_obs=500)
MU, SIG_LO, SIG_HI = 0.0003, 0.008, 0.016   # market drift and regime vols (t6 shocks)
VAR_T6 = 6.0 / 4.0
EXPECTED_M2 = MU ** 2 + VAR_T6 * 0.5 * (SIG_LO ** 2 + SIG_HI ** 2)   # stationary E[m^2]
PUT_STRIKE = -0.015
TRADED_BLOCKS = ('linear', 'conditional', 'tail')


@lru_cache(maxsize=None)
def expected_put(strike=PUT_STRIKE):
    """Stationary E[max(strike - m, 0)] for m = MU + sigma * t6, sigma in {SIG_LO, SIG_HI} 50/50."""
    out = 0.0
    for s in (SIG_LO, SIG_HI):
        f = lambda t: max(strike - MU - s * t, 0.0) * stats.t.pdf(t, 6)
        out += 0.5 * integrate.quad(f, -np.inf, (strike - MU) / s, limit=200)[0]
    return out


# ----------------------------------------------------------------------------- scaffold
def _market(n, rng):
    """Regime-switching market (stay prob 0.98, t6 shocks, 7.6%/yr premium)."""
    idx = pd.bdate_range('2010-01-04', periods=n)
    switch = rng.random(n) >= 0.98
    switch[0] = False
    reg = np.cumsum(switch) % 2
    m = MU + rng.standard_t(6, n) * np.where(reg > 0, SIG_HI, SIG_LO)
    return idx, m


def _factors(m, rng, idx):
    n = len(m)
    return pd.DataFrame({'MKT': m,
                         'SMB': 0.2 * m + rng.standard_t(7, n) * 0.005,
                         'HML': -0.1 * m + rng.standard_t(7, n) * 0.005}, index=idx)


def _factor_means(m_mean, y_mean):
    """Population means of the strategy and of the traded design. Factor x lagged-state
    interactions have population mean 0: the factor means do not depend on yesterday's state."""
    return {'__y__': y_mean, 'MKT': m_mean, 'SMB': 0.2 * m_mean, 'HML': -0.1 * m_mean}


def _pack(y_clean, e, idx, F, truth, **extra):
    inputs = {'strategy_net': pd.Series(y_clean + e, index=idx, name='strategy'), 'factors': F}
    inputs.update(extra)
    truth = dict(truth)
    truth['y_clean'] = pd.Series(y_clean, index=idx)
    return {'inputs': inputs, 'truth': truth}


def _noise(rng, n, noise):
    e = rng.normal(0, 0.003, n)
    return e if noise else np.zeros(n)


# ----------------------------------------------------------------------------- cases
def case_noise(seed, n=DEFAULT_N, noise=True):
    """Factor exposure plus noise. Truth: core 0."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    return _pack(0.3 * m + 0.1 * F['SMB'].values, _noise(rng, n, noise), idx, F, {'core': 0.0})


def case_actuarial_short_variance(seed, n=DEFAULT_N, noise=True):
    """Short variance struck at the realized average variance (no variance premium).
    Truth: conditional traded-model alpha (~0: m^2 is nearly orthogonal to the factors)."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    y = 0.3 * m - 3.0 * (m ** 2 - np.mean(m ** 2))
    return _pack(y, _noise(rng, n, noise), idx, F,
                 {'core': None, 'core_original_definition': 0.0, 'pop_means': _factor_means(MU, 0.3 * MU)})


def case_actuarial_short_put(seed, n=DEFAULT_N, noise=True):
    """Short OTM put (strike -1.5% daily) sold at its realized average payoff (no option premium).
    Truth: conditional traded-model alpha. Negative on average: the put's delta carries market
    risk that the market premium prices, and the seller is not paid for it."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    put = np.maximum(PUT_STRIKE - m, 0.0)
    y = 0.3 * m - 4.0 * (put - put.mean())
    return _pack(y, _noise(rng, n, noise), idx, F,
                 {'core': None, 'core_original_definition': 0.0, 'pop_means': _factor_means(MU, 0.3 * MU)})


def case_peso_short_put(seed, n=DEFAULT_N, noise=True):
    """The same short put sold at its POPULATION expected payoff. With persistent volatility
    regimes the in-sample put P&L swings by +-20%/yr across samples: calm samples look like large,
    'significant' alpha (a peso problem). Truth: conditional traded-model alpha of the sample.
    The slicer cannot tell luck of the regime mix from skill; it must hand every such core to the
    referee's hidden-risk gate (flags), never certify it clean."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    put = np.maximum(PUT_STRIKE - m, 0.0)
    y = 0.3 * m - 4.0 * (put - expected_put())
    return _pack(y, _noise(rng, n, noise), idx, F, {'core': None, 'pop_means': _factor_means(MU, 0.3 * MU)})


def case_timing_skill(seed, n=DEFAULT_N, noise=True):
    """Beta set from a noisy private preview of the market. No selectivity alpha.
    Truth: in-sample timing term Cov(beta_t, m_t) x 252 (alpha in the four-term decomposition)."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    d = (m + rng.normal(0, 0.012, n) > 0).astype(float)
    timing = 0.6 * float(np.mean((d - d.mean()) * (m - m.mean()))) * ANN
    return _pack((0.2 + 0.6 * d) * m, _noise(rng, n, noise), idx, F, {'core': timing})


def case_regime_alpha(seed, n=DEFAULT_N, noise=True):
    """+8%/yr alpha only after high-vol days (the slicer's own lagged HIGH_VOL state).
    Truth: alpha_on 8%, alpha_off 0, unconditional 8% x in-sample share of high-vol days."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    z = build_states(pd.Series(m, index=idx))['HIGH_VOL'].shift(1)
    y = 0.2 * m + (0.08 / ANN) * z.fillna(0.0).values
    return _pack(y, _noise(rng, n, noise), idx, F,
                 {'core': 0.08 * float(z.mean()), 'alpha_on': 0.08, 'alpha_off': 0.0})


def case_crash_premium(seed, n=DEFAULT_N, noise=True, traded=False):
    """Crash-insurance seller: carry 0.08%/day, -4% on crash days (1%/day, with a -2% market
    co-crash). Net premium ~10%/yr.
    traded=False: no traded crash factor; the premium cannot be priced, so the core equals it
                  (conditional truth) and the slicer must flag the unpriced tail risk.
    traded=True : the crash factor C is supplied as a traded tail factor. Truth: core 0."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    crash = rng.random(n) < 0.01
    m = m + np.where(crash, -0.02, 0.0)
    F = _factors(m, rng, idx)
    C = 0.0008 - 0.04 * crash
    y = 0.1 * m + C
    e = _noise(rng, n, noise)
    if traded:
        return _pack(y, e, idx, F, {'core': 0.0}, tail_factors=pd.DataFrame({'CRASH': C}, index=idx))
    m_mean = MU - 0.02 * 0.01
    return _pack(y, e, idx, F, {'core': None,
                                'pop_means': _factor_means(m_mean, 0.1 * m_mean + 0.0008 - 0.04 * 0.01)})


def case_vrp_short_variance(seed, n=DEFAULT_N, noise=True, traded=False):
    """Short variance that EARNS a variance risk premium: the swap is struck 25% above expected
    variance (~4.5%/yr premium).
    traded=False: no traded variance factor; the VRP stays in the core (conditional truth) and
                  must be flagged as unpriced tail risk.
    traded=True : the variance swap P&L is supplied as a traded tail factor. Truth: core 0. The
                  non-traded m^2 basis is then fully spanned and drops out (no collinearity)."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    vs = m ** 2 - 1.25 * EXPECTED_M2          # long variance swap P&L
    y = 0.3 * m - 3.0 * vs
    e = _noise(rng, n, noise)
    if traded:
        return _pack(y, e, idx, F, {'core': 0.0}, tail_factors=pd.DataFrame({'VARSWAP': vs}, index=idx))
    return _pack(y, e, idx, F, {'core': None, 'pop_means': _factor_means(MU, 0.3 * MU + 0.75 * EXPECTED_M2)})


def case_demo(seed, n=3500, noise=True):
    """The package demo (all blocks at once, every exposure traded). Truth: net core 1.0%/yr."""
    d = make_demo_case(n=n, seed=seed)
    inputs = {k: d[k] for k in ('strategy_net', 'factors', 'benchmark', 'strategy_gross',
                                'tail_factors', 'states')}
    return {'inputs': inputs, 'truth': {'core': float(d['truth']['core_alpha_net_ann'])}}


CASES = {
    'noise': (case_noise, {}),
    'actuarial_short_variance': (case_actuarial_short_variance, {}),
    'actuarial_short_put': (case_actuarial_short_put, {}),
    'peso_short_put': (case_peso_short_put, {}),
    'timing_skill': (case_timing_skill, {}),
    'regime_alpha': (case_regime_alpha, {}),
    'crash_premium_unpriced': (case_crash_premium, {'traded': False}),
    'crash_premium_traded': (case_crash_premium, {'traded': True}),
    'vrp_short_variance_unpriced': (case_vrp_short_variance, {'traded': False}),
    'vrp_short_variance_traded': (case_vrp_short_variance, {'traded': True}),
    'demo_known_truth': (case_demo, {}),
}

# ----------------------------------------------------------------------------- luck family (v0.3)
LUCK_CONFIG = SliceConfig(min_obs=500, block_order=('linear',))
K_VARIANTS = 10


def case_best_of_k(seed, n=DEFAULT_N, k=K_VARIANTS, genuine_alpha=0.0):
    """k strategy variants on the same market, the researcher reports the best core.
    genuine_alpha = 0: every variant has zero alpha (pure selection luck).
    genuine_alpha > 0: variant v0 has that annual core alpha (t ~ 4 at 6%), the others none."""
    rng = np.random.default_rng(seed)
    idx, m = _market(n, rng)
    F = _factors(m, rng, idx)
    ys = {}
    for j in range(k):
        a = genuine_alpha / ANN if j == 0 else 0.0
        ys[f'v{j}'] = pd.Series(a + 0.3 * m + 0.1 * F['SMB'].values + rng.normal(0, 0.003, n), index=idx, name=f'v{j}')
    return {'variants': ys, 'factors': F, 'genuine': 'v0' if genuine_alpha else None}


LUCK_CASES = {
    'best_of_10_noise': (case_best_of_k, {'genuine_alpha': 0.0}),
    'best_of_10_one_genuine': (case_best_of_k, {'genuine_alpha': 0.06}),
}

# Fixed before the luck family was first run.
LUCK_TARGETS: Dict[str, List[tuple]] = {
    'best_of_10_noise': [
        ('gated_rate', '<=', 0.06, 'reporting the best of 10 zero-alpha variants is not certified (naive t>=2 is not reported as a target)'),
    ],
    'best_of_10_one_genuine': [
        ('gated_rate', '>=', 0.80, 'a 6%/yr core (t ~ 4) among 10 variants still clears the gate'),
        ('best_is_genuine_rate', '>=', 0.90, 'the best core is the genuine variant'),
    ],
}


def evaluate_luck_one(case: str, seed: int, B: int = 1000) -> dict:
    fn, kw = LUCK_CASES[case]
    d = fn(seed, **kw)
    cores, ts, t_min = {}, {}, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, y in d['variants'].items():
            r = AlphaSlicer(LUCK_CONFIG).fit(y, d['factors'])
            cores[name] = r.core_returns
            ts[name] = float(r.summary['core_alpha_t'])
            t_min = max(t_min, float(r.summary['core_t_critical']))   # v0.4b fixed-b threshold
    g = multi_core_gate(pd.concat(cores, axis=1), pd.Series(ts), t_min=t_min, B=B, seed=seed)
    return {'case': case, 'seed': seed, 'naive_pass': g['naive_pass'], 'gated_pass': g['pass'],
            'best': g['best'], 'best_is_genuine': g['best'] == d['genuine'], 'spa_p': g['spa_p_consistent'],
            'best_t': g['best_hac_t'], 'best_dsr': g['best_dsr'], 'effective_trials': g['effective_trials']}


def _luck_task(args):
    return evaluate_luck_one(*args)


def run_luck_cases(cases: Iterable[str], seeds: Iterable[int], jobs: int = 1) -> pd.DataFrame:
    tasks = [(c, int(s)) for c in cases for s in seeds]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(_luck_task, tasks, chunksize=2))
    else:
        rows = [_luck_task(t) for t in tasks]
    return pd.DataFrame(rows)


def summarize_luck(per_seed: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for case, g in per_seed.groupby('case', sort=False):
        out[case] = {'n_seeds': len(g), 'naive_rate': g['naive_pass'].mean(), 'gated_rate': g['gated_pass'].mean(),
                     'best_is_genuine_rate': g['best_is_genuine'].mean() if g['best_is_genuine'].notna().any() else np.nan,
                     'median_best_t': g['best_t'].median(), 'median_spa_p': g['spa_p'].median(),
                     'median_effective_trials': g['effective_trials'].median()}
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- targets
# (metric, operator, value, why). Thresholds fixed before the evaluation run (see module doc).
TARGETS: Dict[str, List[tuple]] = {
    'noise': [
        ('candidate_rate', '<=', 0.06, 'false core certification near the nominal one-sided 2.3%'),
        ('abs_median_bias', '<=', 0.0075, 'unbiased core'),
        ('coverage95', '>=', 0.85, 'HAC intervals roughly honest'),
        ('flag_rate', '<=', 0.15, 'tail flags rarely fire on symmetric noise'),
    ],
    'actuarial_short_variance': [
        ('abs_median_bias', '<=', 0.01, 'non-traded convexity is not alpha (v0.1: +19%/yr)'),
        ('candidate_rate', '<=', 0.06, 'no certification of a short-vol payoff'),
        ('coverage95', '>=', 0.85, 'HAC intervals roughly honest'),
        ('flag_rate', '>=', 0.80, 'short convexity is flagged as unpriced tail risk'),
    ],
    'actuarial_short_put': [
        ('abs_median_bias', '<=', 0.01, 'put selling is not alpha (v0.1: +188%/yr)'),
        ('candidate_rate', '<=', 0.06, 'no certification of a short-put payoff'),
        ('coverage95', '>=', 0.85, 'HAC intervals roughly honest'),
        ('flag_rate', '>=', 0.80, 'short-put exposure is flagged as unpriced tail risk'),
    ],
    'peso_short_put': [
        ('flag_rate', '>=', 0.80, 'regime-luck put selling is handed to the H2 gate'),
        ('clean_candidate_rate', '<=', 0.10, 'never a clean certification of put selling'),
    ],
    'timing_skill': [
        ('candidate_rate', '>=', 0.90, 'timing skill is alpha and is detected (v0.1: -14%/yr)'),
        ('median_ratio_to_truth', '>=', 0.75, 'core close to the timing term'),
        ('median_ratio_to_truth', '<=', 1.25, 'core close to the timing term'),
        ('flag_rate', '<=', 0.15, 'convex payoffs do not raise tail flags'),
    ],
    'regime_alpha': [
        ('median_alpha_on', '>=', 0.06, 'regime alpha recovered (truth 8%)'),
        ('median_alpha_on', '<=', 0.10, 'regime alpha recovered (truth 8%)'),
        ('abs_median_alpha_off', '<=', 0.015, 'no alpha off-regime (truth 0)'),
        ('diff_sig_rate', '>=', 0.60, 'on-minus-off detected (power)'),
        ('coverage95', '>=', 0.85, 'unconditional core = 8% x share of high-vol days'),
    ],
    'crash_premium_unpriced': [
        ('flag_rate', '>=', 0.90, 'an unpriceable crash premium is handed to the H2 gate'),
        ('clean_candidate_rate', '<=', 0.10, 'never a clean certification of a crash premium'),
    ],
    'crash_premium_traded': [
        ('candidate_rate', '<=', 0.06, 'a priced crash premium is not alpha'),
        ('tail_dominated_share', '>=', 0.80, 'the traded tail block explains the mean'),
        ('abs_median_bias', '<=', 0.01, 'core ~ 0 once the premium is priced'),
        ('flag_rate', '<=', 0.15, 'priced tail risk does not raise unpriced-risk flags'),
    ],
    'vrp_short_variance_unpriced': [
        ('flag_rate', '>=', 0.80, 'an unpriceable variance premium is handed to the H2 gate'),
        ('clean_candidate_rate', '<=', 0.10, 'never a clean certification of a variance premium'),
    ],
    'vrp_short_variance_traded': [
        ('candidate_rate', '<=', 0.06, 'a priced variance premium is not alpha'),
        ('abs_median_bias', '<=', 0.01, 'core ~ 0 once the VRP is priced by a traded swap'),
        ('tail_dominated_share', '>=', 0.80, 'the traded tail block explains the mean'),
        ('flag_rate', '<=', 0.15, 'priced convexity does not raise unpriced-risk flags'),
    ],
    'demo_known_truth': [
        ('abs_median_bias', '<=', 0.01, 'net core 1%/yr recovered with all blocks active'),
        ('coverage95', '>=', 0.85, 'HAC intervals roughly honest'),
        ('flag_rate', '<=', 0.15, 'long convexity + traded tail factors: no unpriced flag'),
    ],
    '*': [
        ('max_addup_error', '<=', 1e-8, 'Shapley contributions + core add up to the raw mean'),
        ('max_core_returns_error', '<=', 1e-8, 'core return series has mean = core alpha'),
    ],
}


# v0.4b targets. Written on 2026-10-19 after the v0.4 review on development seeds 0-99 and
# before the evaluation seeds 1000-1099 were run (see the pre-registration log above).
# Population coverage is measured against E[y] - b'E[f], not the in-sample (conditional) truth.
TARGETS_V04B: Dict[str, List[tuple]] = {
    'actuarial_short_variance': [
        ('coverage95_population', '>=', 0.85, 'decision interval covers the population alpha'),
    ],
    'actuarial_short_put': [
        ('coverage95_population', '>=', 0.85, 'decision interval covers the population alpha'),
    ],
    'peso_short_put': [
        ('coverage95_population', '>=', 0.85, 'rare-crash put: interval covers the population alpha (v0.3 statistic ~15%)'),
        ('candidate_rate', '<=', 0.05, 'rare-crash put is almost never a candidate, flagged or not (v0.3: 16%)'),
    ],
    'crash_premium_unpriced': [
        ('coverage95_population', '>=', 0.85, 'decision interval covers the population alpha'),
    ],
    'vrp_short_variance_unpriced': [
        ('coverage95_population', '>=', 0.85, 'decision interval covers the population alpha'),
    ],
    'regime_alpha': [
        ('candidate_rate', '>=', 0.65, 'power kept on genuine regime alpha (v0.3: 80%; expected ~75% at the fixed-b threshold)'),
    ],
}


# ----------------------------------------------------------------------------- truths
def needs_reference(case: str) -> bool:
    fn, kw = CASES[case]
    return fn(0, n=600, noise=False, **kw)['truth'].get('core', 0.0) is None


def population_reference(case: str, n: int = 500_000, paths: int = 2, seed0: int = 7_000_000) -> dict:
    """Population slopes of the noise-free strategy on the slicer's traded design (long paths)
    and the unconditional population truth E[y] - b'E[f] (annualized) with its MC s.e."""
    fn, kw = CASES[case]
    slopes, vals = [], []
    for k in range(paths):
        data = fn(seed0 + k, n=n, noise=False, **kw)
        inp = {key: v for key, v in data['inputs'].items()
               if key in ('strategy_net', 'factors', 'benchmark', 'states', 'tail_factors')}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            design = AlphaSlicer(MC_CONFIG).prepare(**inp)
        X = design.X_traded
        A = np.column_stack([np.ones(len(design.y)), X.values])
        b, *_ = np.linalg.lstsq(A, design.y.values, rcond=None)
        slopes.append(pd.Series(b[1:], index=X.columns))
        pm = data['truth'].get('pop_means')
        if pm:
            ef = np.array([pm.get(c, 0.0) for c in X.columns])
            vals.append(float((pm['__y__'] - b[1:] @ ef) * ANN))
    vals = np.asarray(vals)
    return {'slopes': pd.concat(slopes, axis=1).mean(axis=1),
            'population_truth': float(vals.mean()) if len(vals) else np.nan,
            'population_truth_se': float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan}


def conditional_truth(y_clean: pd.Series, traded_X: pd.DataFrame, slopes: pd.Series, ann=ANN) -> float:
    """Traded-model alpha of the sample for an oracle that knows the population betas."""
    b = slopes.reindex(traded_X.columns).fillna(0.0)
    return float((y_clean.reindex(traded_X.index).mean() - float(b @ traded_X.mean())) * ann)


# ----------------------------------------------------------------------------- runner
# Cases whose truth is the same number in every sample (so it is also the population truth).
CONSTANT_TRUTH_CASES = ('noise', 'crash_premium_traded', 'vrp_short_variance_traded', 'demo_known_truth')


def evaluate_one(case: str, seed: int, n: Optional[int] = None, cfg: Optional[SliceConfig] = None,
                 slopes: Optional[pd.Series] = None, pop_truth: Optional[float] = None) -> dict:
    fn, kw = CASES[case]
    data = fn(seed, **kw) if n is None else fn(seed, n=n, **kw)
    cfg = cfg or MC_CONFIG
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = AlphaSlicer(cfg).fit(**data['inputs'])
    s = r.summary
    core, t = float(s['core_unexplained_alpha_ann']), float(s['core_alpha_t'])
    se, cv = float(s['core_alpha_se_ann']), float(s['core_ci_critical'])
    se03, t03 = float(s['core_alpha_se_ann_v03']), float(s['core_alpha_t_v03'])
    truth = data['truth']
    truth_core = truth.get('core')
    if truth_core is None:
        if slopes is None:
            truth_core = np.nan
        else:
            Xt = pd.concat([r.feature_blocks[b] for b in TRADED_BLOCKS if b in r.feature_blocks], axis=1)
            truth_core = conditional_truth(truth['y_clean'], Xt, slopes, cfg.annualization)
    if pop_truth is None or not np.isfinite(pop_truth):
        pop_truth = float(truth['core']) if case in CONSTANT_TRUTH_CASES else np.nan
    has_pop = bool(np.isfinite(pop_truth))
    row = {
        'case': case, 'seed': seed, 'core': core, 'core_t': t, 'core_se': se, 'core_cv': cv,
        't_critical': float(s['core_t_critical']), 'truth': truth_core, 'population_truth': pop_truth,
        'bias': core - truth_core, 'covered': bool(abs(core - truth_core) <= cv * se),
        'covered_population': bool(abs(core - pop_truth) <= cv * se) if has_pop else np.nan,
        # pre-v0.4b statistic on the same fit: full design, HAC(hac_lags), normal critical value
        'core_t_v03': t03, 'core_se_v03': se03,
        'covered_v03': bool(abs(core - truth_core) <= 1.96 * se03),
        'covered_population_v03': bool(abs(core - pop_truth) <= 1.96 * se03) if has_pop else np.nan,
        'candidate_v03': bool(core > 0 and t03 >= cfg.core_t_threshold and float(s['core_dsr']) >= cfg.luck_dsr_min),
        'fragile': bool(s['inference_fragile']),
        'candidate': bool(str(s['diagnostic_label']).startswith('core_unexplained_candidate')),
        'flagged': bool(s['unpriced_tail_risk_flag']), 'flags': s['tail_flags_fired'],
        'label': s['diagnostic_label'], 'raw': float(s['raw_net_mean_ann']),
        'core_alt_demeaned': float(s['core_alt_demeaned_controls_ann']),
        'core_v01_convention': float(s['core_if_nontraded_charged_at_raw_mean_ann']),
        'truth_original_definition': truth.get('core_original_definition', np.nan),
        'addup_error': float(abs(r.shapley['contribution_ann'].sum() - s['raw_net_mean_ann'])),
        'core_returns_error': float(abs(r.core_returns.mean() * cfg.annualization - core)),
    }
    if 'alpha_on' in truth:
        ca = r.conditional_alpha
        row.update({'alpha_on': float(ca.loc[('HIGH_VOL', 'on'), 'alpha_ann']),
                    'alpha_off': float(ca.loc[('HIGH_VOL', 'off'), 'alpha_ann']),
                    'diff_t': float(ca.loc[('HIGH_VOL', 'on_minus_off'), 't'])})
    return row


def _task(args):
    return evaluate_one(*args)


def _ref_task(case):
    return case, population_reference(case)


def run_cases(cases: Iterable[str], seeds: Iterable[int], jobs: int = 1, n: Optional[int] = None,
              references: Optional[Dict[str, dict]] = None):
    cases, seeds = list(cases), [int(s) for s in seeds]
    refs = dict(references or {})
    todo = [c for c in cases if c not in refs and needs_reference(c)]
    if todo:
        if jobs and jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                refs.update(dict(ex.map(_ref_task, todo)))
        else:
            refs.update(dict(map(_ref_task, todo)))
    tasks = [(c, s, n, None, refs[c]['slopes'] if c in refs else None,
              refs[c].get('population_truth') if c in refs else None) for c in cases for s in seeds]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(_task, tasks, chunksize=4))
    else:
        rows = [_task(t) for t in tasks]
    return pd.DataFrame(rows), refs


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for case, g in per_seed.groupby('case', sort=False):
        nz = g['truth'].abs() > 1e-12
        labels = g['label'].value_counts()
        d = {
            'n_seeds': len(g),
            'median_truth': g['truth'].median(), 'median_core': g['core'].median(),
            'median_bias': g['bias'].median(), 'abs_median_bias': abs(g['bias'].median()),
            'median_core_t': g['core_t'].median(),
            'coverage95': g['covered'].mean(), 'candidate_rate': g['candidate'].mean(),
            'coverage95_population': (g['covered_population'].dropna().astype(float).mean()
                                      if g['covered_population'].notna().any() else np.nan),
            'coverage95_v03': g['covered_v03'].mean(),
            'coverage95_population_v03': (g['covered_population_v03'].dropna().astype(float).mean()
                                          if g['covered_population_v03'].notna().any() else np.nan),
            'candidate_rate_v03': g['candidate_v03'].mean(),
            'median_se_ratio_v04b_to_v03': (g['core_se'] / g['core_se_v03']).median(),
            'fragile_rate': g['fragile'].mean(),
            'flag_rate': g['flagged'].mean(),
            'clean_candidate_rate': (g['candidate'] & ~g['flagged']).mean(),
            'median_ratio_to_truth': (g.loc[nz, 'core'] / g.loc[nz, 'truth']).median() if nz.any() else np.nan,
            'top_label': labels.index[0], 'top_label_share': labels.iloc[0] / len(g),
            'tail_dominated_share': (g['label'] == 'tail_dominated').mean(),
            'median_core_alt_demeaned': g['core_alt_demeaned'].median(),
            'median_core_v01_convention': g['core_v01_convention'].median(),
            'median_truth_original_definition': g['truth_original_definition'].median(),
            'max_addup_error': g['addup_error'].max(), 'max_core_returns_error': g['core_returns_error'].max(),
        }
        if 'alpha_on' in g and g['alpha_on'].notna().any():
            d.update({'median_alpha_on': g['alpha_on'].median(), 'median_alpha_off': g['alpha_off'].median(),
                      'abs_median_alpha_off': abs(g['alpha_off'].median()),
                      'diff_sig_rate': (g['diff_t'] >= g['t_critical']).mean()})
        out[case] = d
    return pd.DataFrame(out).T


def check_targets(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ops = {'<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b}
    for case in summary.index:
        sets = [('v0.2-v0.3', t) for t in TARGETS.get(case, []) + TARGETS['*']]
        sets += [('v0.4b', t) for t in TARGETS_V04B.get(case, [])]
        for tset, (metric, op, value, why) in sets:
            got = float(summary.loc[case, metric]) if metric in summary.columns else np.nan
            rows.append({'case': case, 'set': tset, 'metric': metric, 'target': f'{op} {value:g}', 'value': got,
                         'pass': bool(np.isfinite(got) and ops[op](got, value)), 'why': why})
    return pd.DataFrame(rows)


def check_luck_targets(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ops = {'<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b}
    for case in summary.index:
        for metric, op, value, why in LUCK_TARGETS.get(case, []):
            got = float(summary.loc[case, metric])
            rows.append({'case': case, 'set': 'v0.3 luck', 'metric': metric, 'target': f'{op} {value:g}', 'value': got,
                         'pass': bool(np.isfinite(got) and ops[op](got, value)), 'why': why})
    return pd.DataFrame(rows)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return 'n/a' if not np.isfinite(v) else f'{v:.4f}'
    return str(v)


def _markdown(summary, checks, refs, seeds, elapsed, luck_summary=None, luck_checks=None):
    n_pass = int(checks['pass'].sum()) + (int(luck_checks['pass'].sum()) if luck_checks is not None else 0)
    n_all = len(checks) + (len(luck_checks) if luck_checks is not None else 0)
    nb = int(((checks['set'] == 'v0.4b') & checks['pass']).sum()) if 'set' in checks else 0
    nb_all = int((checks['set'] == 'v0.4b').sum()) if 'set' in checks else 0
    cols = ['median_truth', 'median_core', 'median_core_t', 'coverage95', 'coverage95_population',
            'candidate_rate', 'flag_rate', 'top_label']
    lines = ['# Monte Carlo known-truth suite', '',
             f'{len(seeds)} seeds per case (seeds {min(seeds)}..{max(seeds)}), n = 3,000 daily obs '
             f'(demo 3,500); runtime {elapsed:.0f}s. Targets, truth definitions and the changes made '
             'after smoke tests are documented in `alpha_slicer/montecarlo.py`.', '',
             f"**Targets passed: {n_pass}/{n_all}** (v0.4b targets: {nb}/{nb_all})", '', '## Results by case', '',
             'Since v0.4b the core\'s t-stat and interval come from the traded-only spanning regression '
             '(the same intercept as the full design), Bartlett HAC with 126 lags and a fixed-b critical '
             'value (about 2.09; the "t >= 2" rule becomes t >= ~2.13). `candidate_rate` counts the slicer\'s '
             'own candidate labels: that statistic plus the single-specification luck gate (deflated Sharpe '
             '>= 0.95, 1 declared trial). `coverage95` is measured against the in-sample (conditional) truth, '
             '`coverage95_population` against the population alpha E[y] - b\'E[f] where one exists.', '',
             '| case | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for case, r in summary.iterrows():
        lines.append(f'| {case} | ' + ' | '.join(_fmt(r.get(c, np.nan)) for c in cols) + ' |')
    cmp_cols = ['coverage95_population_v03', 'coverage95_population', 'candidate_rate_v03', 'candidate_rate',
                'median_se_ratio_v04b_to_v03']
    lines += ['', '## The same fits under the pre-v0.4b statistic', '',
              'v0.3/v0.4 took the core\'s standard error from the full design, where the zero-priced controls '
              'absorb the non-traded payoff\'s variation while its in-sample mean stays in the intercept. '
              'Same datasets, same point estimates; only the standard error and the threshold differ.', '',
              '| case | ' + ' | '.join(cmp_cols) + ' |', '|' + '---|' * (len(cmp_cols) + 1)]
    for case, r in summary.iterrows():
        lines.append(f'| {case} | ' + ' | '.join(_fmt(r.get(c, np.nan)) for c in cmp_cols) + ' |')
    lines.append('')
    if luck_summary is not None and len(luck_summary):
        lines += ['## Luck gate: reporting the best of 10 strategy variants', '',
                  'Each seed builds 10 variants on the same market; the best core is reported. `naive_rate`: best '
                  'variant has core > 0 and t above the fixed-b threshold. `gated_rate`: it also passes Hansen SPA '
                  '(p < 0.05 across the 10) and DSR >= 0.95 at the effective number of variants.', '',
                  '| case | naive_rate | gated_rate | best_is_genuine_rate | median_best_t | median_spa_p | median_effective_trials |',
                  '|---|---|---|---|---|---|---|']
        for case, r in luck_summary.iterrows():
            lines.append(f"| {case} | " + ' | '.join(_fmt(float(r[c])) for c in ['naive_rate', 'gated_rate', 'best_is_genuine_rate',
                                                                                      'median_best_t', 'median_spa_p',
                                                                                      'median_effective_trials']) + ' |')
        lines.append('')
    if refs:
        lines += ['## Population truths of the non-traded payoffs', '',
                  'Unconditional traded-model alpha E[y] - b\'E[f], slopes from 2 x 500,000 noise-free days. '
                  '`coverage95` uses the conditional (in-sample) truths; `coverage95_population` uses these.', '',
                  '| case | population truth | MC s.e. |', '|---|---|---|']
        for c, ref in refs.items():
            lines.append(f"| {c} | {ref['population_truth']:.4f} | {ref['population_truth_se']:.4f} |")
        lines.append('')
    if 'median_truth_original_definition' in summary:
        orig = summary['median_truth_original_definition'].astype(float).dropna()
        if len(orig):
            lines += ['## Against the pre-change truth definition', '',
                      'Before the convention switch, payoffs sold at their actuarial value had truth 0.', '',
                      '| case | original truth | median core | abs diff | original target <= 0.01 |',
                      '|---|---|---|---|---|']
            for c, v in orig.items():
                b = abs(float(summary.loc[c, 'median_core']) - v)
                lines.append(f'| {c} | {v:.4f} | {float(summary.loc[c, "median_core"]):.4f} | {b:.4f} | '
                             f'{"PASS" if b <= 0.01 else "FAIL"} |')
            lines.append('')
    allchecks = checks if luck_checks is None else pd.concat([checks, luck_checks], ignore_index=True)
    lines += ['## Target checks', '', '| case | set | metric | target | value | pass | why |', '|---|---|---|---|---|---|---|']
    for _, c in allchecks.iterrows():
        lines.append(f"| {c['case']} | {c.get('set', '')} | {c['metric']} | {c['target']} | {c['value']:.4g} | "
                     f"{'PASS' if c['pass'] else 'FAIL'} | {c['why']} |")
    return '\n'.join(lines) + '\n'


def run_suite(seeds=range(100), cases=None, jobs=1, out_dir=None, luck=True):
    seeds = list(seeds)
    cases = list(cases or CASES)
    t0 = time.time()
    per_seed, refs = run_cases(cases, seeds, jobs=jobs)
    summary = summarize(per_seed)
    checks = check_targets(summary)
    luck_per_seed = luck_summary = luck_checks = None
    if luck:
        luck_per_seed = run_luck_cases(list(LUCK_CASES), seeds, jobs=jobs)
        luck_summary = summarize_luck(luck_per_seed)
        luck_checks = check_luck_targets(luck_summary)
    elapsed = time.time() - t0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        per_seed.to_csv(os.path.join(out_dir, 'mc_suite_per_seed.csv'), index=False)
        summary.to_csv(os.path.join(out_dir, 'mc_suite_summary.csv'))
        allchecks = checks if luck_checks is None else pd.concat([checks, luck_checks], ignore_index=True)
        allchecks.to_csv(os.path.join(out_dir, 'mc_suite_targets.csv'), index=False)
        if luck_per_seed is not None:
            luck_per_seed.to_csv(os.path.join(out_dir, 'mc_luck_per_seed.csv'), index=False)
            luck_summary.to_csv(os.path.join(out_dir, 'mc_luck_summary.csv'))
        with open(os.path.join(out_dir, 'MC_SUITE.md'), 'w', encoding='utf-8') as fh:
            fh.write(_markdown(summary, checks, refs, seeds, elapsed, luck_summary, luck_checks))
    if luck_checks is not None:
        checks = pd.concat([checks, luck_checks], ignore_index=True)
    return per_seed, summary, checks, refs


def main(argv=None):
    ap = argparse.ArgumentParser(description='Run the known-truth Monte Carlo suite.')
    ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--seed-start', type=int, default=1000)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--no-luck', action='store_true', help='skip the best-of-10 luck family')
    ap.add_argument('--out', default='outputs')
    a = ap.parse_args(argv)
    seeds = range(a.seed_start, a.seed_start + a.seeds)
    _, summary, checks, _ = run_suite(seeds, a.cases, a.jobs, a.out, luck=not a.no_luck)
    pd.set_option('display.width', 220)
    print(summary[['median_truth', 'median_core', 'coverage95', 'candidate_rate', 'flag_rate', 'top_label']].to_string())
    print(f"\nTargets passed: {int(checks['pass'].sum())}/{len(checks)}")
    failed = checks[~checks['pass']]
    if len(failed):
        print(failed.to_string(index=False))
    return 0 if failed.empty else 1


if __name__ == '__main__':
    raise SystemExit(main())
