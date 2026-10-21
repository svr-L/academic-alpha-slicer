from __future__ import annotations

"""Persistence stress test for the core's standard error (v0.4, extended in v0.4b).

Separate from the known-truth pricing suite. It asks one question: when the population alpha is
exactly zero but the conditional mean follows a persistent two-state Markov chain, how often does
each inference method reject zero? v0.4 ran 40 seeds per world and reported 7.5% for HAC(126) and
for a 126-day stationary bootstrap. With 400 seeds per world the same design gives about 16-17%:
a long bandwidth halves the error of HAC(21) but stays well above the nominal 5%, and longer
regimes push it up again. v0.4b adds a fixed-b row and a 200-day-regime world, and reports Wilson
intervals so the number of seeds is visible in the result.

Run:  python -m alpha_slicer.inference_validation --seeds 400 --start-seed 1000 --out outputs
"""

from math import sqrt
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from .regression import hac_ols, prewhitened_hac_ols, fixed_b_critical_value
from .luck import stationary_bootstrap_indices as _sb_paths

WORLDS = {
    'iid': dict(kind='iid'),
    'persistent_100d': dict(kind='persistent', p_stay=0.99),     # mean regime length 100 days (v0.4 design)
    'persistent_200d': dict(kind='persistent', p_stay=0.995),    # mean regime length 200 days
}
METHODS = ('hac21', 'hac126', 'hac126_fixed_b', 'prewhite', 'bootstrap126')


def persistent_zero_alpha(seed: int, n: int = 2500, p_stay: float = 0.99,
                          regime_mu: float = 0.0015, noise_sd: float = 0.005):
    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = rng.choice([-1.0, 1.0])
    for t in range(1, n):
        z[t] = z[t - 1] if rng.random() < p_stay else -z[t - 1]
    # Symmetric Markov chain: stationary population mean is exactly zero.
    y = regime_mu * z + rng.normal(0.0, noise_sd, n)
    return pd.Series(y), rng


def iid_zero_alpha(seed: int, n: int = 2500, noise_sd: float = 0.005):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, noise_sd, n)), rng


def _bootstrap_ci(y: np.ndarray, rng: np.random.Generator, B: int, mean_block: float):
    paths = _sb_paths(len(y), B, mean_block, rng)
    vals = y[paths].mean(axis=1)
    return np.quantile(vals, [0.025, 0.975])


def evaluate_one(seed: int, world: str = 'persistent_100d', B: int = 200, mean_block: int = 126) -> dict:
    spec = WORLDS[world]
    if spec['kind'] == 'iid':
        y, rng = iid_zero_alpha(seed)
    else:
        y, rng = persistent_zero_alpha(seed, p_stay=spec['p_stay'])
    empty = pd.DataFrame(index=y.index)
    h21, h126 = hac_ols(y, empty, lags=21), hac_ols(y, empty, lags=126)
    pw = prewhitened_hac_ols(y, empty, lags='auto')
    lo, hi = _bootstrap_ci(y.to_numpy(float), rng, B, mean_block)
    cv126 = fixed_b_critical_value(127 / len(y))
    t126 = float(h126['tvalues']['const'])
    return {
        'world': world, 'seed': seed, 'sample_mean_ann': float(y.mean() * 252),
        'hac21_reject': bool(h21['pvalues']['const'] < 0.05),
        'hac126_reject': bool(h126['pvalues']['const'] < 0.05),
        'hac126_fixed_b_reject': bool(abs(t126) > cv126),
        'prewhite_reject': bool(pw['pvalues']['const'] < 0.05),
        'bootstrap126_reject': bool(not (lo <= 0 <= hi)),
        'hac21_t': float(h21['tvalues']['const']), 'hac126_t': t126,
        'prewhite_t': float(pw['tvalues']['const']),
        'bootstrap_ci_low_ann': float(lo * 252), 'bootstrap_ci_high_ann': float(hi * 252),
    }


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run(seeds=range(1000, 1400), B: int = 200, mean_block: int = 126, worlds=tuple(WORLDS)):
    rows = [evaluate_one(int(s), w, B=B, mean_block=mean_block) for w in worlds for s in seeds]
    per_seed = pd.DataFrame(rows)
    out = []
    for w, g in per_seed.groupby('world', sort=False):
        for m in METHODS:
            k, n = int(g[f'{m}_reject'].sum()), len(g)
            lo, hi = wilson(k, n)
            out.append({'world': w, 'method': m, 'n_seeds': n, 'false_reject_rate': k / n,
                        'wilson_low': lo, 'wilson_high': hi})
    return per_seed, pd.DataFrame(out)


# ----------------------------------------------------------------------------- decomposition bootstrap
# Coverage of the core interval from the decomposition bootstrap (opt-in in the slicer) on three
# known-truth cases. Targets written on 2026-10-19 before seeds 1000-1099 were run; development
# seeds 0-99 gave 86% / 91% for the first two with mean block 63.
BOOTSTRAP_TARGETS = {
    'peso_short_put': ('coverage_population', '>=', 0.80, 'rare-crash put: bootstrap interval covers the population alpha'),
    'regime_alpha': ('coverage', '>=', 0.85, 'genuine regime alpha: bootstrap interval covers the truth'),
    'noise': ('coverage', '>=', 0.85, 'zero alpha: bootstrap interval covers zero'),
}
POPULATION_TRUTH = {'peso_short_put': -0.0618, 'noise': 0.0}


def decomposition_bootstrap_check(seeds=range(1000, 1100), reps: int = 200, mean_block: float = 63.0,
                                  cases=tuple(BOOTSTRAP_TARGETS)):
    import warnings
    from dataclasses import replace
    from .decomposition import AlphaSlicer
    from .montecarlo import CASES, MC_CONFIG
    cfg = replace(MC_CONFIG, inference_bootstrap_reps=int(reps), inference_bootstrap_mean_block=float(mean_block))
    rows = []
    for case in cases:
        fn, kw = CASES[case]
        for seed in seeds:
            data = fn(int(seed), **kw)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = AlphaSlicer(cfg).fit(**data['inputs'])
            b = r.bootstrap_inference.loc['core_unexplained']
            truth = data['truth'].get('core')
            pop = POPULATION_TRUTH.get(case, np.nan)
            lo, hi = float(b['ci_low_ann']), float(b['ci_high_ann'])
            rows.append({'case': case, 'seed': int(seed), 'core': float(b['estimate_ann']), 'ci_low': lo, 'ci_high': hi,
                         'covered': bool(lo <= truth <= hi) if truth is not None else np.nan,
                         'covered_population': bool(lo <= pop <= hi) if np.isfinite(pop) else np.nan,
                         'candidate': str(r.summary['diagnostic_label']).startswith('core_unexplained_candidate'),
                         'fragile': bool(r.summary['inference_fragile'])})
    per_seed = pd.DataFrame(rows)
    out = []
    for case, g in per_seed.groupby('case', sort=False):
        metric, op, value, why = BOOTSTRAP_TARGETS[case]
        got = float(g['covered_population' if metric == 'coverage_population' else 'covered'].dropna().astype(float).mean())
        out.append({'case': case, 'metric': metric, 'target': f'{op} {value:g}', 'value': got,
                    'pass': bool(got >= value) if op == '>=' else bool(got <= value),
                    'median_ci_width': float((g['ci_high'] - g['ci_low']).median()),
                    'candidate_rate': float(g['candidate'].mean()), 'fragile_rate': float(g['fragile'].mean()),
                    'why': why})
    return per_seed, pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=400)
    ap.add_argument('--start-seed', type=int, default=1000)
    ap.add_argument('--bootstrap-reps', type=int, default=200)
    ap.add_argument('--mean-block', type=int, default=126)
    ap.add_argument('--decomposition-bootstrap', action='store_true',
                    help='also run the decomposition-bootstrap coverage check (100 seeds from --start-seed)')
    ap.add_argument('--out', type=str, default='outputs')
    a = ap.parse_args()
    seeds = range(a.start_seed, a.start_seed + a.seeds)
    per_seed, summary = run(seeds, B=a.bootstrap_reps, mean_block=a.mean_block)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / 'inference_validation_per_seed.csv', index=False)
    summary.to_csv(out / 'inference_validation_summary.csv', index=False)
    print(summary.to_string(index=False))
    if a.decomposition_bootstrap:
        bper, bsum = decomposition_bootstrap_check(range(a.start_seed, a.start_seed + 100))
        bper.to_csv(out / 'bootstrap_coverage_per_seed.csv', index=False)
        bsum.to_csv(out / 'bootstrap_coverage_summary.csv', index=False)
        print(bsum.to_string(index=False))


if __name__ == '__main__':
    main()
