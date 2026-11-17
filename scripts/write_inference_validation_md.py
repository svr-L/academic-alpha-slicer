"""Write outputs/INFERENCE_VALIDATION.md from the stress-test and bootstrap-coverage CSVs.

Usage (from the repo root, after `python -m alpha_slicer.inference_validation --decomposition-bootstrap`):
    python scripts/write_inference_validation_md.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
LABELS = {'hac21': 'HAC(21)', 'hac126': 'HAC(126), 1.96', 'hac126_fixed_b': 'HAC(126), fixed-b',
          'prewhite': 'prewhitened HAC', 'bootstrap126': 'bootstrap, block 126'}
WORLDS = {'iid': 'iid returns', 'persistent_100d': 'two-state mean, regimes ~100 days (v0.4 design)',
          'persistent_200d': 'two-state mean, regimes ~200 days'}


def main():
    s = pd.read_csv(OUT / 'inference_validation_summary.csv')
    n = int(s['n_seeds'].iloc[0])
    lines = ['# Inference validation (v0.4, rerun and extended in v0.4b)', '',
             'Population alpha is exactly zero in every world. The table gives the share of samples in which '
             'each method rejects zero at the nominal 5% level, with a Wilson 95% interval for that share. '
             f'{n} seeds per world (1000 onward), 2,500 daily observations, 200 bootstrap draws.', '',
             '| world | ' + ' | '.join(LABELS.values()) + ' |', '|---|' + '---:|' * len(LABELS)]
    for w, wl in WORLDS.items():
        g = s[s['world'] == w].set_index('method')
        if g.empty:
            continue
        cells = [f"{g.loc[m, 'false_reject_rate']:.1%} [{g.loc[m, 'wilson_low']:.1%}, {g.loc[m, 'wilson_high']:.1%}]"
                 for m in LABELS]
        lines.append(f'| {wl} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'How to read it:', '',
              '- **The v0.4 headline came from 40 seeds.** It reported 7.5% for HAC(126) and for the 126-day bootstrap '
              '(3 of 40; Wilson interval 2.6–19.9%). With the seeds above the same design gives the rates in the second row.',
              '- **A long bandwidth helps but does not reach the nominal 5%** when the mean follows regimes that last '
              'about as long as the bandwidth, and longer regimes push the error up again. Fixed-b critical values '
              'take a little of it back.',
              '- **Prewhitening does not help.** Daily returns here are noise plus a slow regime: their first-order '
              'autocorrelation is about 0.08, so a VAR(1) on the scores sees almost none of the persistence.',
              '- **This is a stress test of the bandwidth, not of the slicer.** The suite\'s own rare-crash put had a '
              'different problem, fixed in v0.4b: its standard error came from a regression that included the zero-priced '
              'controls (see `outputs/MC_SUITE.md` and the CHANGELOG).', '']
    b = OUT / 'bootstrap_coverage_summary.csv'
    if b.exists():
        bs = pd.read_csv(b)
        lines += ['## Decomposition bootstrap: coverage of the core interval', '',
                  'The opt-in stationary bootstrap of the decomposition (200 draws, mean block 63) on three known-truth '
                  'cases, seeds 1000-1099. Targets were written before the run '
                  '(`alpha_slicer/inference_validation.py`, `BOOTSTRAP_TARGETS`).', '',
                  '| case | metric | target | value | pass | median interval width | candidate rate | fragile rate |',
                  '|---|---|---|---:|---|---:|---:|---:|']
        for _, r in bs.iterrows():
            lines.append(f"| {r['case']} | {r['metric']} | {r['target']} | {r['value']:.1%} | "
                         f"{'PASS' if r['pass'] else 'FAIL'} | {r['median_ci_width']:.1%} | {r['candidate_rate']:.0%} | "
                         f"{r['fragile_rate']:.0%} |")
        lines.append('')
    (OUT / 'INFERENCE_VALIDATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('written', OUT / 'INFERENCE_VALIDATION.md')


if __name__ == '__main__':
    main()
