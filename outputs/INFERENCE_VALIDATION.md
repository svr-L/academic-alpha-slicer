# Inference validation (v0.4, rerun and extended in v0.4b)

Population alpha is exactly zero in every world. The table gives the share of samples in which each method rejects zero at the nominal 5% level, with a Wilson 95% interval for that share. 400 seeds per world (1000 onward), 2,500 daily observations, 200 bootstrap draws.

| world | HAC(21) | HAC(126), 1.96 | HAC(126), fixed-b | prewhitened HAC | bootstrap, block 126 |
|---|---:|---:|---:|---:|---:|
| iid returns | 4.5% [2.9%, 7.0%] | 5.0% [3.3%, 7.6%] | 4.2% [2.7%, 6.7%] | 4.2% [2.7%, 6.7%] | 7.2% [5.1%, 10.2%] |
| two-state mean, regimes ~100 days (v0.4 design) | 33.2% [28.8%, 38.0%] | 16.2% [13.0%, 20.2%] | 12.0% [9.2%, 15.6%] | 42.2% [37.5%, 47.1%] | 16.8% [13.4%, 20.7%] |
| two-state mean, regimes ~200 days | 44.5% [39.7%, 49.4%] | 21.8% [18.0%, 26.1%] | 19.0% [15.5%, 23.1%] | 51.7% [46.9%, 56.6%] | 19.5% [15.9%, 23.7%] |

How to read it:

- **The v0.4 headline came from 40 seeds.** It reported 7.5% for HAC(126) and for the 126-day bootstrap (3 of 40; Wilson interval 2.6–19.9%). With the seeds above the same design gives the rates in the second row.
- **A long bandwidth helps but does not reach the nominal 5%** when the mean follows regimes that last about as long as the bandwidth, and longer regimes push the error up again. Fixed-b critical values take a little of it back.
- **Prewhitening does not help.** Daily returns here are noise plus a slow regime: their first-order autocorrelation is about 0.08, so a VAR(1) on the scores sees almost none of the persistence.
- **This is a stress test of the bandwidth, not of the slicer.** The suite's own rare-crash put had a different problem, fixed in v0.4b: its standard error came from a regression that included the zero-priced controls (see `outputs/MC_SUITE.md` and the CHANGELOG).

## Decomposition bootstrap: coverage of the core interval

The opt-in stationary bootstrap of the decomposition (200 draws, mean block 63) on three known-truth cases, seeds 1000-1099. Targets were written before the run (`alpha_slicer/inference_validation.py`, `BOOTSTRAP_TARGETS`).

| case | metric | target | value | pass | median interval width | candidate rate | fragile rate |
|---|---|---|---:|---|---:|---:|---:|
| peso_short_put | coverage_population | >= 0.8 | 91.0% | PASS | 49.7% | 2% | 18% |
| regime_alpha | coverage | >= 0.85 | 97.0% | PASS | 5.4% | 74% | 12% |
| noise | coverage | >= 0.85 | 96.0% | PASS | 5.2% | 2% | 0% |

