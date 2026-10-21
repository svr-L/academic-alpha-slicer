# Monte Carlo known-truth suite

100 seeds per case (seeds 1000..1099), n = 3,000 daily obs (demo 3,500); runtime 1500s. Targets, truth definitions and the changes made after smoke tests are documented in `alpha_slicer/montecarlo.py`.

**Targets passed: 69/70** (v0.4b targets: 7/7)

## Results by case

Since v0.4b the core's t-stat and interval come from the traded-only spanning regression (the same intercept as the full design), Bartlett HAC with 126 lags and a fixed-b critical value (about 2.09; the "t >= 2" rule becomes t >= ~2.13). `candidate_rate` counts the slicer's own candidate labels: that statistic plus the single-specification luck gate (deflated Sharpe >= 0.95, 1 declared trial). `coverage95` is measured against the in-sample (conditional) truth, `coverage95_population` against the population alpha E[y] - b'E[f] where one exists.

| case | median_truth | median_core | median_core_t | coverage95 | coverage95_population | candidate_rate | flag_rate | top_label |
|---|---|---|---|---|---|---|---|---|
| noise | 0.0000 | -0.0003 | -0.0252 | 0.9700 | 0.9700 | 0.0200 | 0.0400 | linear_dominated |
| actuarial_short_variance | -0.0007 | 0.0000 | 0.0003 | 0.9900 | 0.9900 | 0.0100 | 1.0000 | linear_dominated |
| actuarial_short_put | -0.0551 | -0.0453 | -0.3201 | 1.0000 | 0.9900 | 0.0000 | 1.0000 | linear_dominated |
| peso_short_put | -0.0785 | -0.0667 | -0.4978 | 1.0000 | 0.9200 | 0.0200 | 1.0000 | linear_dominated |
| timing_skill | 0.6428 | 0.6369 | 16.3577 | 1.0000 | n/a | 1.0000 | 0.0000 | core_unexplained_candidate |
| regime_alpha | 0.0403 | 0.0404 | 2.8091 | 0.9800 | n/a | 0.7400 | 0.0300 | core_unexplained_candidate |
| crash_premium_unpriced | 0.1036 | 0.1033 | 4.6518 | 1.0000 | 0.9500 | 0.9700 | 1.0000 | core_unexplained_candidate_tail_flagged |
| crash_premium_traded | 0.0000 | -0.0007 | -0.0439 | 0.9500 | 0.9500 | 0.0200 | 0.0500 | tail_dominated |
| vrp_short_variance_unpriced | 0.0452 | 0.0448 | 2.3014 | 0.9900 | 0.9800 | 0.5900 | 1.0000 | core_unexplained_candidate_tail_flagged |
| vrp_short_variance_traded | 0.0000 | -0.0002 | -0.0138 | 0.9600 | 0.9600 | 0.0200 | 0.0100 | tail_dominated |
| demo_known_truth | 0.0100 | 0.0097 | 0.6789 | 1.0000 | 1.0000 | 0.0000 | 0.0100 | linear_dominated |

## The same fits under the pre-v0.4b statistic

v0.3/v0.4 took the core's standard error from the full design, where the zero-priced controls absorb the non-traded payoff's variation while its in-sample mean stays in the intercept. Same datasets, same point estimates; only the standard error and the threshold differ.

| case | coverage95_population_v03 | coverage95_population | candidate_rate_v03 | candidate_rate | median_se_ratio_v04b_to_v03 |
|---|---|---|---|---|---|
| noise | 0.9700 | 0.9700 | 0.0200 | 0.0200 | 1.0055 |
| actuarial_short_variance | 0.9400 | 0.9900 | 0.0200 | 0.0100 | 1.4371 |
| actuarial_short_put | 0.3300 | 0.9900 | 0.0100 | 0.0000 | 6.9895 |
| peso_short_put | 0.1600 | 0.9200 | 0.1300 | 0.0200 | 6.9895 |
| timing_skill | n/a | n/a | 1.0000 | 1.0000 | 2.3681 |
| regime_alpha | n/a | n/a | 0.8400 | 0.7400 | 1.0247 |
| crash_premium_unpriced | 0.9400 | 0.9500 | 0.9700 | 0.9700 | 0.9895 |
| crash_premium_traded | 0.9400 | 0.9500 | 0.0300 | 0.0200 | 0.9696 |
| vrp_short_variance_unpriced | 0.7700 | 0.9800 | 0.8100 | 0.5900 | 1.4371 |
| vrp_short_variance_traded | 0.9700 | 0.9600 | 0.0200 | 0.0200 | 1.0139 |
| demo_known_truth | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9823 |

## Luck gate: reporting the best of 10 strategy variants

Each seed builds 10 variants on the same market; the best core is reported. `naive_rate`: best variant has core > 0 and t above the fixed-b threshold. `gated_rate`: it also passes Hansen SPA (p < 0.05 across the 10) and DSR >= 0.95 at the effective number of variants.

| case | naive_rate | gated_rate | best_is_genuine_rate | median_best_t | median_spa_p | median_effective_trials |
|---|---|---|---|---|---|---|
| best_of_10_noise | 0.2400 | 0.0100 | 0.0000 | 1.6952 | 0.4160 | 10.0000 |
| best_of_10_one_genuine | 0.9900 | 0.8700 | 1.0000 | 4.4300 | 0.0000 | 10.0000 |

## Population truths of the non-traded payoffs

Unconditional traded-model alpha E[y] - b'E[f], slopes from 2 x 500,000 noise-free days. `coverage95` uses the conditional (in-sample) truths; `coverage95_population` uses these.

| case | population truth | MC s.e. |
|---|---|---|
| actuarial_short_variance | 0.0002 | 0.0001 |
| actuarial_short_put | -0.0618 | 0.0001 |
| peso_short_put | -0.0618 | 0.0001 |
| crash_premium_unpriced | 0.0999 | 0.0000 |
| vrp_short_variance_unpriced | 0.0455 | 0.0001 |

## Against the pre-change truth definition

Before the convention switch, payoffs sold at their actuarial value had truth 0.

| case | original truth | median core | abs diff | original target <= 0.01 |
|---|---|---|---|---|
| actuarial_short_variance | 0.0000 | 0.0000 | 0.0000 | PASS |
| actuarial_short_put | 0.0000 | -0.0453 | 0.0453 | FAIL |

## Target checks

| case | set | metric | target | value | pass | why |
|---|---|---|---|---|---|---|
| noise | v0.2-v0.3 | candidate_rate | <= 0.06 | 0.02 | PASS | false core certification near the nominal one-sided 2.3% |
| noise | v0.2-v0.3 | abs_median_bias | <= 0.0075 | 0.0003349 | PASS | unbiased core |
| noise | v0.2-v0.3 | coverage95 | >= 0.85 | 0.97 | PASS | HAC intervals roughly honest |
| noise | v0.2-v0.3 | flag_rate | <= 0.15 | 0.04 | PASS | tail flags rarely fire on symmetric noise |
| noise | v0.2-v0.3 | max_addup_error | <= 1e-08 | 1.388e-17 | PASS | Shapley contributions + core add up to the raw mean |
| noise | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 6.768e-14 | PASS | core return series has mean = core alpha |
| actuarial_short_variance | v0.2-v0.3 | abs_median_bias | <= 0.01 | 0.0007958 | PASS | non-traded convexity is not alpha (v0.1: +19%/yr) |
| actuarial_short_variance | v0.2-v0.3 | candidate_rate | <= 0.06 | 0.01 | PASS | no certification of a short-vol payoff |
| actuarial_short_variance | v0.2-v0.3 | coverage95 | >= 0.85 | 0.99 | PASS | HAC intervals roughly honest |
| actuarial_short_variance | v0.2-v0.3 | flag_rate | >= 0.8 | 1 | PASS | short convexity is flagged as unpriced tail risk |
| actuarial_short_variance | v0.2-v0.3 | max_addup_error | <= 1e-08 | 1.388e-17 | PASS | Shapley contributions + core add up to the raw mean |
| actuarial_short_variance | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 1.475e-13 | PASS | core return series has mean = core alpha |
| actuarial_short_variance | v0.4b | coverage95_population | >= 0.85 | 0.99 | PASS | decision interval covers the population alpha |
| actuarial_short_put | v0.2-v0.3 | abs_median_bias | <= 0.01 | 0.009053 | PASS | put selling is not alpha (v0.1: +188%/yr) |
| actuarial_short_put | v0.2-v0.3 | candidate_rate | <= 0.06 | 0 | PASS | no certification of a short-put payoff |
| actuarial_short_put | v0.2-v0.3 | coverage95 | >= 0.85 | 1 | PASS | HAC intervals roughly honest |
| actuarial_short_put | v0.2-v0.3 | flag_rate | >= 0.8 | 1 | PASS | short-put exposure is flagged as unpriced tail risk |
| actuarial_short_put | v0.2-v0.3 | max_addup_error | <= 1e-08 | 3.469e-17 | PASS | Shapley contributions + core add up to the raw mean |
| actuarial_short_put | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 3.881e-12 | PASS | core return series has mean = core alpha |
| actuarial_short_put | v0.4b | coverage95_population | >= 0.85 | 0.99 | PASS | decision interval covers the population alpha |
| peso_short_put | v0.2-v0.3 | flag_rate | >= 0.8 | 1 | PASS | regime-luck put selling is handed to the H2 gate |
| peso_short_put | v0.2-v0.3 | clean_candidate_rate | <= 0.1 | 0 | PASS | never a clean certification of put selling |
| peso_short_put | v0.2-v0.3 | max_addup_error | <= 1e-08 | 2.776e-17 | PASS | Shapley contributions + core add up to the raw mean |
| peso_short_put | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 3.881e-12 | PASS | core return series has mean = core alpha |
| peso_short_put | v0.4b | coverage95_population | >= 0.85 | 0.92 | PASS | rare-crash put: interval covers the population alpha (v0.3 statistic ~15%) |
| peso_short_put | v0.4b | candidate_rate | <= 0.05 | 0.02 | PASS | rare-crash put is almost never a candidate, flagged or not (v0.3: 16%) |
| timing_skill | v0.2-v0.3 | candidate_rate | >= 0.9 | 1 | PASS | timing skill is alpha and is detected (v0.1: -14%/yr) |
| timing_skill | v0.2-v0.3 | median_ratio_to_truth | >= 0.75 | 0.9989 | PASS | core close to the timing term |
| timing_skill | v0.2-v0.3 | median_ratio_to_truth | <= 1.25 | 0.9989 | PASS | core close to the timing term |
| timing_skill | v0.2-v0.3 | flag_rate | <= 0.15 | 0 | PASS | convex payoffs do not raise tail flags |
| timing_skill | v0.2-v0.3 | max_addup_error | <= 1e-08 | 0 | PASS | Shapley contributions + core add up to the raw mean |
| timing_skill | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 7.152e-13 | PASS | core return series has mean = core alpha |
| regime_alpha | v0.2-v0.3 | median_alpha_on | >= 0.06 | 0.08252 | PASS | regime alpha recovered (truth 8%) |
| regime_alpha | v0.2-v0.3 | median_alpha_on | <= 0.1 | 0.08252 | PASS | regime alpha recovered (truth 8%) |
| regime_alpha | v0.2-v0.3 | abs_median_alpha_off | <= 0.015 | 0.0002155 | PASS | no alpha off-regime (truth 0) |
| regime_alpha | v0.2-v0.3 | diff_sig_rate | >= 0.6 | 0.85 | PASS | on-minus-off detected (power) |
| regime_alpha | v0.2-v0.3 | coverage95 | >= 0.85 | 0.98 | PASS | unconditional core = 8% x share of high-vol days |
| regime_alpha | v0.2-v0.3 | max_addup_error | <= 1e-08 | 1.388e-17 | PASS | Shapley contributions + core add up to the raw mean |
| regime_alpha | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 7.87e-14 | PASS | core return series has mean = core alpha |
| regime_alpha | v0.4b | candidate_rate | >= 0.65 | 0.74 | PASS | power kept on genuine regime alpha (v0.3: 80%; expected ~75% at the fixed-b threshold) |
| crash_premium_unpriced | v0.2-v0.3 | flag_rate | >= 0.9 | 1 | PASS | an unpriceable crash premium is handed to the H2 gate |
| crash_premium_unpriced | v0.2-v0.3 | clean_candidate_rate | <= 0.1 | 0 | PASS | never a clean certification of a crash premium |
| crash_premium_unpriced | v0.2-v0.3 | max_addup_error | <= 1e-08 | 0 | PASS | Shapley contributions + core add up to the raw mean |
| crash_premium_unpriced | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 1.772e-13 | PASS | core return series has mean = core alpha |
| crash_premium_unpriced | v0.4b | coverage95_population | >= 0.85 | 0.95 | PASS | decision interval covers the population alpha |
| crash_premium_traded | v0.2-v0.3 | candidate_rate | <= 0.06 | 0.02 | PASS | a priced crash premium is not alpha |
| crash_premium_traded | v0.2-v0.3 | tail_dominated_share | >= 0.8 | 0.98 | PASS | the traded tail block explains the mean |
| crash_premium_traded | v0.2-v0.3 | abs_median_bias | <= 0.01 | 0.0006742 | PASS | core ~ 0 once the premium is priced |
| crash_premium_traded | v0.2-v0.3 | flag_rate | <= 0.15 | 0.05 | PASS | priced tail risk does not raise unpriced-risk flags |
| crash_premium_traded | v0.2-v0.3 | max_addup_error | <= 1e-08 | 2.776e-17 | PASS | Shapley contributions + core add up to the raw mean |
| crash_premium_traded | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 1.734e-13 | PASS | core return series has mean = core alpha |
| vrp_short_variance_unpriced | v0.2-v0.3 | flag_rate | >= 0.8 | 1 | PASS | an unpriceable variance premium is handed to the H2 gate |
| vrp_short_variance_unpriced | v0.2-v0.3 | clean_candidate_rate | <= 0.1 | 0 | PASS | never a clean certification of a variance premium |
| vrp_short_variance_unpriced | v0.2-v0.3 | max_addup_error | <= 1e-08 | 3.469e-18 | PASS | Shapley contributions + core add up to the raw mean |
| vrp_short_variance_unpriced | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 1.474e-13 | PASS | core return series has mean = core alpha |
| vrp_short_variance_unpriced | v0.4b | coverage95_population | >= 0.85 | 0.98 | PASS | decision interval covers the population alpha |
| vrp_short_variance_traded | v0.2-v0.3 | candidate_rate | <= 0.06 | 0.02 | PASS | a priced variance premium is not alpha |
| vrp_short_variance_traded | v0.2-v0.3 | abs_median_bias | <= 0.01 | 0.0001928 | PASS | core ~ 0 once the VRP is priced by a traded swap |
| vrp_short_variance_traded | v0.2-v0.3 | tail_dominated_share | >= 0.8 | 0.72 | FAIL | the traded tail block explains the mean |
| vrp_short_variance_traded | v0.2-v0.3 | flag_rate | <= 0.15 | 0.01 | PASS | priced convexity does not raise unpriced-risk flags |
| vrp_short_variance_traded | v0.2-v0.3 | max_addup_error | <= 1e-08 | 2.776e-17 | PASS | Shapley contributions + core add up to the raw mean |
| vrp_short_variance_traded | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 7.815e-14 | PASS | core return series has mean = core alpha |
| demo_known_truth | v0.2-v0.3 | abs_median_bias | <= 0.01 | 0.0002845 | PASS | net core 1%/yr recovered with all blocks active |
| demo_known_truth | v0.2-v0.3 | coverage95 | >= 0.85 | 1 | PASS | HAC intervals roughly honest |
| demo_known_truth | v0.2-v0.3 | flag_rate | <= 0.15 | 0.01 | PASS | long convexity + traded tail factors: no unpriced flag |
| demo_known_truth | v0.2-v0.3 | max_addup_error | <= 1e-08 | 1.388e-17 | PASS | Shapley contributions + core add up to the raw mean |
| demo_known_truth | v0.2-v0.3 | max_core_returns_error | <= 1e-08 | 9.353e-14 | PASS | core return series has mean = core alpha |
| best_of_10_noise | v0.3 luck | gated_rate | <= 0.06 | 0.01 | PASS | reporting the best of 10 zero-alpha variants is not certified (naive t>=2 is not reported as a target) |
| best_of_10_one_genuine | v0.3 luck | gated_rate | >= 0.8 | 0.87 | PASS | a 6%/yr core (t ~ 4) among 10 variants still clears the gate |
| best_of_10_one_genuine | v0.3 luck | best_is_genuine_rate | >= 0.9 | 1 | PASS | the best core is the genuine variant |
