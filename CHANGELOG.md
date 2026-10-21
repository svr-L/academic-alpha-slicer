# v0.5.0 — Alpha Survival Curve / expanding priced-factor span

- Added `SliceResult.alpha_survival`: a pre-declared model-depth path that asks how much apparent alpha survives as the **priced** span expands.
- Default hierarchy, when columns exist: `RAW_NET → MKT → FF3 → FF5 / FF3+MOM → FF5+MOM → ALL_LINEAR → +CONDITIONAL → +TRADED_TAIL`. Missing families are skipped and duplicate designs are collapsed.
- Every stage reports annualized intercept, selected-policy HAC SE/t/CI, DSR, stage screen, R²/BIC/condition number, incremental erosion, cumulative erosion and signed survival ratios.
- The survival curve is **not assumed monotone**. Adding a traded factor can raise alpha when the strategy is exposed to a negative-premium factor or when omitted-variable bias changes the intercept.
- Non-traded nonlinear/tail-shape controls are excluded from survival by construction. They can diagnose payoff shape but cannot become a priced slice merely because their sample mean is nonzero.
- Added summary diagnostics: `alpha_survival_final_vs_raw`, `alpha_survival_span_fragile`, first failure stage, deepest passing stage, final screen status and stage count.
- Added custom pre-declared factor hierarchies through `survival_custom_linear_spans`. Custom paths must be nested; users who want a fully custom path disable the defaults rather than mixing non-nested alternatives.
- Added `plot_alpha_survival` and exposed survival tables through `report_tables`.
- Preserved the v0.4c inference-policy layer unchanged. `alpha_survival_span_fragile` is **model/span** sensitivity; `profile_sensitive` / `inference_frontier` remain **inference-standard** sensitivity.
- Regression tests verify that the final survival intercept equals both the full traded-span core and the Shapley `core_unexplained` term exactly.

# v0.4c — explicit inference policy and power/robustness frontier

- Added `inference_profile={discovery,balanced,certification,custom}`. The selected profile drives the Slicer's candidate label; `balanced` exactly reproduces v0.4b.
- `discovery`: HAC(63) with conventional critical values, intentionally higher power.
- `balanced`: HAC(126) with Kiefer–Vogelsang fixed-b critical values.
- `certification`: balanced HAC plus same-sign stationary-bootstrap percentile-CI agreement. It refuses to run as the primary policy if bootstrap inference is disabled.
- Added `inference_profile_precommitted`, an explicit audit declaration recording whether the primary standard was chosen before seeing the result.
- Added `profile_comparison`: all three standard policies are always evaluated side by side, with DSR and candidate status. A custom primary is shown in addition, never instead of the standard presets.
- Added `inference_frontier`: conventional and fixed-b HAC at 21/63/126 lags plus stationary bootstrap, with an evidence ratio relative to each specification's required threshold.
- Added `profile_sensitive` and `inference_frontier_robustness_score` summaries. The latter is descriptive, not a new test.
- Preserved the exact v0.4b balanced t/SE in `core_alpha_t_v04b_balanced` / `core_alpha_se_ann_v04b_balanced` for auditability.
- No change to pricing, Shapley attribution, non-traded conventions, tail flags, DSR/SPA machinery, or the external Tail-Aware Referee verdict logic.

# Changelog

## 0.4.1 (v0.4b): the core's standard error, corrected

### The problem

Up to v0.4 the core's t-stat came from the **full design**: the traded blocks plus the zero-priced non-traded controls (`m²`, `m³`, the downside and tail terms). The controls are orthogonal to the constant and to every traded regressor, so the alpha did not move; that is the point of the v0.2 pricing rule. But they stayed in the regression that measures the alpha's uncertainty:

- they explain most of an option-like payoff's day-to-day variation, so the residuals are small;
- the payoff's in-sample mean, which is part of the core's estimation error (how many crashes the sample happened to contain), sits in the intercept, where those residuals cannot see it.

The standard error therefore missed the main source of error for exactly the strategies the slicer is meant to catch. On the rare-crash put (`peso_short_put`) it was 1.9%/yr against an actual error of about 15%/yr, and the interval covered the population alpha in 16% of samples. A longer HAC bandwidth on the same design changed nothing (ratio 0.95–1.00 in all 11 cases), so v0.4's bandwidth audit could not see the problem, and `inference_fragile` flagged none of the rare-crash put's candidates.

The suite's coverage targets had passed because they were measured against the **conditional** (in-sample) truth, which already contains the sample's crash luck. The v0.2 design put the controls in that regression, and the v0.2–v0.4 documents described the symptom as a bandwidth problem ("HAC(21) understates uncertainty with persistent regimes"). That diagnosis was wrong, and it led v0.4 to look at the bandwidth.

### The fix: an explicit change of the decision statistic

- **Core t, standard error and interval** come from the **traded-only spanning regression**. Its intercept equals the core (Frisch–Waugh–Lovell; checked in every fit), but its residuals keep the non-traded payoff's variation. Bartlett HAC with `core_hac_lags=126` and a Kiefer–Vogelsang **fixed-b** critical value (`fixed_b_critical_value`, checked against simulation). The "t ≥ 2" rule is scaled by cv(b)/1.96, about 2.13 on 2,700 days.
- **The same statistic** feeds the luck gate, `core_significantly_negative`, the specification sweep and the naive rule of the luck family.
- **Waterfall and regime-alpha table** use the same bandwidth and critical value. The regime table leaves out the zero-priced controls: they are not orthogonal to the state, so inside that regression they would be charged a regime-dependent price.
- **`core_inference`** is computed on the traded-only design. `inference_fragile` uses the decision family only: the HAC rows with fixed-b and the bootstrap. Prewhitened HAC and the pre-v0.4b statistic (`v0.3 full design HAC(21)`) are shown for comparison.
- **New summary fields:**
  - `core_alpha_se_ann`, `core_ci_low_ann`, `core_ci_high_ann`;
  - `core_t_critical`, `core_ci_critical`, `core_hac_lags_used`, `core_se_method`;
  - for comparison, `core_alpha_t_v03`, `core_alpha_se_ann_v03` and `core_se_ratio_vs_v03`.

  `core_alpha_p` is size-adjusted to the fixed-b critical value.
- **Faster bootstrap.** All index paths are drawn at once and the blocks are converted to arrays once. The method is the same.

### Validation: pre-registered, on fresh seeds 1000–1099, run once

The targets were written and hashed before the run (`outputs/PREREGISTRATION_V04B.txt`). The development seeds 0–99 had been seen while diagnosing the problem. Results are in [`outputs/MC_SUITE.md`](outputs/MC_SUITE.md).

**69/70 targets pass**, including all 7 new v0.4b targets. The miss is the same as before: in the traded-VRP case the label share is 72% against 80%.

On the same fits, the pre-v0.4b statistic against v0.4b:

| case | interval covers the population alpha | candidate rate |
|---|---|---|
| rare-crash put (`peso_short_put`) | 16% → **92%** | 13% → **2%** |
| put sold at its average payoff | 33% → 99% | 1% → 0% |
| variance premium, no traded swap (always flagged) | 77% → 98% | 81% → 59% |
| alpha only in high-volatility regimes (genuine) | 98% (conditional truth) | 84% → **74%** |
| market timing (genuine) | 100% (conditional truth) | 100% → 100% |
| noise | 97% → 97% | 2% → 2% |

- **The price.** Genuine regime alpha is certified less often (84% → 74%). The long bandwidth and the fixed-b threshold trade a little power for intervals that are honest when the payoff has a non-traded component.
- **Luck family.** The best of 10 zero-alpha variants is certified 1% of the time (target ≤ 6%). A genuine 6%/yr variant hidden among 10 is certified 87% of the time (target ≥ 80%) and is the best of the 10 in 100% of runs.

### Persistence stress test, rerun

This test is separate from the suite. The population alpha is zero and the conditional mean follows a two-state Markov chain. Seeds 1000–1399, 400 per world; Wilson intervals are in [`outputs/INFERENCE_VALIDATION.md`](outputs/INFERENCE_VALIDATION.md). Each cell is the false-rejection rate at the nominal 5%:

| world | HAC(21) | HAC(126) | HAC(126), fixed-b | prewhitened | bootstrap, block 126 |
|---|---:|---:|---:|---:|---:|
| iid | 4.5% | 5.0% | 4.2% | 4.2% | 7.2% |
| regimes ~100 days (v0.4 design) | 33.2% | 16.2% | **12.0%** | 42.2% | 16.8% |
| regimes ~200 days | 44.5% | 21.8% | **19.0%** | 51.7% | 19.5% |

- **Bandwidth.** A long bandwidth halves the error of HAC(21) but does not reach 5% when regimes last about as long as the bandwidth. Fixed-b takes back a few points; longer regimes push the error up again.
- **Prewhitening.** It makes things worse. Daily returns here are mostly noise, so their lag-1 autocorrelation is about 0.08, and a VAR(1) on the scores sees almost none of the persistence.
- **What this tests.** It is a stress test of the bandwidth. The suite's rare-crash put had a different problem, fixed above.

### Decomposition bootstrap: first coverage check

The opt-in bootstrap of the whole decomposition had no coverage test in v0.4. Here it runs with 200 draws and mean block 63 on seeds 1000–1099, against targets written before the run (`BOOTSTRAP_TARGETS` in `alpha_slicer/inference_validation.py`):

| case | coverage of the core interval | target | median interval width |
|---|---:|---|---:|
| rare-crash put (population alpha) | 91% | ≥ 80% | 49.7%/yr |
| alpha only in high-volatility regimes | 97% | ≥ 85% | 5.4%/yr |
| noise | 96% | ≥ 85% | 5.2%/yr |

The bootstrap refits only traded blocks, so it had been measuring the rare-crash put's uncertainty correctly all along. It was simply off by default and outside the decision rule.

### Corrections to earlier documents

- **v0.4 "HAC(126) and the 126-day bootstrap: 7.5% false rejections".** This came from 40 seeds; the Wilson interval for 3 of 40 is 2.6–19.9%. See the stress test above.
- **v0.2–v0.4 "HAC(21) understates the uncertainty of persistent regimes (peso case)".** The cause was the controls in the standard-error regression, not the bandwidth.
- **v0.3 README "95% coverage 92–100%".** This was measured against the in-sample truth. Against the population alpha, the rare-crash put was at 16% and the actuarial put at 33%.

### Breaking changes

- **`summary['core_alpha_t']`** is the v0.4b statistic. The old one is `core_alpha_t_v03`.
- **Candidate threshold.** It is about 2.13 instead of 2.0 (fixed-b at 126 lags). It scales with the sample length through `core_ci_critical`.
- **`conditional_alpha`.**
  - The estimates exclude the zero-priced controls.
  - The standard errors use `core_hac_lags`.
  - New `critical_value` column.
- **`core_inference`.**
  - Computed on the traded-only design.
  - New `critical_value` and `decision_family` columns.
  - New `v0.3 full design HAC(...)` row.
- **The suite's evaluation seeds** are now 1000–1099 (CLI default).
- **The `inference_validation` summary** is one row per world and method, with Wilson intervals.

## 0.4.0: dependence-aware inference for the full alpha decomposition

- **Core inference grid.** `SliceResult.core_inference` compares HAC(21), HAC(63), HAC(126), an automatic Newey-West bandwidth, VAR(1) score-prewhitened HAC, and (when enabled) the stationary bootstrap. The point alpha is unchanged; only uncertainty is audited.
- **Full decomposition bootstrap.** `SliceResult.bootstrap_inference` uses a Politis–Romano stationary bootstrap on the common estimation sample and refits every traded-block spanning regression / exact Shapley subset in every replication. It reports uncertainty for each priced premium, the core, raw/gross mean and implementation drag.
- **No silent rule change.** The v0.3 HAC(21)+DSR/SPA candidate/luck rule remains frozen. v0.4 surfaces `inference_fragile` and related summary fields rather than changing labels without a separately validated rule.
- **Persistence validation.** A fresh 40-seed/world stress test (`outputs/INFERENCE_VALIDATION.md`) gives false-rejection rates of 30% for HAC(21) versus 7.5% for HAC(126) and 7.5% for a 126-day stationary bootstrap in a persistent zero-alpha Markov world; all methods are close to nominal in the iid control. Score prewhitening is deliberately reported as *not* curing the regime-duration problem (40% false rejection).
- **Reporting.** Added `plot_bootstrap_attribution`, `demo_core_inference.csv`, `demo_bootstrap_inference.csv`, and an inference-validation CLI (`python -m alpha_slicer.inference_validation`).
- **Tests.** Added exact bootstrap add-up, stationary-block continuity, AR(1) prewhitened-HAC scale and known-truth bootstrap-CI tests.

## 0.3.0: luck gate on the core, referee-faithful adapter

### Fix 4: luck gate on the core

v0.2 certified a core at HAC t ≥ 2 on whatever specification was reported. Reporting the best of several specifications or strategy variants is a search, and t ≥ 2 on the best of 10 zero-alpha variants happens about a quarter of the time.

- **`alpha_slicer/luck.py`** provides:
  - the probabilistic and deflated Sharpe ratio (Bailey & López de Prado), including the expected maximum Sharpe ratio of N null trials;
  - the Li–Ji effective number of trials;
  - the Politis–Romano stationary bootstrap;
  - Hansen's SPA test (consistent, lower and upper p-values; upper = White's Reality Check);
  - `multi_core_gate` for the best of several cores.
- **Single specification.** A candidate now needs HAC t ≥ 2 **and** a deflated Sharpe ratio ≥ 0.95 at `SliceConfig.luck_n_trials` declared tries (default 1, where the DSR reduces to the PSR against zero). New `summary` fields: `core_sharpe_ann`, `core_psr`, `core_dsr`, `luck_n_trials`, `luck_sr_hurdle_ann`, `luck_gate_pass`.
- **Across specifications: `spec_sweep`.** It fits a grid of specifications (default: with/without the conditional block, two alternative state definitions, with/without the traded tail factors), aligns the core return series and gates the best one. The best core must pass SPA across all specifications (p < 0.05) and have a DSR ≥ 0.95 at the effective number of specifications times `n_trials_extra`. The resulting labels are `core_survives_luck_gate`, `core_survives_luck_gate_tail_flagged`, `core_not_robust_to_specification_search` and `no_candidate`.
- **Monte Carlo luck family.** It covers the best of 10 strategy variants, all zero-alpha or one with a genuine 6%/yr core. Its targets were fixed before the first run (`LUCK_TARGETS`). Results over 100 seeds:
  - all zero-alpha: the naive rule certifies the best variant 20% of the time, the gate **0%** (target ≤ 6%);
  - one genuine variant: the gate certifies it **83%** of the time (target ≥ 80%), and the best variant is the genuine one in 99% of seeds (target ≥ 90%).
- **Candidate counts in the main suite.** `candidate_rate` now counts the slicer's own candidate labels, so it includes the luck gate. The full suite passes **62/63** targets; the miss is the same label share as in v0.2. Single-specification candidate rates drop where the core's returns are skewed or fat-tailed: the peso put falls from 32% to 16%, the actuarial put from 6% to 1%.

### Fix 5: the Tail-Aware adapter calls the referee's own functions

- `run_tailaware_sp100_live` loads the referee's F0 engine — the notebook, or its `.py` export — without running its pipeline (`load_referee_core`, `find_referee_core`). It then calls the referee's `load_panel` → `build_returns` → `build_ladder` → `build_signals` → `backtest` at the referee's verdict level. On the referee's offline panel the P&L is identical to the referee's own `run_f0` output.
- **The v0.2 fork never ran.** It used a log-residual sum, ±0.5/k weights and its own beta loop, and crashed at its default settings (`min_periods` 242 larger than the 231-day window). Patched, it still gave a different P&L (Sharpe 1.61 vs 1.47, half the volatility, correlation 0.92). It is removed.
- **Opt-in parity test.** `tests/test_luck_and_adapter.py::test_tailaware_adapter_matches_the_referee_pipeline` runs when `TAIL_AWARE_CORE` is set.

## 0.2.0: pricing-convention fixes and a known-truth Monte Carlo suite

### Fix 1: non-traded regressors carry a zero price

v0.1 charged every regressor at its raw sample mean. That mean is a price only for **traded** excess returns. For `m²`, `m³`, `min(m,0)` and the tail dummies it is not a price, and charging it moved the intercept by `γ·E[x]`. This created artificial alpha (Jagannathan–Korajczyk 1986; Glosten–Jagannathan 1994):

| case (20 seeds, review) | v0.1 core | truth |
|---|---|---|
| short variance at actuarial value | +19%/yr, certified 20/20 | ~0 |
| short OTM put at actuarial value | +188%/yr, certified 20/20 | negative (see below) |
| genuine timing skill | −14%/yr, "linear_dominated" | positive |

What changed in v0.2:

- **Pricing type per block.** `linear`, `conditional` and `tail` are traded and charged at their mean. `nonlinear` and `tail_shape` are non-traded.
- **Non-traded blocks are orthogonalized** to the constant and to every traded regressor in the model. Alpha is therefore the spanning-regression alpha on the traded factors. By Frisch–Waugh–Lovell, the non-traded blocks explain variance and tail shape but cannot move alpha, whatever basis is chosen. The waterfall, the Shapley attribution and the core are computed on the traded blocks; non-traded blocks get exactly 0.
  - Why orthogonalize rather than just de-mean (the first attempt)? A de-meaned basis absorbs part of an option's market delta, so the delta escapes the market-premium charge. The result then depends on the basis: on the same short-put sample, alpha moved from +5% to 0% when hinge functions were added.
  - De-meaning also breaks when a traded convexity proxy is supplied: `m²` and a variance swap are almost collinear.
- **Unpriced tail-risk flags** that hand off to the referee:
  - `short_convexity_tm` (Treynor–Mazuy t ≤ −2);
  - `excess_left_tail_beta` (left-tail market beta t ≥ 2);
  - `left_tail_asymmetry` (|ES 2.5%| / ES 97.5% of the core return ≥ 1.5).

  All are measured after the traded blocks, so a supplied traded tail factor switches them off.
- **New labels:** `core_unexplained_candidate`, `core_unexplained_candidate_tail_flagged`, `<block>_dominated`, `inconclusive`.
- **New `referee_handoff` field** names the next referee gate:
  - a flagged core goes to the hidden-risk gate (H2) first;
  - a clean core goes to the luck gate **and** H2. In-sample flags only see tail risk that showed up in the P&L: on the referee's own H2 world they flagged 17/20 hidden-crash strategies, and 2/20 came out clean.
- **New `nontraded` table.** Per non-traded term it gives the loading, its t-stat, the raw mean (marked as *not* a price), the charge v0.1 would have applied, and the break-even premium that would kill the core (also as a multiple of the raw mean). Per block it gives a HAC Wald test and the incremental R².
- **Sensitivity fields in `summary`:**
  - `core_alt_demeaned_controls_ann` (de-meaned convention);
  - `nontraded_naive_charge_ann`;
  - `core_if_nontraded_charged_at_raw_mean_ann` (the v0.1 convention on the same design).
- **`nontraded_premia={term: premium}`** is a what-if that prices a non-traded term at an assumed premium.
- **`proxies.variance_swap_pnl`** builds a traded long-variance-swap P&L from the market and an implied vol (e.g. VIX), to pass as `tail_factors`.

**Consequence for "fair" option payoffs.** A put sold at its expected payoff is *underpriced* relative to the traded model: its delta carries market risk that the market premium prices. Its traded-model alpha is about −6%/yr in the suite's world, not 0. A put seller earns positive alpha only if the option premium exceeds that, and it is then flagged.

### Fix 2: conditional block and lagged states

- **Lagged states.** States, both built and supplied, are lagged by `state_lag=1`, so the value applied to the return of date t is known at the close of t−1. In v0.1 the realized vol and drawdown used day t itself. Warm-up dates are NaN instead of being silently "off".
- **No state dummies in the premium block.** `STATE_*` dummies were removed: a state dummy is time-varying alpha, not a premium.
- **Centred managed portfolios.** The conditional block holds `F_k × (z_{t−1} − z̄)`: traded managed portfolios that capture Cov(β_t, λ_t) given public information.
- **Regime-alpha table.** `conditional_alpha` is now a Christopherson–Ferson–Glassman regime-alpha table: `y = a0 + a1·z_{t−1} + b'X + e` on the full design, reporting alpha off / on / on−off / time-weighted with HAC errors. In v0.1 it reported the core residual's mean by regime. That residual already had the dummy slopes removed, so the means were identical by construction (0.009506 in both regimes on the demo).

### Fix 3: known-truth Monte Carlo suite

- `alpha_slicer/montecarlo.py` has 11 cases: noise; actuarial short variance; actuarial short put; peso short put; timing skill; regime alpha; crash premium unpriced and traded; VRP unpriced and traded; demo.
- Targets were written before the evaluation run. The truth definitions, and the changes made after smoke tests on non-evaluation seeds, are logged in the module docstring.
- Run it with `python -m alpha_slicer.montecarlo --seeds 100 --jobs 2 --out outputs`. Results are in `outputs/MC_SUITE.md`.
- `tests/test_montecarlo_quick.py` is a 6-seed smoke version for `pytest`.

### Before / after (identical datasets, seeds 0–19; `outputs/before_after_v01_v02.csv`)

| case | truth | v0.1 | v0.2 |
|---|---|---|---|
| short variance, no premium | −0.1% | +17.5% (certified 100%) | −0.1% (certified 0%, flagged 100%) |
| short put sold at average payoff | −5.6% | +197% (certified 100%) | −4.8% (certified 0%, flagged 100%) |
| timing skill | +63.9% | −15.0% | +63.5% (certified 100%) |
| variance premium, traded swap supplied | 0 | +10.7% (certified 95%) | −0.3% |
| referee H2 world, no crash factor | raw 15.3% | 42.5% (clean 75%) | 14.4% (flagged 85%, clean 10%) |

### Other changes

- **Demo.** All exposures are traded, so the truth is exact (net core 1.0%/yr):
  - the quadratic leg is now a traded variance swap that pays a 25% variance premium;
  - SHORTVOL carries a positive net premium;
  - the MOM premium is higher after high-vol days, which creates a genuine conditional premium.
- **One common estimation sample** for every regression, so all attributions add up exactly.
- **New `AlphaSlicer.prepare()`** returns the priced design on the common sample. The suite uses it for population truths.
- **HAC tooling.** `hac_ols` returns the HAC covariance; `wald_test`, `linear_combination` and `orthogonalize` were added.
- **QuantReg warnings.** Iteration-limit warnings are silenced inside `quantile_beta`.

### Breaking changes

- **Block names.** `tail` now holds only supplied traded tail factors. The non-traded downside basis moved to `tail_shape`. The default `block_order` is `("linear", "conditional", "tail", "nonlinear", "tail_shape")`.
- **Waterfall.** It now has one row per traded block.
- **Shapley.** It includes zero rows for the non-traded blocks.
- **`conditional_alpha` columns.** They changed from `mean_ann, t, n` to `alpha_ann, se_ann, t, p, n_days, share_of_days`.
- **Label.** `core_unexplained_candidate` can now also appear as `core_unexplained_candidate_tail_flagged`.

### Not in this release (open in v0.2, closed in v0.3)

- A luck gate on the core → v0.3 fix 4.
- The Tail-Aware adapter fork → v0.3 fix 5.
- HAC(21) understates uncertainty when payoffs depend on persistent regimes; see `peso_short_put` (still open).
