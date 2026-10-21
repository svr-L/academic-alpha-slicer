# Academic Alpha Slicer

**Asset-pricing diagnostics that split a strategy's apparent alpha into priced premia (static, conditional, tail) and a residual core. When the core comes with unpriced tail risk, it is flagged and handed to the Tail-Aware Alpha Referee.**

This project is a research layer above the **Tail-Aware Alpha Referee**. The referee asks *whether* a signal deserves alpha certification; the slicer asks *where the apparent return comes from*.

> v0.2 fixes the pricing convention of v0.1, which manufactured alpha from non-traded regressors. v0.3 adds a **luck gate** on the core and a referee-faithful Tail-Aware adapter. **v0.4 adds dependence-aware inference for the core and for the full decomposition.** **v0.4b corrects the core's standard error** by moving inference to the traded-only spanning regression. **v0.4c makes the power / false-positive trade-off explicit** through `discovery`, `balanced`, `certification`, and `custom` policies. **v0.5 adds the Alpha Survival Curve:** a pre-declared expanding priced-factor span (MKT → FF3 → FF5/MOM when available → all linear → conditional → traded tail) that shows how much alpha survives as the asset-pricing model gets richer. The curve is deliberately separate from the inference frontier: one changes the economic model, the other changes the uncertainty standard. See [CHANGELOG.md](CHANGELOG.md).

## Research question

> When an academic anomaly or systematic strategy appears to earn alpha, how much is compensation for priced risk (static factor premia, conditional premia, tail premia), how much is implementation drag, and what core survives?

## Pricing convention: what can be removed from alpha

Alpha is defined relative to prices. A premium can be subtracted only if it has a price, and only **traded excess returns** have one: their mean. The slicer therefore types every block:

| block | content | pricing |
|---|---|---|
| `linear` | supplied factors | traded: charged at the factor means |
| `conditional` | `F_k × (z(t−1) − z̄)` managed portfolios on **lagged** public states | traded: charged at their means |
| `tail` | supplied traded tail/convexity P&L (variance swap, put-writing, crash factor) | traded: charged at their means |
| `nonlinear` | `m², m³` | non-traded: **zero price** |
| `tail_shape` | `min(m,0)`, `m·1{m≤q}`, `m²·1{m≤q}` | non-traded: **zero price** |

**Non-traded blocks are orthogonalized** to the constant and to every traded regressor. Whatever part of a non-linear payoff is spanned by traded factors is priced by those factors; the rest carries a zero price. By Frisch–Waugh–Lovell the core alpha equals the spanning-regression alpha on the traded factors, whatever non-traded basis is used. Non-traded blocks explain variance and tail shape, and they feed the flags, but they never move alpha. This closes three failure modes:

1. **Charging a raw mean as if it were a premium.** `γ·E[m²]` is not a price. v0.1 turned a short-variance payoff with no premium into +17.5%/yr of "alpha", and a short put into +197%/yr. It also turned genuine timing skill into −15%/yr (table B below).
2. **Delta hiding.** A merely de-meaned basis absorbs part of an option's market delta, so the delta escapes the market-premium charge, and alpha depends on the chosen basis.
3. **Collinearity with traded proxies.** With a traded variance swap supplied, `m²` is fully spanned and drops out instead of competing with it.

In the four-term decomposition `E[r] = α + E[β]E[λ] + Cov(β_t, λ_t) + Cov(β_t, u_t)`:

- the linear block removes the static premium `E[β]E[λ]`;
- the conditional block removes the conditional premium `Cov(β_t, λ_t)` earned by moving exposure with lagged public information;
- the core keeps selectivity `α` plus genuine timing `Cov(β_t, u_t)`.

**Consequence.** A non-linear premium (variance, crash, put-writing) is removed **only if you supply a traded proxy** for it. Otherwise it stays in the core, and the slicer raises an unpriced-tail-risk flag. That is deliberate: the slicer will not pretend to know a price it does not have.

## Outputs

| output | what it answers |
|---|---|
| `summary` | raw → net, core alpha with its standard error, t and interval (traded-only regression, HAC(126), fixed-b), label, flags, referee hand-off, convention sensitivities, the pre-v0.4b t for comparison |
| `waterfall` | sequential erosion over the traded blocks (declared order) |
| `shapley` | order-robust attribution of the mean over the traded blocks (non-traded get exactly 0) and its order sensitivity |
| `conditional_alpha` | regime alphas with the state on the alpha side: off / on / on−off / time-weighted, on the traded design, HAC(126) with fixed-b p-values |
| `nontraded` | loading on each unspanned non-linear term, the raw mean v0.1 would have charged, and the break-even premium that would kill the core |
| `tail_flags` | `short_convexity_tm`, `excess_left_tail_beta`, `left_tail_asymmetry`, measured after the traded blocks |
| `tail_diagnostics` | Tail-Aware-style downside / q10 beta, coskewness, CDaR beta, ES, max drawdown of the core return |
| `nonlinear_order` | linear vs quadratic vs cubic dependence (R², BIC, t). Alpha is identical across orders by construction |
| `nested_models`, `coefficients` | model audit with HAC inference and condition numbers |
| `summary` luck fields | `core_sharpe_ann`, `core_psr`, `core_dsr` at `luck_n_trials` tries, `luck_sr_hurdle_ann`, `luck_gate_pass` |
| `spec_sweep(...)` | the slicer under a grid of specifications; Hansen SPA and DSR at the effective number of specifications gate the best core |
| `core_inference` | same core intercept on the traded-only design under HAC(126/21/63) with fixed-b, automatic HAC and (when enabled) the stationary bootstrap; prewhitened HAC and the pre-v0.4b statistic for comparison |
| `bootstrap_inference` | uncertainty on the additive decomposition itself: raw/net mean, each traded Shapley premium, core alpha and implementation drag |
| `profile_comparison` | v0.4c power/robustness audit: `discovery`, `balanced`, `certification` side by side, with HAC rule, bootstrap agreement, DSR and candidate decision |
| `inference_frontier` | sensitivity surface over conventional/fixed-b HAC bandwidths plus the stationary bootstrap; reports an evidence ratio and pass/fail under each standard |
| `alpha_survival` | v0.5 model-depth audit: raw mean and intercept as the **priced** span expands through canonical factor families, conditional managed portfolios and traded tail factors; includes primary-policy CI, DSR screen, erosion and survival ratios |

**Labels:**

- `core_unexplained_candidate` — positive core that passes the luck gate (t above the fixed-b threshold, about 2.13 at 126 lags, and deflated Sharpe ≥ 0.95 at `luck_n_trials` declared tries), no flags;
- `core_unexplained_candidate_tail_flagged` — the same, but with unpriced tail exposure;
- `<block>_dominated` — a traded block explains at least 35% of the raw mean;
- `inconclusive`.



## v0.5 Alpha Survival Curve — how deep is the alpha?

The v0.4c inference frontier asks: **does the conclusion survive stricter uncertainty standards?**
The v0.5 survival curve asks a different question: **does the alpha survive a richer priced-factor span?**

By default the engine follows a fixed, non-optimized hierarchy whenever the required factors are present:

```text
RAW_NET → MKT → FF3 → FF5 / FF3+MOM → FF5+MOM → ALL_LINEAR → +CONDITIONAL → +TRADED_TAIL
```

Missing families are skipped. Duplicate spans are collapsed. Non-traded `m²/m³/downside/tail-shape` controls never enter the curve because their means are not prices. The final stage is therefore exactly the same traded span that defines `core_unexplained_alpha_ann`; this equality is regression-tested.

```python
res = AlphaSlicer(SliceConfig(inference_profile='balanced')).fit(...)
res.alpha_survival[[
    'alpha_ann', 'erosion_from_prior_ann', 'survival_ratio_vs_raw',
    't', 't_threshold', 'core_dsr', 'screen_pass'
]]
```

Interpretation:

- `erosion_from_prior_ann` — how much intercept the newly added priced span removes; it can be **negative** if the new factor has a negative premium or corrects omitted-variable bias; the curve is not forced to be monotone.
- `survival_ratio_vs_raw` — signed fraction of the raw net mean still left in the intercept.
- `screen_pass` — stage-level HAC + DSR screen under the selected policy's bandwidth/critical value. It is diagnostic only; the official full-model candidate still comes from the v0.4c policy layer and then the Tail-Aware hand-off.
- `alpha_survival_span_fragile` — the screen changes across pricing-model depth. This is **model/span fragility**, distinct from `profile_sensitive`, which is inferential fragility.

The default path is intentionally not selected by looking at which model makes alpha disappear. For a custom hierarchy:

```python
cfg = SliceConfig(
    survival_include_default_linear_spans=False,
    survival_custom_linear_spans=(
        ('MKT', ('MKT',)),
        ('MKT+MOM', ('MKT','MOM')),
        ('MKT+MOM+QUALITY', ('MKT','MOM','QUALITY')),
    ),
)
```

Custom spans must be nested; otherwise the engine raises an error. This keeps the object a true **survival path**, not an ex-post factor-set tournament.

Validation details and the known-truth curve are in [`outputs/V05_VALIDATION.md`](outputs/V05_VALIDATION.md) and [`outputs/ALPHA_SURVIVAL_VALIDATION.md`](outputs/ALPHA_SURVIVAL_VALIDATION.md).

### v0.4c inference profiles

The primary Slicer label is now controlled by an explicit inference policy:

| profile | rule | intended use |
|---|---|---|
| `discovery` | HAC(63), conventional critical value | higher-power idea screening |
| `balanced` | HAC(126), fixed-b critical value | default research standard; exactly the v0.4b decision rule |
| `certification` | balanced HAC **and** stationary-bootstrap CI agreement | conservative claim / capital-allocation standard |
| `custom` | user-specified HAC/fixed-b/bootstrap requirement | explicit alternative loss preference |

Set `inference_profile_precommitted=True` when the policy was chosen before seeing the result. This is an audit declaration, not something the code can infer. Regardless of the selected profile, `profile_comparison` shows all three standard presets and `inference_frontier` shows the sensitivity surface. `profile_sensitive=True` means the candidate conclusion changes across the standard presets.

This discretion belongs to the **Slicer**, which diagnoses evidence. The Tail-Aware Referee remains a separately frozen certification system.

**`referee_handoff`** names the next gate:

- a flagged candidate goes to the hidden-risk gate (H2) first;
- a clean candidate goes to the luck gate (H1 vs H3) **and still to H2**. The flags see only tail risk that showed up in the strategy's P&L: on the referee's own H2 world they catch 17/20 hidden-crash strategies, and 2/20 come out as clean candidates.

## Validation: Monte Carlo known-truth suite

There are 11 cases × 100 seeds, with n = 3,000 daily observations. Targets were written before each evaluation run; truth definitions and every change made after smoke tests are logged in `alpha_slicer/montecarlo.py`. Since v0.4b the suite runs on **fresh seeds 1000–1099**, with the v0.4b targets written and hashed before that run (`outputs/PREREGISTRATION_V04B.txt`). Full results are in [`outputs/MC_SUITE.md`](outputs/MC_SUITE.md).

**A. v0.4b on 100 fresh seeds per case: 69/70 pre-registered targets pass** (60 v0.2/v0.3 targets, 7 v0.4b targets, 3 luck-family targets)

| case | truth (median) | core (median) | candidate | flagged | covers in-sample truth | covers population alpha |
|---|---|---|---|---|---|---|
| noise | 0 | −0.0% | 2% | 4% | 97% | 97% |
| short variance, no premium | −0.1% | 0.0% | 1% | 100% | 99% | 99% |
| short put sold at average payoff | −5.5% | −4.5% | 0% | 100% | 100% | 99% |
| short put, crashes rare in sample (peso) | −7.9% | −6.7% | 2% | 100% | 100% | 92% |
| timing skill | +64.3% | +63.7% | 100% | 0% | 100% | — |
| alpha only after high-vol days | 4.0% (on 8%, off 0) | 4.0% (on 8.3%, off 0.0%) | 74% | 3% | 98% | — |
| crash premium, no traded factor | +10.4% | +10.3% | 97% | 100% | 100% | 95% |
| crash premium, traded factor supplied | 0 | −0.1% | 2% | 5% | 95% | 95% |
| variance premium, no traded swap | +4.5% | +4.5% | 59% | 100% | 99% | 98% |
| variance premium, traded swap supplied | 0 | −0.0% | 2% | 1% | 96% | 96% |
| demo (all blocks) | +1.0% | +1.0% | 0% | 1% | 100% | 100% |

How to read table A:

- **Population alpha.** It is E[y] − b'E[f], the number an investor cares about. The in-sample truth already contains the sample's luck (how many crashes, which regimes), so covering it says nothing about that luck. Both are reported since v0.4b.
- **The miss.** In the traded-VRP case the label is `tail_dominated` in 72% of seeds against a target of ≥ 80% (73% in v0.3). In the others it is `linear_dominated`, when the realized market premium is as large as the VRP. The core and flag targets for that case pass.
- **Flagged candidates.** The crash and variance premia without a traded factor are real traded-model alpha: the slicer cannot price them, so it certifies none of them clean. Every one is flagged and goes to the referee's hidden-risk gate.

**A2. The same fits under the pre-v0.4b statistic** (full design with the zero-priced controls, HAC(21), t ≥ 2)

| case | covers population alpha: v0.3 → v0.4b | candidate: v0.3 → v0.4b | standard error, v0.4b / v0.3 |
|---|---|---|---|
| short put, crashes rare in sample (peso) | 16% → **92%** | 13% → **2%** | 7.0× |
| short put sold at average payoff | 33% → 99% | 1% → 0% | 7.0× |
| variance premium, no traded swap | 77% → 98% | 81% → 59% | 1.4× |
| alpha only after high-vol days (genuine) | — | 84% → 74% | 1.0× |
| timing skill (genuine) | — | 100% → 100% | 2.4× |
| noise | 97% → 97% | 2% → 2% | 1.0× |

The rare-crash put's actual error across samples is 14.8%/yr. The v0.4b standard error is 13.5%; the pre-v0.4b one was 1.9%. The price of the correction is power on genuine regime alpha (84% → 74%).

**B. Before / after on identical datasets (seeds 0–19): the v0.1 engine vs v0.2**

| case | truth | v0.1 core | v0.2 core | v0.1 certified | v0.2 clean candidate |
|---|---|---|---|---|---|
| short variance, no premium | −0.1% | **+17.5%** | −0.1% | 100% | 0% |
| short put sold at average payoff | −5.6% | **+197%** | −4.8% | 100% | 0% |
| timing skill | +63.9% | **−15.0%** | +63.5% | 0% | 100% |
| alpha only after high-vol days | +4.0% | +0.6% | +3.7% | 5% | 80% |
| crash premium, no traded factor | +10.2% | +16.0% | +10.3% | 75% | 0% (100% flagged) |
| variance premium, traded swap supplied | 0 | **+10.7%** | −0.3% | 95% | 0% |
| demo (all blocks) | +1.0% | −6.7% | +1.2% | 0% | 0% |

The median absolute error of the core falls from 2–200 pp to 0.2–1.6 pp (`outputs/before_after_v01_v02.csv`).

**C. Luck gate: the best of 10 strategy variants (seeds 1000–1099; targets fixed before the first run in v0.3)**

| case | naive: best above the t threshold | gated: + SPA p < 0.05 + DSR ≥ 0.95 at 10 effective tries | best is the genuine variant |
|---|---|---|---|
| 10 zero-alpha variants | 24% | **1%** (target ≤ 6%) | — |
| 9 zero-alpha + 1 with a 6%/yr core (t ≈ 4.4) | 99% | **87%** (target ≥ 80%) | 100% (target ≥ 90%) |

**D. External check: the referee's own H2 world (V1.4 engine, 20 seeds)**

This is a hidden crash premium in a residual-momentum strategy; the raw mean is 15.3%/yr.

| | v0.1 | v0.2 |
|---|---|---|
| core without a crash factor | 42.5%/yr | 14.4%/yr |
| clean candidate | 75% | 10% |
| flagged | — | 85% |
| `tail_dominated` once the traded crash factor is supplied | 20/20 | 20/20 |

The 10% are 2 seeds whose realized P&L shows no tail signature (in one of them the core survives even with the crash factor). This is why a clean candidate still goes through the referee's H2 gate (`outputs/referee_h2_world_v01_v02.csv`).

```bash
python -m alpha_slicer.montecarlo --seeds 100 --seed-start 1000 --jobs 2 --out outputs   # ~25 min on 2 cores
pytest -q                                                              # unit tests + 6-seed smoke suite
```

## Inference: v0.4 audit, v0.4b correction, v0.4c policy layer

v0.3 answered *what the point estimate is*; v0.4 asks *how uncertain every slice is*; v0.4b fixes where the core's uncertainty is measured.

```python
cfg = SliceConfig(
    core_hac_lags=126,               # v0.4b decision statistic (traded-only regression, fixed-b)
    inference_bootstrap_reps=500,    # opt-in bootstrap of the whole decomposition
    inference_bootstrap_mean_block=63,
)
res = AlphaSlicer(cfg).fit(...)
res.summary[['core_unexplained_alpha_ann', 'core_alpha_se_ann', 'core_alpha_t', 'core_t_critical', 'core_alpha_t_v03']]
res.core_inference       # the same core under several dependence models
res.bootstrap_inference  # intervals for each traded premium, the core and implementation drag
```

### Why the core's standard error comes from the traded-only regression (v0.4b)

The zero-priced controls (`m²`, `m³`, downside and tail terms) are orthogonal to the constant and to the traded regressors, so they cannot move the alpha. If they stay in the regression that measures the alpha's uncertainty, they absorb the non-traded payoff's day-to-day variation, while the payoff's **in-sample mean** stays in the intercept. That mean is a large part of the core's error for option-like strategies: how many crashes the sample contained.

On the rare-crash put the full-design standard error was 1.9%/yr against an actual error of 14.8%/yr. The traded-only regression has the same intercept but keeps that variation in its residuals; its standard error is 13.5%/yr (table A2 above).

Under the default `balanced` profile, the decision statistic is:

- Bartlett HAC with `core_hac_lags=126` on the traded-only regression;
- a Kiefer–Vogelsang fixed-b critical value (about 2.09 on 2,700 days);
- the threshold "t ≥ 2" scaled by that value over 1.96, about 2.13.

`discovery` uses HAC(63) with conventional critical values; `certification` additionally requires the stationary-bootstrap interval to agree. The waterfall and regime-alpha table use the selected profile's HAC bandwidth / critical value. The regime table also drops the controls: they are not orthogonal to the state, so it would charge them a regime-dependent price.

### Core-inference audit

`core_inference` reports the identical intercept on the traded-only design under:

- HAC(126), HAC(21) and HAC(63), with fixed-b critical values;
- an automatic Newey–West bandwidth;
- the stationary bootstrap, when enabled.

These rows form the decision family: `inference_fragile` is true when some of them clear their threshold and others do not.

Two rows are for comparison only:

- VAR(1) score-prewhitened HAC (Andrews–Monahan style), which over-rejects when the mean follows regimes (stress test below);
- the pre-v0.4b statistic (`v0.3 full design HAC(21)`).

### Stationary bootstrap of the full priced decomposition

Every bootstrap replication jointly resamples the common estimation sample in geometric blocks, then **refits all traded spanning regressions required by the exact block-Shapley decomposition**. The output gives, for each priced slice and the core:

- point estimate;
- bootstrap mean and bias;
- bootstrap standard error;
- percentile confidence interval;
- sign probability and a two-sided percentile p-value.

The bootstrap holds the chosen feature definitions fixed. Only traded blocks enter the refits, so a non-traded payoff's variation stays in the core's distribution.

On the known-truth demo (500 draws, mean block 63) the add-up identity holds in every draw: traded Shapley premia + core = raw net mean. The net core is 1.17%/yr against a construction truth of 1.00%/yr, with a 95% bootstrap interval of −1.5% to +4.0%. The linear premium, 3.8%/yr, has an interval of 0.5% to 7.3%: the slices of a decomposition are estimates too.

**Coverage check (v0.4b, seeds 1000–1099, 200 draws, block 63; targets written before the run).** The core's bootstrap interval covers:

- the population alpha of the rare-crash put in 91% of samples (median width 50%/yr);
- the truth of the regime-alpha case in 97%;
- zero in the noise case in 96%.

### Persistence stress test

[`outputs/INFERENCE_VALIDATION.md`](outputs/INFERENCE_VALIDATION.md) isolates the bandwidth question in zero-alpha worlds whose mean follows a two-state Markov chain (400 seeds per world, 2,500 observations). Each cell is the false-rejection rate at the nominal 5%:

| world | HAC(21) | HAC(126) | HAC(126), fixed-b | prewhitened HAC | bootstrap, block 126 |
|---|---:|---:|---:|---:|---:|
| iid | 4.5% | 5.0% | 4.2% | 4.2% | 7.2% |
| regimes ~100 days | 33.2% | 16.2% | **12.0%** | 42.2% | 16.8% |
| regimes ~200 days | 44.5% | 21.8% | **19.0%** | 51.7% | 19.5% |

- **v0.4's headline.** v0.4 reported 7.5% for HAC(126) and the bootstrap from 40 seeds, whose Wilson interval runs from 2.6% to 19.9%.
- **What a long bandwidth buys.** It halves the error but does not reach 5% when regimes last about as long as the bandwidth. No estimator can see regimes the sample never contained.
- **The safeguards.** The tail flags and the referee's hidden-risk gate remain separate safeguards.

## Luck gate on the core

A core above the t threshold is evidence only if it was the one specification tried. Declare your search, or let the slicer run it:

```python
res = AlphaSlicer(SliceConfig(luck_n_trials=20)).fit(...)   # 20 = specifications x variants tried before this one
res.summary[['core_alpha_t', 'core_psr', 'core_dsr', 'luck_sr_hurdle_ann', 'luck_gate_pass']]

from alpha_slicer import spec_sweep
sw = spec_sweep(strategy, factors, benchmark, tail_factors=tf, n_trials_extra=5)   # default grid of specifications
sw.table, sw.gate, sw.label    # best core must pass SPA across the grid and DSR at the effective number of specs
```

- **Single specification.** The deflated Sharpe ratio (Bailey & López de Prado) is the probability that the core's true Sharpe ratio exceeds the expected maximum of `luck_n_trials` null tries, corrected for skewness and kurtosis. With one try it is the probabilistic Sharpe ratio against zero.
- **Across a specification sweep.** Hansen's SPA test uses a stationary bootstrap of the aligned core returns, so it handles autocorrelation and the correlation between specifications. The effective number of specifications (Li–Ji) sets the DSR hurdle.

## Traded proxies for non-linear premia

```python
from alpha_slicer import variance_swap_pnl
vs = variance_swap_pnl(spy_returns, vix_close)          # long variance swap P&L: m_t^2 - VIX_{t-1}^2/252
res = AlphaSlicer().fit(strategy, factors, tail_factors=pd.DataFrame({'VARSWAP': vs}))
```

A put-writing or buy-write index (as excess return) can be passed the same way. Without such a proxy, `nontraded` shows how large a premium on the unspanned non-linear exposure would have to be to explain the core. `nontraded_premia={term: premium}` gives the what-if.

## Notebook

`Academic_Alpha_Slicer.ipynb` is the end-to-end research notebook; `Academic_Alpha_Slicer.executed.ipynb` is the executed copy. It runs offline on the known-truth demo, shows four stress cases from the suite, and includes optional live adapters for the author's repositories:

- `run_factor_momentum_live(...)` reuses `cross-sectional-factor-momentum/core.py`, the Yahoo factor ETFs and SPY;
- `run_tailaware_sp100_live(...)` loads the referee's F0 engine (its notebook or core `.py`) and calls the referee's own data loader, residualization, signals and backtest, so the slicer explains exactly the P&L the referee judges. On the referee's offline panel the two P&Ls are identical.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
jupyter notebook Academic_Alpha_Slicer.ipynb
```

## Interpretation discipline

- **"Core unexplained alpha"** means unexplained by the **traded factors supplied**, not "true causal alpha". A positive core can still reflect omitted priced factors, measurement error or data-mining.
- **A flagged core** comes with unpriced tail exposure and is the referee's job (H2), not a discovery.
- **States must be public and lagged.** The engine lags them by `state_lag`; pass states as observed at the close.

## Known limitations

- **The luck gate needs honest bookkeeping.** `luck_n_trials` must count every specification and variant tried; the slicer cannot see searches done outside it. The DSR treats returns as i.i.d. (the HAC t and the SPA's block bootstrap cover autocorrelation).
- **Persistent regimes.** A 126-day bandwidth with fixed-b still rejects a true zero 12% of the time when the mean follows ~100-day regimes and 19% with ~200-day regimes (stress test). A finite sample cannot reveal states it never contained; the flags and the referee remain the safeguard.
- **Power.** The corrected statistic certifies genuine regime alpha in 74% of samples instead of 84%. That is the price of honest intervals for option-like payoffs.
- **The nested-model table** still reports `alpha_t` with the controls in the regression, for continuity. Use `summary['core_alpha_t']` or `core_inference` for inference on the core.
- **`tail_shape` thresholds.** They use a full-sample quantile. That is acceptable because the basis carries a zero price, but it is descriptive only.

## Status

**v0.4c (package 0.4.2):** the v0.4b economic decomposition and validated balanced statistic are unchanged. v0.4c adds an explicit inference-policy layer: higher-power `discovery`, v0.4b-compatible `balanced`, bootstrap-confirmed `certification`, and `custom`. All standard policies plus the inference frontier are always shown, and the report records whether the primary policy was precommitted. The underlying v0.4b Monte Carlo validation remains 69/70 pre-registered targets; v0.4c adds policy-layer regression tests and a deterministic boundary case where discovery passes while balanced fails with the identical alpha estimate. Test suite: 36 passed, plus 1 optional adapter test. `Academic_Alpha_Slicer.executed.ipynb` completes with zero execution errors. No claim of production readiness or causal identification.
