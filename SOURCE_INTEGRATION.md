# Integration with the current research repos

The MVP was built against the supplied current repositories:

- `tail-aware-alpha-referee-main`
- `cross-sectional-factor-momentum-main`

## Factor Momentum adapter

`alpha_slicer.adapters.run_factor_momentum_live` imports the existing `core.py` rather than copying the whole research engine. It reuses the current factor-momentum implementation, AR spread estimator and monthly backtest, then exposes:

- `MomTop1_80_20` net/gross P&L;
- the five ETF sleeve returns;
- SPY as the market benchmark.

This is intentional: the slicer should diagnose the current strategy, not fork it.

## Tail-Aware adapter (v0.3: the referee's own functions)

The Tail-Aware repo keeps its implementation in notebooks rather than a shared Python module. Since v0.3, `run_tailaware_sp100_live`:

1. finds the referee's F0 engine (`notebooks/02_f0_sp100_public_data.ipynb`, or a `f0_core_reference.py` export) and loads only its definitions — imports, functions and call-free constants such as `CONFIG` and `SP100`. The pipeline, prints and plots are not run (`load_referee_core`);
2. calls the referee's own functions in the referee's order: `load_panel` → `build_returns` → `build_ladder` → `build_signals` → `backtest`, at the referee's verdict level (`L2_FF3`), with the referee's `CONFIG`;
3. returns that P&L with the referee's FF3 factors, ready for `AlphaSlicer.fit`.

On the referee's offline panel (`data_source='shaped'`) the adapter's net P&L is identical to the one in the referee's own `run_f0` table (Sharpe 1.464941 in both). `tests/test_luck_and_adapter.py` has an opt-in parity test: set `TAIL_AWARE_CORE` to the notebook or core file.

Up to v0.2 the adapter recreated a simplified fork instead (log-residual sum, ±0.5/k weights, its own beta loop). That fork crashed at its default settings (`min_periods` 242 larger than the 231-day window), so it had never produced a P&L. Once that bug was patched, its P&L on the same offline panel still differed: Sharpe 1.61 vs 1.47, half the volatility, correlation 0.92.

## Separation of responsibilities

- **Tail-Aware Alpha Referee:** certification / rejection of a signal using hidden-luck + hidden-risk gates.
- **Academic Alpha Slicer:** attribution of apparent return across linear, nonlinear, conditional, tail and implementation layers.

The slicer does not replace the referee. Since v0.2 the slicer says which referee gate comes next (`summary['referee_handoff']`):

- `core_unexplained_candidate` (which since v0.3 also passes the slicer's own luck gate on the core) → the referee's luck gate (H1 vs H3) **and** its hidden-risk gate. The slicer's flags only see tail risk that materialized in the P&L: on the referee's H2 world, 2 of 20 hidden-crash strategies came out as clean candidates.
- `core_unexplained_candidate_tail_flagged` → the referee's hidden-risk gate (H2) first. The core comes with unpriced short-convexity or crash exposure that no supplied traded factor prices.

## v0.4/v0.4b inference layer and repo separation

v0.4 and v0.4b do **not** alter either source strategy or the Tail-Aware referee. They sit after the adapters have produced a common daily P&L / factor design and ask a separate question: how uncertain is the decomposition itself?

- `core_inference` reports the same core intercept under several dependence estimators, on the traded-only design since v0.4b.
- `bootstrap_inference` stationary-bootstraps the common priced design and refits every exact Shapley subset, so uncertainty in estimated factor loadings propagates into the linear / conditional / tail / core slices.
- **v0.4b changes the label's statistic, explicitly and with its own held-out validation.** The core's t comes from the traded-only regression with HAC(126) and a fixed-b threshold, because the zero-priced controls had been shrinking the standard error of option-like cores. For the referee this means fewer option-selling cores arrive as candidates. The 2-of-20 clean hidden-crash strategies above still argue for sending every clean candidate to H2. `inference_fragile` remains an audit field.

This separation is deliberate: **Factor Momentum generates a strategy; the Slicer explains it; Tail-Aware decides whether any surviving core deserves certification.**

## v0.5 — Alpha Survival Curve integration

v0.5 does not change either external adapter. Once Factor Momentum or Tail-Aware returns a strategy P&L and factor panel, the Slicer now reports an additional `alpha_survival` table. The path expands only across traded/priced returns. This is useful for the live cases because it separates two questions that were previously conflated:

1. **model-depth robustness** — does the intercept survive MKT/FF-style/managed/tail priced spans? (`alpha_survival`)
2. **inference robustness** — does the same final intercept survive discovery/balanced/certification standards? (`profile_comparison`, `inference_frontier`)

For live Factor Momentum, the supplied ETF sleeve returns are treated as the linear priced span; for Tail-Aware, the adapter's FF residual signal can be tested under any externally supplied traded factor hierarchy. No new data dependency is introduced by v0.5.
