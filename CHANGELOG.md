## v0.4.1 — Fourier and QE-M numerical fixes, QuantLib validation (2026-11-19)

### Fixed

- **Batch Heston pricer bias** (`heston_surface_prices`, used by calibration):
  the single 96-node Gauss–Legendre rule on [0, 180] under-resolved the
  `1/(u²+1/4)` peak of the Lewis integrand and overpriced every option by
  about 2e-4 per 100 of spot. Replaced by composite Gauss–Legendre panels
  (16 nodes each) that start at width 0.5 near `u=0` and are capped in width
  by the phase rate of the integrand.
- **Fourier truncation**: the fixed cut-offs (200 scalar, 180 batch) were too
  short for short expiries with large vol-of-vol (errors up to 1e-2 and
  negative prices, e.g. a 7-day 105 call at −0.007 instead of +0.00002). The
  cut-off is now chosen per maturity from the decay of `|phi(u−i/2)|`
  (target error `rel_tol*spot`, default 1e-10, cap 2**16). Scalar and batch
  pricers share the same numerics.
- **Small vol-of-vol in the characteristic function**: `(a−d)/xi²` and the
  little-trap logarithm are evaluated without cancellation (xi=1e-9 gave 0.84
  instead of 7.36). Continuous with the exact `xi=0` case.
- **QE-M degenerate branch** (`psi < 1e-12`, e.g. xi below ~1e-6 with daily
  steps) kept `v_next = m` but lost the rho² share of the stock variance
  (ATM 5.15 instead of 7.36 at rho=−0.7). It now uses the xi→0 limit of QE-M:
  full trapezoidal integrated variance, correlated with `z_variance`.
- `heston_log_return_cf(-1j, ...)` returned NaN when `kappa < rho*xi`.

### Changed

- New exported `FourierAccuracyWarning`: issued when the estimated truncation
  error or a static no-arbitrage bound violation exceeds `1000*rel_tol*spot`.
  Non-finite values raise `FloatingPointError`. Inside calibration such
  evaluations are treated as failed (penalty residuals), never fitted.
- `heston_european(..., integration_upper=None, n_nodes=16, rel_tol=1e-10)`:
  `None` = automatic cut-off, a float fixes it (and is still checked).
  `epsabs` is ignored and deprecated.
- `heston_surface_prices(..., n_nodes=16, integration_upper=None, rel_tol=1e-10)`:
  `n_nodes` now counts Gauss–Legendre nodes **per panel**.
- `CalibrationConfig` defaults `n_nodes=16, integration_upper=None`;
  `evaluate_heston_fit` defaults `n_nodes=32, integration_upper=None`.
- Batch pricing is about 2.5x slower (≈2 ms for 64 quotes over 8 expiries)
  and calibration about 2x; batch prices move by about −2e-4 per 100 of spot,
  so calibration outputs change in the last digits (EMPIRICAL_SPY.md numbers
  were produced with v0.4.0). Synthetic recovery IV RMSE drops from
  0.0012/0.0036 to below 0.0001 vol points.

### Tests (86 → 190)

- Reference prices from a 40-digit mpmath Lewis integral cross-checked with
  QuantLib (incl. the v0.4.0 failure cases and Andersen 2008 case I), QuantLib
  barrier values; generator in `tests/reference_generator.py`.
- Live QuantLib comparisons over the calibration box, with dated curves and for
  BS barriers (skipped if QuantLib is absent; QuantLib added to `dev` extra).
- Fourier robustness, QE-M small-xi, Monte Carlo accuracy with multi-seed
  statistical tolerances, calibration recovery from the default start (not the
  truth). `FourierAccuracyWarning` is an error in the test suite; two v0.4.0
  tolerances tightened (batch vs scalar 2e-3 → 1e-8; MC vs Fourier +0.2 → +0.02).

## v0.4.0 — dated market curves and point-in-time history (2026-11-14)

- Added `DiscountCurve` and `EquityCarryCurves`, continuous proportional
  carry, log-DF interpolation, explicit as-of checks and no hidden
  extrapolation, or inferred curve inputs.
- Integrated option-maturity forward and discount curves in Heston Fourier
  surface/scalar pricing and IV calibration; integrated time-varying drift
  into the Heston Monte Carlo path simulation, discounting and bumped Greeks.
- Added collector quote filtering using curve-based forward moneyness and IV,
  strict UTC date alignment, and `--curves-json` CLI input (flat mode retained).
- Added in-sample calibration bid/ask coverage and number of quoted spreads;
  chronological `calibrate_history`, warm starts, and next-date parameter
  transfer diagnostics, explicitly NOT a tradable backtest.
- Added dedicated regression tests and synthetic dated curve example.
- Known limits: no independently sourced historic OIS or ETF forward curves;
  no exercise adjustment to American ETF options; no scheduled cash dividends.
- FX and FI repositories not modified. Public v0.3 interfaces preserved.

# Changelog

## 0.3.0 — 2026-11-11

**Additive calibration and data-adapter release.** Existing v0.2 models,
simulation schemes, payoff APIs and defaults remain unchanged.

### Added

- Batch Heston European Fourier pricing across strike/expiry grids with shared
  Gauss–Legendre quadrature (`heston_surface.py`).
- Constrained reproducible multi-start Heston calibration to price,
  Vega-scaled price or European implied-volatility residuals.
- Initial/fit diagnostics, Jacobian condition proxy, price/IV RMSE, and
  within-snapshot holdout evaluation including bid/ask coverage.
- Optional pandas/Parquet snapshot loader for dated Yahoo collector CSV.GZ,
  Parquet and ZIP archives with timestamp-safe selection.
- Quality filters (liquidity, spreads, forward OTM, European IV bounds) and
  strike-only static arbitrage midpoint/bid-ask diagnostics.
- Strict opt-in and visible warning to fit American quotes using a European
  model proxy; explicit caller-supplied flat rate and dividend yield.
- Self-contained synthetic recovery and reproducible SPY empirical demo,
  methodology documentation and expanded tests.

### Results / limitations

- 74 tests passing in the local release environment (59 pre-existing + 15 new).
- Supplied SPY Oct 27, 2026 CSV exploratory proxy example: 96 selected quotes,
  72 train / 24 held out on the same snapshot; train and holdout IV RMSE
  ~0.400 / 0.407 percentage points; holdout bid/ask coverage **1/24**.
- No American Heston solver, true historical rate/dividend curves,
  out-of-time validation, arbitrage-free surface repair, recalibrated market
  Vega or portfolio hedge-P&L framework in this release.

## 0.2.0 — 2026-11-10

**Equity Derivatives Pricing & Risk Engine** (separate standalone repository;
import namespace `equity_derivatives` preserved).

### Added

- Andersen **QE-M** (quadratic-exponential with conditional stock
  martingale correction) as `MCConfig(..., scheme="qe_m")`; variance
  transition matches CIR conditional first two moments. Handles an atom at
  zero and `xi=0` exact deterministic integrated variance.
- Continuous-barrier **conditional Brownian-bridge survival weighting** for
  `Barrier(..., monitoring="continuous_bridge")`. Exact under constant
  Black-Scholes volatility; only **approximate under Heston**.
- Independent exact continuously monitored Black-Scholes barrier benchmark
  `black_scholes_barrier` (zero rebate, up/down, in/out, call/put).
- Validation tests for QE moments, martingality, Feller violation, CRN,
  bridge convergence to exact Black-Scholes and knock-in/out parity.
- Cross-scheme comparison and barrier-benchmark runnable examples; expanded
  methodology, design decisions and recorded QA.

### Preserved

- Full Truncation Euler default and discrete-barrier default.
- v0.1 simulation, payoff, pricing, vanilla, risk and Fourier call signatures.
- No runtime dependency on or modification of the FX/FI repositories.

### Not implemented

- Calibration to market IV/vanilla quotes, market Vega/recalibrated Greeks,
  exact continuous-time Heston barrier crossing, QMC, control variates,
  non-flat rate/dividend curves, American features or production guarantees.

## 0.1.0 — 2026-11-10

Initial **standalone** Equity Derivatives Engine milestone.

- Vanilla Black-76/Black-Scholes European pricing and IV inversion.
- Heston characteristic function and Lewis Fourier European vanilla pricer.
- Correlated Heston full-truncation Euler Monte Carlo with reproducible shocks.
- European, arithmetic Asian, fixed-strike lookback and discretely monitored
  barrier knock-in/knock-out payoff implementations.
- Antithetic simulation and pair-correct sampling error/normal CIs.
- CRN finite-difference Delta/Gamma/initial-variance sensitivity.
- Executable demos, convergence study and Python test suite.
- Upstream FX/FI repositories remain untouched and are not package dependencies.
