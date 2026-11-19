# Equity Derivatives Pricing & Risk Engine — v0.4.1

**New in v0.4.1 — numerical fixes and independent validation.** The Heston
Fourier pricers no longer carry a systematic quadrature bias, choose their
truncation automatically (no more negative prices for short expiries with large
vol-of-vol), are stable for tiny vol-of-vol, and warn instead of returning a
suspicious price; the QE-M degenerate branch keeps the full stock variance.
Prices are checked against QuantLib and 40-digit mpmath references. See the
[Changelog](CHANGELOG.md).

**New in v0.4.0 — dated term structures and chronology.** Added deterministic
risk-free discount factors and proportional-dividend/forward curves (explicit
as-of and no silent extrapolation), integrated consistently into Fourier
Heston vanilla pricing, calibrated IV, Monte Carlo path carry and Greeks.
The daily collector can be read with **externally verified** dated curves,
while American ETF options still require an explicit European-proxy opt-in.
Per-date Heston calibration, chronology guards, in-sample bid/ask coverage
and next-date *parameter-transfer* diagnostics are also included. No
historical curves or European exercise corrections are synthesized.

**All v0.3.0 APIs remain valid**, and flat rate/yield behavior is the default.
See [v0.4 term curves and history](docs/CURVES_AND_HISTORY.md) and
`examples/dated_curves_synthetic.py` for runnable examples.

---

## Project overview

**v0.4.1 — standalone Python research package.** Heston / Black–Scholes
vanilla pricing, Heston Monte Carlo pricing of Asian, lookback and barrier
options, Greeks, implied volatility, calibrated European vanilla surfaces,
source-compatible option-chain snapshot ingestion and numerical validation.

**This is a numerical research engine, not a production trading system.** A
successful optimizer run is not proof of a uniquely identified Heston model;
American ETF options cannot be priced exactly by the European Fourier pricer.
See [Calibration & limitations](docs/CALIBRATION.md).

## Installation

Python >=3.10. Pricing core needs NumPy and SciPy. Pandas and PyArrow are
optional for collector snapshots (PyArrow required for Parquet, not CSV.GZ).

```bash
python -m pip install -e '.[dev,marketdata]'
python -m pytest -q
python examples/price_exotics.py
python examples/compare_schemes.py
python examples/calibrate_synthetic.py
```

Install `pip install -e .` for pricing/calibration from pure Python quotes;
use `pip install -e '.[marketdata]'` for the data adapter. The import name is
**`equity_derivatives`**, unchanged since v0.1. Previous FX and Fixed Income
repositories are neither modified nor runtime dependencies. The Python
distribution name remains `equity-derivatives-pricing-risk-engine` from v0.2.

## The new v0.3 calibration layer

- Batch Heston Fourier European call/put prices, grouping strikes by expiry.
  Since v0.4.1 scalar and batch pricers share composite Gauss–Legendre
  quadrature with an automatic, per-expiry truncation and raise a
  `FourierAccuracyWarning` instead of returning a suspicious price.
- Constrained, reproducible multi-start nonlinear least-squares calibration of
  `(kappa, theta, xi, rho, v0)` to vanilla mids: price RMSE, Vega-scaled price
  residuals (default), or re-inverted implied-vol residuals.
- IVs recomputed internally from quoted mids; optional bid/ask based weights,
  holdout IV/price errors, within-bid/ask coverage, Jacobian condition proxy,
  optimizer starting-point diagnostics.
- Optional data adapter for the user's **existing** daily Yahoo option-chain
  collector: historical CSV.GZ/Parquet or ZIP, deterministic timestamp
  selection (no reliance on stale `latest` files), quote-quality filters,
  OTM-by-forward filtering, and strike monotonicity/convexity diagnostics.
- **Explicit exercise-style control.** For American-style ETF option data,
  `allow_american_proxy=True` must be specified. The output is labelled as a
  European proxy; no early-exercise premium is modelled. Flat rate/dividend
  assumptions must be passed by the caller.

### Minimal synthetic European surface calibration

```python
import numpy as np
from equity_derivatives import (
    EquityMarket, HestonParams, VanillaQuote, CalibrationConfig,
    heston_surface_prices, calibrate_heston,
)
market = EquityMarket(spot=100, rate=0.02, dividend_yield=0.01)
truth = HestonParams(kappa=2.5, theta=.05, xi=.6, rho=-.65, v0=.045)
strikes = np.tile([90., 100., 110.], 3)
maturities = np.repeat([.25, .75, 1.5], 3)
prices = heston_surface_prices(market, truth, strikes, maturities)
quotes = [VanillaQuote(float(k), float(t), float(p))
          for k,t,p in zip(strikes,maturities,prices)]
fit = calibrate_heston(market, quotes, config=CalibrationConfig(n_starts=4))
print(fit.params, fit.market_rmse_iv, fit.jacobian_condition_number)
```

For an **independently integrated synthetic** demo with a held-out subset:

```bash
python examples/calibrate_synthetic.py
```

### Use the supplied historical Yahoo archive

**The raw data are intentionally not bundled in this GitHub repository.**
To reproduce an exploratory proxy fit on the archived SPY CSV dated
2026-10-27, with **illustrative (NOT historically verified) assumptions**
`r=4%`, `q=1%` (continuously compounded):

```bash
python examples/calibrate_snapshot.py 'IV Data(3).zip' \
  --ticker SPY --date 2026-10-27 \
  --rate 0.04 --dividend-yield 0.01 \
  --exercise-style american --allow-american-proxy \
  --output calibration_spy_exploratory.json
```

On this run: **96 selected quotes over eight expiries**, 72 training and 24
held-out within the **same date**; training / holdout IV RMSE approximately
**0.400 / 0.407 volatility percentage points**, with **1/24 holdout predictions
inside bid/ask**. This is *not* a temporal backtest, calibrated European-index
surface, or evidence of executable market pricing. See
[Empirical run notes](docs/EMPIRICAL_SPY.md).

If choosing the newest snapshot directly from the ZIP, `.parquet` reading
requires the `[marketdata]` optional dependencies. The adapter intentionally
does not silently use an older CSV if Parquet support is missing.

## Core v0.1/v0.2 functionality preserved

| Layer | Capability | Qualification |
|---|---|---|
| Vanilla | Black-76 / Black–Scholes, IV inversion, scalar/batch Heston Fourier | Flat or dated deterministic discount/proportional-dividend curves; European |
| Simulation | Correlated Heston Full Truncation Euler / QE-M | Not exact Heston paths |
| Payoffs | European, arithmetic Asian, fixed-strike lookback, up/down in/out barrier | Asian/lookback grid fixings, no rebate |
| Barrier | Discrete or conditional bridge survival | Bridge exact under BS, approximate under Heston |
| Risk | CRN bump spot Delta/Gamma and initial variance sensitivity | **Not** recalibrated market-IV Vega |
| Statistics | Pair-aware antithetic SE, CIs | Does not include time-step/model bias |
| Validation | Heston European and BS barriers vs QuantLib and 40-digit mpmath; CIR moments, martingality | Heston exotics not independently validated |

The **defaults** `MCConfig.scheme='full_truncation_euler'` and
`Barrier.monitoring='discrete'` are unchanged from v0.1. All existing public
APIs remain available. The new data/calibration modules are additive.

## Repository layout

```text
src/equity_derivatives/
    models.py, simulation.py, payoffs.py, pricing.py, greeks.py
    vanilla.py, heston_fourier.py, heston_surface.py
    calibration.py, market_data.py
    barrier_bridge.py, barrier_reference.py
examples/
    price_exotics.py, compare_schemes.py, convergence_study.py
    barrier_validation.py, heston_barrier_grid.py
    calibrate_synthetic.py, calibrate_snapshot.py
tests/
docs/
    METHODOLOGY.md, VALIDATION.md, PROVENANCE.md, DESIGN_DECISIONS.md
    CALIBRATION.md, MARKET_DATA.md, EMPIRICAL_SPY.md
```

## Validation and limitations

- **v0.4.1: 190 tests** (the 86 of v0.4 plus 104 new): reference prices from a
  40-digit mpmath Lewis integral cross-checked with QuantLib, live QuantLib
  comparisons (skipped when QuantLib is not installed), Fourier robustness,
  QE-M small vol-of-vol, multi-seed Monte Carlo accuracy and calibration
  recovery from the default start. Against QuantLib over the whole calibration
  box (1 day to 10 years) Heston European prices agree to about 4e-12 × spot.
- The synthetic scalar-Fourier calibration reproduces Heston inputs closely,
  but model-to-model recovery cannot establish genuine market fit or
  identifiability.
- The actual SPY example shows an IV fit to a European **proxy**, not an
  exercise-adjusted calibration. Data timestamps, flat input curves and thin
  bid/ask coverage need further analysis.
- Static-arbitrage diagnostics cover **strike monotonicity and convexity**,
  not the entire no-static-arbitrage surface or calendar relationships.
- No independent historical curve/forward sourcing, recalibrated market-IV
  Greeks, American exercise pricing, cash-dividend schedule, portfolio risk
  aggregation or market-data downloads are included in v0.4.

See [Changelog](CHANGELOG.md), [Architecture decisions](docs/DESIGN_DECISIONS.md),
[Methodology](docs/METHODOLOGY.md), and [Upstream provenance](docs/PROVENANCE.md).
