from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd

# Pricing type of every block.
#   traded    : excess returns of implementable positions. Their premium is their mean,
#               so they are charged at their sample mean in the alpha decomposition.
#   nontraded : functions of returns that are NOT returns (m^2, min(m,0), tail dummies...).
#               Their raw mean is not a price. The engine ORTHOGONALIZES them to the traded
#               regressors (and the constant): whatever part of a nonlinear payoff is spanned
#               by traded factors is priced by those factors, and the remainder carries a zero
#               price. They explain variance and tail shape but can never move alpha. A priced
#               nonlinear/tail premium needs a traded proxy supplied through `tail_factors`
#               (e.g. variance-swap or put-writing P&L).
BLOCK_TYPES: Dict[str, str] = {
    "linear": "traded",        # supplied common factors
    "conditional": "traded",   # factor x lagged-state managed portfolios
    "tail": "traded",          # supplied traded tail/convexity factors (optional)
    "nonlinear": "nontraded",  # market powers 2..K
    "tail_shape": "nontraded", # downside / left-tail market basis
}


def _unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def build_states(benchmark: pd.Series, vol_lookback=21, threshold_lookback=252,
                 drawdown_threshold=-0.10) -> pd.DataFrame:
    """State variables as observed at the close of each date t (uses data up to and including t).

    These values are NOT yet valid conditioning information for the return of date t.
    The engine shifts them by ``SliceConfig.state_lag`` (default 1 day) before use, so the
    state applied to the return of t is the one known at the close of t-1.
    Warm-up dates without enough history are NaN (not silently classified as "off").
    """
    b = pd.Series(benchmark).astype(float).sort_index()
    rv = b.rolling(vol_lookback, min_periods=max(5, vol_lookback // 2)).std()
    thr = rv.rolling(threshold_lookback, min_periods=max(63, threshold_lookback // 4)).median().shift(1)
    high_vol = (rv > thr).astype(float).where(rv.notna() & thr.notna())
    wealth = (1.0 + b.fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    drawdown = (dd <= drawdown_threshold).astype(float).where(b.notna())
    return pd.DataFrame({"HIGH_VOL": high_vol, "DRAWDOWN": drawdown}, index=b.index)


def lag_states(states: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Shift states so that the value used at date t was observable at the close of t-lag."""
    S = pd.DataFrame(states).astype(float)
    return S.shift(int(lag)) if int(lag) else S


@dataclass
class FeatureDesign:
    blocks: Dict[str, pd.DataFrame]
    block_types: Dict[str, str]
    states: pd.DataFrame
    nontraded_raw_means: pd.Series
    tail_threshold: float
    dropped_columns: List[str] = field(default_factory=list)


def build_feature_blocks(factors: pd.DataFrame, benchmark: pd.Series | None = None,
                         states: pd.DataFrame | None = None, market_col="MKT",
                         nonlinear_order=3, tail_quantile=0.10,
                         tail_factors: pd.DataFrame | None = None) -> FeatureDesign:
    """Build the regression design on a FIXED estimation sample.

    All inputs must already be aligned on the estimation sample (no missing values) and
    ``states`` must already be lagged. State centring uses that sample.

    Blocks (see BLOCK_TYPES for the pricing convention):
      linear      : supplied factors (traded)
      conditional : F_k,t * (z_{t-1} - mean z) for every factor k and lagged state z (traded
                    managed portfolios). State dummies themselves are NOT here: a state
                    dummy is time-varying alpha and is reported by the regime-alpha table.
      tail        : supplied traded tail/convexity factors, renamed TAILFACTOR_* (traded)
      nonlinear   : market powers 2..K (non-traded; returned RAW)
      tail_shape  : min(m,0), m*1{m<=q}, m^2*1{m<=q} (non-traded; returned RAW)

    Non-traded blocks are returned raw; the engine orthogonalizes them to the traded
    regressors that are actually in the model (see ``orthogonalize``).
    """
    F = pd.DataFrame(factors).astype(float).sort_index().copy()
    if benchmark is None:
        if market_col in F:
            b = F[market_col].copy()
        else:
            raise ValueError("benchmark is required when market_col is absent from factors")
    else:
        b = pd.Series(benchmark, name="BENCH").astype(float).sort_index().reindex(F.index)
    market = F[market_col] if market_col in F.columns else b

    linear = F.copy()

    S = pd.DataFrame(index=F.index) if states is None else pd.DataFrame(states).reindex(F.index).astype(float)
    conditional = pd.DataFrame(index=F.index)
    for c in S.columns:
        zc = S[c] - S[c].mean()
        for f in F.columns:
            conditional[f"{f}_X_{c}"] = F[f] * zc

    tail = pd.DataFrame(index=F.index)
    if tail_factors is not None:
        tf = pd.DataFrame(tail_factors).reindex(F.index).astype(float)
        tail = tf.rename(columns=lambda c: f"TAILFACTOR_{c}")

    nonlinear_raw = pd.DataFrame(index=F.index)
    for p in range(2, int(nonlinear_order) + 1):
        nonlinear_raw[f"{market_col}_POW{p}"] = market ** p

    # Full-sample quantile: acceptable here because the basis is non-traded and carries a
    # zero premium, so a look-ahead in its definition cannot leak a tradable mean.
    q = float(market.dropna().quantile(tail_quantile))
    tail_dummy = (market <= q).astype(float)
    tail_shape_raw = pd.DataFrame({
        f"{market_col}_DOWNSIDE": market.clip(upper=0.0),
        f"{market_col}_TAIL": market * tail_dummy,
        f"{market_col}_TAIL_SQ": (market ** 2) * tail_dummy,
    }, index=F.index)

    raw_means = pd.concat([nonlinear_raw, tail_shape_raw], axis=1).mean()

    blocks = {
        "linear": _unique_columns(linear),
        "conditional": _unique_columns(conditional),
        "tail": _unique_columns(tail),
        "nonlinear": _unique_columns(nonlinear_raw),
        "tail_shape": _unique_columns(tail_shape_raw),
    }
    dropped = []
    for name, X in blocks.items():
        if X.shape[1] == 0:
            continue
        sd = X.std(ddof=0)
        bad = list(sd.index[~(sd > 1e-14)])
        if bad:
            dropped += bad
            blocks[name] = X.drop(columns=bad)
    return FeatureDesign(blocks=blocks, block_types=dict(BLOCK_TYPES), states=S,
                         nontraded_raw_means=raw_means, tail_threshold=q, dropped_columns=dropped)
