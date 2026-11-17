from __future__ import annotations

"""Alpha survival across an expanding, pre-declared priced-factor span.

The survival curve answers a model-depth question that is deliberately separate from the
inference-policy question introduced in v0.4c:

    how much apparent alpha remains as the *priced* span is enriched?

Only traded returns enter a survival span. Non-traded nonlinear/tail-shape controls are excluded
because their raw means are not prices. Adding a span is therefore an economic pricing-model
change, not a variance-fitting trick.
"""

from collections import OrderedDict
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .luck import luck_gate
from .regression import NORMAL_CV, hac_ols, size_adjusted_p


SpanSpec = Tuple[str, Tuple[str, ...]]


def _dedupe_spans(spans: Sequence[SpanSpec]) -> list[SpanSpec]:
    out: list[SpanSpec] = []
    seen: set[frozenset[str]] = set()
    for name, cols in spans:
        cols = tuple(dict.fromkeys(cols))
        key = frozenset(cols)
        if not cols or key in seen:
            continue
        seen.add(key)
        out.append((str(name), cols))
    return out


def default_linear_spans(columns: Iterable[str], market_col: str = "MKT") -> list[SpanSpec]:
    """Canonical nested factor spans, included only when the required columns exist.

    The function never searches for the span that minimizes alpha. It follows a fixed hierarchy:
    market -> FF3 -> FF5 -> +MOM -> all supplied linear factors. Missing families are skipped.
    If FF5 is unavailable but MOM is available, FF3+MOM is reported.
    """
    cols = list(dict.fromkeys(str(c) for c in columns))
    have = set(cols)
    spans: list[SpanSpec] = []

    if market_col in have:
        spans.append(("MKT", (market_col,)))

    ff3 = (market_col, "SMB", "HML")
    if set(ff3).issubset(have):
        spans.append(("FF3", ff3))

    ff5 = (market_col, "SMB", "HML", "RMW", "CMA")
    if set(ff5).issubset(have):
        spans.append(("FF5", ff5))
        if "MOM" in have:
            spans.append(("FF5+MOM", ff5 + ("MOM",)))
    elif set(ff3).issubset(have) and "MOM" in have:
        spans.append(("FF3+MOM", ff3 + ("MOM",)))

    all_linear = tuple(cols)
    if all_linear:
        spans.append(("ALL_LINEAR", all_linear))

    return _dedupe_spans(spans)


def resolve_linear_spans(factors: pd.DataFrame, market_col: str = "MKT",
                         custom_spans: Optional[Sequence[SpanSpec]] = None,
                         include_defaults: bool = True) -> list[SpanSpec]:
    """Validate custom spans and combine them with the canonical hierarchy.

    Custom spans are interpreted as ex-ante pricing-model choices, not optimizer candidates.
    Each item is ``(name, (factor1, factor2, ...))``. Columns must exist in ``factors``.
    """
    F = pd.DataFrame(factors)
    spans: list[SpanSpec] = default_linear_spans(F.columns, market_col) if include_defaults else []
    if custom_spans:
        for name, cols in custom_spans:
            cols = tuple(str(c) for c in cols)
            missing = [c for c in cols if c not in F.columns]
            if missing:
                raise ValueError(f"survival span {name!r} references missing factor columns {missing}")
            spans.append((str(name), cols))
    spans = _dedupe_spans(spans)
    # A survival curve is an expanding span, not an arbitrary model tournament. If a user wants
    # a custom hierarchy that is not an extension of the defaults, disable the defaults and
    # provide the complete nested custom path explicitly.
    prev: set[str] = set()
    for name, cols in spans:
        cur = set(cols)
        if prev and not prev.issubset(cur):
            raise ValueError(
                f"survival span {name!r} is not nested: it drops {sorted(prev-cur)}. "
                "Use survival_include_default_linear_spans=False for a fully custom nested path."
            )
        prev = cur
    return spans


def _fit_stage(y: pd.Series, X: pd.DataFrame, ann: int, hac_lags: int,
               cv: float, base_t_threshold: float, luck_n_trials: int,
               luck_dsr_min: float) -> dict:
    X = pd.DataFrame(X, index=y.index)
    fit = hac_ols(y, X if X.shape[1] else None, lags=hac_lags)
    alpha = float(fit["params"]["const"] * ann)
    se = float(np.sqrt(max(fit["cov"].loc["const", "const"], 0.0)) * ann)
    t = float(fit["tvalues"]["const"])
    tcrit = float(base_t_threshold * cv / NORMAL_CV)
    p = float(size_adjusted_p(t, cv))

    if X.shape[1]:
        beta = fit["params"].drop("const")
        charge = X[beta.index].mul(beta, axis=1).sum(axis=1)
        residual = (y - charge).rename("span_core")
    else:
        residual = y.rename("span_core")

    luck = luck_gate(residual, t, luck_n_trials, luck_dsr_min, tcrit)
    hac_pass = bool(alpha > 0 and np.isfinite(t) and t >= tcrit)
    dsr_pass = bool(luck["pass"])
    return {
        "alpha_ann": alpha,
        "se_ann": se,
        "t": t,
        "p": p,
        "critical_value": float(cv),
        "t_threshold": tcrit,
        "ci_low_ann": alpha - cv * se,
        "ci_high_ann": alpha + cv * se,
        "r2": float(fit["r2"]),
        "adj_r2": float(fit["adj_r2"]),
        "bic": float(fit["bic"]),
        "condition_number": float(fit["condition_number"]),
        "n": int(fit["n"]),
        "hac_evidence_pass": hac_pass,
        "core_dsr": float(luck["dsr"]),
        "luck_gate_pass": dsr_pass,
        # Screen only. The official candidate label remains the full slicer's inference policy +
        # tail/referee hand-off; intermediate spans are model-depth diagnostics.
        "screen_pass": bool(hac_pass and dsr_pass),
        "residual": residual,
    }


def alpha_survival_curve(y: pd.Series, factors: pd.DataFrame,
                         conditional: Optional[pd.DataFrame] = None,
                         tail: Optional[pd.DataFrame] = None,
                         market_col: str = "MKT", annualization: int = 252,
                         hac_lags: int = 126, critical_value: float = NORMAL_CV,
                         base_t_threshold: float = 2.0, luck_n_trials: int = 1,
                         luck_dsr_min: float = 0.95,
                         custom_linear_spans: Optional[Sequence[SpanSpec]] = None,
                         include_default_linear_spans: bool = True,
                         include_conditional: bool = True,
                         include_tail: bool = True) -> pd.DataFrame:
    """Build the alpha-survival curve over an expanding priced span.

    Notes
    -----
    * The curve is not assumed to be monotone. Collinearity and omitted-variable interactions can
      make an intercept rise when a factor is added; that is information, not an error.
    * Non-traded shape controls never enter this curve.
    * ``screen_pass`` is a descriptive stage-by-stage HAC+DSR screen. It does not replace the
      v0.4c profile comparison or the Tail-Aware Referee verdict.
    """
    y = pd.Series(y).astype(float)
    F = pd.DataFrame(factors).astype(float).reindex(y.index)
    C = pd.DataFrame(index=y.index) if conditional is None else pd.DataFrame(conditional).astype(float).reindex(y.index)
    T = pd.DataFrame(index=y.index) if tail is None else pd.DataFrame(tail).astype(float).reindex(y.index)

    # All designs must share the already-prepared sample; fail loudly if the caller did not align.
    if y.isna().any() or F.isna().any().any() or C.isna().any().any() or T.isna().any().any():
        raise ValueError("alpha_survival_curve requires a common complete estimation sample")

    spans = resolve_linear_spans(F, market_col, custom_linear_spans, include_default_linear_spans)

    stages: list[tuple[str, pd.DataFrame, str]] = [("RAW_NET", pd.DataFrame(index=y.index), "raw")]
    current_cols: list[str] = []
    for name, cols in spans:
        current_cols = list(cols)
        stages.append((name, F.loc[:, current_cols], "linear"))

    # The full supplied linear span is the base for managed and tail premia. If defaults/customs
    # omitted it, add it explicitly before richer blocks.
    if len(F.columns):
        all_cols = tuple(F.columns)
        if not stages or set(stages[-1][1].columns) != set(all_cols):
            stages.append(("ALL_LINEAR", F.copy(), "linear"))

    X = F.copy()
    if include_conditional and C.shape[1]:
        X = pd.concat([X, C], axis=1)
        stages.append(("+CONDITIONAL", X.copy(), "conditional"))
    if include_tail and T.shape[1]:
        X = pd.concat([X, T], axis=1)
        stages.append(("+TRADED_TAIL", X.copy(), "tail"))

    # Deduplicate identical designs while preserving the economic labels.
    deduped: list[tuple[str, pd.DataFrame, str]] = []
    seen_designs: set[frozenset[str]] = set()
    for name, Xs, family in stages:
        key = frozenset(map(str, Xs.columns))
        if key in seen_designs and name != "RAW_NET":
            continue
        seen_designs.add(key)
        deduped.append((name, Xs, family))

    rows = []
    raw_alpha = np.nan
    prior_alpha = np.nan
    first_priced_alpha = np.nan
    for depth, (name, Xs, family) in enumerate(deduped):
        st = _fit_stage(y, Xs, annualization, hac_lags, critical_value, base_t_threshold,
                        luck_n_trials, luck_dsr_min)
        alpha = st.pop("alpha_ann")
        residual = st.pop("residual")
        if depth == 0:
            raw_alpha = alpha
        elif not np.isfinite(first_priced_alpha):
            first_priced_alpha = alpha
        row = {
            "stage": name,
            "depth": depth,
            "family_added": family,
            "n_priced_terms": int(Xs.shape[1]),
            "priced_terms": ",".join(map(str, Xs.columns)),
            "alpha_ann": alpha,
            "erosion_from_prior_ann": np.nan if depth == 0 else prior_alpha - alpha,
            "cumulative_erosion_ann": raw_alpha - alpha,
            "survival_ratio_vs_raw": alpha / raw_alpha if np.isfinite(raw_alpha) and abs(raw_alpha) > 1e-12 else np.nan,
            "survival_ratio_vs_first_priced": alpha / first_priced_alpha if np.isfinite(first_priced_alpha) and abs(first_priced_alpha) > 1e-12 else np.nan,
            **st,
        }
        # Keep residual out of the public table; it is an implementation detail for DSR only.
        rows.append(row)
        prior_alpha = alpha

    out = pd.DataFrame(rows).set_index("stage")
    return out


def survival_summary(curve: pd.DataFrame) -> pd.Series:
    if curve is None or curve.empty:
        return pd.Series(dtype=object)
    c = curve.copy()
    priced = c.iloc[1:] if len(c) > 1 else c.iloc[0:0]
    screen = priced["screen_pass"].astype(bool) if len(priced) else pd.Series(dtype=bool)
    first_fail = next((idx for idx, val in screen.items() if not val), "none")
    last_pass = next((idx for idx, val in reversed(list(screen.items())) if val), "none")
    final = c.iloc[-1]
    raw = c.iloc[0]
    return pd.Series({
        "alpha_survival_stages": int(len(c)),
        "alpha_survival_final_stage": str(c.index[-1]),
        "alpha_survival_raw_ann": float(raw["alpha_ann"]),
        "alpha_survival_final_ann": float(final["alpha_ann"]),
        "alpha_survival_final_vs_raw": float(final["survival_ratio_vs_raw"]) if np.isfinite(final["survival_ratio_vs_raw"]) else np.nan,
        "alpha_survival_screen_pass_share": float(screen.mean()) if len(screen) else np.nan,
        "alpha_survival_span_fragile": bool(screen.nunique() > 1) if len(screen) else False,
        "alpha_survival_first_failure_stage": str(first_fail),
        "alpha_survival_deepest_screen_pass_stage": str(last_pass),
        "alpha_survival_final_screen_pass": bool(final["screen_pass"]),
    })
