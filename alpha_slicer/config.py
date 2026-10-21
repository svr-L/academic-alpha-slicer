from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class SliceConfig:
    """Configuration for the alpha decomposition engine.

    The decomposition is intentionally diagnostic, not causal.

    Pricing convention (v0.2): only TRADED blocks (linear factors, factor x lagged-state
    managed portfolios, supplied traded tail factors) are charged at their mean. NON-TRADED
    blocks (market powers, downside/tail basis) enter de-meaned, i.e. at a zero premium: they
    explain variance and tail shape but never absorb mean return. ``block_order`` lists the
    blocks to use; the traded ones, in this order, form the readable waterfall, and a
    block-Shapley decomposition over the traded blocks exposes ordering dependence.
    """
    annualization: int = 252
    hac_lags: int = 21
    market_col: str = "MKT"
    nonlinear_max_order: int = 3
    tail_quantile: float = 0.10
    vol_lookback: int = 21
    vol_threshold_lookback: int = 252
    drawdown_threshold: float = -0.10
    # States (built or supplied) are observed at the close of t; the value applied to the
    # return of date t is the one from t - state_lag. Set 0 only if you pass pre-lagged states.
    state_lag: int = 1
    min_obs: int = 252
    min_regime_obs: int = 63
    block_order: Tuple[str, ...] = ("linear", "conditional", "tail", "nonlinear", "tail_shape")
    shapley_max_blocks: int = 5
    core_t_threshold: float = 2.0
    material_share: float = 0.35
    # Unpriced tail-risk flags (hand-off to the referee's hidden-risk gate)
    flag_t_threshold: float = 2.0
    flag_es_q: float = 0.025
    flag_es_ratio: float = 1.5
    # Luck gate on the core (v0.3): a candidate needs HAC t >= the core threshold AND a deflated
    # Sharpe ratio >= luck_dsr_min after luck_n_trials tries (specifications x strategy variants
    # you tried before reporting this one). With 1 trial the DSR is the probabilistic SR vs 0.
    luck_n_trials: int = 1
    luck_dsr_min: float = 0.95

    # v0.4b core inference (an explicit, separately validated change of the decision statistic).
    # The core's standard error comes from the TRADED-ONLY spanning regression. Its intercept is
    # the core itself (Frisch-Waugh-Lovell: the zero-priced controls are orthogonal to the
    # constant and to every traded regressor), but its residuals keep the variation of the
    # non-traded payoff. Up to v0.4 the standard error came from the full design, where those
    # controls soak up that variation while their in-sample mean, which is part of the core's
    # estimation error, stays in the intercept: on the rare-crash put the standard error was
    # 1.9%/yr against an actual error of 16%/yr.
    # core_hac_lags: Bartlett bandwidth for the core, the waterfall and the regime-alpha table.
    # core_fixed_b: Kiefer-Vogelsang fixed-b critical values for that bandwidth; the decision
    #   threshold becomes core_t_threshold * cv(b) / 1.96 (about 2.14 at 126 lags on 2,700 days).
    core_hac_lags: int = 126
    core_fixed_b: bool = True

    # v0.4 inference audit (computed on the same traded-only regression since v0.4b). These
    # settings only report sensitivity; the decision statistic is set above.
    inference_hac_lags: Tuple[int, ...] = (21, 63, 126)
    inference_include_auto_hac: bool = True
    inference_include_prewhite_hac: bool = True
    inference_bootstrap_reps: int = 0
    inference_bootstrap_mean_block: float = 63.0
    inference_bootstrap_ci: float = 0.95
    inference_bootstrap_seed: int = 1729
    inference_store_bootstrap_draws: bool = False

    # v0.4c: explicit power-vs-false-positive preference. The selected profile drives the Slicer's
    # candidate label, while all profiles and the full frontier are still reported.
    # discovery     = HAC(63), conventional critical value (higher power)
    # balanced      = v0.4b rule: HAC(core_hac_lags), fixed-b
    # certification = balanced + stationary-bootstrap CI agreement
    # custom        = use core_hac_lags/core_fixed_b plus inference_custom_require_bootstrap
    inference_profile: str = "balanced"
    inference_profile_precommitted: bool = False
    inference_custom_require_bootstrap: bool = False
    inference_frontier_hac_lags: Tuple[int, ...] = (21, 63, 126)

    # v0.5: alpha survival across an expanding, pre-declared priced-factor span.
    # The default hierarchy is market -> FF3 -> FF5(+MOM) when available -> all supplied
    # linear factors -> managed/conditional portfolios -> supplied traded tail factors.
    # Custom spans are additive, never selected by minimizing alpha.
    survival_enabled: bool = True
    survival_include_default_linear_spans: bool = True
    survival_custom_linear_spans: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    survival_include_conditional: bool = True
    survival_include_tail: bool = True
