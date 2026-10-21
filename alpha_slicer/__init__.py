"""Academic Alpha Slicer."""
from .config import SliceConfig
from .decomposition import AlphaSlicer, SliceResult
from .demo import make_demo_case
from .features import BLOCK_TYPES, build_states, lag_states
from .proxies import variance_swap_pnl
from .luck import deflated_sharpe, spa_test, effective_trials
from .sweep import spec_sweep, SweepResult
from .inference import stationary_bootstrap_decomposition, core_inference_grid, automatic_hac_lags
from .regression import fixed_b_critical_value
from .policy import InferencePolicy, policy_presets, resolve_policy, evaluate_inference_profiles, inference_frontier
from .survival import alpha_survival_curve, survival_summary, default_linear_spans, resolve_linear_spans

__all__ = ["SliceConfig", "AlphaSlicer", "SliceResult", "make_demo_case", "BLOCK_TYPES",
           "build_states", "lag_states", "variance_swap_pnl", "deflated_sharpe", "spa_test",
           "effective_trials", "spec_sweep", "SweepResult", "stationary_bootstrap_decomposition",
           "core_inference_grid", "automatic_hac_lags", "fixed_b_critical_value", "InferencePolicy",
           "policy_presets", "resolve_policy", "evaluate_inference_profiles", "inference_frontier",
           "alpha_survival_curve", "survival_summary", "default_linear_spans", "resolve_linear_spans"]
__version__ = "0.5.0"
