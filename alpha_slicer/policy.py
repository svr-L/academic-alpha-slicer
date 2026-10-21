from __future__ import annotations

"""Inference-policy presets for Academic Alpha Slicer v0.4c.

The user chooses a primary policy before looking at the result. The engine still evaluates all
presets and an inference frontier, so selecting a higher-power rule cannot hide how the claim
behaves under stricter standards.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .regression import NORMAL_CV, fixed_b_critical_value, hac_ols, size_adjusted_p


@dataclass(frozen=True)
class InferencePolicy:
    name: str
    hac_lags: int
    fixed_b: bool
    require_bootstrap_agreement: bool
    description: str
    conservativeness: int


def policy_presets(balanced_lags: int = 126) -> Dict[str, InferencePolicy]:
    """Default power/false-positive trade-off presets.

    Discovery intentionally accepts a shorter bandwidth and conventional critical values.
    Balanced is the v0.4b standard. Certification uses the balanced HAC rule and additionally
    requires the stationary-bootstrap percentile interval to exclude zero with the same sign.
    """
    return {
        "discovery": InferencePolicy(
            "discovery", 63, False, False,
            "higher-power screening: HAC(63), conventional critical value", 1,
        ),
        "balanced": InferencePolicy(
            "balanced", int(balanced_lags), True, False,
            f"balanced evidence: HAC({int(balanced_lags)}) with fixed-b critical value", 2,
        ),
        "certification": InferencePolicy(
            "certification", int(balanced_lags), True, True,
            f"capital/certification standard: HAC({int(balanced_lags)}) fixed-b + bootstrap agreement", 3,
        ),
    }


def resolve_policy(profile: str, balanced_lags: int = 126, custom_lags: Optional[int] = None,
                   custom_fixed_b: bool = True, custom_require_bootstrap: bool = False) -> InferencePolicy:
    name = str(profile).strip().lower()
    presets = policy_presets(balanced_lags)
    if name in presets:
        return presets[name]
    if name == "custom":
        lags = int(balanced_lags if custom_lags is None else custom_lags)
        return InferencePolicy(
            "custom", lags, bool(custom_fixed_b), bool(custom_require_bootstrap),
            f"custom policy: HAC({lags}), fixed_b={bool(custom_fixed_b)}, "
            f"bootstrap_agreement={bool(custom_require_bootstrap)}", 2,
        )
    raise ValueError(f"Unknown inference_profile={profile!r}; use discovery, balanced, certification, or custom")


def _bootstrap_core_row(bootstrap_inference: Optional[pd.DataFrame]):
    if bootstrap_inference is None or bootstrap_inference.empty or "core_unexplained" not in bootstrap_inference.index:
        return None
    return bootstrap_inference.loc["core_unexplained"]


def evaluate_inference_profiles(y: pd.Series, X: pd.DataFrame, annualization: int,
                                base_t_threshold: float, balanced_lags: int,
                                bootstrap_inference: Optional[pd.DataFrame], luck_stats: dict,
                                primary_profile: str = "balanced", custom_lags: Optional[int] = None,
                                custom_fixed_b: bool = True,
                                custom_require_bootstrap: bool = False) -> pd.DataFrame:
    """Evaluate all standard profiles and the selected custom profile if requested.

    ``candidate_pass`` combines the profile-specific inference rule with the same DSR hurdle used
    elsewhere. This makes the power choice transparent without changing the economic decomposition.
    """
    y = pd.Series(y).astype(float)
    X = pd.DataFrame(X).reindex(y.index)
    policies = list(policy_presets(balanced_lags).values())
    if str(primary_profile).lower() == "custom":
        policies.append(resolve_policy("custom", balanced_lags, custom_lags,
                                       custom_fixed_b, custom_require_bootstrap))
    b = _bootstrap_core_row(bootstrap_inference)
    rows = []
    for pol in policies:
        L = int(max(1, min(pol.hac_lags, max(1, len(y)//4))))
        fit = hac_ols(y, X, lags=L)
        alpha = float(fit["params"]["const"] * annualization)
        se = float(np.sqrt(max(fit["cov"].loc["const", "const"], 0.0)) * annualization)
        t = float(fit["tvalues"]["const"])
        cv = fixed_b_critical_value((L + 1) / len(y)) if pol.fixed_b else NORMAL_CV
        tcrit = float(base_t_threshold * cv / NORMAL_CV)
        hac_pass = bool(alpha > 0 and np.isfinite(t) and t >= tcrit)
        if b is None:
            boot_pass = False
            boot_available = False
        else:
            lo, hi = float(b["ci_low_ann"]), float(b["ci_high_ann"])
            boot_pass = bool((alpha > 0 and lo > 0) or (alpha < 0 and hi < 0))
            boot_available = True
        if pol.require_bootstrap_agreement and not boot_available:
            inference_pass = np.nan
        else:
            inference_pass = bool(hac_pass and (boot_pass if pol.require_bootstrap_agreement else True))
        dsr_pass = bool(np.isfinite(luck_stats.get("dsr", np.nan)) and luck_stats.get("dsr", np.nan) >= luck_stats.get("dsr_min", 0.95))
        candidate = np.nan if not np.isfinite(inference_pass) else bool(alpha > 0 and inference_pass and dsr_pass)
        rows.append({
            "profile": pol.name, "description": pol.description, "conservativeness": pol.conservativeness,
            "hac_lags": L, "fixed_b": pol.fixed_b, "alpha_ann": alpha, "se_ann": se, "t": t,
            "critical_value": cv, "t_threshold": tcrit, "p": size_adjusted_p(t, cv),
            "hac_pass": hac_pass, "bootstrap_required": pol.require_bootstrap_agreement,
            "bootstrap_available": boot_available, "bootstrap_pass": boot_pass,
            "dsr": luck_stats.get("dsr", np.nan), "dsr_min": luck_stats.get("dsr_min", np.nan),
            "dsr_pass": dsr_pass, "inference_pass": inference_pass, "candidate_pass": candidate,
            "primary": pol.name == str(primary_profile).lower(),
        })
    return pd.DataFrame(rows).set_index("profile")


def inference_frontier(y: pd.Series, X: pd.DataFrame, annualization: int, base_t_threshold: float,
                       hac_lags=(21, 63, 126), bootstrap_inference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Sensitivity surface across bandwidth and fixed-b choices, plus the stationary bootstrap.

    This is descriptive, not a new hypothesis test. ``positive_pass`` asks whether a positive core
    clears that specification's threshold. ``evidence_ratio`` is t / required-t for positive alpha.
    """
    y = pd.Series(y).astype(float)
    X = pd.DataFrame(X).reindex(y.index)
    rows = []
    rank = 0
    for fixed_b in (False, True):
        for lag in sorted(set(int(x) for x in hac_lags)):
            L = int(max(1, min(lag, max(1, len(y)//4))))
            fit = hac_ols(y, X, lags=L)
            alpha = float(fit["params"]["const"] * annualization)
            se = float(np.sqrt(max(fit["cov"].loc["const", "const"], 0.0)) * annualization)
            t = float(fit["tvalues"]["const"])
            cv = fixed_b_critical_value((L + 1) / len(y)) if fixed_b else NORMAL_CV
            tcrit = float(base_t_threshold * cv / NORMAL_CV)
            rank += 1
            rows.append({
                "spec": f"HAC({L})" + (" fixed-b" if fixed_b else " conventional"),
                "family": "HAC", "hac_lags": L, "fixed_b": fixed_b,
                "alpha_ann": alpha, "se_ann": se, "t": t, "t_threshold": tcrit,
                "evidence_ratio": t / tcrit if tcrit else np.nan,
                "positive_pass": bool(alpha > 0 and t >= tcrit), "rank": rank,
            })
    b = _bootstrap_core_row(bootstrap_inference)
    if b is not None:
        a, se = float(b["estimate_ann"]), float(b["bootstrap_se_ann"])
        lo, hi = float(b["ci_low_ann"]), float(b["ci_high_ann"])
        rows.append({
            "spec": "Stationary bootstrap", "family": "bootstrap", "hac_lags": np.nan,
            "fixed_b": np.nan, "alpha_ann": a, "se_ann": se,
            "t": a/se if se > 0 else np.nan, "t_threshold": NORMAL_CV,
            "evidence_ratio": (a/se)/NORMAL_CV if se > 0 else np.nan,
            "positive_pass": bool(a > 0 and lo > 0), "rank": rank + 1,
        })
    return pd.DataFrame(rows).set_index("spec")
