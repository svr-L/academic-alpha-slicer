# v0.5.1 Reference-Anchored Academic Alpha: validation record

## Scientific correction

- Primary estimand: the intercept of the ex-ante selected academic reference regression, **not** raw net mean.
- The added factor blocks, plus the expanded-model residual intercept, must sum to that original alpha.
- The separate structural identity includes reference-beta reallocation; zero is **not** assumed to be the economic premium of the untraded nonlinear/tail-shape diagnostics.
- An explained component is **still a component of the original reference-model alpha**. No conclusion about skill follows from its explainability alone.

## Tests actually executed

- `python -m pytest -q tests/test_v051_academic_alpha.py tests/test_slicer.py tests/test_v05_alpha_survival.py tests/test_v04b_core_inference.py tests/test_v04c_inference_policy.py`: **38 passed**.
- First eight cells of the package notebook executed with nbclient without errors.
- First nine cells of the separate standalone notebook executed in an empty working directory, verifying the embedded package, with no `atlas` dependency: PASS.
- The entire historical Monte Carlo test suite has **not been re-run** for this patch (a full-suite attempt exceeded the execution window). Its results are retained as historical v0.5 evidence, not claimed to be new v0.5.1 validation.

## Synthetic demo, seed 14, 3,000 sessions, baseline MKT

| Annualized estimand | Value |
|---|---:|
| Raw mean, legacy chart | +5.6863% |
| Original CAPM academic alpha, **primary** | +0.7067% |
| Extra linear factor contribution (Shapley) | +0.4090% |
| Conditional factor contribution (Shapley) | +1.1877% |
| Traded tail contribution (Shapley) | -1.8036% |
| Expanded-model residual intercept | +0.9136% |
| **Add-up** | **+0.7067%** |

All values in this table come from a **synthetic** known-construction demonstration; **not** from live ETF, MacroAnchor or TAA results. The synthetic illustration shows why decomposing the raw mean and decomposing reference alpha are not interchangeable. Signs and percentages can differ with the specified baseline and sample.

For 50 geometric-block bootstrap draws of this illustrative synthetic case, the alpha-reference 95% percentile interval included zero. The **50-draw result is a smoke test**, not an adequately precise published confidence interval. Set `academic_bootstrap_reps` to a larger precommitted value for empirical publication.

## Release status

- New v0.5.1 copy, leaving the frozen original v0.5 ZIP untouched.
- No remote GitHub repositories changed.
- Cross-project factor-momentum/MacroAnchor/TAA alpha findings from v0.5 need to be *re-expressed relative to a stated reference academic benchmark* before presenting them as dissected academic alpha.
