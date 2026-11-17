# Academic Alpha Slicer — realignment of the scientific question

**Source of the correction:** *JPrep.pdf*, Projects Defense discussion, 16 November 2026, pp. 64–72. User's original idea: dissect academic alpha, not subtract everything explainable to manufacture a putative `genuine alpha`.

## 1. The object being decomposed

For excess returns `y` and ex-ante chosen academic benchmark factors `B`, run `y = a0 + B b0 + e0`. The intercept `a0` is **academic alpha relative to B**, not the expected total return. It may contain timing alpha, compensated nonlinear risks, omitted factor exposures, or sampling noise. The identity and terminology **do not pronounce whether the manager has skill**.

## 2. Regression expansion and an exact identity

Let `X` be additional *traded* linear factors, factor-managed/conditional strategies, and option/tail payoff returns. Fit on the EXACT same observations:

`y = a1 + B b1 + X theta + e1`

Then the sample means satisfy the identity:

`a0 = a1 + (b1 - b0)' mean(B) + theta' mean(X)`.

This **reference-beta reallocation** is essential. It is an omitted-variable accounting effect, not an independently priced source of alpha. Omitting it causes the subcomponents not to sum to the original academic alpha.

## 3. Two complementary estimands

**Structural identity:** assigns factor terms `theta' mean(X)`, adjustment `(b1 - b0)' mean(B)`, and `a1`. Useful for audit; attribution depends on specification and factor correlation.

**Reference-anchored Shapley:** for additional blocks `S`, define `v(S) = a0 - a(B + S)`; the Shapley contributions across `additional_linear`, `conditional`, `tail` plus the final `a1` sum EXACTLY to `a0`. The academic baseline is always included. This is order-invariant as an accounting rule, **not causal identification**. Shared information and benchmark-beta changes are allocated over added blocks.

Both decompositions are signed; an explained component may increase or decrease the reference alpha. A 0 residual after expanding the model does **not** mean the original alpha never existed or cannot represent genuine discovery/skill.

## 4. Untraded nonlinear exposures

Orthogonalized `m²`, `m³`, downside and crash transformations remain valuable for detecting nonlinear risks and asymmetries in the observed strategy. By Frisch–Waugh–Lovell, they cannot change the reference-anchored `a1` when constructed orthogonally to the constant and traded span on the SAME sample. That is a numerical/statistical convention and does not establish zero economic premia. The new `academic_alpha.diagnostics` denotes **unidentified**, not `0`, when assigning an economic alpha component. Existing `result.nontraded` reports loadings, HAC tests and hypothetical break-even premia.

To attribute a nontraded shape component economically one must introduce appropriately priced traded payoff(s), or develop an explicit separate pricing model (e.g. Q/P), with assumptions and uncertainty recorded. Never add arbitrary `gamma * mean(m²)` to the priced attribution.

## 5. Model credibility and scope

The legacy v0.5 raw-mean waterfall, raw-mean Shapley and the Alpha Survival Curve remain available as supplementary analyses. All original core inferential checks remain valid **for the intercept of the expanded traded model**, not automatically for every historical alpha component. The optional new stationary bootstrap refits the reference and all added factor subset regressions jointly. The baseline intercept also receives HAC/fixed-b-calibrated confidence intervals. This measures sampling uncertainty conditional on the selected factor sets; it does not establish causal identification or control the whole history of research trial selection.

A user may choose `academic_reference_factors=('MKT',)` or a precommitted multi-factor benchmark. It is not legitimate to choose it after seeing which yields the most persuasive attribution. Refit of states themselves under bootstrap is not included; state definitions and the common sample are conditioned on.

## 6. Research-governance rules

- Keep original v0.5 untouched as a reproducible baseline.
- Any new modelling of priced risk (conditional factor pricing, option P&L, P vs Q, spectral diagnostics) must be reviewed as a distinct extension, not backfilled into the meaning of current outputs.
- No claim of actual ETF/MacroAnchor/TAA performance is created by this mathematical repair; empirical strategy cases must be rerun against the new reference-anchored estimand.
- Treat alpha, economically explained return, compensated risk, and skill as four different questions. Do not use `unexplained` as a synonym for `genuine`.
