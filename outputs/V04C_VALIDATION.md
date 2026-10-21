# v0.4c validation

## Regression suite

- `36 passed`
- `1 skipped` optional Tail-Aware parity test when `TAIL_AWARE_CORE` is not supplied
- End-to-end `Restart & Run All`: zero cell execution errors

## Compatibility

`balanced` is defined to reproduce the v0.4b decision statistic: traded-only spanning regression, HAC(126), fixed-b critical value. The v0.4b t and SE remain explicitly stored as audit fields.

## Policy-boundary regression test

A deterministic AR(1) alpha series is intentionally placed near the decision boundary. The alpha point estimate is identical under every profile.

| profile | t | required t | DSR | candidate |
|---|---:|---:|---:|---|
| discovery | 2.01045 | 2.00000 | 0.99943 | yes |
| balanced | 2.00312 | 2.15499 | 0.99943 | no |
| certification | 2.00312 | 2.15499 | 0.99943 | unavailable without bootstrap in this standalone boundary example |

This is the intended v0.4c behavior: the engine exposes the power/false-positive trade-off instead of letting an estimator switch silently change the claim.

## Interpretation rule

The selected Slicer profile is a declared research preference, not a way to choose whichever rule passes. `inference_profile_precommitted=True` records an ex-ante choice. Every result still reports all standard presets and the full frontier. The Tail-Aware Referee remains a separate frozen certification layer.
