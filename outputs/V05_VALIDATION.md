# v0.5 validation

## Regression suite

- **43 passed, 1 skipped** with plugin autoload disabled.
- The skipped test is the optional external Tail-Aware parity test when its path is not supplied.
- New v0.5 survival tests: canonical hierarchy, full-core equality, exclusion of non-traded controls, custom-span validation, descriptive non-monotonicity, and missing-factor errors.

## Notebook

- `Academic_Alpha_Slicer.ipynb`: **Restart & Run All successful**.
- Executed code cells: **22/22**.
- Execution errors: **0**.

## Known-truth survival check

- Construction net core: **1.00%/yr**.
- Final survival/core estimate: **1.17%/yr**.
- Final survival intercept equals `summary.core_unexplained_alpha_ann`: **True**.
- Final survival intercept equals Shapley `core_unexplained`: **True**.
- Non-traded nonlinear/tail-shape terms are absent from every priced survival span: **True**.
- Final/raw survival ratio on this demo: **0.234**.

The curve is not constrained to fall monotonically. In the demo, adding the traded-tail block raises the intercept because the strategy has exposure to a negative-premium variance leg; v0.5 reports that economic effect rather than forcing one-way alpha erosion.
