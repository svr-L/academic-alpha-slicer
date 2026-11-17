# v0.5 Alpha Survival validation

Known-truth demo, seed 14, balanced policy. The survival path is pre-declared and uses priced/traded returns only.

| stage        |   depth | family_added   |   n_priced_terms |   alpha_ann |   erosion_from_prior_ann |   survival_ratio_vs_raw |        t |   t_threshold |   core_dsr | screen_pass   |
|:-------------|--------:|:---------------|-----------------:|------------:|-------------------------:|------------------------:|---------:|--------------:|-----------:|:--------------|
| RAW_NET      |       0 | raw            |                0 |     0.04986 |                nan       |                 1       |  1.74681 |       2.11055 |    0.97366 | False         |
| MKT          |       1 | linear         |                1 |     0.00052 |                  0.04934 |                 0.0104  |  0.02604 |       2.11055 |    0.51028 | False         |
| FF3          |       2 | linear         |                3 |    -0.00284 |                  0.00336 |                -0.05699 | -0.14022 |       2.11055 |    0.44304 | False         |
| FF3+MOM      |       3 | linear         |                4 |    -0.00488 |                  0.00204 |                -0.09793 | -0.29379 |       2.11055 |    0.39322 | False         |
| +CONDITIONAL |       4 | conditional    |                8 |    -0.00492 |                  4e-05   |                -0.09874 | -0.30664 |       2.11055 |    0.38841 | False         |
| +TRADED_TAIL |       5 | tail           |               10 |     0.01169 |                 -0.01661 |                 0.23439 |  0.82995 |       2.11055 |    0.78021 | False         |

## Structural checks

- final_equals_core: **True**
- final_equals_shapley_core: **True**
- nontraded_absent: **True**

- construction net core truth: **1.00%/yr**
- estimated final core: **1.17%/yr**
- final/raw survival ratio: **0.234**

The curve is not required to be monotone. In this demo the traded variance leg has a negative premium, so adding the traded-tail block can raise the residual intercept; the engine reports that rather than imposing a one-way erosion narrative.
