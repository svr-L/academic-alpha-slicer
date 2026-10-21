from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Mapping, Optional
import warnings
import numpy as np
import pandas as pd

from .config import SliceConfig
from .features import BLOCK_TYPES, build_feature_blocks, build_states, lag_states
from .regression import NORMAL_CV, core_hac_settings, hac_ols, orthogonalize, size_adjusted_p, wald_test
from .nonlinear import nonlinear_order_table
from .conditional import conditional_alpha_table
from .tail import tail_diagnostics, tail_risk_flags
from .luck import luck_gate
from .inference import core_inference_grid, stationary_bootstrap_decomposition, inference_fragility_summary
from .policy import resolve_policy, evaluate_inference_profiles, inference_frontier
from .survival import alpha_survival_curve, survival_summary

PRICING_CONVENTION = ("alpha = intercept on traded blocks (charged at their sample mean); non-traded "
                      "blocks orthogonalized to them (zero price, cannot move alpha); states lagged; "
                      "state dummies reported as regime alphas")
CORE_SE_METHOD = ("traded-only spanning regression (same intercept as the full design), Bartlett HAC, "
                  "fixed-b critical value")


@dataclass
class SliceResult:
    waterfall: pd.DataFrame
    shapley: pd.DataFrame
    nested_models: pd.DataFrame
    nonlinear_order: pd.DataFrame
    conditional_alpha: pd.DataFrame
    tail_diagnostics: pd.Series
    coefficients: pd.DataFrame
    core_returns: pd.Series
    states: pd.DataFrame
    feature_blocks: Dict[str, pd.DataFrame]
    summary: pd.Series
    nontraded: pd.DataFrame
    tail_flags: pd.DataFrame
    block_types: Dict[str, str]
    core_inference: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_inference: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_draws: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    inference_frontier: pd.DataFrame = field(default_factory=pd.DataFrame)
    alpha_survival: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PreparedDesign:
    """Everything the regressions need, on one common estimation sample."""
    y: pd.Series
    factors: pd.DataFrame
    benchmark: pd.Series
    market: pd.Series
    states: pd.DataFrame
    blocks: Dict[str, pd.DataFrame]          # traded raw + non-traded orthogonalized
    nontraded_raw: Dict[str, pd.DataFrame]   # non-traded blocks before orthogonalization
    nontraded_raw_means: pd.Series
    names: List[str]
    traded: List[str]
    controls: List[str]
    block_types: Dict[str, str]
    core_inference: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_inference: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_draws: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped_columns: List[str] = field(default_factory=list)

    @property
    def X_traded(self) -> pd.DataFrame:
        return _concat_blocks(self.blocks, self.traded, self.y.index)

    @property
    def X_full(self) -> pd.DataFrame:
        return _concat_blocks(self.blocks, self.names, self.y.index)


def _concat_blocks(blocks, names, index):
    xs = [blocks[n] for n in names if n in blocks and blocks[n].shape[1] > 0]
    return pd.concat(xs, axis=1) if xs else pd.DataFrame(index=index)


def _alpha_of(fit, ann):
    return (float(fit['params']['const'] * ann), float(fit['tvalues']['const']),
            float(fit['pvalues']['const']))


def _waterfall(y, blocks, traded, cfg, lags=None, cv=None):
    """Sequential erosion over the traded blocks, in the declared order.

    Non-traded blocks have no stage: they are orthogonal to the traded regressors and to the
    constant, so they cannot move the intercept at any stage. Each stage is a traded-only
    regression; since v0.4b its t uses the core bandwidth and p the matching critical value.
    """
    ann = cfg.annualization
    lags = cfg.hac_lags if lags is None else int(lags)
    cv = NORMAL_CV if cv is None else float(cv)

    def _stats(fit):
        a, t, _ = _alpha_of(fit, ann)
        return a, t, size_adjusted_p(t, cv)

    a, t, p = _stats(hac_ols(y, None, lags=lags))
    rows = [{'stage': 'raw_net_mean', 'alpha_ann': a, 'alpha_t': t, 'p': p,
             'erosion_from_prior': np.nan, 'n': len(y)}]
    prev, used = a, []
    for name in traded:
        used.append(name)
        fit = hac_ols(y, _concat_blocks(blocks, used, y.index), lags=lags)
        a, t, p = _stats(fit)
        rows.append({'stage': f'after_{name}', 'alpha_ann': a, 'alpha_t': t, 'p': p,
                     'erosion_from_prior': prev - a, 'n': fit['n']})
        prev = a
    return pd.DataFrame(rows).set_index('stage')


def _shapley_alpha(y, blocks, traded, controls, cfg):
    """Shapley attribution of the mean across TRADED blocks.

    v(S) = raw annualized mean - annualized intercept after the traded blocks in S. Averaging
    marginal contributions over all orderings makes the grouped attribution invariant to the
    presentation order. Non-traded blocks receive exactly zero (zero price by construction).
    """
    ann = cfg.annualization
    if len(traded) > cfg.shapley_max_blocks:
        raise ValueError(f'{len(traded)} traded blocks exceed shapley_max_blocks={cfg.shapley_max_blocks}')
    raw = float(y.mean() * ann)
    cache = {(): raw}

    def alpha_after(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            X = _concat_blocks(blocks, list(key), y.index)
            cache[key] = float(hac_ols(y, X, lags=cfg.hac_lags)['params']['const'] * ann)
        return cache[key]

    contrib = {n: [] for n in traded}
    for perm in permutations(traded):
        S, a_prev = [], raw
        for n in perm:
            S2 = S + [n]
            a_new = alpha_after(S2)
            contrib[n].append(a_prev - a_new)
            S, a_prev = S2, a_new
    rows = []
    for n in traded:
        arr = np.asarray(contrib[n], float)
        rows.append({'block': n, 'pricing': 'traded: charged at mean', 'contribution_ann': arr.mean(),
                     'min_order_contribution': arr.min(), 'max_order_contribution': arr.max(),
                     'order_sensitivity': arr.max() - arr.min()})
    for n in controls:
        rows.append({'block': n, 'pricing': 'non-traded: zero price', 'contribution_ann': 0.0,
                     'min_order_contribution': 0.0, 'max_order_contribution': 0.0, 'order_sensitivity': 0.0})
    core = alpha_after(traded)
    rows.append({'block': 'core_unexplained', 'pricing': 'unexplained intercept',
                 'contribution_ann': core, 'min_order_contribution': core,
                 'max_order_contribution': core, 'order_sensitivity': 0.0})
    out = pd.DataFrame(rows).set_index('block')
    out['share_of_raw_mean'] = out['contribution_ann'] / raw if abs(raw) > 1e-12 else np.nan
    return out


def _nested_models(y, blocks, names, block_types, cfg):
    """Nested model audit: R2, BIC and alpha stability as blocks are added.

    Rows that include zero-priced controls show an ``alpha_t`` that treats the controls' in-sample
    means as known; it is reported for continuity only. Inference on the core uses the traded-only
    regression (``summary['core_alpha_t']``, ``core_inference``).
    """
    rows, coefs, used = [], [], []
    term_block = {c: n for n in names for c in blocks[n].columns}
    for name in names:
        used.append(name)
        fit = hac_ols(y, _concat_blocks(blocks, used, y.index), lags=cfg.hac_lags)
        model = '+'.join(used)
        rows.append({'model': model, 'alpha_ann': fit['params']['const'] * cfg.annualization,
                     'alpha_t': fit['tvalues']['const'], 'alpha_p': fit['pvalues']['const'],
                     'r2': fit['r2'], 'adj_r2': fit['adj_r2'], 'bic': fit['bic'],
                     'condition_number': fit['condition_number'], 'n': fit['n']})
        cc = pd.DataFrame({'coef': fit['params'], 't': fit['tvalues'], 'p': fit['pvalues']})
        cc['model'] = model
        cc['term'] = cc.index
        cc['block'] = cc['term'].map(term_block).fillna('intercept')
        cc['pricing'] = cc['block'].map(lambda b: block_types.get(b, 'intercept'))
        coefs.append(cc.reset_index(drop=True))
    return pd.DataFrame(rows).set_index('model'), pd.concat(coefs, ignore_index=True)


def _nontraded_table(y, blocks, names, controls, fullfit, raw_means, core_ann, cfg):
    """Exposure to non-traded (zero-price) regressors and what pricing them would imply.

    Term rows (coefficients are on the orthogonalized regressor, which by Frisch-Waugh-Lovell
    equals the coefficient on the raw regressor in a regression that includes the traded blocks):
      coef, t                 loading in the full model
      raw_mean_ann            sample mean of the RAW regressor x 252 (NOT a price)
      naive_charge_ann        coef * raw_mean_ann: what charging the raw mean as if it were a
                              premium would remove (the v0.1 convention; source of artificial alpha)
      breakeven_premium_ann   premium per unit of the unspanned regressor (long side, annualized)
                              that would drive the core to zero: core / coef
      breakeven_vs_raw_mean   breakeven premium as a multiple of the raw mean; e.g. -0.3 on
                              MKT_POW2 = a variance premium of 30% of realized variance
    Block rows: HAC Wald test of the block and its incremental R2.
    """
    ann = cfg.annualization
    rows = []
    for b in controls:
        terms = list(blocks[b].columns)
        for term in terms:
            g = float(fullfit['params'][term])
            mu = float(raw_means.get(term, np.nan))
            be = core_ann / g if abs(g) > 1e-12 else np.nan
            rows.append({'block': b, 'term': term, 'coef': g, 't': float(fullfit['tvalues'][term]),
                         'raw_mean_ann': mu * ann, 'naive_charge_ann': g * mu * ann,
                         'breakeven_premium_ann': be,
                         'breakeven_vs_raw_mean': be / (mu * ann) if abs(mu) > 1e-15 and np.isfinite(be) else np.nan})
        W, p, k = wald_test(fullfit, terms)
        others = [n for n in names if n != b]
        fit_wo = hac_ols(y, _concat_blocks(blocks, others, y.index), lags=cfg.hac_lags)
        rows.append({'block': b, 'term': '(block)', 'wald_chi2_hac': W, 'wald_p_hac': p, 'n_terms': k,
                     'incremental_r2': fullfit['r2'] - fit_wo['r2'],
                     'naive_charge_ann': float(sum(r['naive_charge_ann'] for r in rows
                                                   if r['block'] == b and r['term'] != '(block)'))})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(['block', 'term'])


class AlphaSlicer:
    def __init__(self, config: Optional[SliceConfig] = None):
        self.config = config or SliceConfig()

    def prepare(self, strategy_net: pd.Series, factors: pd.DataFrame,
                benchmark: Optional[pd.Series] = None, states: Optional[pd.DataFrame] = None,
                tail_factors: Optional[pd.DataFrame] = None) -> PreparedDesign:
        """Align inputs, lag states, fix the common sample and build the priced design."""
        cfg = self.config
        y = pd.Series(strategy_net, name=getattr(strategy_net, 'name', None) or 'strategy_net').astype(float).sort_index()
        F = pd.DataFrame(factors).astype(float).sort_index()
        if benchmark is None:
            if cfg.market_col not in F.columns:
                raise ValueError('benchmark required when market_col not present')
            bench = F[cfg.market_col].rename('benchmark')
        else:
            bench = pd.Series(benchmark, name='benchmark').astype(float).sort_index()
        idx = y.index.intersection(F.index).intersection(bench.index)
        y, F, bench = y.loc[idx], F.loc[idx], bench.loc[idx]

        unknown = [b for b in cfg.block_order if b not in BLOCK_TYPES]
        if unknown:
            raise ValueError(f'Unknown blocks {unknown}; valid: {list(BLOCK_TYPES)}')
        names = list(dict.fromkeys(cfg.block_order))
        if 'tail' in names and tail_factors is None and 'tail_shape' not in names:
            warnings.warn("block 'tail' holds only supplied traded tail factors since v0.2; the "
                          "non-traded downside basis is the separate block 'tail_shape'.")

        if states is None:
            states_obs = build_states(bench, cfg.vol_lookback, cfg.vol_threshold_lookback,
                                      cfg.drawdown_threshold)
        else:
            states_obs = pd.DataFrame(states).astype(float).sort_index().reindex(idx)
        states_obs = states_obs.loc[:, states_obs.notna().any()]
        S = lag_states(states_obs, cfg.state_lag).reindex(idx)
        TF = None
        if tail_factors is not None and 'tail' in names:
            TF = pd.DataFrame(tail_factors).astype(float).sort_index().reindex(idx)

        # One common estimation sample for every regression, so that all attributions add up.
        parts = [y.rename('__y__'), F.add_prefix('__F__'), bench.rename('__bench__'), S.add_prefix('__S__')]
        if TF is not None:
            parts.append(TF.add_prefix('__TF__'))
        sample = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna().index
        if len(sample) < cfg.min_obs:
            raise ValueError(f'Need at least {cfg.min_obs} aligned observations; got {len(sample)}')
        y, F, bench, S = y.loc[sample], F.loc[sample], bench.loc[sample], S.loc[sample]
        if TF is not None:
            TF = TF.loc[sample]

        fd = build_feature_blocks(F, bench, S if 'conditional' in names else None,
                                  cfg.market_col, cfg.nonlinear_max_order, cfg.tail_quantile, TF)
        blocks = {n: fd.blocks[n] for n in names if fd.blocks[n].shape[1] > 0}
        names = [n for n in names if n in blocks]
        block_types = {n: BLOCK_TYPES[n] for n in names}
        traded = [n for n in names if block_types[n] == 'traded']
        controls = [n for n in names if block_types[n] == 'nontraded']
        nontraded_raw = {n: blocks[n] for n in controls}
        X_traded = _concat_blocks(blocks, traded, sample)
        dropped = list(fd.dropped_columns)
        for n in controls:
            Z = orthogonalize(blocks[n], X_traded)
            # a non-traded column fully spanned by traded factors (e.g. m^2 when a traded
            # variance swap is supplied) is entirely priced by them: nothing is left to keep.
            keep = Z.std(ddof=0) > 1e-10 * blocks[n].std(ddof=0).replace(0, 1.0)
            dropped += list(Z.columns[~keep])
            blocks[n] = Z.loc[:, keep]
        names = [n for n in names if blocks[n].shape[1] > 0]
        controls = [n for n in controls if n in names]
        block_types = {n: block_types[n] for n in names}
        market = F[cfg.market_col] if cfg.market_col in F.columns else bench
        return PreparedDesign(y=y, factors=F, benchmark=bench, market=market, states=S, blocks=blocks,
                              nontraded_raw=nontraded_raw, nontraded_raw_means=fd.nontraded_raw_means,
                              names=names, traded=traded, controls=controls, block_types=block_types,
                              dropped_columns=dropped)

    def fit(self, strategy_net: pd.Series, factors: pd.DataFrame,
            benchmark: Optional[pd.Series] = None, strategy_gross: Optional[pd.Series] = None,
            costs: Optional[pd.Series] = None, states: Optional[pd.DataFrame] = None,
            tail_factors: Optional[pd.DataFrame] = None,
            nontraded_premia: Optional[Mapping[str, float]] = None) -> SliceResult:
        """Slice the mean net return of a strategy.

        strategy_net     daily net excess returns
        factors          daily TRADED factor excess returns (include MKT or pass benchmark)
        benchmark        market/benchmark return (defaults to factors[market_col])
        strategy_gross   optional gross returns (implementation drag)
        costs            optional daily cost series (alternative to strategy_gross)
        states           optional state variables AS OBSERVED AT THE CLOSE of each date; the
                         engine lags them by config.state_lag. Default: HIGH_VOL / DRAWDOWN.
        tail_factors     optional TRADED tail/convexity excess returns (e.g. a variance-swap or
                         put-writing P&L); charged at their mean like any traded factor
        nontraded_premia optional {non-traded term: assumed annualized premium per unit of its
                         unspanned part, long side}; a what-if, never used by the main numbers
        """
        cfg = self.config
        ann = cfg.annualization
        selected_policy = resolve_policy(
            cfg.inference_profile, balanced_lags=cfg.core_hac_lags,
            custom_lags=cfg.core_hac_lags, custom_fixed_b=cfg.core_fixed_b,
            custom_require_bootstrap=cfg.inference_custom_require_bootstrap,
        )
        if selected_policy.require_bootstrap_agreement and int(cfg.inference_bootstrap_reps) <= 0:
            raise ValueError("certification/custom bootstrap-required profile needs inference_bootstrap_reps > 0")
        d = self.prepare(strategy_net, factors, benchmark, states, tail_factors)
        y, F, bench, S, blocks = d.y, d.factors, d.benchmark, d.states, d.blocks
        names, traded, controls, block_types = d.names, d.traded, d.controls, d.block_types
        sample = y.index

        X_full = d.X_full
        X_traded = d.X_traded
        fullfit = hac_ols(y, X_full, lags=cfg.hac_lags)
        core_ann, core_t_v03, _ = _alpha_of(fullfit, ann)
        core_se_v03 = float(np.sqrt(max(fullfit['cov'].loc['const', 'const'], 0.0)) * ann)

        # v0.4c primary decision statistic: profile-selected bandwidth/critical value.
        # Balanced exactly reproduces v0.4b; certification adds bootstrap agreement later.
        L_core, cv_core = core_hac_settings(len(y), selected_policy.hac_lags, selected_policy.fixed_b)
        t_crit = float(cfg.core_t_threshold * cv_core / NORMAL_CV)
        corefit = hac_ols(y, X_traded, lags=L_core)
        core_t = float(corefit['tvalues']['const'])
        core_se = float(np.sqrt(max(corefit['cov'].loc['const', 'const'], 0.0)) * ann)
        core_p = size_adjusted_p(core_t, cv_core)

        # Preserve the exact v0.4b statistic as an audit anchor even when another profile is primary.
        L_v04b, cv_v04b = core_hac_settings(len(y), cfg.core_hac_lags, True)
        fit_v04b = hac_ols(y, X_traded, lags=L_v04b)
        core_t_v04b = float(fit_v04b['tvalues']['const'])
        core_se_v04b = float(np.sqrt(max(fit_v04b['cov'].loc['const', 'const'], 0.0)) * ann)
        fwl_gap = abs(float(corefit['params']['const']) * ann - core_ann)
        if fwl_gap > 1e-8 * max(1.0, abs(core_ann)):
            warnings.warn(f'traded-only and full-design intercepts differ by {fwl_gap:.2e}; '
                          'the non-traded controls are not orthogonal to the traded design')
        traded_terms = list(X_traded.columns)
        if traded_terms:
            charge = X_traded[traded_terms].mul(fullfit['params'][traded_terms], axis=1).sum(axis=1)
            core_returns = (y - charge).rename(y.name)
        else:
            core_returns = y.copy()

        waterfall = _waterfall(y, blocks, traded, cfg, L_core, cv_core)
        shapley = _shapley_alpha(y, blocks, traded, controls, cfg)
        nested, coefs = _nested_models(y, blocks, names, block_types, cfg)
        nonlinear = nonlinear_order_table(y, F, bench, cfg.market_col, cfg.nonlinear_max_order,
                                          cfg.hac_lags, ann)
        cond = conditional_alpha_table(y, X_traded, S, L_core, ann, cfg.min_regime_obs, crit=cv_core)
        tail = tail_diagnostics(core_returns, bench, cfg.tail_quantile)
        flags = tail_risk_flags(y, X_traded, d.market, core_returns, cfg.hac_lags, cfg.tail_quantile,
                                cfg.flag_t_threshold, cfg.flag_es_q, cfg.flag_es_ratio)
        nontraded = _nontraded_table(y, blocks, names, controls, fullfit, d.nontraded_raw_means,
                                     core_ann, cfg)

        # Sensitivity to the pricing convention of the non-traded blocks (same sample, same terms).
        core_demeaned = core_ann
        naive_total = 0.0
        if controls:
            kept = {n: list(blocks[n].columns) for n in controls}
            raw_kept = pd.concat([d.nontraded_raw[n][kept[n]] for n in controls], axis=1)
            fit_dm = hac_ols(y, pd.concat([X_traded, raw_kept - raw_kept.mean()], axis=1), lags=cfg.hac_lags)
            core_demeaned = float(fit_dm['params']['const'] * ann)
            naive_total = float(sum(fit_dm['params'][c] * raw_kept[c].mean() * ann for c in raw_kept.columns))

        core_after_premia = np.nan
        if nontraded_premia:
            nt_terms = [c for b in controls for c in blocks[b].columns]
            bad = [k for k in nontraded_premia if k not in nt_terms]
            if bad:
                raise ValueError(f'nontraded_premia keys {bad} not in non-traded terms {nt_terms}')
            core_after_premia = core_ann - float(sum(fullfit['params'][k] * v for k, v in nontraded_premia.items()))

        raw_ann = float(y.mean() * ann)
        gross_ann = np.nan
        cost_drag = np.nan
        g_aligned = None
        c_aligned = None
        if strategy_gross is not None:
            g_aligned = pd.Series(strategy_gross).astype(float).reindex(sample)
            common = g_aligned.dropna().index
            gross_ann = float(g_aligned.loc[common].mean() * ann)
            cost_drag = float((g_aligned.loc[common] - y.loc[common]).mean() * ann)
        elif costs is not None:
            c_aligned = pd.Series(costs).astype(float).reindex(sample)
            common = c_aligned.dropna().index
            cost_drag = float(c_aligned.loc[common].mean() * ann)
            gross_ann = float(y.loc[common].mean() * ann + cost_drag)

        # Inference audit (v0.4, on the traded-only design since v0.4b): sensitivity of the core's
        # uncertainty to the dependence model, plus bootstrap bands around the decomposition.
        bootstrap_inference = pd.DataFrame()
        bootstrap_draws = pd.DataFrame()
        if int(cfg.inference_bootstrap_reps) > 0:
            bootstrap_inference, bootstrap_draws = stationary_bootstrap_decomposition(
                y=y, blocks=blocks, traded=traded, annualization=ann,
                reps=cfg.inference_bootstrap_reps, mean_block=cfg.inference_bootstrap_mean_block,
                ci=cfg.inference_bootstrap_ci, seed=cfg.inference_bootstrap_seed,
                strategy_gross=g_aligned, costs=c_aligned, point_shapley=shapley, raw_point=raw_ann,
            )
        core_inference = core_inference_grid(
            y=y, X=X_traded, annualization=ann, primary_lags=L_core,
            hac_lag_grid=[lag for lag in cfg.inference_hac_lags if lag <= max(1, len(y) // 4)],
            include_auto=cfg.inference_include_auto_hac,
            include_prewhite=cfg.inference_include_prewhite_hac,
            bootstrap_inference=bootstrap_inference, fixed_b=cfg.core_fixed_b,
            X_full=X_full, v03_lags=cfg.hac_lags,
        )
        infer_summary = inference_fragility_summary(core_inference, threshold=cfg.core_t_threshold)

        # v0.5: economic/model-span robustness. This is deliberately separate from the v0.4c
        # inference frontier: the point question is how much alpha survives richer *priced* spans.
        alpha_survival = pd.DataFrame()
        survival_stats = pd.Series(dtype=object)
        if cfg.survival_enabled:
            alpha_survival = alpha_survival_curve(
                y=y, factors=F,
                conditional=blocks.get('conditional', pd.DataFrame(index=y.index)),
                tail=blocks.get('tail', pd.DataFrame(index=y.index)),
                market_col=cfg.market_col, annualization=ann, hac_lags=L_core,
                critical_value=cv_core, base_t_threshold=cfg.core_t_threshold,
                luck_n_trials=cfg.luck_n_trials, luck_dsr_min=cfg.luck_dsr_min,
                custom_linear_spans=cfg.survival_custom_linear_spans,
                include_default_linear_spans=cfg.survival_include_default_linear_spans,
                include_conditional=cfg.survival_include_conditional,
                include_tail=cfg.survival_include_tail,
            )
            survival_stats = survival_summary(alpha_survival)

        abs_contrib = shapley.loc[traded, 'contribution_ann'].abs() if traded else pd.Series(dtype=float)
        dominant = abs_contrib.idxmax() if len(abs_contrib) else 'none'
        fired = list(flags.index[flags['fired']])
        unpriced_tail = bool(fired)
        luck = luck_gate(core_returns, core_t, cfg.luck_n_trials, cfg.luck_dsr_min, t_crit)
        luck_for_profiles = dict(luck)
        luck_for_profiles['dsr_min'] = cfg.luck_dsr_min
        profile_comparison = evaluate_inference_profiles(
            y=y, X=X_traded, annualization=ann, base_t_threshold=cfg.core_t_threshold,
            balanced_lags=cfg.core_hac_lags, bootstrap_inference=bootstrap_inference,
            luck_stats=luck_for_profiles, primary_profile=cfg.inference_profile,
            custom_lags=cfg.core_hac_lags, custom_fixed_b=cfg.core_fixed_b,
            custom_require_bootstrap=cfg.inference_custom_require_bootstrap,
        )
        frontier = inference_frontier(
            y=y, X=X_traded, annualization=ann, base_t_threshold=cfg.core_t_threshold,
            hac_lags=cfg.inference_frontier_hac_lags, bootstrap_inference=bootstrap_inference,
        )
        primary_name = str(cfg.inference_profile).lower()
        if primary_name not in profile_comparison.index:
            raise RuntimeError(f'Primary inference profile {primary_name!r} missing from profile comparison')
        primary_row = profile_comparison.loc[primary_name]
        candidate = bool(primary_row['candidate_pass'])
        standard_profiles = [x for x in ('discovery', 'balanced', 'certification') if x in profile_comparison.index]
        comparable = profile_comparison.loc[standard_profiles, 'candidate_pass'].dropna() if standard_profiles else pd.Series(dtype=object)
        profile_sensitive = bool(comparable.nunique() > 1) if len(comparable) else False
        robustness_score = float(frontier['positive_pass'].mean()) if len(frontier) else np.nan
        if candidate:
            label = 'core_unexplained_candidate_tail_flagged' if unpriced_tail else 'core_unexplained_candidate'
        elif len(abs_contrib) and abs_contrib.max() >= cfg.material_share * max(abs(raw_ann), 1e-6):
            label = f'{dominant}_dominated'
        else:
            label = 'inconclusive'
        if candidate and unpriced_tail:
            handoff = 'referee: hidden-risk gate (H2) first - core comes with unpriced tail exposure'
        elif candidate:
            # In-sample flags only see tail risk that showed up in the P&L: on the referee's H2
            # world they catch ~90%, so a clean candidate still needs the referee's H2 gate.
            handoff = ('referee: luck gate (H1 vs H3) and hidden-risk gate (H2) - no unpriced tail '
                       'exposure visible in-sample, which does not rule out unrealized crash risk')
        elif unpriced_tail:
            handoff = 'no core to certify; unpriced tail exposure present (price it with a traded tail factor)'
        else:
            handoff = 'none'

        sel = nonlinear.index[nonlinear.selected_bic]
        _date = lambda x: str(x.date()) if hasattr(x, 'date') else str(x)
        summary = pd.Series({
            'n_obs': len(y), 'sample_start': _date(sample[0]), 'sample_end': _date(sample[-1]),
            'raw_net_mean_ann': raw_ann, 'gross_mean_ann': gross_ann,
            'implementation_cost_drag_ann': cost_drag,
            'traded_premia_explained_ann': raw_ann - core_ann,
            'core_unexplained_alpha_ann': core_ann, 'core_alpha_t': core_t, 'core_alpha_p': core_p,
            'core_alpha_se_ann': core_se,
            'core_ci_low_ann': core_ann - cv_core * core_se, 'core_ci_high_ann': core_ann + cv_core * core_se,
            'core_t_critical': t_crit, 'core_ci_critical': cv_core, 'core_hac_lags_used': L_core,
            'core_se_method': f"traded-only spanning regression; profile={selected_policy.name}; HAC({L_core}); fixed_b={selected_policy.fixed_b}",
            'inference_profile': selected_policy.name,
            'inference_profile_precommitted': bool(cfg.inference_profile_precommitted),
            'inference_profile_requires_bootstrap': bool(selected_policy.require_bootstrap_agreement),
            'primary_profile_candidate_pass': candidate,
            'profile_sensitive': profile_sensitive,
            'inference_frontier_robustness_score': robustness_score,
            'inference_frontier_specs': int(len(frontier)),
            'alpha_survival_stages': survival_stats.get('alpha_survival_stages', 0),
            'alpha_survival_final_stage': survival_stats.get('alpha_survival_final_stage', 'none'),
            'alpha_survival_raw_ann': survival_stats.get('alpha_survival_raw_ann', np.nan),
            'alpha_survival_final_ann': survival_stats.get('alpha_survival_final_ann', np.nan),
            'alpha_survival_final_vs_raw': survival_stats.get('alpha_survival_final_vs_raw', np.nan),
            'alpha_survival_screen_pass_share': survival_stats.get('alpha_survival_screen_pass_share', np.nan),
            'alpha_survival_span_fragile': survival_stats.get('alpha_survival_span_fragile', False),
            'alpha_survival_first_failure_stage': survival_stats.get('alpha_survival_first_failure_stage', 'none'),
            'alpha_survival_deepest_screen_pass_stage': survival_stats.get('alpha_survival_deepest_screen_pass_stage', 'none'),
            'alpha_survival_final_screen_pass': survival_stats.get('alpha_survival_final_screen_pass', False),
            'core_alpha_t_v04b_balanced': core_t_v04b, 'core_alpha_se_ann_v04b_balanced': core_se_v04b,
            'core_alpha_t_v03': core_t_v03, 'core_alpha_se_ann_v03': core_se_v03,
            'core_se_ratio_vs_v03': core_se / core_se_v03 if core_se_v03 > 0 else np.nan,
            'core_t_min_inference': infer_summary.get('core_t_min_inference', np.nan),
            'core_t_max_inference': infer_summary.get('core_t_max_inference', np.nan),
            'core_abs_t_min_inference': infer_summary.get('core_abs_t_min_inference', np.nan),
            'core_abs_t_max_inference': infer_summary.get('core_abs_t_max_inference', np.nan),
            'core_p_max_inference': infer_summary.get('core_p_max_inference', np.nan),
            'core_ci_all_exclude_zero': infer_summary.get('core_ci_all_exclude_zero', False),
            'core_inference_sign_consistent': infer_summary.get('core_inference_sign_consistent', False),
            'core_abs_t_all_above_threshold': infer_summary.get('core_abs_t_all_above_threshold', False),
            'inference_fragile': infer_summary.get('inference_fragile', False),
            'inference_bootstrap_reps': int(cfg.inference_bootstrap_reps),
            'inference_bootstrap_mean_block': float(cfg.inference_bootstrap_mean_block),
            'core_sharpe_ann': luck['sr'] * np.sqrt(ann) if np.isfinite(luck['sr']) else np.nan,
            'core_psr': luck['psr'], 'core_dsr': luck['dsr'], 'luck_n_trials': cfg.luck_n_trials,
            'luck_sr_hurdle_ann': luck['sr_star'] * np.sqrt(ann) if np.isfinite(luck['sr_star']) else np.nan,
            'luck_gate_pass': luck['pass'],
            'core_alt_demeaned_controls_ann': core_demeaned,
            'nontraded_naive_charge_ann': naive_total,
            'core_if_nontraded_charged_at_raw_mean_ann': core_demeaned - naive_total,
            'core_after_assumed_nontraded_premia_ann': core_after_premia,
            'full_model_r2': fullfit['r2'], 'full_model_condition_number': fullfit['condition_number'],
            'dominant_explanatory_block': dominant, 'diagnostic_label': label,
            'unpriced_tail_risk_flag': unpriced_tail, 'tail_flags_fired': ','.join(fired) if fired else 'none',
            'core_significantly_negative': bool(core_t <= -t_crit),
            'referee_handoff': handoff,
            'selected_nonlinear_order_bic': int(sel[0]) if len(sel) else np.nan,
            'state_lag': cfg.state_lag, 'dropped_columns': ','.join(d.dropped_columns) if d.dropped_columns else 'none',
            'pricing_convention': PRICING_CONVENTION,
        })
        if not cfg.inference_store_bootstrap_draws:
            bootstrap_draws = pd.DataFrame()
        return SliceResult(
            waterfall=waterfall, shapley=shapley, nested_models=nested, nonlinear_order=nonlinear,
            conditional_alpha=cond, tail_diagnostics=tail, coefficients=coefs, core_returns=core_returns,
            states=S, feature_blocks=blocks, summary=summary, nontraded=nontraded, tail_flags=flags,
            block_types=block_types, core_inference=core_inference,
            bootstrap_inference=bootstrap_inference, bootstrap_draws=bootstrap_draws,
            profile_comparison=profile_comparison, inference_frontier=frontier,
            alpha_survival=alpha_survival,
        )
