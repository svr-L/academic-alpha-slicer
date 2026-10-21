"""Specification sweep: fit the slicer under several specifications and gate the best core.

Reporting the best of several factor sets / state definitions / tail-factor choices is a search.
``spec_sweep`` runs every specification, aligns the core return series on common dates and applies
``luck.multi_core_gate``: the best core survives only if it beats the best-of-K luck benchmark
(Hansen SPA across all specifications and a deflated Sharpe ratio at the effective number of
specifications times the strategy-level trials already spent).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Optional
import warnings
import pandas as pd

from .config import SliceConfig
from .decomposition import AlphaSlicer, SliceResult
from .luck import multi_core_gate


@dataclass
class SweepResult:
    table: pd.DataFrame
    core_returns: pd.DataFrame
    gate: Dict[str, object]
    results: Dict[str, SliceResult]

    @property
    def label(self) -> str:
        g = self.gate
        if not g['naive_pass']:
            return 'no_candidate'
        if not g['pass']:
            return 'core_not_robust_to_specification_search'
        flagged = bool(self.table.loc[g['best'], 'unpriced_tail_risk_flag'])
        return 'core_survives_luck_gate_tail_flagged' if flagged else 'core_survives_luck_gate'


def default_specs(has_tail_factors: bool = False) -> Dict[str, dict]:
    """A small default grid over the choices that move the core under the traded convention:
    which traded blocks are used and how the lagged states are defined."""
    specs = {
        'linear_only': {'config': {'block_order': ('linear', 'tail', 'nonlinear', 'tail_shape')}},
        'conditional_default': {'config': {}},
        'conditional_vol63': {'config': {'vol_lookback': 63}},
        'conditional_dd20': {'config': {'drawdown_threshold': -0.20}},
    }
    if has_tail_factors:
        specs['no_tail_factors'] = {'config': {}, 'fit': {'tail_factors': None}}
    return specs


def spec_sweep(strategy_net, factors, benchmark=None, specs: Optional[Dict[str, dict]] = None,
               base_config: Optional[SliceConfig] = None, n_trials_extra: float = 1,
               alpha: float = 0.05, B: int = 2000, mean_block: float = 21, seed: int = 0,
               **fit_kwargs) -> SweepResult:
    """Fit every specification and apply the luck gate to the best core.

    specs: {name: {'config': {SliceConfig overrides}, 'factors': optional DataFrame of traded
            factors, 'fit': {extra or overriding fit kwargs}}}; default ``default_specs``.
    n_trials_extra: trials spent before this sweep (e.g. strategy variants); multiplies the
            effective number of specifications in the deflated Sharpe ratio.
    """
    base = base_config or SliceConfig()
    specs = specs or default_specs(fit_kwargs.get('tail_factors') is not None)
    results, rows, cores = {}, [], {}
    for name, spec in specs.items():
        cfg = replace(base, **spec.get('config', {}))
        kw = dict(fit_kwargs)
        kw.update(spec.get('fit', {}))
        F = spec.get('factors', factors)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = AlphaSlicer(cfg).fit(strategy_net, F, benchmark, **kw)
        results[name] = r
        s = r.summary
        cores[name] = r.core_returns
        rows.append({'spec': name, 'core_ann': s['core_unexplained_alpha_ann'], 'core_t': s['core_alpha_t'],
                     'core_t_critical': s['core_t_critical'],
                     'core_dsr_single': s['core_dsr'], 'label_single': s['diagnostic_label'],
                     'unpriced_tail_risk_flag': s['unpriced_tail_risk_flag'], 'n_obs': s['n_obs']})
    table = pd.DataFrame(rows).set_index('spec')
    R = pd.concat(cores, axis=1).dropna(how='any')
    # v0.4b: each spec's t is the traded-only, long-bandwidth statistic; the naive rule uses the
    # largest fixed-b threshold across specs (they differ only through their sample length).
    gate = multi_core_gate(R, table['core_t'], n_trials_extra=n_trials_extra, alpha=alpha,
                           dsr_min=base.luck_dsr_min, t_min=float(table['core_t_critical'].max()), B=B,
                           mean_block=mean_block, seed=seed)
    table['is_best'] = table.index == gate['best']
    return SweepResult(table=table, core_returns=R, gate=gate, results=results)
