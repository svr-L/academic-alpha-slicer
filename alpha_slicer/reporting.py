from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def plot_waterfall(result, title='Sequential alpha erosion (traded blocks)'):
    w=result.waterfall.copy()
    ax=w['alpha_ann'].plot(kind='bar',figsize=(9,4),title=title)
    ax.axhline(0,linewidth=1)
    ax.set_ylabel('Annualized intercept / mean')
    ax.set_xlabel('')
    plt.tight_layout()
    return ax


def plot_shapley(result, title='Order-robust attribution of the mean'):
    s=result.shapley.copy()
    ax=s['contribution_ann'].plot(kind='bar',figsize=(9,4),title=title)
    ax.axhline(0,linewidth=1)
    ax.set_ylabel('Annualized mean contribution')
    ax.set_xlabel('')
    plt.tight_layout()
    return ax


def report_tables(result):
    return {
        'summary':result.summary.to_frame('value'),
        'waterfall':result.waterfall,
        'shapley':result.shapley,
        'nonlinear_order':result.nonlinear_order,
        'conditional_alpha':result.conditional_alpha,
        'tail_diagnostics':result.tail_diagnostics.to_frame('value'),
        'tail_flags':result.tail_flags,
        'nontraded':result.nontraded,
        'nested_models':result.nested_models,
        'core_inference':result.core_inference,
        'bootstrap_inference':result.bootstrap_inference,
        'profile_comparison':result.profile_comparison,
        'inference_frontier':result.inference_frontier,
        'alpha_survival':result.alpha_survival,
    }



def plot_bootstrap_attribution(result, title='Bootstrap uncertainty of alpha slices'):
    """Point estimates and percentile confidence intervals for the additive decomposition."""
    t = result.bootstrap_inference.copy()
    if t.empty:
        raise ValueError('No bootstrap inference stored; set inference_bootstrap_reps > 0')
    rows = [i for i in t.index if i.startswith('premium_') or i == 'core_unexplained']
    t = t.loc[rows]
    yerr = [t['estimate_ann'] - t['ci_low_ann'], t['ci_high_ann'] - t['estimate_ann']]
    ax = t['estimate_ann'].plot(kind='bar', yerr=yerr, capsize=3, figsize=(9,4), title=title)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel('Annualized contribution')
    ax.set_xlabel('')
    plt.tight_layout()
    return ax


def plot_inference_frontier(result, title='Inference robustness frontier'):
    """Evidence ratio across inference specifications; 1 is the pass threshold."""
    t = result.inference_frontier.copy()
    if t.empty:
        raise ValueError('No inference frontier available')
    ax = t['evidence_ratio'].plot(kind='bar', figsize=(10,4), title=title)
    ax.axhline(1.0, linewidth=1)
    ax.set_ylabel('Evidence ratio (statistic / required threshold)')
    ax.set_xlabel('')
    plt.tight_layout()
    return ax


def plot_alpha_survival(result, title='Alpha survival across expanding priced span'):
    """Annualized intercept and primary-policy confidence band across pricing-model depth.

    The line is not required to be monotone. A rise after adding a factor is an economically
    meaningful model-span effect, not a plotting error.
    """
    t = result.alpha_survival.copy()
    if t.empty:
        raise ValueError('No alpha survival curve available; set survival_enabled=True')
    x = range(len(t))
    ax = plt.figure(figsize=(10,4)).gca()
    ax.plot(x, t['alpha_ann'].values, marker='o')
    ax.fill_between(x, t['ci_low_ann'].values, t['ci_high_ann'].values, alpha=0.15)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(t.index), rotation=35, ha='right')
    ax.set_ylabel('Annualized alpha / mean')
    ax.set_xlabel('Expanding priced span')
    ax.set_title(title)
    plt.tight_layout()
    return ax
