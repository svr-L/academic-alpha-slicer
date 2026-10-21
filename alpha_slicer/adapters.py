from __future__ import annotations
from pathlib import Path
import ast, importlib.util, json, types
import numpy as np
import pandas as pd

FACTOR_ETFS={"Value":"RPV","Size":"SIZE","Momentum":"MTUM","Quality":"SPHQ","LowVol":"SPLV"}


def _import_module(path, name='external_core'):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run_factor_momentum_live(mom_repo, start='2000-01-01', variant='6-1',
                             implementation_lag=1, use_ar_costs=True):
    """Run the user's current Factor Momentum core.py on live Yahoo data.

    Returns a dict ready for AlphaSlicer.fit. Network/yfinance required.
    """
    import yfinance as yf
    mom_repo=Path(mom_repo)
    core=_import_module(mom_repo/'core.py','factor_momentum_core')
    tick=list(FACTOR_ETFS.values())
    raw=yf.download(tick,start=start,auto_adjust=True,progress=False)
    if not isinstance(raw.columns,pd.MultiIndex): raise ValueError('Expected Yahoo MultiIndex')
    rename={v:k for k,v in FACTOR_ETFS.items()}
    close=raw['Close'].rename(columns=rename); high=raw['High'].rename(columns=rename); low=raw['Low'].rename(columns=rename)
    rets=close.pct_change(fill_method=None).dropna(how='any')
    costs=None
    if use_ar_costs:
        spreads=core.abdi_ranaldo_panel(high,low,close,tickers=list(FACTOR_ETFS))
        costs=core.spread_panel_to_one_way_cost(spreads)
    out=core.run_factor_mom_from_returns(rets,variant=variant,cost_oneway_daily=costs,
                                         implementation_lag=implementation_lag)
    spy=yf.download('SPY',start=start,auto_adjust=True,progress=False)['Close']
    if isinstance(spy,pd.DataFrame): spy=spy.iloc[:,0]
    spy=spy.pct_change(fill_method=None).dropna().rename('MKT')
    idx=out['rets_net'].index.intersection(spy.index)
    # Static ETF sleeves are the linear spanning block; SPY is added explicitly as MKT.
    F=rets.reindex(idx).copy(); F.insert(0,'MKT',spy.reindex(idx))
    return {'strategy_net':out['rets_net']['MomTop1_80_20'].reindex(idx),
            'strategy_gross':out['rets_gross']['MomTop1_80_20'].reindex(idx),
            'factors':F,'benchmark':spy.reindex(idx),'raw_output':out}


# ----------------------------------------------------------------------------- Tail-Aware referee
def _extract_literal_from_notebook(ipynb_path, variable):
    nb=json.load(open(ipynb_path,encoding='utf-8'))
    for cell in nb.get('cells',[]):
        if cell.get('cell_type')!='code': continue
        src=''.join(cell.get('source',[]))
        try: tree=ast.parse(src)
        except SyntaxError: continue
        for node in tree.body:
            if isinstance(node,ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt,ast.Name) and tgt.id==variable:
                        return ast.literal_eval(node.value)
    raise KeyError(f'{variable} not found in {ipynb_path}')


def extract_sp100_from_tail_repo(tail_repo):
    p=Path(tail_repo)/'notebooks'/'02_f0_sp100_public_data.ipynb'
    return _extract_literal_from_notebook(p,'SP100')


def load_referee_core(path):
    """Load the referee's DEFINITIONS from its F0 engine without running its pipeline.

    ``path`` is a notebook (.ipynb, code cells concatenated) or a Python export of it. Only
    imports, function/class definitions and call-free constant assignments (CONFIG, the ticker
    list, URLs) are executed; prints, plots and the pipeline run are skipped. The slicer therefore
    uses the referee's own data loader, residualization, signals and backtest, not a copy.
    """
    p=Path(path)
    if p.suffix=='.ipynb':
        nb=json.load(open(p,encoding='utf-8'))
        src='\n'.join(''.join(c.get('source',[])) for c in nb.get('cells',[]) if c.get('cell_type')=='code')
    else:
        src=p.read_text(encoding='utf-8')
    src='\n'.join(l for l in src.splitlines() if not l.lstrip().startswith(('%','!')))
    keep=[]
    for node in ast.parse(src).body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            keep.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)) and node.value is not None \
                and not any(isinstance(n,ast.Call) for n in ast.walk(node.value)):
            keep.append(node)
    mod=types.ModuleType('tail_aware_referee_core'); mod.__file__=str(p)
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),mod.__dict__)
    missing=[f for f in ('CONFIG','load_panel','build_returns','build_ladder','build_signals','backtest')
             if f not in mod.__dict__]
    if missing:
        raise ValueError(f'{p} is not a Tail-Aware F0 engine: missing {missing}')
    return mod


def find_referee_core(tail_repo):
    """Locate the referee's F0 S&P100 engine inside its repository."""
    r=Path(tail_repo)
    candidates=[r/'notebooks'/'02_f0_sp100_public_data.ipynb', r/'f0_core_reference.py',
                r/'src'/'f0_core_reference.py', *sorted(r.glob('**/RUN_ME_*F0_REAL_DATA*.ipynb'))]
    for c in candidates:
        if c.exists(): return c
    raise FileNotFoundError(f'No Tail-Aware F0 engine found under {r} (looked for {[str(c) for c in candidates[:3]]})')


def run_tailaware_sp100_live(tail_repo=None, start='2007-01-01', end='2025-01-01', core_path=None,
                             signal='S1_resmom', level=None, data_source='yfinance', config_overrides=None):
    """Reproduce the referee's own residual-momentum P&L and return it ready for slicing.

    Calls the referee's functions in its own order - load_panel -> build_returns -> build_ladder
    -> build_signals -> backtest - with the referee's CONFIG (only start/end/data_source and
    explicit ``config_overrides`` change). Nothing is re-implemented here, so the slicer explains
    the same P&L the referee judges. v0.2 and earlier recreated a simplified fork instead (log-
    residual sum, +-0.5/k weights).

    level defaults to the referee's verdict level (L2_FF3). data_source='shaped' runs the
    referee's offline panel (no network). Network + yfinance required otherwise.
    """
    core=load_referee_core(core_path or find_referee_core(tail_repo))
    cfg=dict(core.CONFIG); cfg.update(start=start,end=end,data_source=data_source); cfg.update(config_overrides or {})
    prices,ff=core.load_panel(cfg)
    rets,excess,ff=core.build_returns(prices,ff,cfg)
    ladder=core.build_ladder(excess,ff,cfg)
    lvl=level or cfg['verdict_level']
    sigs=core.build_signals({lvl:ladder[lvl]},cfg)
    sig=sigs[(lvl,signal)]
    bt=core.backtest(sig,rets,cfg)
    F=ff[['MKT','SMB','HML']]
    idx=bt['net'].dropna().index.intersection(F.dropna().index)
    return {'strategy_net':bt['net'].reindex(idx).rename(f'{signal}_{lvl}_net'),
            'strategy_gross':bt['gross'].reindex(idx).rename(f'{signal}_{lvl}_gross'),
            'factors':F.reindex(idx),'benchmark':F['MKT'].reindex(idx),
            'signal':sig,'referee_core':str(core.__file__),'referee_config':cfg,'universe':list(prices.columns)}
