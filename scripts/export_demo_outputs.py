"""Regenerate outputs/demo_*.csv from the known-truth demo (same call as the notebook).

Usage (from the repo root):  python scripts/export_demo_outputs.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_slicer import AlphaSlicer, SliceConfig, make_demo_case  # noqa: E402


def main(out_dir=ROOT / 'outputs'):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    d = make_demo_case(n=3500, seed=14)
    r = AlphaSlicer(SliceConfig(min_obs=500, inference_bootstrap_reps=500,
                                      inference_bootstrap_mean_block=63, inference_bootstrap_seed=20261019)).fit(
        d['strategy_net'], d['factors'], d['benchmark'], strategy_gross=d['strategy_gross'],
        tail_factors=d['tail_factors'], states=d['states'])
    r.summary.to_frame('value').to_csv(out_dir / 'demo_summary.csv')
    r.waterfall.to_csv(out_dir / 'demo_waterfall.csv')
    r.shapley.to_csv(out_dir / 'demo_shapley.csv')
    r.nonlinear_order.to_csv(out_dir / 'demo_nonlinear_order.csv')
    r.conditional_alpha.to_csv(out_dir / 'demo_conditional_alpha.csv')
    r.tail_diagnostics.to_frame('value').to_csv(out_dir / 'demo_tail_diagnostics.csv')
    r.tail_flags.to_csv(out_dir / 'demo_tail_flags.csv')
    r.nontraded.to_csv(out_dir / 'demo_nontraded.csv')
    r.core_inference.to_csv(out_dir / 'demo_core_inference.csv')
    r.bootstrap_inference.to_csv(out_dir / 'demo_bootstrap_inference.csv')
    print('demo outputs written to', out_dir)


if __name__ == '__main__':
    main()
