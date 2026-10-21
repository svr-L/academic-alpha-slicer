from __future__ import annotations
import numpy as np
import pandas as pd

VAR_T6 = 6.0 / 4.0   # variance of a Student-t with 6 degrees of freedom


def make_demo_case(n=3500, seed=11):
    """Known-truth synthetic case for pipeline validation (v0.2 pricing convention).

    Ingredient                                            How the slicer must treat it
    ----------------------------------------------------  ----------------------------------------
    static exposure to MKT, SMB, HML, MOM                  traded: charged at the factor means
    extra MOM exposure after high-vol days, when the MOM   traded managed portfolio MOM x z_{t-1}:
      premium is also higher                               a conditional premium, not alpha
    0.22 x SHORTVOL, a traded short-vol factor whose       traded tail factor: charged at its mean
      carry exceeds its crash losses (~ +5%/yr)
    2.0 x VARSWAP, a long variance swap struck 25% above   traded convexity factor: the variance
      expected variance (the strategy pays a VRP)          premium paid is charged, not called alpha
    core alpha 1.8%/yr gross, costs 0.8%/yr                net core alpha 1.0%/yr

    Every exposure is traded, so the truth is exact under the traded-factor convention. The
    strong quadratic market dependence shows up in the nonlinear-order table (factors only),
    while in the full model m^2 is entirely spanned by the traded VARSWAP and drops out.

    The regime is observed at the close of each day; the strategy acts on yesterday's regime
    and the slicer lags the supplied states by one day, so both use z_{t-1}.
    """
    rng=np.random.default_rng(seed)
    idx=pd.bdate_range('2012-01-02',periods=n)
    regime=np.zeros(n)
    for t in range(1,n):
        regime[t]=regime[t-1] if rng.random()<0.97 else 1-regime[t-1]
    z_lag=np.r_[0.0,regime[:-1]]
    vol=np.where(regime>0,0.017,0.008)
    mkt=0.00025+rng.standard_t(6,n)*vol
    smb=0.00005+0.25*mkt+rng.standard_t(7,n)*0.005
    hml=0.00007-0.10*mkt+rng.standard_t(7,n)*0.005
    # MOM premium is higher after high-vol days: a conditional premium a strategy can earn by
    # scaling MOM exposure with yesterday's (public) regime.
    mom=0.00010+0.00030*z_lag-0.05*mkt+rng.standard_t(6,n)*0.006
    crash=mkt < np.quantile(mkt,0.05)
    # Positive-carry / crash-loss factor: a stylised traded short-vol premium (~+5%/yr net).
    shortvol=np.full(n,0.00060)+rng.normal(0,0.0012,n)
    shortvol[crash] += -0.0045 - 0.10*np.abs(mkt[crash])
    # Long variance swap struck 25% above the stationary expected variance (a 25% VRP).
    expected_var=0.00025**2+VAR_T6*0.5*(0.017**2+0.008**2)
    varswap=mkt**2-1.25*expected_var
    F=pd.DataFrame({'MKT':mkt,'SMB':smb,'HML':hml,'MOM':mom},index=idx)
    tailF=pd.DataFrame({'SHORTVOL':shortvol,'VARSWAP':varswap},index=idx)
    core=0.018/252
    eps=rng.standard_t(7,n)*0.0030
    eps = eps - eps.mean()  # exact zero sample mean for a clean known-truth intercept test
    gross=(core + 0.20*mkt + 0.12*smb - 0.08*hml + 0.10*mom
           + 0.30*mom*z_lag + 0.22*shortvol + 2.0*varswap + eps)
    cost=np.full(n,0.008/252)
    net=gross-cost
    truth=pd.Series({'core_alpha_gross_ann':0.018,'cost_drag_ann':0.008,
                     'core_alpha_net_ann':0.010,
                     'conditional_premium_population_ann':0.30*0.00030*0.25*252,
                     'shortvol_premium_in_sample_ann':float(0.22*shortvol.mean()*252),
                     'variance_premium_paid_in_sample_ann':float(2.0*varswap.mean()*252),
                     'notes':'linear + MOM x lagged regime + SHORTVOL + VARSWAP (VRP paid) + core alpha'})
    return {
        'strategy_gross':pd.Series(gross,index=idx,name='DemoGross'),
        'strategy_net':pd.Series(net,index=idx,name='DemoNet'),
        'factors':F,'benchmark':F['MKT'],'tail_factors':tailF,
        'costs':pd.Series(cost,index=idx,name='cost'),
        'states':pd.DataFrame({'HIGH_VOL':regime},index=idx),
        'truth':truth,
    }
