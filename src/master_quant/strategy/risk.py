"""
risk.py
=======
Everything here decides HOW MUCH capital to put to work — as opposed to
scoring.py / portfolio.py, which decide WHICH stocks. This is the layer
that was completely missing from the first version of the strategy: v1
was always ~100% invested regardless of market conditions, which is why
it fell about as hard as the overall market in downturns.

Two independent safety mechanisms, stacked on top of each other:

  1. regime_exposure_for   — proactive: reduce exposure BEFORE trouble hits,
                              based on the current market regime
  2. apply_drawdown_cut    — reactive: cut exposure FURTHER after the
                              account has already lost a meaningful amount
                              from its peak value, regardless of regime
"""


def regime_exposure_for(regime: str, cfg) -> float:
    """How much of total capital to deploy, given the current regime."""
    return cfg.regime_exposure.get(regime, cfg.regime_exposure["neutral"])


def apply_drawdown_cut(exposure: float, current_equity: float, equity_peak: float, cfg):
    """
    Drawdown = how far current account value has fallen from its
    highest-ever value ("peak"), as a percentage. If that drop exceeds
    cfg.dd_trigger (e.g. 15%), cut whatever exposure was about to be used
    in half (cfg.dd_cut) as a circuit breaker — reduce risk further while
    the account is already hurting, rather than staying fully committed
    and hoping for a quick recovery.

    Returns (adjusted_exposure, current_drawdown).
    """
    if equity_peak <= 0:
        return exposure, 0.0

    drawdown = 1.0 - (current_equity / equity_peak)
    if drawdown > cfg.dd_trigger:
        exposure = exposure * cfg.dd_cut

    return exposure, drawdown


def update_equity_peak(current_equity: float, equity_peak: float) -> float:
    """Track the running high-water mark of account value."""
    return max(current_equity, equity_peak)
