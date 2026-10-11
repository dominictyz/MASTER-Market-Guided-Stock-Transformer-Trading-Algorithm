"""
regime.py
=========
This is the "Market-Guided Gating" layer of the MASTER architecture: before
scoring any individual stock, first figure out what kind of market we're
in, then let that answer decide how much weight each signal gets.

Two inputs decide the regime:
  - VIX level        → how much investors currently expect prices to swing
                        (a "fear gauge"; high VIX = more expected turbulence)
  - SPY 20-day return → is the overall market trending up or down right now

Four regimes come out of combining those two:
  bull_low_vol   — market rising, calm            → lean into momentum
  bull_high_vol  — market rising, choppy           → balance signals
  bear_low_vol   — market slipping, calm           → lean into mean-reversion
  bear_high_vol  — market falling, turbulent       → lean hardest into
                                                       mean-reversion, get defensive

Kept as pure functions (no QuantConnect objects) so the regime logic can be
unit tested with plain numbers.
"""


def classify_regime(vix_val: float, spy_ret_20: float, vix_high_threshold: float) -> str:
    high_vol = vix_val > vix_high_threshold
    bull_mkt = spy_ret_20 > 0

    if bull_mkt and not high_vol:
        return "bull_low_vol"
    elif bull_mkt and high_vol:
        return "bull_high_vol"
    elif not bull_mkt and not high_vol:
        return "bear_low_vol"
    else:
        return "bear_high_vol"


def compute_market_gating(vix_val: float, spy_ret_20: float, cfg) -> dict:
    """
    Returns a dict of factor weights (w_short, w_mid, w_long, w_rev) plus
    the regime label itself, so downstream code can also use the regime
    name for exposure sizing (see risk.py).
    """
    regime = classify_regime(vix_val, spy_ret_20, cfg.vix_high_threshold)
    weights = dict(cfg.regime_factor_weights.get(regime, cfg.regime_factor_weights["neutral"]))
    weights["regime"] = regime
    return weights
