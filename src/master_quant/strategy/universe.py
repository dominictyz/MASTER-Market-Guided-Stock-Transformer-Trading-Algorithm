"""
universe.py
===========
Decides which stocks are even eligible to be considered each day, before
any scoring happens. Two-stage QuantConnect filter:

  coarse_filter — cheap, broad checks (has fundamental data, price floor,
                  minimum dollar volume) run across the whole US equity
                  market, kept to the top N most liquid names
  fine_filter   — more detailed check (common stock only — excludes ETFs,
                  REITs, preferred shares, etc.) run on the smaller
                  coarse-filtered list
"""


def coarse_filter(coarse, cfg):
    filtered = [
        c for c in coarse if c.HasFundamentalData and c.Price > 5 and c.DollarVolume > 5_000_000
    ]
    sorted_by_vol = sorted(filtered, key=lambda c: c.DollarVolume, reverse=True)
    return [c.Symbol for c in sorted_by_vol[: cfg.universe_size]]


def fine_filter(fine):
    return [
        f.Symbol
        for f in fine
        if f.SecurityReference.SecurityType == "ST00000001"  # common stock
    ]
