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

Both functions only read attributes off the objects passed in, so they can
be unit tested with simple stand-in objects instead of QuantConnect types.
"""

from collections.abc import Iterable
from typing import Any

COMMON_STOCK_TYPE = "ST00000001"  # Morningstar security-type code for common stock


def coarse_filter(coarse: Iterable[Any], cfg: Any) -> list[Any]:
    filtered = [
        c
        for c in coarse
        if c.HasFundamentalData
        and c.Price > cfg.min_price
        and c.DollarVolume > cfg.min_dollar_volume
    ]
    sorted_by_vol = sorted(filtered, key=lambda c: c.DollarVolume, reverse=True)
    return [c.Symbol for c in sorted_by_vol[: cfg.universe_size]]


def fine_filter(fine: Iterable[Any]) -> list[Any]:
    return [f.Symbol for f in fine if f.SecurityReference.SecurityType == COMMON_STOCK_TYPE]
