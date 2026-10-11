"""
scoring.py
==========
This module holds two things that have NO dependency on QuantConnect's
engine, which is exactly why they're separated out — they can be unit
tested with plain Python floats and lists, no broker/backtester needed.

  1. SymbolData    — a rolling window of a stock's closing prices, and
                      the feature calculations (return, volatility,
                      return series) built on top of it.
  2. compute_stock_score — combines those features into one number
                      that ranks how attractive a stock is right now.

This is the "intra-stock aggregation" + "temporal aggregation" layer from
the MASTER architecture: for a single stock, look at multiple lookback
windows (short/mid/long) and combine them into one composite view.
"""

import numpy as np


class SymbolData:
    """Rolling close-price history for a single symbol, plus derived features."""

    def __init__(self, symbol, max_window):
        self.symbol = symbol
        self.max_window = max_window
        self.closes = []  # list[float], oldest first

    def update(self, close_price):
        self.closes.append(float(close_price))
        # Trim to keep memory bounded — we only ever need max_window + a
        # small buffer of history.
        if len(self.closes) > self.max_window + 10:
            self.closes = self.closes[-(self.max_window + 10) :]

    def is_ready(self, n):
        return len(self.closes) >= n

    def return_(self, n):
        """Simple n-day percentage return: (price_now - price_n_days_ago) / price_n_days_ago."""
        if len(self.closes) < n + 1:
            return None
        p_now = self.closes[-1]
        p_past = self.closes[-(n + 1)]
        if p_past == 0:
            return None
        return (p_now - p_past) / p_past

    def volatility(self, n):
        """
        Annualised volatility of daily log returns over the last n days.
        Volatility here is just "how much the price wiggles" — a bigger
        number means a bumpier, less predictable stock.
        """
        if len(self.closes) < n + 1:
            return None
        prices = self.closes[-(n + 1) :]
        rets = [np.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        if len(rets) < 2:
            return None
        vol = float(np.std(rets)) * np.sqrt(252)  # 252 trading days/year
        return vol if vol > 0 else None

    def return_series(self, n):
        """Daily log-return series of length n — used for correlation, not scoring."""
        if len(self.closes) < n + 1:
            return None
        prices = self.closes[-(n + 1) :]
        rets = [np.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        return np.array(rets)


def compute_stock_score(
    sd: SymbolData, lookback_short: int, lookback_mid: int, lookback_long: int, factor_weights: dict
):
    """
    Combine short/mid/long momentum and a short-term mean-reversion signal
    into one composite score for a stock, using whatever factor weights
    the current market regime says matter right now.

    Each raw return is divided by volatility first (a "Sharpe-like"
    adjustment) so that a 5% move in a calm stock counts for more than
    the same 5% move in a wild, choppy one — it's a cleaner signal.

    Returns None if there isn't enough history yet, or if the stock has
    (near) zero volatility, which would blow up the division.
    """
    if sd is None:
        return None

    try:
        ret_short = sd.return_(lookback_short)
        ret_mid = sd.return_(lookback_mid)
        ret_long = sd.return_(lookback_long)
        vol = sd.volatility(lookback_mid)

        if any(v is None for v in [ret_short, ret_mid, ret_long, vol]):
            return None
        if vol < 1e-9:
            return None

        sharpe_short = ret_short / vol
        sharpe_mid = ret_mid / vol
        sharpe_long = ret_long / vol
        rev_signal = -ret_short / vol  # mean-reversion: fade a recent short-term move

        score = (
            factor_weights["w_short"] * sharpe_short
            + factor_weights["w_mid"] * sharpe_mid
            + factor_weights["w_long"] * sharpe_long
            + factor_weights["w_rev"] * rev_signal
        )
        return float(score)

    except Exception:
        return None
