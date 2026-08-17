"""
strategy_config.py
===================
Every tunable knob for the strategy lives here, in one place, instead of
being scattered through the algorithm as magic numbers. This is what makes
the strategy "config-driven": you can back-test a different rebalance
schedule, a tighter correlation threshold, or a different drawdown trigger
by editing one file, without touching any logic.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StrategyConfig:
    # ── Backtest window / capital ───────────────────────────────────────
    start_date: tuple = (2010, 1, 1)
    end_date: tuple = (2026, 12, 31)
    starting_cash: float = 100_000  # raised from 10,000 — see README "Fix 1a"

    # ── Universe ─────────────────────────────────────────────────────────
    universe_size: int = 200     # top N liquid names kept after coarse filter
    portfolio_size: int = 20     # number of stocks actually held at once

    # ── Lookback windows (in trading days) ─────────────────────────────
    lookback_short: int = 5
    lookback_mid: int = 20
    lookback_long: int = 60
    corr_window: int = 40
    min_history: int = 70        # bars required before a stock is "ready"

    # ── Trading frequency / turnover control ────────────────────────────
    rebalance_days: int = 10     # was 5 — see README "Fix 1b"
    min_score_edge: float = 0.15  # turnover band — see README "Fix 1c"

    # ── Diversification ──────────────────────────────────────────────────
    corr_threshold: float = 0.75  # reject a candidate if avg |corr| exceeds this

    # ── Regime detection ─────────────────────────────────────────────────
    vix_high_threshold: float = 22.0

    # Factor weights per regime — how much each momentum/reversion signal
    # counts toward a stock's score, depending on what kind of market we're in.
    regime_factor_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "bull_low_vol":  {"w_short": 0.15, "w_mid": 0.45, "w_long": 0.30, "w_rev": 0.10},
        "bull_high_vol": {"w_short": 0.25, "w_mid": 0.35, "w_long": 0.20, "w_rev": 0.20},
        "bear_low_vol":  {"w_short": 0.20, "w_mid": 0.25, "w_long": 0.20, "w_rev": 0.35},
        "bear_high_vol": {"w_short": 0.30, "w_mid": 0.15, "w_long": 0.10, "w_rev": 0.45},
        "neutral":       {"w_short": 0.20, "w_mid": 0.40, "w_long": 0.25, "w_rev": 0.15},
    })

    # How much of total capital to deploy per regime — see README "Fix 2"
    regime_exposure: Dict[str, float] = field(default_factory=lambda: {
        "bull_low_vol":  1.00,
        "bull_high_vol": 0.85,
        "bear_low_vol":  0.65,
        "bear_high_vol": 0.40,
        "neutral":       0.80,
    })

    # ── Drawdown circuit breaker — see README "Fix 4" ───────────────────
    dd_trigger: float = 0.15   # cut exposure further if equity is down >15% from peak
    dd_cut: float = 0.5        # ...cut it in half when triggered
