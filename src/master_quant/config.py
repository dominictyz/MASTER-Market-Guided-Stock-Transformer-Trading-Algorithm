"""
config.py
=========
Every tunable knob for the strategy lives here, in one place, instead of
being scattered through the algorithm as magic numbers. This is what makes
the strategy "config-driven": you can back-test a different rebalance
schedule, a tighter correlation threshold, or a different drawdown trigger
by editing one file, without touching any logic.
"""

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    # ── Backtest window / capital ───────────────────────────────────────
    start_date: tuple[int, int, int] = (2010, 1, 1)
    end_date: tuple[int, int, int] = (2026, 12, 31)
    starting_cash: float = 100_000

    # ── Universe ─────────────────────────────────────────────────────────
    universe_size: int = 200  # top N liquid names kept after coarse filter
    portfolio_size: int = 20  # number of stocks actually held at once
    min_price: float = 5.0  # coarse filter: minimum share price (USD)
    min_dollar_volume: float = 5_000_000  # coarse filter: minimum daily $ volume

    # ── Lookback windows (in trading days) ─────────────────────────────
    lookback_short: int = 5
    lookback_mid: int = 20
    lookback_long: int = 60
    corr_window: int = 40
    min_history: int = 70  # bars required before a stock is "ready"

    # ── Trading frequency / turnover control ────────────────────────────
    rebalance_days: int = 10  # trading days between rebalances
    min_score_edge: float = 0.15  # turnover band: z-score edge needed to swap a holding

    # ── Diversification ──────────────────────────────────────────────────
    corr_threshold: float = 0.75  # reject a candidate if avg |corr| exceeds this

    # ── Regime detection ─────────────────────────────────────────────────
    vix_high_threshold: float = 22.0

    # Factor weights per regime — how much each momentum/reversion signal
    # counts toward a stock's score, depending on what kind of market we're in.
    regime_factor_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "bull_low_vol": {"w_short": 0.15, "w_mid": 0.45, "w_long": 0.30, "w_rev": 0.10},
            "bull_high_vol": {"w_short": 0.25, "w_mid": 0.35, "w_long": 0.20, "w_rev": 0.20},
            "bear_low_vol": {"w_short": 0.20, "w_mid": 0.25, "w_long": 0.20, "w_rev": 0.35},
            "bear_high_vol": {"w_short": 0.30, "w_mid": 0.15, "w_long": 0.10, "w_rev": 0.45},
            "neutral": {"w_short": 0.20, "w_mid": 0.40, "w_long": 0.25, "w_rev": 0.15},
        }
    )

    # Fraction of total capital to deploy in each regime.
    regime_exposure: dict[str, float] = field(
        default_factory=lambda: {
            "bull_low_vol": 1.00,
            "bull_high_vol": 0.85,
            "bear_low_vol": 0.65,
            "bear_high_vol": 0.40,
            "neutral": 0.80,
        }
    )

    # ── Drawdown circuit breaker ────────────────────────────────────────
    dd_trigger: float = 0.15  # cut exposure further if equity is down >15% from peak
    dd_cut: float = 0.5  # ...cut it in half when triggered

    def __post_init__(self) -> None:
        """Fail fast on an invalid configuration instead of mid-backtest."""
        if self.portfolio_size <= 0 or self.universe_size < self.portfolio_size:
            raise ValueError("need 0 < portfolio_size <= universe_size")
        if not self.lookback_short < self.lookback_mid < self.lookback_long:
            raise ValueError("lookbacks must satisfy short < mid < long")
        if self.min_history <= self.lookback_long:
            raise ValueError("min_history must exceed lookback_long")
        if not 0.0 < self.corr_threshold <= 1.0:
            raise ValueError("corr_threshold must be in (0, 1]")
        if not 0.0 < self.dd_trigger < 1.0 or not 0.0 <= self.dd_cut <= 1.0:
            raise ValueError("dd_trigger must be in (0, 1) and dd_cut in [0, 1]")
        for regime, w in self.regime_factor_weights.items():
            if abs(sum(w.values()) - 1.0) > 1e-9:
                raise ValueError(f"factor weights for {regime!r} must sum to 1.0")
        for regime, e in self.regime_exposure.items():
            if not 0.0 <= e <= 1.0:
                raise ValueError(f"exposure for {regime!r} must be in [0, 1]")
