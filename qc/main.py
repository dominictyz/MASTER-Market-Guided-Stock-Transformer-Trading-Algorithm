"""
main.py
=======
QuantConnect entrypoint. This file is intentionally thin — it owns
scheduling and order execution, and delegates every actual decision
(regime detection, scoring, diversification, sizing, risk) to the modules
in this package. That separation is what makes each piece independently
testable and independently swappable.

To run on QuantConnect Cloud, upload this file as the project's main.py
and upload the `src/master_quant/` folder alongside it (so the project
contains `main.py` and `master_quant/`). QC multi-file Python projects
support local package imports exactly like this.
"""

import numpy as np
from AlgorithmImports import *
from scipy.stats import zscore

from master_quant.config import StrategyConfig
from master_quant.strategy.portfolio import (
    apply_turnover_band,
    correlation_diversify,
    inverse_vol_weights,
)
from master_quant.strategy.regime import compute_market_gating
from master_quant.strategy.risk import apply_drawdown_cut, regime_exposure_for, update_equity_peak
from master_quant.strategy.scoring import SymbolData, compute_stock_score
from master_quant.strategy.universe import coarse_filter, fine_filter


class MASTERStrategy(QCAlgorithm):
    def Initialize(self):
        self.cfg = StrategyConfig()

        self.SetStartDate(*self.cfg.start_date)
        self.SetEndDate(*self.cfg.end_date)
        self.SetCash(self.cfg.starting_cash)

        self.UniverseSettings.Resolution = Resolution.Daily

        # ── Market index for regime detection ──────────────────────────
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        self.spy_data = SymbolData(self.spy, self.cfg.lookback_mid + 5)

        # ── Universe selection ───────────────────────────────────────────
        self.AddUniverse(
            lambda coarse: coarse_filter(coarse, self.cfg),
            fine_filter,
        )

        # ── State ────────────────────────────────────────────────────────
        self.symbol_data = {}  # symbol -> SymbolData
        self.day_counter = 0
        self.last_rebal = -999
        self.current_holdings = set()  # for the turnover band
        self.equity_peak = float(self.cfg.starting_cash)

        self.SetWarmUp(self.cfg.min_history + 5)
        self.SetBenchmark("SPY")

    # ══════════════════════════════════════════════════════════════════
    # UNIVERSE CHANGED
    # ══════════════════════════════════════════════════════════════════

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            sym = security.Symbol
            if sym not in self.symbol_data and sym not in (self.spy, self.vix):
                self.symbol_data[sym] = SymbolData(sym, self.cfg.lookback_long + 5)

        for security in changes.RemovedSecurities:
            sym = security.Symbol
            if sym in self.symbol_data:
                del self.symbol_data[sym]
                if self.Portfolio[sym].Invested:
                    self.Liquidate(sym)
                self.current_holdings.discard(sym)

    # ══════════════════════════════════════════════════════════════════
    # DAILY EVENT
    # ══════════════════════════════════════════════════════════════════

    def OnData(self, data):
        if data.Bars.ContainsKey(self.spy):
            self.spy_data.update(data.Bars[self.spy].Close)

        for sym, sd in self.symbol_data.items():
            if data.Bars.ContainsKey(sym):
                sd.update(data.Bars[sym].Close)

        current_equity = float(self.Portfolio.TotalPortfolioValue)
        self.equity_peak = update_equity_peak(current_equity, self.equity_peak)

        if self.IsWarmingUp:
            return

        self.day_counter += 1
        if (self.day_counter - self.last_rebal) < self.cfg.rebalance_days:
            return

        self.last_rebal = self.day_counter
        self.Rebalance(data)

    # ══════════════════════════════════════════════════════════════════
    # REBALANCE
    # ══════════════════════════════════════════════════════════════════

    def Rebalance(self, data):
        # ── Step 1: Market-Guided Gating ──────────────────────────────
        vix_val = self.Securities[self.vix].Price if self.vix in self.Securities else 20.0
        spy_ret_20 = 0.0
        if self.spy_data.is_ready(25):
            r = self.spy_data.return_series(20)
            spy_ret_20 = float(np.sum(r)) if r is not None else 0.0

        market_weights = compute_market_gating(vix_val, spy_ret_20, self.cfg)
        regime = market_weights["regime"]

        # ── Steps 2-4: score every ready stock ──────────────────────────
        ready_syms = [
            sym
            for sym, sd in self.symbol_data.items()
            if sd.is_ready(self.cfg.min_history) and data.Bars.ContainsKey(sym)
        ]
        if len(ready_syms) < 10:
            return

        returns_matrix = {}
        for sym in ready_syms:
            sd = self.symbol_data[sym]
            rs = sd.return_series(self.cfg.corr_window)
            if rs is not None:
                returns_matrix[sym] = rs

        raw_scores = {}
        for sym in ready_syms:
            score = compute_stock_score(
                self.symbol_data[sym],
                self.cfg.lookback_short,
                self.cfg.lookback_mid,
                self.cfg.lookback_long,
                market_weights,
            )
            if score is not None:
                raw_scores[sym] = score
        if len(raw_scores) < 10:
            return

        vals = np.array(list(raw_scores.values()))
        syms = list(raw_scores.keys())
        if vals.std() < 1e-9:
            return
        z = zscore(vals)
        scores = {sym: z[i] for i, sym in enumerate(syms)}

        # ── Step 5: diversify, then apply the turnover band ────────────
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = correlation_diversify(
            ranked, returns_matrix, self.cfg.portfolio_size, self.cfg.corr_threshold
        )
        selected = apply_turnover_band(
            self.current_holdings,
            candidates,
            scores,
            self.cfg.portfolio_size,
            self.cfg.min_score_edge,
        )

        # ── Risk sizing: regime exposure, then drawdown circuit breaker ─
        exposure = regime_exposure_for(regime, self.cfg)
        current_equity = float(self.Portfolio.TotalPortfolioValue)
        exposure, drawdown = apply_drawdown_cut(
            exposure, current_equity, self.equity_peak, self.cfg
        )

        # ── Position sizing: inverse-volatility weights within exposure ─
        weights = inverse_vol_weights(selected, self.symbol_data, self.cfg.lookback_mid, exposure)

        # ── Execute trades ────────────────────────────────────────────
        existing = set(self.Portfolio.Keys)
        for sym in existing:
            if self.Portfolio[sym].Invested and sym not in weights:
                self.Liquidate(sym)

        for sym, w in weights.items():
            self.SetHoldings(sym, w)

        self.current_holdings = set(weights.keys())

        self.Log(
            f"Rebalanced: {len(weights)} positions | Regime: {regime} | "
            f"Exposure: {exposure:.2f} | Drawdown: {drawdown:.2%} | "
            f"Top: {[str(s) for s in list(weights.keys())[:5]]}"
        )
