"""Shared pytest fixtures for the master_quant test suite."""

import numpy as np
import pytest

from master_quant.config import StrategyConfig
from master_quant.strategy.scoring import SymbolData


@pytest.fixture
def cfg() -> StrategyConfig:
    """A fresh default config for every test (dataclass fields are mutable)."""
    return StrategyConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator so every test run is reproducible."""
    return np.random.default_rng(seed=42)


def make_symbol_data(prices, symbol="TEST", max_window=200) -> SymbolData:
    """Build a SymbolData pre-loaded with a list of closing prices."""
    sd = SymbolData(symbol, max_window)
    for p in prices:
        sd.update(p)
    return sd


def geometric_prices(n: int, daily_return: float, start: float = 100.0) -> list[float]:
    """Price path that compounds at a constant daily return."""
    return [start * (1 + daily_return) ** i for i in range(n)]


def random_walk_prices(rng: np.random.Generator, n: int, vol: float = 0.01) -> list[float]:
    """Log-normal random walk with the given daily volatility."""
    log_rets = rng.normal(0.0, vol, size=n - 1)
    return list(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])))
