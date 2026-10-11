"""Tests for SymbolData features and the composite stock score."""

import math

import numpy as np
import pytest

from master_quant.strategy.scoring import SymbolData, compute_stock_score
from tests.conftest import geometric_prices, make_symbol_data, random_walk_prices

NEUTRAL_WEIGHTS = {"w_short": 0.20, "w_mid": 0.40, "w_long": 0.25, "w_rev": 0.15}


# ── SymbolData ──────────────────────────────────────────────────────────


def test_rolling_window_is_bounded():
    sd = make_symbol_data(range(1, 1001), max_window=50)
    assert len(sd.closes) == 60  # max_window + 10 buffer
    assert sd.closes[-1] == 1000.0


def test_is_ready():
    sd = make_symbol_data([100.0] * 10)
    assert sd.is_ready(10)
    assert not sd.is_ready(11)


def test_return_matches_hand_calculation():
    sd = make_symbol_data([100, 102, 101, 105, 110, 99])
    assert sd.return_(1) == pytest.approx((99 - 110) / 110)
    assert sd.return_(5) == pytest.approx((99 - 100) / 100)


def test_return_needs_n_plus_one_prices():
    sd = make_symbol_data([100, 101, 102])
    assert sd.return_(2) is not None
    assert sd.return_(3) is None


def test_return_with_zero_past_price_is_none():
    sd = make_symbol_data([0.0, 10.0])
    assert sd.return_(1) is None


def test_volatility_matches_numpy(rng):
    prices = random_walk_prices(rng, 30, vol=0.02)
    sd = make_symbol_data(prices)
    log_rets = np.diff(np.log(prices[-21:]))
    assert sd.volatility(20) == pytest.approx(np.std(log_rets) * math.sqrt(252))


def test_volatility_is_roughly_annualised(rng):
    sd = make_symbol_data(random_walk_prices(rng, 2001, vol=0.01), max_window=2100)
    # 1% daily vol ~= 15.9% annualised; allow sampling error
    assert sd.volatility(2000) == pytest.approx(0.01 * math.sqrt(252), rel=0.05)


def test_volatility_of_flat_series_is_none():
    assert make_symbol_data([50.0] * 30).volatility(20) is None


def test_volatility_insufficient_history_is_none():
    assert make_symbol_data([100, 101]).volatility(20) is None


def test_return_series_length_and_values():
    sd = make_symbol_data([100, 110, 99])
    rs = sd.return_series(2)
    assert isinstance(rs, np.ndarray)
    np.testing.assert_allclose(rs, [math.log(1.1), math.log(0.9)])
    assert sd.return_series(3) is None


# ── compute_stock_score ─────────────────────────────────────────────────


def test_score_none_for_missing_symbol_data():
    assert compute_stock_score(None, 5, 20, 60, NEUTRAL_WEIGHTS) is None


def test_score_none_with_insufficient_history(rng):
    sd = make_symbol_data(random_walk_prices(rng, 40))
    assert compute_stock_score(sd, 5, 20, 60, NEUTRAL_WEIGHTS) is None


def test_score_none_for_zero_volatility():
    sd = make_symbol_data([100.0] * 80)
    assert compute_stock_score(sd, 5, 20, 60, NEUTRAL_WEIGHTS) is None


def test_score_none_for_malformed_weights(rng):
    sd = make_symbol_data(random_walk_prices(rng, 80))
    assert compute_stock_score(sd, 5, 20, 60, {"w_short": 1.0}) is None


def test_score_matches_formula(rng):
    sd = make_symbol_data(random_walk_prices(rng, 80))
    vol = sd.volatility(20)
    r5, r20, r60 = sd.return_(5), sd.return_(20), sd.return_(60)
    w = NEUTRAL_WEIGHTS
    expected = (
        w["w_short"] * r5 / vol
        + w["w_mid"] * r20 / vol
        + w["w_long"] * r60 / vol
        - w["w_rev"] * r5 / vol
    )
    assert compute_stock_score(sd, 5, 20, 60, w) == pytest.approx(expected)


def test_uptrend_outscores_downtrend(rng):
    noise = rng.normal(0, 0.005, 80)
    up = make_symbol_data(np.array(geometric_prices(80, 0.003)) * np.exp(noise))
    down = make_symbol_data(np.array(geometric_prices(80, -0.003)) * np.exp(noise))
    s_up = compute_stock_score(up, 5, 20, 60, NEUTRAL_WEIGHTS)
    s_down = compute_stock_score(down, 5, 20, 60, NEUTRAL_WEIGHTS)
    assert s_up > 0 > s_down


def test_pure_reversion_weights_reward_recent_losers(rng):
    rev_only = {"w_short": 0.0, "w_mid": 0.0, "w_long": 0.0, "w_rev": 1.0}
    base = random_walk_prices(rng, 75, vol=0.01)
    loser = make_symbol_data(base + [base[-1] * 0.97**i for i in range(1, 6)])
    assert compute_stock_score(loser, 5, 20, 60, rev_only) > 0


def test_symbol_data_keeps_symbol():
    assert SymbolData("AAPL", 10).symbol == "AAPL"
