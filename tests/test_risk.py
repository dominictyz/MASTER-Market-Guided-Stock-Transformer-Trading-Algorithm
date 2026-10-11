"""Tests for exposure sizing and the drawdown circuit breaker."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from master_quant.config import StrategyConfig
from master_quant.strategy.risk import (
    apply_drawdown_cut,
    regime_exposure_for,
    update_equity_peak,
)


@pytest.mark.parametrize(
    ("regime", "expected"),
    [("bull_low_vol", 1.0), ("bear_high_vol", 0.40), ("unknown_regime", 0.80)],
)
def test_regime_exposure_for(cfg, regime, expected):
    assert regime_exposure_for(regime, cfg) == expected


def test_drawdown_just_below_trigger_does_not_cut(cfg):
    exposure, dd = apply_drawdown_cut(1.0, current_equity=85_010, equity_peak=100_000, cfg=cfg)
    assert dd == pytest.approx(0.1499)
    assert exposure == 1.0


def test_drawdown_just_above_trigger_cuts_in_half(cfg):
    exposure, dd = apply_drawdown_cut(1.0, current_equity=84_990, equity_peak=100_000, cfg=cfg)
    assert dd == pytest.approx(0.1501)
    assert exposure == pytest.approx(0.5)


def test_drawdown_exactly_at_trigger_does_not_cut(cfg):
    exposure, _ = apply_drawdown_cut(0.8, current_equity=85_000, equity_peak=100_000, cfg=cfg)
    assert exposure == 0.8


def test_no_drawdown_at_new_high(cfg):
    exposure, dd = apply_drawdown_cut(0.65, current_equity=120_000, equity_peak=120_000, cfg=cfg)
    assert dd == 0.0
    assert exposure == 0.65


def test_zero_peak_is_handled_safely(cfg):
    assert apply_drawdown_cut(1.0, current_equity=0, equity_peak=0, cfg=cfg) == (1.0, 0.0)


def test_update_equity_peak_is_a_high_water_mark():
    peak = 100.0
    for equity in [105, 103, 110, 90, 108]:
        peak = update_equity_peak(equity, peak)
    assert peak == 110


@given(
    exposure=st.floats(0.0, 1.0),
    peak=st.floats(1.0, 1e9),
    frac=st.floats(0.0, 2.0),
)
def test_drawdown_cut_never_increases_exposure(exposure, peak, frac):
    # Built inside the test (not via the fixture): hypothesis runs this body
    # many times and function-scoped fixtures are not re-created between runs.
    cfg = StrategyConfig()
    new_exposure, _ = apply_drawdown_cut(exposure, peak * frac, peak, cfg)
    assert 0.0 <= new_exposure <= exposure
