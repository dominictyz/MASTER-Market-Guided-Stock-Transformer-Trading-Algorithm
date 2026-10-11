"""Tests for StrategyConfig validation."""

import pytest

from master_quant.config import StrategyConfig


def test_default_config_is_valid(cfg):
    assert cfg.portfolio_size == 20
    assert cfg.universe_size >= cfg.portfolio_size


@pytest.mark.parametrize(
    "regime", ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "neutral"]
)
def test_every_regime_has_weights_and_exposure(cfg, regime):
    assert regime in cfg.regime_factor_weights
    assert regime in cfg.regime_exposure
    assert sum(cfg.regime_factor_weights[regime].values()) == pytest.approx(1.0)


def test_exposure_decreases_as_regime_gets_riskier(cfg):
    e = cfg.regime_exposure
    assert e["bull_low_vol"] > e["bull_high_vol"] > e["bear_low_vol"] > e["bear_high_vol"]


def test_configs_do_not_share_mutable_state():
    a, b = StrategyConfig(), StrategyConfig()
    a.regime_exposure["neutral"] = 0.0
    assert b.regime_exposure["neutral"] == 0.80


@pytest.mark.parametrize(
    "overrides",
    [
        {"portfolio_size": 0},
        {"portfolio_size": 300, "universe_size": 200},
        {"lookback_short": 30},  # short must be < mid
        {"min_history": 60},  # must exceed lookback_long
        {"corr_threshold": 0.0},
        {"dd_trigger": 1.5},
        {"dd_cut": -0.1},
        {"regime_exposure": {"neutral": 1.2}},
        {"regime_factor_weights": {"neutral": {"w_short": 0.5, "w_mid": 0.1}}},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ValueError):
        StrategyConfig(**overrides)
