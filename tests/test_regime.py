"""Tests for market regime classification (Market-Guided Gating layer)."""

import pytest

from master_quant.strategy.regime import classify_regime, compute_market_gating


@pytest.mark.parametrize(
    ("vix", "spy_ret", "expected"),
    [
        (15.0, 0.02, "bull_low_vol"),
        (30.0, 0.02, "bull_high_vol"),
        (15.0, -0.02, "bear_low_vol"),
        (30.0, -0.02, "bear_high_vol"),
        # Boundaries: both comparisons are strict '>'
        (22.0, 0.01, "bull_low_vol"),  # VIX exactly at threshold is NOT high vol
        (22.01, 0.01, "bull_high_vol"),
        (15.0, 0.0, "bear_low_vol"),  # flat market counts as NOT bull
    ],
)
def test_classify_regime(vix, spy_ret, expected):
    assert classify_regime(vix, spy_ret, vix_high_threshold=22.0) == expected


def test_market_gating_returns_regime_weights(cfg):
    out = compute_market_gating(vix_val=35.0, spy_ret_20=-0.05, cfg=cfg)
    assert out["regime"] == "bear_high_vol"
    for k, v in cfg.regime_factor_weights["bear_high_vol"].items():
        assert out[k] == v


def test_market_gating_does_not_mutate_config(cfg):
    before = dict(cfg.regime_factor_weights["bull_low_vol"])
    compute_market_gating(vix_val=12.0, spy_ret_20=0.03, cfg=cfg)
    assert cfg.regime_factor_weights["bull_low_vol"] == before
    assert "regime" not in cfg.regime_factor_weights["bull_low_vol"]


def test_bear_high_vol_favours_mean_reversion_over_momentum(cfg):
    w = compute_market_gating(vix_val=40.0, spy_ret_20=-0.10, cfg=cfg)
    assert w["w_rev"] > w["w_mid"]
