"""Tests for the coarse/fine universe filters, using stand-in objects."""

from types import SimpleNamespace

from master_quant.strategy.universe import COMMON_STOCK_TYPE, coarse_filter, fine_filter


def _coarse(sym, price=50.0, dollar_volume=1e8, has_fundamentals=True):
    return SimpleNamespace(
        Symbol=sym, Price=price, DollarVolume=dollar_volume, HasFundamentalData=has_fundamentals
    )


def test_coarse_filter_drops_illiquid_cheap_and_non_fundamental(cfg):
    coarse = [
        _coarse("GOOD"),
        _coarse("PENNY", price=2.0),
        _coarse("ILLIQUID", dollar_volume=1e6),
        _coarse("ETF", has_fundamentals=False),
    ]
    assert coarse_filter(coarse, cfg) == ["GOOD"]


def test_coarse_filter_sorts_by_dollar_volume_and_caps(cfg):
    cfg.universe_size = 2
    coarse = [
        _coarse("LOW", dollar_volume=1e7),
        _coarse("HIGH", dollar_volume=1e9),
        _coarse("MID", dollar_volume=1e8),
    ]
    assert coarse_filter(coarse, cfg) == ["HIGH", "MID"]


def test_fine_filter_keeps_common_stock_only():
    fine = [
        SimpleNamespace(
            Symbol="AAPL", SecurityReference=SimpleNamespace(SecurityType=COMMON_STOCK_TYPE)
        ),
        SimpleNamespace(
            Symbol="PREF", SecurityReference=SimpleNamespace(SecurityType="ST00000002")
        ),
    ]
    assert fine_filter(fine) == ["AAPL"]
