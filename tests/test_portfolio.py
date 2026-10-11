"""Tests for diversification, the turnover band and position sizing."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from master_quant.strategy.portfolio import (
    apply_turnover_band,
    correlation_diversify,
    inverse_vol_weights,
)

# ── correlation_diversify ───────────────────────────────────────────────


def test_rejects_perfectly_correlated_candidate(rng):
    base = rng.normal(0, 0.01, 40)
    returns = {"A": base, "B": base * 2.0, "C": rng.normal(0, 0.01, 40)}
    ranked = [("A", 3.0), ("B", 2.0), ("C", 1.0)]
    assert correlation_diversify(ranked, returns, n=2, corr_threshold=0.75) == ["A", "C"]


def test_negative_correlation_also_counts_as_redundant(rng):
    base = rng.normal(0, 0.01, 40)
    returns = {"A": base, "B": -base, "C": rng.normal(0, 0.01, 40)}
    ranked = [("A", 3.0), ("B", 2.0), ("C", 1.0)]
    assert correlation_diversify(ranked, returns, n=2) == ["A", "C"]


def test_tops_up_when_everything_is_correlated(rng):
    base = rng.normal(0, 0.01, 40)
    returns = {s: base * (i + 1) for i, s in enumerate("ABCD")}
    ranked = [(s, 4.0 - i) for i, s in enumerate("ABCD")]
    out = correlation_diversify(ranked, returns, n=3)
    assert out == ["A", "B", "C"]  # topped up best-score-first


def test_symbol_without_returns_is_accepted(rng):
    returns = {"A": rng.normal(0, 0.01, 40)}
    out = correlation_diversify([("A", 2.0), ("NEW", 1.0)], returns, n=2)
    assert out == ["A", "NEW"]


def test_short_overlap_is_ignored(rng):
    base = rng.normal(0, 0.01, 40)
    returns = {"A": base, "B": base[:5]}  # only 5 overlapping obs -> not trusted
    assert correlation_diversify([("A", 2.0), ("B", 1.0)], returns, n=2) == ["A", "B"]


def test_flat_return_series_does_not_crash(rng):
    returns = {"A": rng.normal(0, 0.01, 40), "B": np.zeros(40)}
    assert correlation_diversify([("A", 2.0), ("B", 1.0)], returns, n=2) == ["A", "B"]


def test_never_returns_more_than_n(rng):
    returns = {s: rng.normal(0, 0.01, 40) for s in "ABCDEFG"}
    ranked = [(s, float(i)) for i, s in enumerate("ABCDEFG")]
    assert len(correlation_diversify(ranked, returns, n=4)) == 4


# ── apply_turnover_band ─────────────────────────────────────────────────


def test_keeps_holding_when_edge_is_too_small():
    scores = {"A": 1.0, "B": 0.9, "NEW": 1.0}
    out = apply_turnover_band(
        {"A", "B"}, ["A", "NEW"], scores, portfolio_size=2, min_score_edge=0.15
    )
    assert set(out) == {"A", "B"}  # 1.0 - 0.9 = 0.10 < 0.15 -> keep B


def test_swaps_when_edge_is_large_enough():
    scores = {"A": 1.0, "B": 0.5, "NEW": 1.2}
    out = apply_turnover_band(
        {"A", "B"}, ["NEW", "A"], scores, portfolio_size=2, min_score_edge=0.15
    )
    assert set(out) == {"A", "NEW"}


def test_replaces_weakest_holding_first():
    scores = {"A": 0.2, "B": 0.8, "C": 0.5, "NEW": 2.0}
    out = apply_turnover_band({"A", "B", "C"}, ["NEW", "B", "C"], scores, portfolio_size=3)
    assert set(out) == {"NEW", "B", "C"}  # A (weakest) is the one swapped out


def test_fills_empty_portfolio_from_candidates():
    scores = {"A": 1.0, "B": 0.5}
    assert apply_turnover_band(set(), ["A", "B"], scores, portfolio_size=2) == ["A", "B"]


def test_unscored_holding_is_released():
    scores = {"A": 1.0, "B": 0.8}
    out = apply_turnover_band({"DELISTED", "A"}, ["A", "B"], scores, portfolio_size=2)
    assert set(out) == {"A", "B"}


def test_output_is_sorted_and_capped():
    scores = {s: float(i) for i, s in enumerate("ABCDEF")}
    out = apply_turnover_band(set("ABCDEF"), list("FEDCBA"), scores, portfolio_size=3)
    assert len(out) == 3
    assert out == sorted(out, key=lambda s: -scores[s])


def test_truncation_keeps_highest_scores():
    """Regression: the old version truncated an unordered set, so which names
    survived depended on Python's hash seed and changed between runs."""
    scores = {f"S{i:02d}": float(i) for i in range(30)}
    holdings = set(scores)  # 30 holdings, all still candidates, only 5 slots
    out = apply_turnover_band(holdings, sorted(scores), scores, portfolio_size=5)
    assert out == ["S29", "S28", "S27", "S26", "S25"]


def test_ties_are_broken_deterministically():
    scores = {f"S{i:02d}": 1.0 for i in range(30)}
    out = apply_turnover_band(set(scores), sorted(scores), scores, portfolio_size=3)
    assert out == ["S00", "S01", "S02"]  # equal scores -> alphabetical


# ── inverse_vol_weights ─────────────────────────────────────────────────


class _FakeSD:
    def __init__(self, vol):
        self._vol = vol

    def volatility(self, _n):
        return self._vol


def test_weights_sum_to_exposure():
    sds = {"A": _FakeSD(0.2), "B": _FakeSD(0.4), "C": _FakeSD(0.1)}
    w = inverse_vol_weights(["A", "B", "C"], sds, 20, total_exposure=0.65)
    assert sum(w.values()) == pytest.approx(0.65)


def test_calmer_stock_gets_more_weight():
    sds = {"CALM": _FakeSD(0.1), "WILD": _FakeSD(0.4)}
    w = inverse_vol_weights(["CALM", "WILD"], sds, 20, total_exposure=1.0)
    assert w["CALM"] == pytest.approx(0.8)
    assert w["WILD"] == pytest.approx(0.2)


def test_unknown_vol_gets_zero_weight():
    sds = {"A": _FakeSD(0.2), "B": _FakeSD(None)}
    w = inverse_vol_weights(["A", "B", "MISSING"], sds, 20, total_exposure=1.0)
    assert w == {"A": pytest.approx(1.0), "B": 0.0, "MISSING": 0.0}


def test_falls_back_to_equal_weight():
    w = inverse_vol_weights(["A", "B"], {}, 20, total_exposure=0.5)
    assert w == {"A": 0.25, "B": 0.25}


def test_empty_selection():
    assert inverse_vol_weights([], {}, 20, total_exposure=1.0) == {}


@given(
    vols=st.lists(st.floats(0.01, 2.0), min_size=1, max_size=30),
    exposure=st.floats(0.0, 1.0),
)
def test_weights_are_non_negative_and_never_exceed_exposure(vols, exposure):
    syms = [f"S{i}" for i in range(len(vols))]
    sds = {s: _FakeSD(v) for s, v in zip(syms, vols, strict=True)}
    w = inverse_vol_weights(syms, sds, 20, exposure)
    assert all(x >= 0 for x in w.values())
    assert sum(w.values()) <= exposure + 1e-9
