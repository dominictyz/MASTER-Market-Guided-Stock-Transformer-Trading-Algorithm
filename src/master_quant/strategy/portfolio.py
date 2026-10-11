"""
portfolio.py
============
Turns a ranked list of stock scores into an actual set of positions with
weights. This is the "Inter-Stock Aggregation" layer of MASTER — deciding
not just which stocks look good individually, but which combination of
them is good together (i.e. not all secretly betting on the same thing).

Three responsibilities, kept as separate pure functions so each is
independently testable:

  1. correlation_diversify — reject candidates too correlated with names
                              already picked (diversification)
  2. apply_turnover_band    — resist replacing a held position unless a
                              new candidate is CLEARLY better (cost control)
  3. inverse_vol_weights    — size positions so calmer stocks get more
                              capital than wilder ones, for a given total
                              amount of exposure
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

MIN_OVERLAP = 10  # minimum overlapping return observations to trust a correlation


def correlation_diversify(
    ranked_scores: Sequence[tuple[Hashable, float]],
    returns_matrix: Mapping[Hashable, np.ndarray],
    n: int,
    corr_threshold: float = 0.75,
) -> list[Hashable]:
    """
    Greedy selection: walk down the ranked list, keep a stock only if its
    average absolute correlation with everything already selected is below
    corr_threshold. Two highly correlated stocks add little real
    diversification even if both score well individually.

    If too many candidates are rejected, the remainder is topped up from the
    ranked list (best score first) so the portfolio is never under-sized.

    ranked_scores:  list of (symbol, score) tuples, best first
    returns_matrix: dict[symbol -> np.ndarray of daily returns]
    """
    selected: list[Hashable] = []

    for sym, _score in ranked_scores:
        if len(selected) >= n:
            break
        if sym not in returns_matrix or not selected:
            selected.append(sym)
            continue

        corrs = []
        r1 = returns_matrix[sym]
        for sel_sym in selected:
            if sel_sym not in returns_matrix:
                continue
            r2 = returns_matrix[sel_sym]
            min_len = min(len(r1), len(r2))
            if min_len <= MIN_OVERLAP:
                continue
            a, b = r1[-min_len:], r2[-min_len:]
            if np.std(a) == 0 or np.std(b) == 0:
                continue  # correlation undefined for a flat series
            c = np.corrcoef(a, b)[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))

        avg_corr = float(np.mean(corrs)) if corrs else 0.0
        if avg_corr < corr_threshold:
            selected.append(sym)

    if len(selected) < n:
        for sym, _score in ranked_scores:
            if len(selected) >= n:
                break
            if sym not in selected:
                selected.append(sym)

    return selected


def apply_turnover_band(
    current_holdings: Iterable[Hashable],
    candidate_selection: Sequence[Hashable],
    scores: Mapping[Hashable, float],
    portfolio_size: int,
    min_score_edge: float = 0.15,
) -> list[Hashable]:
    """
    Every rebalance, a fresh ranking will always look at least a little
    different from the last one — some of that difference is real
    information, some of it is just noise. Rebuilding the whole portfolio
    every time chases the noise too, which racks up trading costs for no
    real benefit.

    Rule: keep a currently-held stock unless a new candidate's score beats
    the *weakest* currently-held stock's score by more than min_score_edge.

    Returns at most `portfolio_size` symbols, sorted best score first.
    The output is fully deterministic: the same inputs always produce the
    same portfolio (earlier versions truncated an unordered set, so which
    names survived could change from run to run).
    """
    holdings = set(current_holdings)
    candidate_set = set(candidate_selection)

    # Held names that no longer have a score (e.g. dropped from the universe)
    # cannot be compared, so they are released.
    held = [s for s in holdings if s in scores]
    held_sorted = sorted(held, key=lambda s: (scores[s], str(s)))  # weakest first

    final = holdings & candidate_set  # names in both lists: keep automatically
    remaining_candidates = [s for s in candidate_selection if s not in final]
    remaining_held = [s for s in held_sorted if s not in final]

    for cand in remaining_candidates:
        if len(final) >= portfolio_size:
            break
        if len(final) + len(remaining_held) < portfolio_size:
            final.add(cand)  # portfolio isn't full yet, no need to replace anything
            continue
        if remaining_held and scores[cand] - scores[remaining_held[0]] > min_score_edge:
            remaining_held.pop(0)
            final.add(cand)

    final.update(remaining_held)

    if len(final) < portfolio_size:
        for sym in candidate_selection:
            if len(final) >= portfolio_size:
                break
            final.add(sym)

    ranked = sorted(final, key=lambda s: (-scores.get(s, float("-inf")), str(s)))
    return ranked[:portfolio_size]


def inverse_vol_weights(
    selected: Sequence[Hashable],
    symbol_data_map: Mapping[Hashable, Any],
    lookback_mid: int,
    total_exposure: float,
) -> dict[Hashable, float]:
    """
    Split total_exposure (e.g. 0.65 = 65% of the account) across the
    selected names inversely to each one's own volatility: a calmer stock
    gets a bigger slice, a wilder one gets a smaller slice. The goal is a
    book where no single jumpy name can swing total performance on its own.

    Names with unknown volatility get zero weight; if no name has a usable
    volatility, exposure is split equally.
    """
    if not selected:
        return {}

    inv_vols: dict[Hashable, float] = {}
    for sym in selected:
        sd = symbol_data_map.get(sym)
        vol = sd.volatility(lookback_mid) if sd else None
        inv_vols[sym] = (1.0 / vol) if (vol and vol > 1e-6) else 0.0

    total_inv_vol = sum(inv_vols.values())
    if total_inv_vol <= 0:
        return {sym: total_exposure / len(selected) for sym in selected}

    return {sym: total_exposure * (iv / total_inv_vol) for sym, iv in inv_vols.items()}
