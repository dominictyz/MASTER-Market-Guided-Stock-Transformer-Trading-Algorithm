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

import numpy as np


def correlation_diversify(ranked_scores, returns_matrix, n, corr_threshold=0.75):
    """
    Greedy selection: walk down the ranked list, keep a stock only if its
    average correlation with everything already selected is below
    corr_threshold. Correlation here means "do these two stocks tend to
    move together" — 1.0 is lockstep, 0 is unrelated, -1.0 is opposite.
    Two highly correlated stocks add little real diversification even if
    both score well individually.

    ranked_scores: list of (symbol, score) tuples, best first
    returns_matrix: dict[symbol -> np.array of daily returns]
    """
    selected = []

    for sym, score in ranked_scores:
        if len(selected) >= n:
            break
        if sym not in returns_matrix:
            selected.append(sym)
            continue
        if len(selected) == 0:
            selected.append(sym)
            continue

        corrs = []
        for sel_sym in selected:
            if sel_sym in returns_matrix:
                try:
                    r1 = returns_matrix[sym]
                    r2 = returns_matrix[sel_sym]
                    min_len = min(len(r1), len(r2))
                    if min_len > 10:
                        c = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                        if not np.isnan(c):
                            corrs.append(abs(c))
                except Exception:
                    pass

        avg_corr = np.mean(corrs) if corrs else 0.0
        if avg_corr < corr_threshold:
            selected.append(sym)

    # If too many candidates got rejected for being correlated, top up
    # with whatever's left so the portfolio isn't under-sized.
    if len(selected) < n:
        for sym, score in ranked_scores:
            if sym not in selected:
                selected.append(sym)
            if len(selected) >= n:
                break

    return selected


def apply_turnover_band(
    current_holdings, candidate_selection, scores, portfolio_size, min_score_edge=0.15
):
    """
    Every rebalance, a fresh ranking will always look at least a little
    different from the last one — some of that difference is real
    information, some of it is just noise. Rebuilding the whole portfolio
    every time chases the noise too, which racks up trading costs for no
    real benefit.

    Rule: keep a currently-held stock unless a new candidate's score beats
    the *worst* currently-held stock's score by more than min_score_edge.
    Only swap when the edge is big enough to be worth paying a commission for.
    """
    candidate_set = set(candidate_selection)
    held = [s for s in current_holdings if s in scores]
    held_sorted = sorted(held, key=lambda s: scores[s])  # weakest held name first

    final = set(current_holdings) & candidate_set  # names in both lists: keep automatically

    remaining_candidates = [s for s in candidate_selection if s not in final]
    remaining_held = [s for s in held_sorted if s not in final]

    for cand in remaining_candidates:
        if len(final) >= portfolio_size:
            break
        if len(final) + len(remaining_held) < portfolio_size:
            final.add(cand)  # portfolio isn't full yet, no need to replace anything
            continue
        if remaining_held:
            weakest = remaining_held[0]
            if scores[cand] - scores[weakest] > min_score_edge:
                remaining_held.pop(0)
                final.add(cand)

    final.update(remaining_held)

    if len(final) < portfolio_size:
        for sym in candidate_selection:
            if sym not in final:
                final.add(sym)
            if len(final) >= portfolio_size:
                break

    return list(final)[:portfolio_size]


def inverse_vol_weights(selected, symbol_data_map, lookback_mid, total_exposure):
    """
    Split total_exposure (e.g. 0.65 = 65% of the account) across the
    selected names inversely to each one's own volatility: a calmer stock
    gets a bigger slice, a wilder one gets a smaller slice. The goal is a
    book where no single jumpy name can swing total performance on its own.
    """
    inv_vols = {}
    for sym in selected:
        sd = symbol_data_map.get(sym)
        vol = sd.volatility(lookback_mid) if sd else None
        inv_vols[sym] = (1.0 / vol) if (vol and vol > 1e-6) else 0.0

    total_inv_vol = sum(inv_vols.values())
    if total_inv_vol <= 0:
        n = len(selected)
        return {sym: total_exposure / n for sym in selected} if n > 0 else {}

    return {sym: total_exposure * (iv / total_inv_vol) for sym, iv in inv_vols.items()}
