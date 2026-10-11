# MASTER: Market-Guided Stock Transformer Trading System

[![CI](https://github.com/dominictyz/MASTER-Market-Guided-Stock-Transformer-Trading-Algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/dominictyz/MASTER-Market-Guided-Stock-Transformer-Trading-Algorithm/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A cross-sectional equity research and trading framework for S&P 500 stocks,
inspired by **MASTER** (Li et al., *"MASTER: Market-Guided Stock Transformer
for Stock Price Forecasting"*, AAAI 2024).

The current release (`v0.1.0`) contains a rule-based, MASTER-inspired factor
strategy that runs on QuantConnect. It serves as the **baseline** that the
PyTorch transformer implementation (planned) must beat.

## How the strategy works

| Layer | MASTER concept | Module |
|---|---|---|
| 1 | Market-guided gating: VIX level + SPY 20-day trend → one of 4 regimes, each with its own factor weights | `strategy/regime.py` |
| 2 | Intra-stock / temporal aggregation: volatility-adjusted 5/20/60-day momentum + short-term reversal | `strategy/scoring.py` |
| 3 | Inter-stock aggregation: greedy correlation filter, turnover band, inverse-volatility sizing | `strategy/portfolio.py` |
| 4 | Risk: regime-based gross exposure + 15% drawdown circuit breaker | `strategy/risk.py` |

All decision logic is pure Python with no QuantConnect dependency, so every
layer is unit tested in isolation. `qc/main.py` is a thin adapter that only
handles scheduling and order execution.

## Project structure

```
.
├── src/master_quant/          # installable Python package
│   ├── config.py              # every tunable parameter, validated on load
│   └── strategy/
│       ├── regime.py          # market regime detection
│       ├── scoring.py         # rolling features + composite stock score
│       ├── portfolio.py       # diversification, turnover control, sizing
│       ├── risk.py            # exposure sizing + drawdown circuit breaker
│       └── universe.py        # liquidity / security-type filters
├── qc/main.py                 # QuantConnect Cloud entrypoint
├── tests/                     # pytest + hypothesis test suite
├── .github/workflows/ci.yml   # lint, type-check and test on every push / PR
└── pyproject.toml
```

## Quick start

```bash
git clone https://github.com/dominictyz/MASTER-Market-Guided-Stock-Transformer-Trading-Algorithm.git
cd MASTER-Market-Guided-Stock-Transformer-Trading-Algorithm
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install

pytest            # 79 tests, coverage report printed at the end
ruff check .      # lint
mypy              # static type check
```

## Testing

79 tests (unit, parametrised boundary and property-based tests using
[Hypothesis](https://hypothesis.readthedocs.io/)) with **94% branch coverage**;
CI fails below 85%. The suite found and now guards against two bugs in the
original version:

- **Non-reproducible portfolios:** the turnover band truncated an unordered
  `set`, so which stocks survived depended on Python's hash seed and changed
  between runs. Output is now sorted by score with deterministic tie-breaking.
- **Floating-point drawdown error:** `1 - 85_000/100_000` evaluates to
  `0.15000000000000002`, so the 15% circuit breaker fired exactly *at* the
  trigger. Drawdown is now computed as `(peak - equity) / peak`.

## Running on QuantConnect

Create a QuantConnect Cloud Python project, then upload `qc/main.py` as the
project's `main.py` and the `src/master_quant/` folder next to it.

## Roadmap

- [x] `v0.1.0`: package layout, test suite, CI
- [ ] PySpark data pipeline (bronze / silver / gold) with data validation
- [ ] Point-in-time feature store (Alpha158-style factors)
- [ ] Independent vectorised backtester with transaction costs
- [ ] MASTER transformer in PyTorch + baselines, tracked with MLflow
- [ ] Risk engine: VaR / CVaR, stress tests, factor exposures
- [ ] OMS, reconciliation and paper trading
- [ ] FastAPI signal service, Docker, dashboard

## License

MIT. See [LICENSE](LICENSE).
