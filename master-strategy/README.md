# Project structure

master-strategy/
├── README.md                    # this file
├── requirements.txt              # numpy, scipy, pytest
├── config/
│   └── strategy_config.py       # every tunable parameter, in one place
├── src/
│   ├── main.py                  # QuantConnect entrypoint — scheduling & order execution only
│   ├── universe.py              # coarse/fine stock universe filters
│   ├── regime.py                # market regime detection (Layer 1)
│   ├── scoring.py                # SymbolData + per-stock scoring (Layers 2 & 4)
│   ├── portfolio.py             # diversification, turnover control, position sizing (Layer 3)
│   └── risk.py                   # regime exposure + drawdown circuit breaker
├── tests/
│   ├── test_scoring.py
│   ├── test_regime.py
│   ├── test_portfolio.py
│   └── test_risk.py
└── results/                       # backtest reports go here (see §7)



# Running it

```bash
pip install -r requirements.txt
pytest tests/ -v
```

37 tests currently cover: score calculation edge cases (zero volatility,
insufficient history), regime classification boundaries, correlation-based
diversification, the turnover band's swap/keep logic, inverse-volatility
weighting, and both risk-cut mechanisms.

# Backtest

Upload `src/`, `config/`, and `requirements.txt` as a QuantConnect Cloud
project and run a backtest from the QC IDE. 

