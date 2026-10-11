"""
main.py
=======
QuantConnect entrypoint. This file is intentionally thin — it owns
scheduling and order execution, and delegates every actual decision
(regime detection, scoring, diversification, sizing, risk) to the modules
in this package. That separation is what makes each piece independently
testable and independently swappable.

To run on QuantConnect Cloud, upload this file as the project's main.py
and upload the `src/master_quant/` folder alongside it (so the project
contains `main.py` and `master_quant/`). QC multi-file Python projects
support local package imports exactly like this.
"""

import numpy as np
from AlgorithmImports import *
from scipy.stats import zscore

from master_quant.config import StrategyConfig
from master_quant.strategy.portfolio import (
    apply_turnover_band,
    correlation_diversify,
    inverse_vol_weights,
)
from master_quant.strategy.regime import compute_market_gating
from master_quant.strategy.risk import apply_drawdown_cut, regime_exposure_for, update_equity_peak
from master_quant.strategy.scoring import SymbolData, compute_stock_score
from master_quant.strategy.universe import coarse_filter, fine_filter
