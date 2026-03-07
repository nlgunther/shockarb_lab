"""
ShockArb — Geopolitical crisis mispricing detector.

The model decomposes stock returns into a macro-factor-explained component
(broad market, sector, cross-asset moves) and a residual.  Stocks whose
actual returns deviate significantly from factor-implied returns during a
crisis are flagged as candidates for mean-reversion trades.

Quick start
-----------
    import shockarb.pipeline as pipeline
    from shockarb.config import US_UNIVERSE

    model = pipeline.build(US_UNIVERSE)
    scores = model.score(today_etf_returns, today_stock_returns)
    print(scores.head(10))

Architecture
------------
    config.py    Universe definitions and execution settings.
    engine.py    Pure math: SVD factor extraction, OLS projection, scoring.
    cache.py     Intelligent OHLCV parquet caching with incremental updates.
    pipeline.py  Data I/O: fetching, caching, model persistence — no math.
    report.py    Terminal formatting and display — no math, no I/O.
    cli.py       Command-line interface for all operations.

Design rule
-----------
engine.py contains zero I/O.  It takes DataFrames in and returns DataFrames
out.  All file and network interaction is in pipeline.py and cache.py.
"""

__version__ = "3.0.0"

# Public API
from shockarb.engine import FactorDiagnostics, FactorModel
from shockarb.config import (
    ExecutionConfig,
    GLOBAL_UNIVERSE,
    UniverseConfig,
    US_UNIVERSE,
)
import shockarb.pipeline as pipeline  # preferred: `pipeline.build(...)` etc.

from shockarb.names import TickerReferenceResolver
from shockarb.backtest import Backtest, BacktestConfig, BacktestResults

__all__ = [
    "FactorModel",
    "FactorDiagnostics",
    "UniverseConfig",
    "ExecutionConfig",
    "US_UNIVERSE",
    "GLOBAL_UNIVERSE",
    "pipeline",
    "TickerReferenceResolver",
    "Backtest",
    "BacktestConfig",
    "BacktestResults",
]
