"""
ShockArb Factor Model
=====================

A quantitative system for identifying stocks mispriced by geopolitical panic selling.

The model decomposes daily stock returns into:
  1. A macro-factor-explained portion (broad market, sector, cross-asset moves)
  2. A residual (unexplained by factors)

Stocks whose actual returns deviate significantly from factor-implied returns during 
a crisis are flagged as mispricing candidates.

Quick Start
-----------
    from shockarb import FactorModel, Pipeline, US_UNIVERSE
    
    # Build and fit a model
    model = Pipeline.build(US_UNIVERSE)
    
    # Score live returns
    scores = model.score(today_etf_returns, today_stock_returns)
    print(scores.head(10))  # Top 10 mispriced stocks

Architecture
------------
    config.py   - Universe definitions and execution settings
    engine.py   - Pure math: SVD factor extraction, OLS projection, scoring
    pipeline.py - Data I/O: fetching, caching, persistence
    cli.py      - Command-line interface for all operations

Design Principle: The engine module contains zero I/O. It takes DataFrames in and 
returns DataFrames out. All file/API interaction lives in pipeline.py.
"""

__version__ = "2.1.0"

# Public API - these are the only imports users need
from shockarb.engine import FactorModel, FactorDiagnostics
from shockarb.pipeline import Pipeline
from shockarb.config import (
    UniverseConfig,
    ExecutionConfig,
    US_UNIVERSE,
    GLOBAL_UNIVERSE,
)

__all__ = [
    "FactorModel",
    "FactorDiagnostics", 
    "Pipeline",
    "UniverseConfig",
    "ExecutionConfig",
    "US_UNIVERSE",
    "GLOBAL_UNIVERSE",
]
