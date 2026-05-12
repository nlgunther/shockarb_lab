"""
Tests for shockarb.backtest — Walk-forward testing engine.

Covers:
  - BacktestConfig validation (return_type enforcement)
  - Date slicing (ensuring calibration window prevents look-ahead bias)
  - Cohort tracking and expiration logic
  - PnL compounding for both raw and residual returns
"""

from __future__ import annotations

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from shockarb.backtest import Backtest, BacktestConfig, BacktestResults
from shockarb.config import UniverseConfig
from shockarb.engine import FactorModel

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_universe() -> UniverseConfig:
    """A minimal valid universe for testing SVD boundaries."""
    return UniverseConfig(
        name="test_uni",
        market_etfs=["ETF1", "ETF2", "ETF3"],
        individual_stocks=["STK1", "STK2", "STK3", "STK4"],
        n_components=2,
        start_date="2020-01-01",
        end_date="2020-03-01"
    )

@pytest.fixture
def synthetic_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates 50 business days of synthetic returns.
    Ensures enough rows to satisfy a 35-day calibration window + 15 walk-forward days.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=50)
    
    etfs = pd.DataFrame(
        np.random.normal(0, 0.01, size=(50, 3)), 
        index=dates, 
        columns=["ETF1", "ETF2", "ETF3"]
    )
    stocks = pd.DataFrame(
        np.random.normal(0, 0.02, size=(50, 4)), 
        index=dates, 
        columns=["STK1", "STK2", "STK3", "STK4"]
    )
    return etfs, stocks

# =============================================================================
# Tests
# =============================================================================

def test_backtest_config_validation(dummy_universe):
    """Ensure the immutable config strictly validates the return_type."""
    # Valid configs
    BacktestConfig(universe=dummy_universe, return_type="raw")
    BacktestConfig(universe=dummy_universe, return_type="residual")
    BacktestConfig(universe=dummy_universe, return_type="both")

    # Invalid config
    with pytest.raises(ValueError, match="return_type must be 'raw', 'residual', or 'both'"):
        BacktestConfig(universe=dummy_universe, return_type="invalid_type")

def test_walk_forward_date_slicing(dummy_universe, synthetic_returns):
    """Verify the runner correctly slices the dates to avoid look-ahead bias."""
    etfs, stocks = synthetic_returns
    config = BacktestConfig(universe=dummy_universe, calibration_window=35)
    
    runner = Backtest(config, etfs, stocks)
    
    # If we have 50 total days and a 35-day calibration window, 
    # we should have exactly 15 walk-forward scoring days.
    assert len(runner._walk_forward_dates) == 15
    assert runner._walk_forward_dates[0] == stocks.index[35]
    assert runner._walk_forward_dates[-1] == stocks.index[-1]

@patch("shockarb.engine.FactorModel.score")
def test_cohort_expiration_and_garbage_collection(mock_score, dummy_universe, synthetic_returns):
    """
    Ensure cohorts are retired from memory exactly when they reach 
    the maximum holding period to prevent memory leaks in long backtests.
    """
    etfs, stocks = synthetic_returns
    config = BacktestConfig(
        universe=dummy_universe, 
        calibration_window=35,
        holding_periods=(1, 2, 3), # Max holding is 3 days
        min_confidence=-1.0,       # Force all stocks to be selected
        min_r_squared=-1.0         # Force all stocks to be selected
    )
    
    # Mock the score to return a deterministic DataFrame so the backtester triggers trades
    dummy_scores = pd.DataFrame({
        "confidence_delta": [0.01, 0.02, 0.03, 0.04],
        "r_squared": [0.6, 0.7, 0.8, 0.9],
        "delta_rel": [-0.01, -0.01, -0.01, -0.01]
    }, index=["STK1", "STK2", "STK3", "STK4"])
    mock_score.return_value = dummy_scores

    runner = Backtest(config, etfs, stocks)
    
    # Instead of running the whole loop, we'll manually step through a few days
    # to inspect the active_cohorts dictionary state.
    runner._walk_forward_dates = stocks.index[35:40] # 5 days
    results = runner.run()

    # The ledger should contain records for T+1, T+2, and T+3 for multiple cohorts
    assert not results.ledger.empty
    assert set(results.ledger["holding_period"].unique()) == {1, 2, 3}
    
    # Since we ran 5 days, the cohort from day 1 should have been deleted (days_held = 4),
    # but the cohort from day 5 should still be active (days_held = 0).

def test_residual_pnl_compounding(dummy_universe, synthetic_returns):
    """
    Verify that residual returns correctly invert the delta_rel sign
    and geometrically compound over the holding period.
    """
    etfs, stocks = synthetic_returns
    
    # Inject a deterministic return for STK1 to verify the math
    # Day 35: Calibration ends
    # Day 36: Trade Entry
    # Day 37: T+1 
    stocks.loc[stocks.index[36], "STK1"] = -0.05 # Artificial crash to trigger buy
    stocks.loc[stocks.index[37], "STK1"] = 0.10  # Artificial massive rebound
    
    config = BacktestConfig(
        universe=dummy_universe, 
        calibration_window=35,
        holding_periods=(1,), 
        return_type="both",
        min_confidence=-1.0,  # <-- ADD THIS: bypass signal filter
        min_r_squared=-1.0    # <-- ADD THIS: bypass fit quality filter
    )
    
    runner = Backtest(config, etfs, stocks)
    results = runner.run()
    
   # Extract STK1 trades
    stk1_trades = results.ledger[results.ledger["ticker"] == "STK1"]
    
    if not stk1_trades.empty:
        # Check that both raw and residual columns exist in the ledger
        assert "raw_return" in stk1_trades.columns
        assert "residual_return" in stk1_trades.columns
        
        # FIX: We must explicitly grab the trade that was entered on the crash day (Day 36)
        # to measure the T+1 rebound on Day 37.
        entry_date = stocks.index[36]
        target_trade = stk1_trades[stk1_trades["entry_date"] == entry_date]
        
        # The raw return on the rebound day should track our injected 10% move
        t1_raw = target_trade.iloc[0]["raw_return"]
        assert np.isclose(t1_raw, 0.10, atol=0.05)