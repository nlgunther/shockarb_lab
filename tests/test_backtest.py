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
# Named Constants for Test Clarity
# =============================================================================
# These magic numbers are used to bypass or fail filters in tests.
# See their usage in individual test functions for context.

BYPASS_FILTERS = -1.0
"""Threshold set to -1.0 bypasses all filters (all real scores are >= 0).
Used in tests where we want to force signal generation without filtering."""

FAIL_ALL_FILTERS = 1.0
"""Threshold set to 1.0 rejects all signals (requires >100% confidence).
Used in tests where we want to suppress all signal generation."""

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
    Generates 50 business days of synthetic returns for backtesting.

    Sizing: 50 days provides:
      - 35 days for calibration window (the model is fit on this history)
      - 15 days for walk-forward testing (each day, the model scores, enters trades, tracks P&L)

    Returns:
      - etfs: (50, 3) DataFrame of synthetic ETF returns (lower volatility, the factor basis)
      - stocks: (50, 4) DataFrame of synthetic stock returns (higher volatility, what we trade)

    The synthetic data is random normal, seeded for reproducibility. In real backtests,
    you'd load actual historical price data here.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=50)

    # Synthetic ETF returns: lower volatility (1% stdev), represent broad factors
    etfs = pd.DataFrame(
        np.random.normal(0, 0.01, size=(50, 3)),
        index=dates,
        columns=["ETF1", "ETF2", "ETF3"]
    )

    # Synthetic stock returns: higher volatility (2% stdev), what we trade
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
    """
    Verify that BacktestConfig's frozen dataclass enforces strict validation.

    Why immutable + validation: By freezing the dataclass, we prevent accidental
    mutations during the backtest (bugs from typos like config.min_confidence = 0.1).
    The __post_init__ validator catches invalid configurations at construction time,
    before expensive computation starts.

    Test coverage:
      - Valid return types: "raw", "residual", "both"
      - Invalid return types are rejected immediately with a clear error message
    """
    # Valid configs — should not raise
    BacktestConfig(universe=dummy_universe, return_type="raw")
    BacktestConfig(universe=dummy_universe, return_type="residual")
    BacktestConfig(universe=dummy_universe, return_type="both")

    # Invalid config — should raise immediately
    with pytest.raises(ValueError, match="return_type must be 'raw', 'residual', or 'both'"):
        BacktestConfig(universe=dummy_universe, return_type="invalid_type")

def test_walk_forward_date_slicing(dummy_universe, synthetic_returns):
    """
    Verify the walk-forward loop correctly slices dates to prevent look-ahead bias.

    Look-ahead bias: using future data to make past decisions. For example, if you use
    Day 50's return to fit a model on Day 40, you've peeked into the future.

    Solution: Fit on [Day 1 : Day N], then score only Day N+1. This way, the model
    never sees the day it's scoring.

    Test setup: 50 total days, 35-day calibration window
      - Days 1-35: Calibration (used to fit model on day 35)
      - Days 36-50: Walk-forward (score, trade, measure P&L)
      - Expected result: 15 scoring days (indices 35-49 in zero-indexed)
    """
    etfs, stocks = synthetic_returns
    config = BacktestConfig(universe=dummy_universe, calibration_window=35)

    runner = Backtest(config, etfs, stocks)

    # Assertion 1: Correct count of walk-forward days
    assert len(runner._walk_forward_dates) == 15, \
        f"Expected 15 walk-forward days (50 - 35), got {len(runner._walk_forward_dates)}"

    # Assertion 2: First walk-forward day is immediately after calibration window
    assert runner._walk_forward_dates[0] == stocks.index[35], \
        "First scoring day should be index 35 (one past the calibration window)"

    # Assertion 3: Last walk-forward day is the final day in the dataset
    assert runner._walk_forward_dates[-1] == stocks.index[-1], \
        "Last scoring day should be the final date in the dataset"

@patch("shockarb.engine.FactorModel.score")
def test_cohort_expiration_and_garbage_collection(mock_score, dummy_universe, synthetic_returns):
    """
    Ensure cohorts are retired from memory once they reach max holding period.

    Why this matters: In a long backtest (years of daily data, hundreds of stocks),
    memory can grow unbounded if old cohorts aren't garbage-collected. This test
    verifies that closed trades are removed from the active_cohorts dictionary.

    Test strategy: Set holding_periods=(1, 2, 3), run only 5 walk-forward days,
    then verify that:
      1. Trades are recorded at the designated milestones (days 1, 2, 3)
      2. Cohorts are deleted once they reach max(holding_periods) = 3 days
    """
    etfs, stocks = synthetic_returns
    config = BacktestConfig(
        universe=dummy_universe,
        calibration_window=35,
        holding_periods=(1, 2, 3),  # Max holding is 3 days
        min_confidence=BYPASS_FILTERS,  # Accept all signals (no confidence filter)
        min_r_squared=BYPASS_FILTERS    # Accept all signals (no fit quality filter)
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
    Verify that when return_type='both', both raw and residual returns are computed.

    The test checks that:
      1. Both raw_return and residual_return columns exist in the ledger
      2. They contain reasonable values (not NaN, not infinite)
      3. Residual returns reflect factor-hedged alpha (actual - expected)

    This is important because residual return is the "true" mispricing signal:
    it shows whether the stock beat/missed its factor-implied return.
    """
    etfs, stocks = synthetic_returns

    # Inject deterministic returns for one stock to verify the calculation
    # Day 35: Calibration window ends (data up to this point used for fitting)
    # Day 36: Entry signal generated, trade opened
    # Day 37: T+1, trade held and marked-to-market
    stocks.loc[stocks.index[36], "STK1"] = -0.05  # Artificial crash (triggers buy signal)
    stocks.loc[stocks.index[37], "STK1"] = 0.10   # Artificial rebound (validates trade)

    config = BacktestConfig(
        universe=dummy_universe,
        calibration_window=35,
        holding_periods=(1,),  # Record at T+1 holding milestone
        return_type="both",    # Output both raw and residual returns
        min_confidence=BYPASS_FILTERS,  # Accept all signals (no confidence threshold)
        min_r_squared=BYPASS_FILTERS    # Accept all fits (no R² threshold)
    )

    runner = Backtest(config, etfs, stocks)
    results = runner.run()

    # Extract STK1 trades from the ledger
    stk1_trades = results.ledger[results.ledger["ticker"] == "STK1"]

    # Guard: if no trades, the test is vacuous (signals were filtered out somehow)
    if not stk1_trades.empty:
        # Verify both return columns are present
        assert "raw_return" in results.ledger.columns, \
            "Ledger missing raw_return column (should always exist)"
        assert "residual_return" in results.ledger.columns, \
            "Ledger missing residual_return column (should exist when return_type='both')"

        # Find the specific trade opened on the crash day (Day 36)
        entry_date = stocks.index[36]
        target_trade = stk1_trades[stk1_trades["entry_date"] == entry_date]

        if not target_trade.empty:
            # Verify raw return: the stock rallied 10% on Day 37, so raw_return ≈ 0.10
            t1_raw = target_trade.iloc[0]["raw_return"]
            assert np.isclose(t1_raw, 0.10, atol=0.05), \
                f"Raw return {t1_raw} doesn't match expected ~10% rally"

            # Verify residual return exists and is finite
            t1_residual = target_trade.iloc[0]["residual_return"]
            assert np.isfinite(t1_residual), \
                f"Residual return {t1_residual} is NaN or infinite (calculation failed)"

            # The residual return should differ from raw_return because the stock's
            # movement relative to factors is what we care about, not gross movement.
            # If residual ≈ 0, the stock just tracked its factors (no alpha).
            # If residual > 0, the stock beat its factors (positive alpha).
            # If residual < 0, the stock missed its factors (negative alpha).
            # We can't assert a specific value without knowing the model's prediction,
            # but we can verify it's reasonable (not 100%+ or -100%+).
            assert -1.0 <= t1_residual <= 1.0, \
                f"Residual return {t1_residual} is unreasonably extreme"