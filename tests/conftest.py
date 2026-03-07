"""
Shared pytest fixtures for the ShockArb test suite.

All fixtures are session-scoped where the data is read-only, and
function-scoped (the default) where tests might mutate the object.

Fixture hierarchy
-----------------
  sample_etf_returns    — (36 days × 5 ETFs) synthetic return DataFrame
  sample_stock_returns  — (36 days × 5 stocks) aligned with above
  fitted_model          — FactorModel fitted with n_components=2
  temp_dir              — temporary directory, cleaned up after each test
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from shockarb.engine import FactorModel


# ---------------------------------------------------------------------------
# Market-data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_etf_returns() -> pd.DataFrame:
    """
    Synthetic ETF returns with realistic crisis-like structure.

    Five ETFs covering different risk/factor axes:
      VOO — broad equity (pure market beta)
      VDE — energy (geopolitical crisis beneficiary)
      TLT — long bonds (negative beta / flight-to-safety)
      GLD — gold (negative beta)
      ITA — defence (mild positive crisis drift)
    """
    np.random.seed(42)
    dates = pd.bdate_range("2022-02-10", "2022-03-31")
    market = np.random.normal(0, 0.01, len(dates))
    return pd.DataFrame(
        {
            "VOO": market + np.random.normal(0, 0.005, len(dates)),
            "VDE": 0.5 * market + np.random.normal(0.002, 0.015, len(dates)),
            "TLT": -0.3 * market + np.random.normal(0.001, 0.008, len(dates)),
            "GLD": -0.2 * market + np.random.normal(0.001, 0.010, len(dates)),
            "ITA": 0.6 * market + np.random.normal(0.001, 0.012, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def sample_stock_returns(sample_etf_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic stock returns aligned with sample_etf_returns.

    Five stocks with varying factor exposures:
      V    — payment processor, moderate beta
      MSFT — tech, high beta
      LMT  — defence, crisis beneficiary
      CVX  — energy, positive crisis drift
      UNH  — healthcare, defensive
    """
    np.random.seed(43)
    market = sample_etf_returns["VOO"].values
    n = len(market)
    return pd.DataFrame(
        {
            "V":    0.8 * market + np.random.normal(-0.001, 0.012, n),
            "MSFT": 1.1 * market + np.random.normal(-0.001, 0.015, n),
            "LMT":  0.4 * market + np.random.normal(0.002, 0.010, n),
            "CVX":  0.3 * market + np.random.normal(0.003, 0.018, n),
            "UNH":  0.5 * market + np.random.normal(0, 0.008, n),
        },
        index=sample_etf_returns.index,
    )


@pytest.fixture
def fitted_model(
    sample_etf_returns: pd.DataFrame,
    sample_stock_returns: pd.DataFrame,
) -> FactorModel:
    """A FactorModel fitted with 2 components, ready for scoring tests."""
    return FactorModel(sample_etf_returns, sample_stock_returns).fit(n_components=2)


# ---------------------------------------------------------------------------
# Infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir() -> str:
    """Isolated temporary directory, deleted after each test."""
    d = tempfile.mkdtemp(prefix="shockarb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_model(temp_dir) -> "FactorModel":
    """Minimal 3-ETF / 2-stock fitted model saved in temp_dir as 'us'."""
    import numpy as np
    import pandas as pd
    from shockarb.engine import FactorModel
    from shockarb.config import ExecutionConfig
    import shockarb.pipeline as pipeline

    np.random.seed(42)
    dates = pd.bdate_range("2022-02-10", "2022-03-31")
    n = len(dates)
    etf = pd.DataFrame(
        {"VOO": np.random.randn(n) * 0.01,
         "TLT": np.random.randn(n) * 0.01,
         "GLD": np.random.randn(n) * 0.01},
        index=dates,
    )
    stk = pd.DataFrame(
        {"AAPL": np.random.randn(n) * 0.015,
         "MSFT": np.random.randn(n) * 0.015},
        index=dates,
    )
    model = FactorModel(etf, stk).fit(n_components=2)
    cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
    pipeline.save_model(model, "us", cfg)
    return model
