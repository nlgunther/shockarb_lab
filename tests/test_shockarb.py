#!/usr/bin/env python3
"""
Comprehensive Test Suite for ShockArb Factor Model
===================================================

Tests cover:
  - Unit tests for each module (config, engine, pipeline)
  - Integration tests for end-to-end workflows
  - Edge cases and error handling
  - Mathematical correctness validation

Run with pytest:
    pytest tests/ -v
    pytest tests/ -v --cov=shockarb --cov-report=term-missing

Or run directly:
    python tests/test_shockarb.py
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shockarb.config import UniverseConfig, ExecutionConfig, US_UNIVERSE, GLOBAL_UNIVERSE
from shockarb.engine import FactorModel, FactorDiagnostics
from shockarb.pipeline import Pipeline


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_etf_returns():
    """Generate synthetic ETF return data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-02-10", "2022-03-31")
    n_days = len(dates)
    n_etfs = 5
    
    # Create correlated returns with realistic structure
    market_factor = np.random.normal(0, 0.01, n_days)
    
    etfs = ["VOO", "VDE", "TLT", "GLD", "ITA"]
    data = {}
    
    # VOO: pure market
    data["VOO"] = market_factor + np.random.normal(0, 0.005, n_days)
    # VDE: energy, positive crisis drift
    data["VDE"] = 0.5 * market_factor + np.random.normal(0.002, 0.015, n_days)
    # TLT: bonds, negative beta
    data["TLT"] = -0.3 * market_factor + np.random.normal(0.001, 0.008, n_days)
    # GLD: gold, negative beta
    data["GLD"] = -0.2 * market_factor + np.random.normal(0.001, 0.010, n_days)
    # ITA: defense, slight positive drift
    data["ITA"] = 0.6 * market_factor + np.random.normal(0.001, 0.012, n_days)
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_stock_returns(sample_etf_returns):
    """Generate synthetic stock return data aligned with ETF data."""
    np.random.seed(43)
    dates = sample_etf_returns.index
    n_days = len(dates)
    
    # Extract implicit market factor from ETFs
    market = sample_etf_returns["VOO"].values
    
    stocks = ["V", "MSFT", "LMT", "CVX", "UNH"]
    data = {}
    
    # V: payment processor, moderate market beta
    data["V"] = 0.8 * market + np.random.normal(-0.001, 0.012, n_days)
    # MSFT: tech, high beta
    data["MSFT"] = 1.1 * market + np.random.normal(-0.001, 0.015, n_days)
    # LMT: defense, benefits from crisis
    data["LMT"] = 0.4 * market + np.random.normal(0.002, 0.010, n_days)
    # CVX: energy, high positive drift in crisis
    data["CVX"] = 0.3 * market + np.random.normal(0.003, 0.018, n_days)
    # UNH: healthcare, defensive
    data["UNH"] = 0.5 * market + np.random.normal(0, 0.008, n_days)
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def fitted_model(sample_etf_returns, sample_stock_returns):
    """Create a fitted FactorModel for testing."""
    model = FactorModel(sample_etf_returns, sample_stock_returns)
    model.fit(n_components=2)
    return model


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    tmp_dir = tempfile.mkdtemp(prefix="shockarb_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def minimal_universe():
    """Create a minimal universe for fast testing."""
    return UniverseConfig(
        name="test",
        market_etfs=["VOO", "TLT", "GLD"],
        individual_stocks=["AAPL", "MSFT"],
        n_components=2,
        start_date="2022-02-10",
        end_date="2022-03-31",
    )


# =============================================================================
# Config Module Tests
# =============================================================================

class TestUniverseConfig:
    """Tests for UniverseConfig dataclass."""
    
    def test_basic_creation(self):
        """Test basic universe config creation."""
        config = UniverseConfig(
            name="test",
            market_etfs=["SPY", "TLT"],
            individual_stocks=["AAPL"],
            n_components=2,
            start_date="2022-01-01",
            end_date="2022-03-31",
        )
        
        assert config.name == "test"
        assert len(config.market_etfs) == 2
        assert config.n_components == 2
    
    def test_frozen_immutability(self):
        """Test that UniverseConfig is immutable."""
        config = UniverseConfig(
            name="test",
            market_etfs=["SPY"],
            individual_stocks=["AAPL"],
            n_components=2,
            start_date="2022-01-01",
            end_date="2022-03-31",
        )
        
        with pytest.raises(AttributeError):
            config.name = "modified"
    
    def test_validation_empty_etfs(self):
        """Test that empty ETF list raises error."""
        with pytest.raises(ValueError, match="market_etfs cannot be empty"):
            UniverseConfig(
                name="test",
                market_etfs=[],
                individual_stocks=["AAPL"],
                n_components=2,
                start_date="2022-01-01",
                end_date="2022-03-31",
            )
    
    def test_validation_empty_stocks(self):
        """Test that empty stock list raises error."""
        with pytest.raises(ValueError, match="individual_stocks cannot be empty"):
            UniverseConfig(
                name="test",
                market_etfs=["SPY"],
                individual_stocks=[],
                n_components=2,
                start_date="2022-01-01",
                end_date="2022-03-31",
            )
    
    def test_validation_zero_components(self):
        """Test that zero components raises error."""
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            UniverseConfig(
                name="test",
                market_etfs=["SPY"],
                individual_stocks=["AAPL"],
                n_components=0,
                start_date="2022-01-01",
                end_date="2022-03-31",
            )
    
    def test_prebuilt_us_universe(self):
        """Test that US_UNIVERSE is properly configured."""
        assert US_UNIVERSE.name == "us"
        assert len(US_UNIVERSE.market_etfs) > 10
        assert len(US_UNIVERSE.individual_stocks) > 30
        assert US_UNIVERSE.n_components == 3
        assert "VOO" in US_UNIVERSE.market_etfs
        assert "V" in US_UNIVERSE.individual_stocks
    
    def test_prebuilt_global_universe(self):
        """Test that GLOBAL_UNIVERSE is properly configured."""
        assert GLOBAL_UNIVERSE.name == "global"
        assert "VGK" in GLOBAL_UNIVERSE.market_etfs  # European ETF
        assert "BNO" in GLOBAL_UNIVERSE.market_etfs  # Brent crude


class TestExecutionConfig:
    """Tests for ExecutionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        # Ensure no env var pollution
        old_val = os.environ.pop("SHOCK_ARB_DATA_DIR", None)
        try:
            config = ExecutionConfig()
            
            assert config.use_cache is True
            assert config.log_to_file is True
            assert config.log_level == "INFO"
            assert "data" in config.data_dir
        finally:
            if old_val:
                os.environ["SHOCK_ARB_DATA_DIR"] = old_val
    
    def test_custom_data_dir(self):
        """Test custom data directory."""
        config = ExecutionConfig(data_dir="/custom/path")
        assert config.data_dir == "/custom/path"
    
    def test_env_var_data_dir(self):
        """Test environment variable for data directory."""
        with patch.dict(os.environ, {"SHOCK_ARB_DATA_DIR": "/env/path"}):
            config = ExecutionConfig()
            assert config.data_dir == "/env/path"
    
    def test_resolve_path(self, temp_data_dir):
        """Test path resolution creates directory."""
        config = ExecutionConfig(data_dir=temp_data_dir)
        
        # Resolve a path in a subdirectory
        subdir = os.path.join(temp_data_dir, "subdir")
        config_sub = ExecutionConfig(data_dir=subdir)
        
        path = config_sub.resolve_path("test.json")
        
        assert os.path.exists(subdir)
        assert path == os.path.join(subdir, "test.json")
    
    def test_configure_logger_idempotent(self):
        """Test that logger configuration is idempotent."""
        config = ExecutionConfig(log_to_file=False)
        
        config.configure_logger()
        config.configure_logger()  # Second call should not error
        
        assert config._logger_configured is True


# =============================================================================
# Engine Module Tests
# =============================================================================

class TestFactorDiagnostics:
    """Tests for FactorDiagnostics dataclass."""
    
    def test_summary_output(self, fitted_model):
        """Test that summary produces readable output."""
        summary = fitted_model.diagnostics.summary()
        
        assert "FactorModel Diagnostics" in summary
        assert "days" in summary
        assert "ETFs" in summary
        assert "Stocks" in summary
        assert "Variance" in summary


class TestFactorModel:
    """Tests for FactorModel class."""
    
    def test_initialization(self, sample_etf_returns, sample_stock_returns):
        """Test model initialization."""
        model = FactorModel(sample_etf_returns, sample_stock_returns)
        
        assert model._fitted is False
        assert model.loadings is None
        assert model.diagnostics is None
    
    def test_initialization_index_mismatch(self, sample_etf_returns):
        """Test that mismatched indices raise error."""
        bad_stocks = pd.DataFrame(
            {"AAPL": [0.01, 0.02]},
            index=pd.date_range("2020-01-01", periods=2)
        )
        
        with pytest.raises(ValueError, match="Index mismatch"):
            FactorModel(sample_etf_returns, bad_stocks)
    
    def test_fit_basic(self, sample_etf_returns, sample_stock_returns):
        """Test basic model fitting."""
        model = FactorModel(sample_etf_returns, sample_stock_returns)
        result = model.fit(n_components=2)
        
        # Check return value for chaining
        assert result is model
        
        # Check fitted state
        assert model._fitted is True
        assert model._Vt is not None
        assert model._F is not None
        assert model.loadings is not None
        assert model.diagnostics is not None
    
    def test_fit_components_validation(self, sample_etf_returns, sample_stock_returns):
        """Test that too many components raises error."""
        model = FactorModel(sample_etf_returns, sample_stock_returns)
        
        # n_components must be < min(T, N_etf)
        with pytest.raises(ValueError, match="n_components"):
            model.fit(n_components=100)
    
    def test_fit_diagnostics_shape(self, fitted_model):
        """Test diagnostics have correct shapes."""
        diag = fitted_model.diagnostics
        
        assert diag.n_factors == 2
        assert len(diag.explained_variance_ratio) == 2
        assert len(diag.stock_r_squared) == 5  # 5 stocks
        assert len(diag.residual_vol) == 5
    
    def test_fit_variance_explained(self, fitted_model):
        """Test that explained variance is reasonable."""
        diag = fitted_model.diagnostics
        
        # Variance ratios should sum to cumulative
        assert abs(sum(diag.explained_variance_ratio) - diag.cumulative_variance) < 1e-10
        
        # Each ratio should be between 0 and 1
        assert all(0 <= r <= 1 for r in diag.explained_variance_ratio)
        
        # Cumulative should be significant (>30% for synthetic data)
        assert diag.cumulative_variance > 0.3
    
    def test_fit_r_squared_bounds(self, fitted_model):
        """Test that R² values are in valid range."""
        r2 = fitted_model.diagnostics.stock_r_squared
        
        # R² should be between 0 and 1
        assert all(0 <= v <= 1 for v in r2.values)
    
    def test_etf_basis_property(self, fitted_model):
        """Test etf_basis property."""
        basis = fitted_model.etf_basis
        
        assert isinstance(basis, pd.DataFrame)
        assert basis.shape == (5, 2)  # 5 ETFs, 2 factors
        assert list(basis.columns) == ["Factor_1", "Factor_2"]
    
    def test_factor_returns_property(self, fitted_model):
        """Test factor_returns property."""
        f_returns = fitted_model.factor_returns
        
        assert isinstance(f_returns, pd.DataFrame)
        assert f_returns.shape[1] == 2  # 2 factors
        assert list(f_returns.columns) == ["Factor_1", "Factor_2"]
    
    def test_require_fitted_guard(self, sample_etf_returns, sample_stock_returns):
        """Test that unfitted model raises error on access."""
        model = FactorModel(sample_etf_returns, sample_stock_returns)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.etf_basis
        
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.factor_returns
    
    def test_score_basic(self, fitted_model, sample_etf_returns, sample_stock_returns):
        """Test basic scoring functionality."""
        # Use last day of data as "today"
        today_etfs = sample_etf_returns.iloc[-1]
        today_stocks = sample_stock_returns.iloc[-1]
        
        scores = fitted_model.score(today_etfs, today_stocks)
        
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) == 5  # 5 stocks
        assert "actual_return" in scores.columns
        assert "expected_return" in scores.columns
        assert "delta" in scores.columns
        assert "r_squared" in scores.columns
        assert "confidence_delta" in scores.columns
    
    def test_score_sorted_by_confidence(self, fitted_model, sample_etf_returns, sample_stock_returns):
        """Test that scores are sorted by confidence_delta descending."""
        today_etfs = sample_etf_returns.iloc[-1]
        today_stocks = sample_stock_returns.iloc[-1]
        
        scores = fitted_model.score(today_etfs, today_stocks)
        
        # Check descending order
        conf_deltas = scores["confidence_delta"].values
        assert all(conf_deltas[i] >= conf_deltas[i+1] for i in range(len(conf_deltas)-1))
    
    def test_score_with_missing_tickers(self, fitted_model):
        """Test scoring with partial ticker coverage."""
        # Only provide subset of ETFs and stocks
        today_etfs = pd.Series({"VOO": -0.01, "TLT": 0.005})
        today_stocks = pd.Series({"V": -0.02, "MSFT": -0.015})
        
        scores = fitted_model.score(today_etfs, today_stocks)
        
        # Should still work with available tickers
        assert len(scores) == 2  # Only V and MSFT
    
    def test_score_confidence_delta_formula(self, fitted_model, sample_etf_returns, sample_stock_returns):
        """Test that confidence_delta = delta * r_squared."""
        today_etfs = sample_etf_returns.iloc[-1]
        today_stocks = sample_stock_returns.iloc[-1]
        
        scores = fitted_model.score(today_etfs, today_stocks)
        
        for _, row in scores.iterrows():
            expected = row["delta"] * row["r_squared"]
            assert abs(row["confidence_delta"] - expected) < 1e-10
    
    def test_project_security(self, fitted_model, sample_stock_returns):
        """Test out-of-sample security projection."""
        # Use one of the existing stocks as a "new" security
        returns = sample_stock_returns["V"]
        
        loadings = fitted_model.project_security("V_NEW", returns)
        
        assert isinstance(loadings, pd.Series)
        assert loadings.name == "V_NEW"
        assert len(loadings) == 2  # 2 factors
    
    def test_project_security_insufficient_overlap(self, fitted_model):
        """Test that insufficient overlap raises error."""
        # Create returns with very different dates
        bad_returns = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.date_range("2020-01-01", periods=3)
        )
        
        with pytest.raises(ValueError, match="overlap"):
            fitted_model.project_security("BAD", bad_returns)
    
    def test_to_dict_and_from_dict(self, fitted_model):
        """Test serialization round-trip."""
        # Serialize
        d = fitted_model.to_dict()
        
        assert "metadata" in d
        assert "Vt" in d
        assert "loadings" in d
        
        # Deserialize
        restored = FactorModel.from_dict(d)
        
        assert restored._fitted is True
        assert restored.diagnostics.n_factors == fitted_model.diagnostics.n_factors
        
        # Check loadings match
        pd.testing.assert_frame_equal(restored.loadings, fitted_model.loadings)
    
    def test_mathematical_orthogonality(self, fitted_model):
        """Test that factors are orthogonal."""
        # Factor returns should be uncorrelated
        f_returns = fitted_model.factor_returns
        corr = f_returns.corr()
        
        # Off-diagonal should be near zero
        for i in range(len(corr)):
            for j in range(len(corr)):
                if i != j:
                    assert abs(corr.iloc[i, j]) < 0.1  # Allow small numerical error
    
    def test_mathematical_mean_centering(self, sample_etf_returns, sample_stock_returns):
        """Test that data is properly mean-centered during fit."""
        model = FactorModel(sample_etf_returns, sample_stock_returns)
        model.fit(n_components=2)
        
        # The stored means should match input means
        pd.testing.assert_series_equal(
            model._etf_mean, 
            sample_etf_returns.mean(),
            check_names=False
        )


# =============================================================================
# Pipeline Module Tests
# =============================================================================

class TestPipeline:
    """Tests for Pipeline class."""
    
    def test_prices_to_returns_basic(self):
        """Test basic price to return conversion."""
        prices = pd.DataFrame({
            "A": [100, 101, 102, 103],
            "B": [50, 51, 50, 52],
        }, index=pd.date_range("2022-01-01", periods=4))
        
        returns = Pipeline.prices_to_returns(prices)
        
        # Should have one less row than prices
        assert len(returns) == 3
        
        # Check first return calculation
        assert abs(returns.iloc[0]["A"] - 0.01) < 1e-10
    
    def test_prices_to_returns_drops_low_coverage(self):
        """Test that low-coverage tickers are dropped."""
        prices = pd.DataFrame({
            "GOOD": [100, 101, 102, 103, 104],
            "BAD": [100, np.nan, np.nan, np.nan, np.nan],
        }, index=pd.date_range("2022-01-01", periods=5))
        
        returns = Pipeline.prices_to_returns(prices, min_coverage=0.8)
        
        assert "GOOD" in returns.columns
        assert "BAD" not in returns.columns
    
    def test_prices_to_returns_forward_fills(self):
        """Test that small gaps are forward-filled."""
        prices = pd.DataFrame({
            "A": [100, np.nan, 102, 103, 104],  # Single gap
        }, index=pd.date_range("2022-01-01", periods=5))
        
        returns = Pipeline.prices_to_returns(prices, min_coverage=0.5)
        
        # Should have returns for all days after first
        assert len(returns) == 4
        assert not returns.isna().any().any()
    
    def test_synthetic_prices_reproducible(self):
        """Test that synthetic data is reproducible."""
        tickers = ["VOO", "TLT", "GLD"]
        
        prices1 = Pipeline._synthetic_prices(tickers, "2022-02-10", "2022-03-31")
        prices2 = Pipeline._synthetic_prices(tickers, "2022-02-10", "2022-03-31")
        
        pd.testing.assert_frame_equal(prices1, prices2)
    
    def test_synthetic_prices_starts_at_100(self):
        """Test that synthetic prices start at 100 (identification marker)."""
        tickers = ["VOO", "TLT"]
        prices = Pipeline._synthetic_prices(tickers, "2022-02-10", "2022-03-31")
        
        # All prices should start near 100 (within 5% due to first day's return)
        for ticker in tickers:
            assert abs(prices[ticker].iloc[0] - 100) < 5
    
    def test_save_and_load_model(self, fitted_model, temp_data_dir):
        """Test model save and load round-trip."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        
        # Save
        path = Pipeline.save_model(fitted_model, "test", exec_cfg)
        
        assert os.path.exists(path)
        assert path.endswith(".json")
        
        # Load
        loaded = Pipeline.load_model(path)
        
        assert loaded._fitted is True
        pd.testing.assert_frame_equal(loaded.loadings, fitted_model.loadings)
    
    def test_save_model_creates_valid_json(self, fitted_model, temp_data_dir):
        """Test that saved model is valid JSON."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        path = Pipeline.save_model(fitted_model, "test", exec_cfg)
        
        with open(path) as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "created_at" in data["metadata"]
        assert "Vt" in data
        assert "loadings" in data
    
    def test_find_latest_model(self, fitted_model, temp_data_dir):
        """Test finding the latest model file."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        
        # Save multiple models
        path1 = Pipeline.save_model(fitted_model, "test", exec_cfg)
        import time
        time.sleep(0.1)  # Ensure different timestamps
        path2 = Pipeline.save_model(fitted_model, "test", exec_cfg)
        
        # Find latest
        latest = Pipeline.find_latest_model("test", exec_cfg)
        
        assert latest == path2  # Should be the second one
    
    def test_find_latest_model_none_found(self, temp_data_dir):
        """Test that missing model returns None."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        
        result = Pipeline.find_latest_model("nonexistent", exec_cfg)
        
        assert result is None
    
    def test_export_csvs(self, fitted_model, temp_data_dir):
        """Test CSV export functionality."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        
        basis_path, loadings_path = Pipeline.export_csvs(fitted_model, "test", exec_cfg)
        
        assert os.path.exists(basis_path)
        assert os.path.exists(loadings_path)
        
        # Verify CSV contents
        basis_df = pd.read_csv(basis_path, index_col=0)
        assert "Factor_1" in basis_df.columns
        
        loadings_df = pd.read_csv(loadings_path, index_col=0)
        assert "R_squared" in loadings_df.columns
        assert "Residual_Vol" in loadings_df.columns
    
    @patch('shockarb.pipeline.yf.download')
    def test_fetch_prices_with_cache(self, mock_download, temp_data_dir):
        """Test that caching works correctly."""
        # Setup mock
        mock_data = pd.DataFrame({
            "Adj Close": {"AAPL": [150, 151, 152]},
        }, index=pd.date_range("2022-01-01", periods=3))
        mock_data.columns = pd.MultiIndex.from_tuples([("Adj Close", "AAPL")])
        mock_download.return_value = mock_data
        
        cache_path = os.path.join(temp_data_dir, "test_cache.parquet")
        
        # First call should download
        prices1 = Pipeline.fetch_prices(["AAPL"], "2022-01-01", "2022-01-10", cache_path)
        assert mock_download.called
        
        # Reset mock
        mock_download.reset_mock()
        
        # Second call should use cache
        prices2 = Pipeline.fetch_prices(["AAPL"], "2022-01-01", "2022-01-10", cache_path)
        assert not mock_download.called
    
    @patch('shockarb.pipeline.yf.download')
    def test_fetch_prices_fallback_to_synthetic(self, mock_download):
        """Test fallback to synthetic data on download failure."""
        mock_download.return_value = pd.DataFrame()  # Empty = failure
        
        prices = Pipeline.fetch_prices(["VOO", "TLT"], "2022-02-10", "2022-03-31")
        
        # Should return synthetic data
        assert not prices.empty
        assert abs(prices.iloc[0, 0] - 100) < 1  # Starts at 100


class TestLiveDataFetching:
    """Tests for live data fetching utilities."""
    
    @patch('shockarb.pipeline.yf.download')
    def test_pipeline_fetch_live_returns_basic(self, mock_download):
        """Test basic live return fetching."""
        mock_data = pd.DataFrame({
            ("Adj Close", "AAPL"): [150, 151, 152, 153, 154],
            ("Adj Close", "MSFT"): [300, 303, 306, 309, 312],
        }, index=pd.date_range("2022-01-01", periods=5))
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data
        
        returns = Pipeline.fetch_live_returns(["AAPL", "MSFT"])
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == 2
    
    @patch('shockarb.pipeline.yf.download')
    def test_pipeline_fetch_live_returns_empty_response(self, mock_download):
        """Test error handling for empty response."""
        mock_download.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="no data"):
            Pipeline.fetch_live_returns(["AAPL"])


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_workflow_with_synthetic_data(self, temp_data_dir):
        """Test complete workflow: build → save → load → score."""
        # Create minimal universe
        universe = UniverseConfig(
            name="integration_test",
            market_etfs=["VOO", "TLT", "GLD"],
            individual_stocks=["AAPL", "MSFT"],
            n_components=2,
            start_date="2022-02-10",
            end_date="2022-03-31",
        )
        
        exec_cfg = ExecutionConfig(
            data_dir=temp_data_dir,
            use_cache=True,
            log_to_file=False,
        )
        
        # Build (will use synthetic data due to network restrictions)
        with patch('shockarb.pipeline.yf.download', return_value=pd.DataFrame()):
            model = Pipeline.build(universe, exec_cfg)
        
        assert model._fitted is True
        
        # Save
        path = Pipeline.save_model(model, universe.name, exec_cfg)
        assert os.path.exists(path)
        
        # Load
        loaded = Pipeline.load_model(path)
        assert loaded._fitted is True
        
        # Score with synthetic returns
        etf_returns = pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015})
        stock_returns = pd.Series({"AAPL": -0.03, "MSFT": -0.025})
        
        scores = loaded.score(etf_returns, stock_returns)
        
        assert len(scores) == 2
        assert "confidence_delta" in scores.columns
    
    def test_model_consistency_across_save_load(self, fitted_model, temp_data_dir):
        """Test that model produces same results after save/load."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        
        # Get scores before save
        etf_returns = pd.Series({"VOO": -0.02, "VDE": 0.03, "TLT": 0.01, "GLD": 0.02, "ITA": 0.025})
        stock_returns = pd.Series({"V": -0.025, "MSFT": -0.03, "LMT": 0.02, "CVX": 0.035, "UNH": -0.01})
        
        scores_before = fitted_model.score(etf_returns, stock_returns)
        
        # Save and load
        path = Pipeline.save_model(fitted_model, "consistency_test", exec_cfg)
        loaded = Pipeline.load_model(path)
        
        # Get scores after load
        scores_after = loaded.score(etf_returns, stock_returns)
        
        # Should be identical
        pd.testing.assert_frame_equal(scores_before, scores_after)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_factor_model(self, sample_etf_returns, sample_stock_returns):
        """Test model with single factor."""
        model = FactorModel(sample_etf_returns, sample_stock_returns)
        model.fit(n_components=1)
        
        assert model.diagnostics.n_factors == 1
        assert model.loadings.shape[1] == 1
    
    def test_minimum_data_points(self):
        """Test model with minimum viable data."""
        # Need at least n_components + 1 observations
        dates = pd.bdate_range("2022-01-01", periods=5)
        
        etf_returns = pd.DataFrame({
            "A": np.random.randn(5) * 0.01,
            "B": np.random.randn(5) * 0.01,
        }, index=dates)
        
        stock_returns = pd.DataFrame({
            "X": np.random.randn(5) * 0.01,
        }, index=dates)
        
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)  # Should work
        
        assert model._fitted is True
    
    def test_identical_returns(self):
        """Test handling of identical (zero-variance) returns."""
        dates = pd.bdate_range("2022-01-01", periods=10)
        
        etf_returns = pd.DataFrame({
            "A": np.zeros(10),  # Zero variance
            "B": np.random.randn(10) * 0.01,
        }, index=dates)
        
        stock_returns = pd.DataFrame({
            "X": np.random.randn(10) * 0.01,
        }, index=dates)
        
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)
        
        # Should handle gracefully (no NaN in loadings)
        assert not model.loadings.isna().any().any()
    
    def test_scoring_with_extreme_returns(self, fitted_model):
        """Test scoring handles extreme return values."""
        # Very large positive/negative returns
        etf_returns = pd.Series({
            "VOO": -0.20,  # -20%
            "VDE": 0.50,   # +50%
            "TLT": 0.10,
            "GLD": 0.15,
            "ITA": 0.30,
        })
        
        stock_returns = pd.Series({
            "V": -0.30,
            "MSFT": -0.25,
            "LMT": 0.40,
            "CVX": 0.60,
            "UNH": -0.10,
        })
        
        scores = fitted_model.score(etf_returns, stock_returns)
        
        # Should produce valid (non-NaN) results
        assert not scores.isna().any().any()
    
    def test_empty_intersection_in_scoring(self, fitted_model):
        """Test scoring when no tickers overlap."""
        etf_returns = pd.Series({"UNKNOWN1": 0.01})
        stock_returns = pd.Series({"UNKNOWN2": 0.01})
        
        scores = fitted_model.score(etf_returns, stock_returns)
        
        # Should return empty DataFrame
        assert len(scores) == 0


# =============================================================================
# CLI Tests
# =============================================================================

class TestCLI:
    """Tests for command-line interface."""
    
    def test_cli_imports(self):
        """Test that CLI module imports correctly."""
        from shockarb.cli import main, cmd_build, cmd_score, cmd_export, cmd_show
        
        assert callable(main)
        assert callable(cmd_build)
    
    def test_universe_registry(self):
        """Test universe lookup function."""
        from shockarb.cli import get_universe, UNIVERSES
        
        assert "us" in UNIVERSES
        assert "global" in UNIVERSES
        
        us = get_universe("US")  # Case insensitive
        assert us.name == "us"
        
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("nonexistent")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
