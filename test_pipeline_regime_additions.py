"""
Additional tests for regime support in shockarb.pipeline module.

These tests should be ADDED to the existing tests/test_pipeline.py file.
They test the regime parameter flow through build(), save_model(), and find_latest_model().
"""

import glob
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

from shockarb.config import ExecutionConfig, UniverseConfig
from shockarb.regimes import UKRAINE_SHOCK, get_regime


class TestRegimeIntegration:
    """Test regime parameter integration in pipeline functions."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    @pytest.fixture
    def mock_coordinator(self):
        """Mock DataCoordinator that returns realistic price DataFrames."""
        # Create mock PRICES (not returns) - build() calls prices_to_returns
        # Need at least 3 ETFs and 10 time periods to use n_components=2
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        
        # Create realistic prices (around $100-500 with small movements)
        etf_prices = pd.DataFrame(
            100 + np.random.randn(10, 5).cumsum(axis=0),
            index=dates,
            columns=['VOO', 'TLT', 'GLD', 'VDE', 'XLF']
        )
        stock_prices = pd.DataFrame(
            200 + np.random.randn(10, 5).cumsum(axis=0) * 2,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        
        mock = MagicMock()
        mock.register = MagicMock()
        mock.fulfill = MagicMock(return_value={
            "test.etf": etf_prices,
            "test.stock": stock_prices,
        })
        return mock

    def test_build_without_regime(self, temp_exec_config, mock_coordinator):
        """build() works without regime parameter (backward compat)."""
        from shockarb import pipeline
        
        universe = UniverseConfig(
            name="test",
            market_etfs=["VOO", "TLT", "GLD", "VDE", "XLF"],
            individual_stocks=["AAPL", "MSFT"],
            n_components=2,
            start_date="2022-01-01",
            end_date="2022-01-31",
        )
        
        with patch.object(pipeline, "_coordinator", return_value=mock_coordinator):
            # Should not raise - regime is optional
            model = pipeline.build(universe, temp_exec_config, regime=None)
            assert model is not None
            assert model.diagnostics.n_factors == 2

    def test_build_with_regime(self, temp_exec_config):
        """build() accepts regime parameter."""
        from shockarb import pipeline
        
        regime = UKRAINE_SHOCK
        
        # Create a mock coordinator that matches the requester names from regime.universe
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        etf_prices = pd.DataFrame(
            100 + np.random.randn(10, 5).cumsum(axis=0),
            index=dates,
            columns=['VOO', 'TLT', 'GLD', 'VDE', 'XLF']
        )
        stock_prices = pd.DataFrame(
            200 + np.random.randn(10, 5).cumsum(axis=0) * 2,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        
        mock_coord = MagicMock()
        mock_coord.register = MagicMock()
        # Return data keyed by universe.name (which is "us" for UKRAINE_SHOCK)
        mock_coord.fulfill = MagicMock(return_value={
            f"{regime.universe.name}.etf": etf_prices,
            f"{regime.universe.name}.stock": stock_prices,
        })
        
        with patch.object(pipeline, "_coordinator", return_value=mock_coord):
            model = pipeline.build(regime.universe, temp_exec_config, regime=regime)
            assert model is not None
            # Can't predict exact n_factors due to random data, just check model exists
            assert model.diagnostics.n_factors > 0

    def test_save_model_without_regime(self, temp_exec_config):
        """save_model() without regime uses old filename pattern."""
        from shockarb import pipeline
        from shockarb.engine import FactorModel
        
        # Create and fit minimal model with n_components=1
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        etf_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates,
            columns=['VOO', 'TLT', 'GLD']
        )
        stock_returns = pd.DataFrame(
            np.random.randn(10, 2) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT']
        )
        
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)  # n_components must be < min(T=10, N_etf=3)
        
        path = pipeline.save_model(model, "test", temp_exec_config, regime=None)
        
        # Check filename pattern: {name}_{timestamp}.json
        filename = os.path.basename(path)
        assert filename.startswith("test_")
        assert filename.endswith(".json")
        assert "ukraine_shock" not in filename  # No regime prefix

    def test_save_model_with_regime(self, temp_exec_config):
        """save_model() with regime uses new filename pattern."""
        from shockarb import pipeline
        from shockarb.engine import FactorModel
        
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        etf_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates,
            columns=['VOO', 'TLT', 'GLD']
        )
        stock_returns = pd.DataFrame(
            np.random.randn(10, 2) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT']
        )
        
        regime = UKRAINE_SHOCK
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)
        
        path = pipeline.save_model(model, "us", temp_exec_config, regime=regime)
        
        # Check filename pattern: {regime}_{name}_{timestamp}.json
        filename = os.path.basename(path)
        assert filename.startswith("ukraine_shock_us_")
        assert filename.endswith(".json")

    def test_save_model_embeds_regime_metadata(self, temp_exec_config):
        """save_model() embeds regime metadata in JSON."""
        from shockarb import pipeline
        from shockarb.engine import FactorModel
        
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        etf_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates,
            columns=['VOO', 'TLT', 'GLD']
        )
        stock_returns = pd.DataFrame(
            np.random.randn(10, 2) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT']
        )
        
        regime = UKRAINE_SHOCK
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)
        
        path = pipeline.save_model(model, "us", temp_exec_config, regime=regime)
        
        # Load and check metadata
        with open(path) as f:
            payload = json.load(f)
        
        assert "metadata" in payload
        assert payload["metadata"]["regime_name"] == "ukraine_shock"
        assert payload["metadata"]["regime_description"] == regime.description

    def test_find_latest_model_without_regime(self, temp_exec_config):
        """find_latest_model() without regime parameter finds all matching files."""
        from shockarb import pipeline
        from shockarb.engine import FactorModel
        import time
        
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        etf_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates,
            columns=['VOO', 'TLT', 'GLD']
        )
        stock_returns = pd.DataFrame(
            np.random.randn(10, 2) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT']
        )
        
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)
        
        # Create three model files
        path1 = pipeline.save_model(model, "us", temp_exec_config, regime=None)
        time.sleep(0.01)  # Ensure different timestamps
        path2 = pipeline.save_model(model, "us", temp_exec_config, regime=UKRAINE_SHOCK)
        time.sleep(0.01)
        path3 = pipeline.save_model(model, "us", temp_exec_config, regime=None)
        
        # Without regime, should find the latest (path3)
        latest = pipeline.find_latest_model("us", temp_exec_config, regime=None)
        assert latest == path3

    def test_find_latest_model_with_regime(self, temp_exec_config):
        """find_latest_model() with regime finds regime-specific files."""
        from shockarb import pipeline
        from shockarb.engine import FactorModel
        import time
        
        dates = pd.date_range('2022-01-01', periods=10, freq='D')
        etf_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates,
            columns=['VOO', 'TLT', 'GLD']
        )
        stock_returns = pd.DataFrame(
            np.random.randn(10, 2) * 0.02,
            index=dates,
            columns=['AAPL', 'MSFT']
        )
        
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=1)
        
        gulf_war = get_regime("gulf_war_recovery")
        
        # Create models for different regimes
        path_ukraine = pipeline.save_model(model, "us", temp_exec_config, regime=UKRAINE_SHOCK)
        time.sleep(0.01)
        path_gulf = pipeline.save_model(model, "us_recovery", temp_exec_config, regime=gulf_war)
        
        # Search with regime should find only that regime's models
        latest_ukraine = pipeline.find_latest_model("us", temp_exec_config, regime="ukraine_shock")
        assert latest_ukraine == path_ukraine
        
        latest_gulf = pipeline.find_latest_model("us_recovery", temp_exec_config, regime="gulf_war_recovery")
        assert latest_gulf == path_gulf

    def test_find_latest_model_no_matches(self, temp_exec_config):
        """find_latest_model() returns None when no files match."""
        from shockarb import pipeline
        
        result = pipeline.find_latest_model("nonexistent", temp_exec_config)
        assert result is None

    def test_backward_compatibility_old_files(self, temp_exec_config):
        """Old model files (without regime prefix) are still found."""
        from shockarb import pipeline
        
        # Create an old-style file manually
        old_filename = "us_20220101_120000.json"
        old_path = os.path.join(temp_exec_config.data_dir, old_filename)
        
        with open(old_path, "w") as f:
            json.dump({"metadata": {}, "diagnostics": {}}, f)
        
        # Should find it when regime=None
        found = pipeline.find_latest_model("us", temp_exec_config, regime=None)
        assert found == old_path


class TestRegimeFilenamePatterns:
    """Test filename pattern generation and parsing."""

    def test_regime_prefix_in_filename(self):
        """Regime name is correctly prefixed to filename."""
        from shockarb import pipeline
        import tempfile
        from shockarb.config import ExecutionConfig
        from shockarb.engine import FactorModel
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_cfg = ExecutionConfig(data_dir=tmpdir, log_to_file=False)
            
            dates = pd.date_range('2022-01-01', periods=10, freq='D')
            etf_returns = pd.DataFrame(
                np.random.randn(10, 3) * 0.01,
                index=dates,
                columns=['VOO', 'TLT', 'GLD']
            )
            stock_returns = pd.DataFrame(
                np.random.randn(10, 2) * 0.02,
                index=dates,
                columns=['AAPL', 'MSFT']
            )
            
            model = FactorModel(etf_returns, stock_returns)
            model.fit(n_components=1)
            
            regime = get_regime("liberation_day_recovery")
            path = pipeline.save_model(model, "us_lib_day", exec_cfg, regime=regime)
            
            filename = os.path.basename(path)
            assert filename.startswith("liberation_day_recovery_us_lib_day_")

    def test_glob_pattern_specificity(self):
        """Glob patterns correctly isolate regime-specific files."""
        import tempfile
        from shockarb.config import ExecutionConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_cfg = ExecutionConfig(data_dir=tmpdir, log_to_file=False)
            
            # Create files for different regimes
            open(os.path.join(tmpdir, "ukraine_shock_us_20220101_120000.json"), "w").close()
            open(os.path.join(tmpdir, "gulf_war_recovery_us_recovery_20220102_120000.json"), "w").close()
            open(os.path.join(tmpdir, "us_20220103_120000.json"), "w").close()  # Old style
            
            # Pattern for ukraine_shock should match only that regime
            ukraine_pattern = os.path.join(tmpdir, "ukraine_shock_us_*.json")
            ukraine_files = glob.glob(ukraine_pattern)
            assert len(ukraine_files) == 1
            assert "ukraine_shock_us_" in ukraine_files[0]
            
            # Pattern for all us models (regime=None) should match all
            all_pattern = os.path.join(tmpdir, "*us_*.json")
            all_files = glob.glob(all_pattern)
            assert len(all_files) >= 2  # ukraine_shock and old style
