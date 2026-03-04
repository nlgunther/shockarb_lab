#!/usr/bin/env python3
"""
Additional CLI Tests for ShockArb
=================================

Tests focused on command-line interface functionality.
"""

import os
import sys
import tempfile
import shutil
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from shockarb.cli import (
    main, 
    cmd_build, 
    cmd_score, 
    cmd_export, 
    cmd_show,
    get_universe,
    UNIVERSES,
    _print_scores,
    _fetch_historical,
)
from shockarb.config import ExecutionConfig, US_UNIVERSE
from shockarb.engine import FactorModel
from shockarb.pipeline import Pipeline


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    tmp_dir = tempfile.mkdtemp(prefix="shockarb_cli_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mock_fitted_model():
    """Create a minimal fitted model for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-02-10", "2022-03-31")
    n_days = len(dates)
    
    etf_returns = pd.DataFrame({
        "VOO": np.random.randn(n_days) * 0.01,
        "TLT": np.random.randn(n_days) * 0.01,
        "GLD": np.random.randn(n_days) * 0.01,
    }, index=dates)
    
    stock_returns = pd.DataFrame({
        "AAPL": np.random.randn(n_days) * 0.015,
        "MSFT": np.random.randn(n_days) * 0.015,
    }, index=dates)
    
    model = FactorModel(etf_returns, stock_returns)
    model.fit(n_components=2)
    return model


class TestGetUniverse:
    """Tests for universe lookup."""
    
    def test_get_universe_us(self):
        """Test getting US universe."""
        universe = get_universe("us")
        assert universe.name == "us"
    
    def test_get_universe_global(self):
        """Test getting global universe."""
        universe = get_universe("global")
        assert universe.name == "global"
    
    def test_get_universe_case_insensitive(self):
        """Test case insensitivity."""
        assert get_universe("US").name == "us"
        assert get_universe("Us").name == "us"
        assert get_universe("GLOBAL").name == "global"
    
    def test_get_universe_invalid(self):
        """Test invalid universe raises error."""
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("nonexistent")


class TestPrintScores:
    """Tests for score printing function."""
    
    def test_print_scores_with_actionable(self, capsys):
        """Test printing scores with actionable signals."""
        scores = pd.DataFrame({
            "actual_return": [-0.02, -0.03, -0.01],
            "expected_return": [0.01, 0.02, 0.005],
            "delta": [0.03, 0.05, 0.015],
            "r_squared": [0.6, 0.5, 0.7],
            "residual_vol": [0.15, 0.18, 0.12],
            "confidence_delta": [0.018, 0.025, 0.0105],
        }, index=["AAPL", "MSFT", "GOOGL"])
        
        _print_scores(scores, "TEST", top_n=10)
        
        captured = capsys.readouterr()
        assert "SHOCKARB SCORES" in captured.out
        assert "TEST" in captured.out
        assert "AAPL" in captured.out or "MSFT" in captured.out
    
    def test_print_scores_no_actionable(self, capsys):
        """Test printing when no actionable signals."""
        scores = pd.DataFrame({
            "actual_return": [-0.02],
            "expected_return": [-0.02],
            "delta": [0.0001],  # Too small
            "r_squared": [0.2],  # Too low
            "residual_vol": [0.15],
            "confidence_delta": [0.00002],
        }, index=["AAPL"])
        
        _print_scores(scores, "TEST", top_n=10)
        
        captured = capsys.readouterr()
        assert "No actionable signals" in captured.out
    
    def test_print_scores_with_worst(self, capsys):
        """Test printing bottom performers."""
        scores = pd.DataFrame({
            "actual_return": [0.02, 0.03],
            "expected_return": [-0.01, -0.02],
            "delta": [-0.03, -0.05],
            "r_squared": [0.6, 0.5],
            "residual_vol": [0.15, 0.18],
            "confidence_delta": [-0.018, -0.025],
        }, index=["AAPL", "MSFT"])
        
        _print_scores(scores, "TEST", top_n=10)
        
        captured = capsys.readouterr()
        assert "avoid" in captured.out.lower() or "Bottom" in captured.out


class TestCmdBuild:
    """Tests for build command."""
    
    @patch('shockarb.pipeline.yf.download')
    def test_cmd_build_basic(self, mock_download, temp_data_dir, capsys):
        """Test basic build command."""
        # Mock empty download (will trigger synthetic fallback)
        mock_download.return_value = pd.DataFrame()
        
        class Args:
            universe = "us"
            data_dir = temp_data_dir
            no_cache = False
            no_log = True
        
        # This will use synthetic data
        cmd_build(Args())
        
        # Check that model was saved
        files = os.listdir(temp_data_dir)
        assert any(f.endswith(".json") for f in files)
        
        captured = capsys.readouterr()
        assert "Model saved" in captured.out or "✅" in captured.out


class TestCmdShow:
    """Tests for show command."""
    
    def test_cmd_show_basic(self, mock_fitted_model, temp_data_dir, capsys):
        """Test basic show command."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        Pipeline.save_model(mock_fitted_model, "us", exec_cfg)
        
        class Args:
            universe = "us"
            data_dir = temp_data_dir
            verbose = False
        
        cmd_show(Args())
        
        captured = capsys.readouterr()
        assert "SHOCKARB MODEL" in captured.out
        assert "US" in captured.out
    
    def test_cmd_show_verbose(self, mock_fitted_model, temp_data_dir, capsys):
        """Test verbose show command."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        Pipeline.save_model(mock_fitted_model, "us", exec_cfg)
        
        class Args:
            universe = "us"
            data_dir = temp_data_dir
            verbose = True
        
        cmd_show(Args())
        
        captured = capsys.readouterr()
        assert "Factor Loadings" in captured.out
        assert "R²" in captured.out
    
    def test_cmd_show_no_model(self, temp_data_dir, capsys):
        """Test show command when no model exists."""
        class Args:
            universe = "us"
            data_dir = temp_data_dir
            verbose = False
        
        with pytest.raises(SystemExit):
            cmd_show(Args())


class TestCmdExport:
    """Tests for export command."""
    
    def test_cmd_export_basic(self, mock_fitted_model, temp_data_dir, capsys):
        """Test basic export command."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        Pipeline.save_model(mock_fitted_model, "us", exec_cfg)
        
        class Args:
            universe = "us"
            data_dir = temp_data_dir
        
        cmd_export(Args())
        
        # Check CSVs were created
        files = os.listdir(temp_data_dir)
        assert any("etf_basis.csv" in f for f in files)
        assert any("stock_loadings.csv" in f for f in files)
        
        captured = capsys.readouterr()
        assert "Exported" in captured.out


class TestCmdScore:
    """Tests for score command."""
    
    @patch('shockarb.cli.fetch_live_returns')
    def test_cmd_score_live(self, mock_fetch, mock_fitted_model, temp_data_dir, capsys):
        """Test live scoring command."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        Pipeline.save_model(mock_fitted_model, "us", exec_cfg)
        
        # Mock live returns - note the fixture has VOO, TLT, GLD as ETFs
        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015}),
            pd.Series({"AAPL": -0.03, "MSFT": -0.025}),
        ]
        
        class Args:
            universe = "us"
            data_dir = temp_data_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
        
        # Set the env var for data dir
        os.environ["SHOCK_ARB_DATA_DIR"] = temp_data_dir
        try:
            cmd_score(Args())
        finally:
            del os.environ["SHOCK_ARB_DATA_DIR"]
        
        captured = capsys.readouterr()
        assert "SHOCKARB SCORES" in captured.out
    
    @patch('shockarb.cli.fetch_live_returns')
    def test_cmd_score_with_output(self, mock_fetch, mock_fitted_model, temp_data_dir, capsys):
        """Test scoring with CSV output."""
        exec_cfg = ExecutionConfig(data_dir=temp_data_dir, log_to_file=False)
        Pipeline.save_model(mock_fitted_model, "us", exec_cfg)
        
        output_path = os.path.join(temp_data_dir, "results.csv")
        
        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015}),
            pd.Series({"AAPL": -0.03, "MSFT": -0.025}),
        ]
        
        class Args:
            universe = "us"
            data_dir = temp_data_dir
            date = None
            model = None
            output = output_path
            top = 20
            no_log = True
        
        os.environ["SHOCK_ARB_DATA_DIR"] = temp_data_dir
        try:
            cmd_score(Args())
        finally:
            del os.environ["SHOCK_ARB_DATA_DIR"]
        
        assert os.path.exists(output_path)
        
        # Verify CSV contents
        df = pd.read_csv(output_path, index_col=0)
        assert "confidence_delta" in df.columns


class TestFetchHistorical:
    """Tests for historical data fetching."""
    
    @patch('yfinance.download')
    def test_fetch_historical_basic(self, mock_download):
        """Test basic historical fetch."""
        # Create mock data with proper structure (DataFrame with Close column)
        dates = pd.date_range("2022-02-01", "2022-02-15")
        
        # yf.download returns DataFrame with multi-level columns for multiple tickers
        # For single ticker queries, we return simple DataFrame
        mock_df = pd.DataFrame({
            "Close": [150 + i for i in range(len(dates))],
        }, index=dates)
        
        # The function calls download twice - return DataFrame each time
        mock_download.return_value = mock_df
        
        etf_returns, stock_returns = _fetch_historical(
            ["AAPL"], ["MSFT"], "2022-02-10"
        )
        
        assert isinstance(etf_returns, pd.Series)
        assert isinstance(stock_returns, pd.Series)
    
    @patch('yfinance.download')
    def test_fetch_historical_weekend_snap(self, mock_download):
        """Test that weekend dates snap to Friday."""
        # Create mock data for weekdays only
        dates = pd.bdate_range("2022-02-01", "2022-02-15")
        mock_df = pd.DataFrame({
            "Close": [150 + i for i in range(len(dates))],
        }, index=dates)
        mock_download.return_value = mock_df
        
        # Request a Saturday (2022-02-12) - should snap to Friday
        etf_returns, _ = _fetch_historical(["AAPL"], ["AAPL"], "2022-02-12")
        
        # Should still return data (snapped to Friday)
        assert isinstance(etf_returns, pd.Series)


class TestMainEntrypoint:
    """Tests for main CLI entrypoint."""
    
    def test_main_no_args(self):
        """Test main with no arguments exits with error."""
        with patch('sys.argv', ['shockarb']):
            with pytest.raises(SystemExit):
                main()
    
    def test_main_help(self, capsys):
        """Test main with --help."""
        with patch('sys.argv', ['shockarb', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    def test_main_build_help(self, capsys):
        """Test build subcommand help."""
        with patch('sys.argv', ['shockarb', 'build', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestArgumentParsing:
    """Tests for CLI argument parsing."""
    
    @patch('shockarb.cli.cmd_build')
    def test_build_args_parsed(self, mock_cmd, temp_data_dir):
        """Test that build arguments are parsed correctly."""
        with patch('sys.argv', [
            'shockarb',
            '--data-dir', temp_data_dir,
            'build', 
            '--universe', 'us',
            '--no-cache',
            '--no-log'
        ]):
            main()
        
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert args.universe == 'us'
        assert args.no_cache is True
        assert args.no_log is True
    
    @patch('shockarb.cli.cmd_score')
    def test_score_args_parsed(self, mock_cmd, temp_data_dir):
        """Test that score arguments are parsed correctly."""
        with patch('sys.argv', [
            'shockarb',
            '--data-dir', temp_data_dir,
            'score',
            '--universe', 'us',
            '--date', '2022-03-01',
            '--top', '10'
        ]):
            main()
        
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert args.universe == 'us'
        assert args.date == '2022-03-01'
        assert args.top == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
