"""
Additional tests for regime support in shockarb.cli module.

These tests should be ADDED to the existing tests/test_cli.py file.
They test the regime CLI commands and sticky file behavior.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from shockarb.cli import (
    _get_sticky_regime,
    _resolve_regime,
    _set_sticky_regime,
    cmd_list_regimes,
    cmd_set_regime,
    cmd_show_regime,
)
from shockarb.config import ExecutionConfig
from shockarb.regimes import UKRAINE_SHOCK


class TestStickyRegimeFile:
    """Test sticky regime file read/write operations."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    def test_get_sticky_no_file(self, temp_exec_config):
        """Returns None when sticky file doesn't exist."""
        result = _get_sticky_regime(temp_exec_config)
        assert result is None

    def test_set_and_get_sticky(self, temp_exec_config):
        """Can set and retrieve sticky regime."""
        _set_sticky_regime("ukraine_shock", temp_exec_config)
        result = _get_sticky_regime(temp_exec_config)
        assert result == "ukraine_shock"

    def test_set_sticky_creates_file(self, temp_exec_config):
        """set_sticky creates .shockarb_regime file."""
        _set_sticky_regime("ukraine_shock", temp_exec_config)
        
        sticky_file = os.path.join(temp_exec_config.data_dir, ".shockarb_regime")
        assert os.path.exists(sticky_file)

    def test_set_sticky_overwrites(self, temp_exec_config):
        """Setting sticky regime overwrites previous value."""
        _set_sticky_regime("ukraine_shock", temp_exec_config)
        _set_sticky_regime("gulf_war_recovery", temp_exec_config)
        
        result = _get_sticky_regime(temp_exec_config)
        assert result == "gulf_war_recovery"

    def test_set_sticky_validates_regime(self, temp_exec_config):
        """set_sticky raises ValueError for invalid regime."""
        with pytest.raises(ValueError, match="Unknown regime"):
            _set_sticky_regime("invalid_regime", temp_exec_config)

    def test_sticky_file_format(self, temp_exec_config):
        """Sticky file contains just the regime name."""
        _set_sticky_regime("ukraine_shock", temp_exec_config)
        
        sticky_file = os.path.join(temp_exec_config.data_dir, ".shockarb_regime")
        with open(sticky_file) as f:
            content = f.read()
        
        assert content.strip() == "ukraine_shock"


class TestResolveRegime:
    """Test regime resolution from args and sticky file."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    @pytest.fixture
    def args_with_regime(self):
        """Mock args with --regime flag."""
        args = MagicMock()
        args.regime = "ukraine_shock"
        args.universe = None
        return args

    @pytest.fixture
    def args_with_universe(self):
        """Mock args with --universe flag (legacy)."""
        args = MagicMock()
        args.regime = None
        args.universe = "us"
        return args

    @pytest.fixture
    def args_no_regime(self):
        """Mock args without regime or universe."""
        args = MagicMock()
        args.regime = None
        args.universe = None
        return args

    def test_resolve_from_flag(self, args_with_regime, temp_exec_config):
        """Resolves regime from --regime flag."""
        regime = _resolve_regime(args_with_regime, temp_exec_config, require=True)
        assert regime.name == "ukraine_shock"

    def test_resolve_from_universe_flag(self, args_with_universe, temp_exec_config):
        """Resolves regime from --universe flag (legacy mapping)."""
        regime = _resolve_regime(args_with_universe, temp_exec_config, require=True)
        assert regime.name == "ukraine_shock"

    def test_resolve_from_sticky(self, args_no_regime, temp_exec_config):
        """Resolves regime from sticky file when no flags."""
        _set_sticky_regime("gulf_war_recovery", temp_exec_config)
        regime = _resolve_regime(args_no_regime, temp_exec_config, require=True)
        assert regime.name == "gulf_war_recovery"

    def test_resolve_priority_flag_over_sticky(self, args_with_regime, temp_exec_config):
        """--regime flag takes priority over sticky file."""
        _set_sticky_regime("gulf_war_recovery", temp_exec_config)
        regime = _resolve_regime(args_with_regime, temp_exec_config, require=True)
        assert regime.name == "ukraine_shock"  # From flag, not sticky

    def test_resolve_no_regime_require_true(self, args_no_regime, temp_exec_config):
        """Exits with error when no regime found and require=True."""
        with pytest.raises(SystemExit):
            _resolve_regime(args_no_regime, temp_exec_config, require=True)

    def test_resolve_no_regime_require_false(self, args_no_regime, temp_exec_config):
        """Returns None when no regime found and require=False."""
        regime = _resolve_regime(args_no_regime, temp_exec_config, require=False)
        assert regime is None


class TestSetRegimeCommand:
    """Test set-regime CLI command."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    def test_cmd_set_regime_valid(self, temp_exec_config, capsys):
        """cmd_set_regime sets the sticky regime."""
        args = MagicMock()
        args.regime_name = "ukraine_shock"
        args.data_dir = temp_exec_config.data_dir
        
        cmd_set_regime(args)
        
        # Check output
        captured = capsys.readouterr()
        assert "✅ Active regime set to: ukraine_shock" in captured.out
        
        # Check sticky file
        result = _get_sticky_regime(temp_exec_config)
        assert result == "ukraine_shock"

    def test_cmd_set_regime_invalid(self, temp_exec_config):
        """cmd_set_regime exits with error for invalid regime."""
        args = MagicMock()
        args.regime_name = "invalid_regime"
        args.data_dir = temp_exec_config.data_dir
        
        with pytest.raises(SystemExit):
            cmd_set_regime(args)


class TestShowRegimeCommand:
    """Test show-regime CLI command."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    def test_cmd_show_regime_with_sticky(self, temp_exec_config, capsys):
        """cmd_show_regime displays the sticky regime."""
        _set_sticky_regime("ukraine_shock", temp_exec_config)
        
        args = MagicMock()
        args.data_dir = temp_exec_config.data_dir
        
        cmd_show_regime(args)
        
        captured = capsys.readouterr()
        assert "✅ Current regime: ukraine_shock" in captured.out
        assert "Russia-Ukraine invasion" in captured.out

    def test_cmd_show_regime_no_sticky(self, temp_exec_config):
        """cmd_show_regime exits with error when no sticky regime."""
        args = MagicMock()
        args.data_dir = temp_exec_config.data_dir
        
        with pytest.raises(SystemExit):
            cmd_show_regime(args)


class TestListRegimesCommand:
    """Test list-regimes CLI command."""

    def test_cmd_list_regimes(self, capsys):
        """cmd_list_regimes displays all available regimes."""
        args = MagicMock()
        args.data_dir = None
        
        cmd_list_regimes(args)
        
        captured = capsys.readouterr()
        assert "AVAILABLE REGIMES" in captured.out
        assert "ukraine_shock" in captured.out
        assert "gulf_war_recovery" in captured.out
        assert "liberation_day_recovery" in captured.out


class TestBuildCommandWithRegime:
    """Test build command with regime support."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    @patch("shockarb.cli.pipeline.build")
    @patch("shockarb.cli.pipeline.save_model")
    @patch("shockarb.cli.pipeline.export_csvs")
    def test_build_with_regime_flag(self, mock_export, mock_save, mock_build, temp_exec_config):
        """build command uses --regime flag."""
        from shockarb.cli import cmd_build
        
        args = MagicMock()
        args.regime = "ukraine_shock"
        args.universe = None
        args.data_dir = temp_exec_config.data_dir
        args.no_log = True
        
        mock_model = MagicMock()
        mock_model.diagnostics.n_factors = 3
        mock_model.diagnostics.cumulative_variance = 0.85
        mock_model.diagnostics.n_stocks = 80
        mock_build.return_value = mock_model
        
        cmd_build(args)
        
        # Check that build was called with regime
        assert mock_build.called
        call_args = mock_build.call_args
        assert call_args.kwargs.get("regime") is not None
        assert call_args.kwargs["regime"].name == "ukraine_shock"

    @patch("shockarb.cli.pipeline.build")
    @patch("shockarb.cli.pipeline.save_model")
    @patch("shockarb.cli.pipeline.export_csvs")
    def test_build_with_sticky_regime(self, mock_export, mock_save, mock_build, temp_exec_config):
        """build command uses sticky regime when no flag."""
        from shockarb.cli import cmd_build
        
        _set_sticky_regime("gulf_war_recovery", temp_exec_config)
        
        args = MagicMock()
        args.regime = None
        args.universe = None
        args.data_dir = temp_exec_config.data_dir
        args.no_log = True
        
        mock_model = MagicMock()
        mock_model.diagnostics.n_factors = 4
        mock_model.diagnostics.cumulative_variance = 0.82
        mock_model.diagnostics.n_stocks = 15
        mock_build.return_value = mock_model
        
        cmd_build(args)
        
        # Check that build was called with gulf_war_recovery
        call_args = mock_build.call_args
        assert call_args.kwargs["regime"].name == "gulf_war_recovery"


class TestScoreCommandWithRegime:
    """Test score command with regime support."""

    @pytest.fixture
    def temp_exec_config(self):
        """Execution config with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExecutionConfig(data_dir=tmpdir, log_to_file=False)

    @patch("shockarb.cli.pipeline.score_universe")
    @patch("shockarb.cli.pipeline.find_latest_model")
    @patch("shockarb.cli.pipeline.load_model")
    def test_score_regime_specific_model_search(
        self, mock_load, mock_find, mock_score, temp_exec_config
    ):
        """score command searches for regime-specific models."""
        from shockarb.cli import cmd_score
        import pandas as pd
        
        args = MagicMock()
        args.regime = "ukraine_shock"
        args.universe = None
        args.data_dir = temp_exec_config.data_dir
        args.no_log = True
        args.date = None
        args.model = None
        args.output = None
        args.top = 20
        args.use_prior_close = False
        args.from_open = False
        args.save_tape = False
        
        mock_find.return_value = "/fake/ukraine_shock_us_20220101_120000.json"
        mock_model = MagicMock()
        mock_model.etf_returns.columns = ["VOO"]
        mock_model.stock_returns.columns = ["AAPL"]
        mock_load.return_value = mock_model
        
        mock_scores = pd.DataFrame({"ticker": ["AAPL"], "delta": [-0.05]})
        mock_prov = MagicMock()
        mock_score.return_value = (mock_scores, mock_prov)
        
        with patch("shockarb.cli.print_scores"):
            cmd_score(args)
        
        # Verify find_latest_model was called with regime parameter
        assert mock_find.called
        call_args = mock_find.call_args
        assert call_args.kwargs.get("regime") == "ukraine_shock"
