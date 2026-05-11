"""
Integration tests for shockarb.cli — command parsing and command execution.

These tests exercise the CLI layer against real (mocked) pipeline calls.
All yfinance calls are intercepted; filesystem writes go to temp_dir.

Coverage areas
--------------
  TestGetUniverse       — registry lookup, case-insensitivity, error handling
  TestPrintScores       — report module output: correct columns, thresholds
  TestCmdBuild          — end-to-end build command creates JSON and prints success
  TestCmdShow           — show command: compact and verbose modes, missing model
  TestCmdExport         — export command creates CSV files
  TestCmdScore          — live score and historical score commands
  TestFetchHistorical   — date snapping for weekends / holidays
  TestMain              — argparse wiring, --help, subcommand dispatch
"""

from __future__ import annotations

# from conftest import InMemoryStore
class InMemoryStore:
    """Shared test double for DataStore."""
    def __init__(self):
        self._data = {}

    def write(self, key, df, meta):
        ticker = key.split("/")[-1]
        if "adj_close" in df.columns:
            self._data[key] = df[["adj_close"]]
        elif ticker in df.columns:
            self._data[key] = df[[ticker]].rename(columns={ticker: "adj_close"})
        else:
            for col in df.columns:
                self._data[f"daily/{col}"] = df[[col]].rename(columns={col: "adj_close"})

    def read(self, key, start, end):
        df = self._data.get(key)
        if df is None: return None
        try: return df.loc[start:end]
        except Exception: return df

    def coverage(self, key):
        df = self._data.get(key)
        if df is None or df.empty: return None
        return (str(df.index.min().date()), str(df.index.max().date()))

    def sweep(self, retention, before):
        return []

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import shockarb.pipeline as pipeline
from shockarb.pipeline import ScoreProvenance
from shockarb.cli import (
    UNIVERSES,
    _fetch_historical,
    _resolve_regime,
    cmd_build,
    cmd_export,
    cmd_score,
    cmd_show,
    get_universe,
    main,
)
from shockarb.config import ExecutionConfig
from shockarb.report import print_scores


# =============================================================================
# Universe registry
# =============================================================================

class TestGetUniverse:

    def test_us_lookup(self):
        assert get_universe("us").name == "us"

    def test_global_lookup(self):
        assert get_universe("global").name == "global"

    def test_case_insensitive(self):
        assert get_universe("US").name == "us"
        assert get_universe("GLOBAL").name == "global"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("nonexistent")

    def test_registry_has_expected_keys(self):
        assert "us" in UNIVERSES
        assert "global" in UNIVERSES


# =============================================================================
# _resolve_regime — universe flag mapping and deprecation warning
# =============================================================================

class TestResolveRegime:
    """Test that --universe flag maps correctly and emits deprecation warnings."""

    def _make_args(self, regime=None, universe=None):
        import argparse
        return argparse.Namespace(regime=regime, universe=universe)

    def _exec_config(self, tmp_path):
        return ExecutionConfig(data_dir=str(tmp_path))

    def test_universe_global_maps_to_global_ukraine_shock(self, tmp_path):
        """--universe global must resolve to global_ukraine_shock (bug fix)."""
        args = self._make_args(universe="global")
        regime = _resolve_regime(args, self._exec_config(tmp_path))
        assert regime.name == "global_ukraine_shock"

    def test_universe_us_maps_to_ukraine_shock(self, tmp_path):
        """--universe us still maps to ukraine_shock."""
        args = self._make_args(universe="us")
        regime = _resolve_regime(args, self._exec_config(tmp_path))
        assert regime.name == "ukraine_shock"

    def test_universe_global_emits_deprecation_warning(self, tmp_path):
        """--universe flag must call logger.warning with 'DEPRECATED'."""
        args = self._make_args(universe="global")
        with patch("shockarb.cli.logger") as mock_logger:
            _resolve_regime(args, self._exec_config(tmp_path))
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("DEPRECATED" in c for c in warning_calls)

    def test_universe_us_emits_deprecation_warning(self, tmp_path):
        """--universe us also calls logger.warning with 'DEPRECATED'."""
        args = self._make_args(universe="us")
        with patch("shockarb.cli.logger") as mock_logger:
            _resolve_regime(args, self._exec_config(tmp_path))
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("DEPRECATED" in c for c in warning_calls)

    def test_regime_flag_takes_priority_over_universe(self, tmp_path):
        """--regime flag overrides --universe when both are provided."""
        args = self._make_args(regime="ukraine_shock", universe="global")
        regime = _resolve_regime(args, self._exec_config(tmp_path))
        assert regime.name == "ukraine_shock"

    def test_regime_flag_no_warning(self, tmp_path):
        """Using --regime directly produces no deprecation warning."""
        args = self._make_args(regime="ukraine_shock")
        with patch("shockarb.cli.logger") as mock_logger:
            _resolve_regime(args, self._exec_config(tmp_path))
        mock_logger.warning.assert_not_called()


# =============================================================================
# print_scores (report module column validation)
# =============================================================================

class TestPrintScores:

    def _scores(self, n=3, conf=0.02):
        return pd.DataFrame(
            {
                "actual_return":    [-0.02] * n,
                "expected_rel":     [0.01] * n,
                "expected_abs":     [0.01] * n,
                "delta_rel":        [0.03] * n,
                "delta_abs":        [0.03] * n,
                "r_squared":        [0.60] * n,
                "residual_vol":     [0.15] * n,
                "confidence_delta": [conf] * n,
            },
            index=[f"T{i}" for i in range(n)],
        )

    def test_header_printed(self, capsys):
        print_scores(self._scores(), "TEST")
        assert "SHOCKARB SCORES" in capsys.readouterr().out

    def test_tickers_displayed(self, capsys):
        print_scores(self._scores(), "TEST")
        assert "T0" in capsys.readouterr().out

    def test_no_actionable_signals_message(self, capsys):
        print_scores(self._scores(conf=0.00001), "TEST", min_confidence=0.001)
        assert "No actionable signals" in capsys.readouterr().out

    def test_bottom_signals_section_shown(self, capsys):
        print_scores(self._scores(conf=-0.02), "TEST")
        out = capsys.readouterr().out
        assert "Bottom" in out or "avoid" in out.lower()


# =============================================================================
# cmd_build
# =============================================================================


def _mock_score_universe_return():
    """Return a (scores, prov) tuple suitable for patching score_universe.

    scores must have all columns that print_scores / report.py reads:
    actual_return, expected_rel, expected_abs, delta_rel, delta_abs,
    r_squared, residual_vol, confidence_delta.
    """
    import pandas as pd
    scores = pd.DataFrame(
        {
            "actual_return":    [-0.025, -0.030],
            "expected_rel":     [ 0.010,  0.005],
            "expected_abs":     [ 0.011,  0.006],
            "delta_rel":        [ 0.035,  0.035],
            "delta_abs":        [ 0.036,  0.036],
            "r_squared":        [ 0.800,  0.700],
            "residual_vol":     [ 0.200,  0.250],
            "confidence_delta": [ 0.050, -0.030],
        },
        index=["AAPL", "MSFT"],
    )
    prov = ScoreProvenance(universe="us", provider="yfinance", n_etfs=5, n_stocks=5)
    prov.path = "daily"
    prov.return_formula = "adj_close / prev_adj_close - 1"
    return scores, prov

class TestCmdBuild:

    def test_creates_json_and_prints_success(self, temp_dir, capsys):
        from datamgr.coordinator import DataCoordinator
        from datamgr.providers.mock import MockProvider

        def _fresh_coord(_exec_cfg=None):
            return DataCoordinator(InMemoryStore(), provider=MockProvider())

        class Args:
            universe = "us"
            data_dir = temp_dir
            no_log   = True
            use_prior_close = False
            from_open = False

        with patch.object(pipeline, "_coordinator", side_effect=_fresh_coord):
            cmd_build(Args())

        assert any(f.endswith(".json") for f in os.listdir(temp_dir))
        assert "✅" in capsys.readouterr().out


# =============================================================================
# cmd_show
# =============================================================================

class TestCmdShow:

    def test_compact_output(self, fitted_model, temp_dir, capsys):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)

        class Args:
            universe = "us"
            data_dir = temp_dir
            verbose = False
            use_prior_close = False
            from_open = False

        cmd_show(Args())
        out = capsys.readouterr().out
        assert "SHOCKARB MODEL" in out
        assert "US" in out

    def test_verbose_output(self, fitted_model, temp_dir, capsys):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)

        class Args:
            universe = "us"
            data_dir = temp_dir
            verbose = True
            use_prior_close = False
            from_open = False

        cmd_show(Args())
        out = capsys.readouterr().out
        # Verbose path calls print_model_state which includes factor tables
        assert "BASIS" in out or "FACTOR" in out or "LOADINGS" in out

    def test_missing_model_exits(self, temp_dir):
        class Args:
            universe = "us"
            data_dir = temp_dir
            verbose = False
            use_prior_close = False
            from_open = False

        with pytest.raises(SystemExit):
            cmd_show(Args())


# =============================================================================
# cmd_export
# =============================================================================

class TestCmdExport:

    def test_creates_csv_files(self, fitted_model, temp_dir, capsys):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)

        class Args:
            universe = "us"
            data_dir = temp_dir
            use_prior_close = False
            from_open = False

        cmd_export(Args())
        files = os.listdir(temp_dir)
        assert any("etf_basis.csv" in f for f in files)
        assert any("stock_loadings.csv" in f for f in files)
        assert "Exported" in capsys.readouterr().out


# =============================================================================
# cmd_score
# =============================================================================

class TestCmdScore:

    @patch("shockarb.pipeline.score_universe", return_value=_mock_score_universe_return())
    def test_live_score_prints_table(self, mock_score, fitted_model, temp_dir, capsys):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
            use_prior_close = False
            from_open = False

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            os.environ.pop("SHOCK_ARB_DATA_DIR", None)

        assert "SHOCKARB SCORES" in capsys.readouterr().out

    @patch("shockarb.pipeline.score_universe", return_value=_mock_score_universe_return())
    def test_output_csv_saved(self, mock_score, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)
        output_path = os.path.join(temp_dir, "results.csv")

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = output_path
            top = 20
            no_log = True
            use_prior_close = False
            from_open = False

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            os.environ.pop("SHOCK_ARB_DATA_DIR", None)

        df = pd.read_csv(output_path, index_col=0)
        assert "confidence_delta" in df.columns


# =============================================================================
# _fetch_historical
# =============================================================================

class TestFetchHistorical:

    @patch("yfinance.download")
    def test_returns_two_series(self, mock_dl):
        dates = pd.bdate_range("2022-02-01", "2022-02-15")
        mock_dl.return_value = pd.DataFrame(
            {"Close": [150.0 + i for i in range(len(dates))]}, index=dates
        )
        etf_ret, stk_ret = _fetch_historical(["AAPL"], ["MSFT"], "2022-02-10")
        assert isinstance(etf_ret, pd.Series)
        assert isinstance(stk_ret, pd.Series)

    @patch("yfinance.download")
    def test_weekend_snaps_to_nearest_weekday(self, mock_dl):
        dates = pd.bdate_range("2022-02-01", "2022-02-15")
        mock_dl.return_value = pd.DataFrame(
            {"Close": [150.0 + i for i in range(len(dates))]}, index=dates
        )
        # 2022-02-12 is a Saturday
        etf_ret, _ = _fetch_historical(["AAPL"], ["AAPL"], "2022-02-12")
        assert isinstance(etf_ret, pd.Series)


# =============================================================================
# main() — argparse wiring
# =============================================================================

class TestMain:

    def test_no_args_exits_nonzero(self):
        with patch("sys.argv", ["shockarb"]):
            with pytest.raises(SystemExit):
                main()

    def test_help_exits_zero(self):
        with patch("sys.argv", ["shockarb", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    @patch("shockarb.cli.cmd_build")
    def test_build_subcommand_parsed(self, mock_cmd, temp_dir):
        with patch("sys.argv", [
            "shockarb", "--data-dir", temp_dir,
            "build", "--universe", "us", "--no-log",
        ]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.universe == "us"
        assert args.no_log is True

    @patch("shockarb.cli.cmd_score")
    def test_score_subcommand_parsed(self, mock_cmd, temp_dir):
        with patch("sys.argv", [
            "shockarb", "--data-dir", temp_dir,
            "score", "--universe", "us", "--date", "2022-03-01", "--top", "10",
        ]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.date == "2022-03-01"
        assert args.top == 10


# =============================================================================
# --save-tape flag in cmd_score
# =============================================================================

class TestCmdScoreSaveTape:

    @patch("shockarb.pipeline.score_universe", return_value=_mock_score_universe_return())
    @patch("shockarb.pipeline.save_live_tape")
    def test_save_tape_flag_calls_save_live_tape(
        self, mock_tape, mock_score, mock_model, temp_dir, capsys
    ):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(mock_model, "us", cfg)

        mock_tape.return_value = pd.DataFrame(
            {("Close", "VOO"): [100.0, 101.0]},
            index=pd.bdate_range("2022-03-14", periods=2),
        )
        mock_tape.return_value.columns = pd.MultiIndex.from_tuples(
            mock_tape.return_value.columns
        )

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
            save_tape = True
            use_prior_close = False
            from_open = False

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            del os.environ["SHOCK_ARB_DATA_DIR"]

        assert mock_tape.called

    @patch("shockarb.pipeline.score_universe", return_value=_mock_score_universe_return())
    @patch("shockarb.pipeline.save_live_tape")
    def test_tape_path_contains_universe_name(
        self, mock_tape, mock_score, mock_model, temp_dir
    ):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(mock_model, "us", cfg)

        mock_tape.return_value = None   # simulate failure — score should still proceed

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
            save_tape = True
  

class TestCmdAddAsset:
    """Tests for the add-asset subcommand."""

    def test_add_asset_adds_ticker_to_model(self, mock_model, temp_dir):
        """add-asset should call pipeline.add_assets() and print the summary."""
        import numpy as np, pandas as pd
        from unittest.mock import patch, MagicMock
        from shockarb.config import ExecutionConfig
        from shockarb.regimes import get_regime
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        # Patch add_assets to return a synthetic summary
        summary_df = pd.DataFrame(
            {"Factor_1": [0.5], "Factor_2": [0.3],
             "r_squared": [0.72], "residual_vol": [0.18]},
            index=pd.Index(["SHOP"], name="ticker"),
        )
        with patch.object(pipeline, "add_assets", return_value=summary_df) as mock_add, \
             patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch("sys.exit") as mock_exit:
            from shockarb.cli import cmd_add_asset
            import argparse
            args = argparse.Namespace(
                tickers=["SHOP"],
                regime="ukraine_shock",
                model=None,
                save=False,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_add_asset(args)

        mock_add.assert_called_once()
        called_tickers = mock_add.call_args[0][0]
        assert "SHOP" in called_tickers

    def test_add_asset_save_flag_calls_save_model(self, mock_model, temp_dir):
        """--save should call pipeline.save_model() after adding assets."""
        import numpy as np, pandas as pd
        from unittest.mock import patch, MagicMock
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        summary_df = pd.DataFrame(
            {"Factor_1": [0.5], "r_squared": [0.72], "residual_vol": [0.18]},
            index=pd.Index(["SHOP"], name="ticker"),
        )
        with patch.object(pipeline, "add_assets", return_value=summary_df), \
             patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch.object(pipeline, "save_model", return_value=f"{temp_dir}/saved.json") as mock_save, \
             patch.object(pipeline, "export_csvs"):
            from shockarb.cli import cmd_add_asset
            import argparse
            args = argparse.Namespace(
                tickers=["SHOP"],
                regime="ukraine_shock",
                model=None,
                save=True,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_add_asset(args)

        mock_save.assert_called_once()

    def test_add_asset_parser_registered(self):
        """add-asset subcommand must be registered in the argument parser."""
        from shockarb.cli import _build_parser
        parser = _build_parser()
        # argparse registers subcommands; exercise it
        args = parser.parse_args(
            ["add-asset", "SHOP", "--regime", "ukraine_shock", "--save"]
        )
        assert args.tickers == ["SHOP"]
        assert args.regime == "ukraine_shock"
        assert args.save is True


class TestRemoveAssetCLI:
    """Tests for the remove-asset subcommand."""

    def test_remove_asset_removes_ticker(self, mock_model, temp_dir):
        """remove-asset should call model.remove_asset() for each requested ticker."""
        from unittest.mock import patch
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime, cmd_remove_asset
        import argparse

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        # Pick a ticker that actually exists in mock_model
        ticker = mock_model.loadings.index[0]

        with patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch("sys.exit"):
            args = argparse.Namespace(
                tickers=[ticker],
                regime="ukraine_shock",
                model=None,
                save=False,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_remove_asset(args)

        assert ticker not in mock_model.loadings.index

    def test_remove_asset_save_flag_calls_save_model(self, mock_model, temp_dir):
        """--save should call pipeline.save_model() after removing assets."""
        from unittest.mock import patch
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime, cmd_remove_asset
        import argparse

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        ticker = mock_model.loadings.index[0]

        with patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch.object(pipeline, "save_model", return_value=f"{temp_dir}/saved.json") as mock_save, \
             patch.object(pipeline, "export_csvs"):
            args = argparse.Namespace(
                tickers=[ticker],
                regime="ukraine_shock",
                model=None,
                save=True,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_remove_asset(args)

        mock_save.assert_called_once()

    def test_remove_asset_missing_ticker_skipped(self, mock_model, temp_dir, capsys):
        """Tickers not in the model should be skipped with a warning, not raise."""
        from unittest.mock import patch
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime, cmd_remove_asset
        import argparse

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        with patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch("sys.exit"):
            args = argparse.Namespace(
                tickers=["DOESNOTEXIST"],
                regime="ukraine_shock",
                model=None,
                save=False,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_remove_asset(args)

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower() or "nothing" in captured.out.lower()

    def test_remove_asset_parser_registered(self):
        """remove-asset subcommand must be registered in the argument parser."""
        from shockarb.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["remove-asset", "RTX", "--regime", "global_ukraine_shock", "--save"]
        )
        assert args.tickers == ["RTX"]
        assert args.regime == "global_ukraine_shock"
        assert args.save is True


class TestRemoveAssetCLI:
    """Tests for the remove-asset subcommand."""

    def test_remove_asset_removes_ticker(self, mock_model, temp_dir):
        """remove-asset should call model.remove_asset() for each requested ticker."""
        from unittest.mock import patch
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime, cmd_remove_asset
        import argparse

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        ticker = mock_model.loadings.index[0]

        with patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch("sys.exit"):
            args = argparse.Namespace(
                tickers=[ticker],
                regime="ukraine_shock",
                model=None,
                save=False,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_remove_asset(args)

        assert ticker not in mock_model.loadings.index

    def test_remove_asset_save_flag_calls_save_model(self, mock_model, temp_dir):
        """--save should call pipeline.save_model() after removing assets."""
        from unittest.mock import patch
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime, cmd_remove_asset
        import argparse

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        ticker = mock_model.loadings.index[0]

        with patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch.object(pipeline, "save_model", return_value=f"{temp_dir}/saved.json") as mock_save, \
             patch.object(pipeline, "export_csvs"):
            args = argparse.Namespace(
                tickers=[ticker],
                regime="ukraine_shock",
                model=None,
                save=True,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_remove_asset(args)

        mock_save.assert_called_once()

    def test_remove_asset_missing_ticker_skipped(self, mock_model, temp_dir, capsys):
        """Tickers not in the model should be skipped with a warning, not raise."""
        from unittest.mock import patch
        from shockarb.config import ExecutionConfig
        import shockarb.pipeline as pipeline
        from shockarb.cli import _set_sticky_regime, cmd_remove_asset
        import argparse

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _set_sticky_regime("ukraine_shock", cfg)

        with patch.object(pipeline, "load_model", return_value=mock_model), \
             patch.object(pipeline, "find_latest_model", return_value=f"{temp_dir}/us_fake.json"), \
             patch("sys.exit"):
            args = argparse.Namespace(
                tickers=["DOESNOTEXIST"],
                regime="ukraine_shock",
                model=None,
                save=False,
                no_log=True,
                data_dir=temp_dir,
            )
            cmd_remove_asset(args)

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower() or "nothing" in captured.out.lower()

    def test_remove_asset_parser_registered(self):
        """remove-asset subcommand must be registered in the argument parser."""
        from shockarb.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["remove-asset", "RTX", "--regime", "global_ukraine_shock", "--save"]
        )
        assert args.tickers == ["RTX"]
        assert args.regime == "global_ukraine_shock"
        assert args.save is True
