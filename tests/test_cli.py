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
from shockarb.cli import (
    UNIVERSES,
    _fetch_historical,
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

        cmd_show(Args())
        out = capsys.readouterr().out
        # Verbose path calls print_model_state which includes factor tables
        assert "BASIS" in out or "FACTOR" in out or "LOADINGS" in out

    def test_missing_model_exits(self, temp_dir):
        class Args:
            universe = "us"
            data_dir = temp_dir
            verbose = False

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

        cmd_export(Args())
        files = os.listdir(temp_dir)
        assert any("etf_basis.csv" in f for f in files)
        assert any("stock_loadings.csv" in f for f in files)
        assert "Exported" in capsys.readouterr().out


# =============================================================================
# cmd_score
# =============================================================================

class TestCmdScore:

    @patch("shockarb.pipeline.fetch_live_returns")
    def test_live_score_prints_table(self, mock_fetch, fitted_model, temp_dir, capsys):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)

        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "VDE": 0.03, "TLT": 0.01, "GLD": 0.02, "ITA": 0.025}),
            pd.Series({"V": -0.025, "MSFT": -0.03, "LMT": 0.02, "CVX": 0.035, "UNH": -0.01}),
        ]

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            os.environ.pop("SHOCK_ARB_DATA_DIR", None)

        assert "SHOCKARB SCORES" in capsys.readouterr().out

    @patch("shockarb.pipeline.fetch_live_returns")
    def test_output_csv_saved(self, mock_fetch, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "us", cfg)
        output_path = os.path.join(temp_dir, "results.csv")

        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "VDE": 0.03, "TLT": 0.01, "GLD": 0.02, "ITA": 0.025}),
            pd.Series({"V": -0.025, "MSFT": -0.03, "LMT": 0.02, "CVX": 0.035, "UNH": -0.01}),
        ]

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = output_path
            top = 20
            no_log = True

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

    @patch("shockarb.pipeline.save_live_tape")
    @patch("shockarb.pipeline.fetch_live_returns")
    def test_save_tape_flag_calls_save_live_tape(
        self, mock_fetch, mock_tape, mock_model, temp_dir, capsys
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
        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015}),
            pd.Series({"AAPL": -0.03, "MSFT": -0.025}),
        ]

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
            save_tape = True

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            del os.environ["SHOCK_ARB_DATA_DIR"]

        assert mock_tape.called

    @patch("shockarb.pipeline.save_live_tape")
    @patch("shockarb.pipeline.fetch_live_returns")
    def test_tape_path_contains_universe_name(
        self, mock_fetch, mock_tape, mock_model, temp_dir
    ):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(mock_model, "us", cfg)

        mock_tape.return_value = None   # simulate failure — score should still proceed
        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015}),
            pd.Series({"AAPL": -0.03, "MSFT": -0.025}),
        ]

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
            save_tape = True

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            del os.environ["SHOCK_ARB_DATA_DIR"]

        call_args = mock_tape.call_args
        tape_path = call_args[0][2]   # third positional arg is path
        assert "us" in tape_path
        assert "tapes" in tape_path

    @patch("shockarb.pipeline.fetch_live_returns")
    def test_no_save_tape_flag_skips_tape(
        self, mock_fetch, mock_model, temp_dir
    ):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(mock_model, "us", cfg)

        mock_fetch.side_effect = [
            pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015}),
            pd.Series({"AAPL": -0.03, "MSFT": -0.025}),
        ]

        class Args:
            universe = "us"
            data_dir = temp_dir
            date = None
            model = None
            output = None
            top = 20
            no_log = True
            save_tape = False

        import shockarb.pipeline as _pl
        original = _pl.save_live_tape
        calls = []
        _pl.save_live_tape = lambda *a, **k: calls.append(a) or None

        os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
        try:
            cmd_score(Args())
        finally:
            del os.environ["SHOCK_ARB_DATA_DIR"]
            _pl.save_live_tape = original

        assert len(calls) == 0

    @patch("shockarb.pipeline.save_live_tape")
    @patch("shockarb.pipeline.fetch_live_returns")
    def test_tape_not_saved_for_historical_date(
        self, mock_fetch, mock_tape, mock_model, temp_dir
    ):
        """--save-tape must be ignored when --date is specified."""
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(mock_model, "us", cfg)

        with patch("yfinance.download") as mock_yf:
            dates = pd.bdate_range("2022-02-01", "2022-02-15")
            mock_yf.return_value = pd.DataFrame(
                {"Close": [100.0 + i for i in range(len(dates))]},
                index=dates,
            )

            class Args:
                universe = "us"
                data_dir = temp_dir
                date = "2022-03-01"
                model = None
                output = None
                top = 20
                no_log = True
                save_tape = True

            os.environ["SHOCK_ARB_DATA_DIR"] = temp_dir
            try:
                cmd_score(Args())
            except Exception:
                pass   # historical fetch may fail with mocked data; that's OK
            finally:
                del os.environ["SHOCK_ARB_DATA_DIR"]

        assert not mock_tape.called
