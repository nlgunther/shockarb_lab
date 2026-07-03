"""
Tests for price_trend.py DataCoordinator integration.

All tests inject a mock coordinator so no network calls or parquet I/O occur.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from price_trend import run


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_coordinator(tickers: list[str], n_rows: int = 5) -> MagicMock:
    """
    Return a mock DataCoordinator whose fulfill() returns a plausible adj-close
    DataFrame keyed by 'price_trend'.
    """
    idx = pd.bdate_range("2026-06-01", periods=n_rows)
    data = {t: [100.0 + i for i in range(n_rows)] for t in tickers}
    closes = pd.DataFrame(data, index=idx)

    mock = MagicMock()
    mock.fulfill.return_value = {"price_trend": closes}
    return mock


# =============================================================================
# Coordinator wiring
# =============================================================================

class TestCoordinatorWiring:
    """price_trend.run() registers the right DataRequest and calls fulfill()."""

    def test_register_called_with_correct_tickers(self):
        mock_coord = _make_mock_coordinator(["MSFT", "BLK"])
        run(["MSFT", "BLK"], days=3, save_csv=False, save_daily=False,
            coordinator=mock_coord)
        # register() must have been called once
        assert mock_coord.register.call_count == 1
        req = mock_coord.register.call_args[0][0]
        assert set(req.tickers) == {"MSFT", "BLK"}

    def test_fulfill_called_once(self):
        mock_coord = _make_mock_coordinator(["MSFT"])
        run(["MSFT"], days=3, save_csv=False, save_daily=False,
            coordinator=mock_coord)
        mock_coord.fulfill.assert_called_once()

    def test_requester_label_is_price_trend(self):
        mock_coord = _make_mock_coordinator(["VOO"])
        run(["VOO"], days=3, save_csv=False, save_daily=False,
            coordinator=mock_coord)
        req = mock_coord.register.call_args[0][0]
        assert req.requester == "price_trend"

    def test_frequency_is_daily(self):
        from datamgr.requests import Frequency
        mock_coord = _make_mock_coordinator(["TLT"])
        run(["TLT"], days=3, save_csv=False, save_daily=False,
            coordinator=mock_coord)
        req = mock_coord.register.call_args[0][0]
        assert req.frequency == Frequency.DAILY


# =============================================================================
# Output shape
# =============================================================================

class TestOutputShape:
    """The closes matrix passed to display is trimmed to `days` rows."""

    def test_window_trimmed_to_days(self, capsys):
        # Provide 10 rows; request days=3 → only 3 rows used
        mock_coord = _make_mock_coordinator(["MSFT"], n_rows=10)
        run(["MSFT"], days=3, save_csv=False, save_daily=False,
            coordinator=mock_coord)
        out = capsys.readouterr().out
        # Header says "3 sessions"
        assert "3 sessions" in out

    def test_empty_result_exits(self):
        mock_coord = MagicMock()
        mock_coord.fulfill.return_value = {"price_trend": pd.DataFrame()}
        with pytest.raises(SystemExit):
            run(["ZZZZZ"], days=5, save_csv=False, save_daily=False,
                coordinator=mock_coord)


# =============================================================================
# File save flags
# =============================================================================

class TestFileSave:
    """--csv and --daily flags write the correct files."""

    def test_daily_save_writes_to_price_trend_daily(self, tmp_path, monkeypatch):
        import paths
        monkeypatch.setattr(paths, "PRICE_TREND_DAILY", tmp_path / "price_trend_daily.csv")
        # Also patch the module-level import in price_trend
        import price_trend as pt
        monkeypatch.setattr(pt, "PRICE_TREND_DAILY", tmp_path / "price_trend_daily.csv")

        mock_coord = _make_mock_coordinator(["MSFT", "BLK"], n_rows=5)
        run(["MSFT", "BLK"], days=5, save_csv=False, save_daily=True,
            coordinator=mock_coord)

        out_file = tmp_path / "price_trend_daily.csv"
        assert out_file.exists()
        df = pd.read_csv(str(out_file), index_col=0)
        assert "MSFT" in df.columns
        assert "BLK" in df.columns

    def test_csv_save_writes_summary(self, tmp_path, monkeypatch):
        import price_trend as pt
        monkeypatch.setattr(pt, "PRICE_TREND_SUMMARY", tmp_path / "price_trend.csv")

        mock_coord = _make_mock_coordinator(["MSFT"], n_rows=5)
        run(["MSFT"], days=5, save_csv=True, save_daily=False,
            coordinator=mock_coord)

        out_file = tmp_path / "price_trend.csv"
        assert out_file.exists()
        df = pd.read_csv(str(out_file))
        assert "Ticker" in df.columns
        assert "Chg_pct" in df.columns
        assert "MSFT" in df["Ticker"].values
