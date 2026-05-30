"""
Tests for utils/fundamental_scanner.py.

All tests use injected fakes — no yfinance network calls.
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# fundamental_scanner lives in utils/, not a package — add to path via conftest
# or import directly.  sys.path manipulation is in conftest.py; if not present,
# use importlib here.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from fundamental_scanner import (
    _next_dividend,
    _next_earnings,
    _safe_get,
    fetch_fundamentals,
    print_fundamentals,
)


# =============================================================================
# _safe_get
# =============================================================================

class TestSafeGet:
    """_safe_get returns formatted values or '—' for missing/bad data."""

    def test_returns_formatted_float(self):
        assert _safe_get({"price": 123.456}, "price", ".2f") == "123.46"

    def test_returns_str_no_fmt(self):
        assert _safe_get({"name": "AAPL"}, "name") == "AAPL"

    def test_missing_key(self):
        assert _safe_get({}, "price") == "—"

    def test_none_value(self):
        assert _safe_get({"price": None}, "price") == "—"

    def test_nan_value(self):
        assert _safe_get({"price": float("nan")}, "price") == "—"

    def test_zero_is_not_missing(self):
        """Zero is a valid value — should not be treated as missing."""
        assert _safe_get({"eps": 0.0}, "eps", ".2f") == "0.00"

    def test_bad_format_falls_back_to_str(self):
        """If format string fails, returns str() of value."""
        result = _safe_get({"x": "hello"}, "x", ".2f")
        assert result == "hello"


# =============================================================================
# _next_dividend
# =============================================================================

class TestNextDividend:
    """_next_dividend parses ex-date timestamp and dividend rate."""

    def test_both_present(self):
        ts = int(datetime(2026, 6, 15).timestamp())
        ex_str, amt_str = _next_dividend({"exDividendDate": ts, "dividendRate": 2.5})
        assert ex_str == "2026-06-15"
        assert amt_str == "$2.50"

    def test_no_ex_date(self):
        ex_str, amt_str = _next_dividend({"dividendRate": 1.0})
        assert ex_str == "—"

    def test_no_dividend_rate(self):
        ts = int(datetime(2026, 6, 15).timestamp())
        ex_str, amt_str = _next_dividend({"exDividendDate": ts})
        assert amt_str == "—"

    def test_both_missing(self):
        ex_str, amt_str = _next_dividend({})
        assert ex_str == "—"
        assert amt_str == "—"


# =============================================================================
# _next_earnings
# =============================================================================

class TestNextEarnings:
    """_next_earnings reads ticker.earnings_dates and returns future row."""

    def _make_ticker(self, future_date: str, est_eps: float | None = 1.23):
        """Build a minimal mock yf.Ticker with earnings_dates."""
        future_idx = pd.DatetimeIndex([pd.Timestamp(future_date, tz="UTC")])
        df = pd.DataFrame(
            {"EPS Estimate": [est_eps], "Reported EPS": [float("nan")]},
            index=future_idx,
        )
        t = MagicMock()
        t.earnings_dates = df
        return t

    def test_returns_date_and_est(self):
        t = self._make_ticker("2026-08-01", est_eps=2.50)
        date_str, est_str = _next_earnings(t)
        assert date_str == "2026-08-01"
        assert est_str == "$2.50"

    def test_missing_est_eps(self):
        t = self._make_ticker("2026-08-01", est_eps=None)
        date_str, est_str = _next_earnings(t)
        assert date_str == "2026-08-01"
        assert est_str == "—"

    def test_no_future_rows(self):
        """All rows have Reported EPS — no upcoming earnings."""
        idx = pd.DatetimeIndex([pd.Timestamp("2026-02-01", tz="UTC")])
        df = pd.DataFrame(
            {"EPS Estimate": [1.0], "Reported EPS": [0.95]},
            index=idx,
        )
        t = MagicMock()
        t.earnings_dates = df
        date_str, est_str = _next_earnings(t)
        assert date_str == "—"
        assert est_str == "—"

    def test_none_earnings_dates(self):
        t = MagicMock()
        t.earnings_dates = None
        date_str, est_str = _next_earnings(t)
        assert date_str == "—"
        assert est_str == "—"

    def test_exception_returns_dashes(self):
        t = MagicMock()
        t.earnings_dates = MagicMock(side_effect=RuntimeError("boom"))
        date_str, est_str = _next_earnings(t)
        assert date_str == "—"
        assert est_str == "—"


# =============================================================================
# fetch_fundamentals
# =============================================================================

class TestFetchFundamentals:
    """fetch_fundamentals returns one dict per ticker."""

    def _make_ticker_mock(
        self,
        symbol: str,
        price: float = 100.0,
        fwd_pe: float = 20.0,
        ttm_eps: float = 5.0,
        fwd_eps: float = 5.5,
        target: float = 120.0,
    ):
        info = {
            "currentPrice":    price,
            "forwardPE":       fwd_pe,
            "trailingEps":     ttm_eps,
            "forwardEps":      fwd_eps,
            "targetMeanPrice": target,
        }
        t = MagicMock()
        t.info = info
        # earnings_dates: one future row
        idx = pd.DatetimeIndex([pd.Timestamp("2026-09-01", tz="UTC")])
        t.earnings_dates = pd.DataFrame(
            {"EPS Estimate": [5.5], "Reported EPS": [float("nan")]},
            index=idx,
        )
        return t

    @patch("fundamental_scanner.yf.Ticker")
    def test_single_ticker_keys(self, mock_ticker_cls):
        mock_ticker_cls.return_value = self._make_ticker_mock("AAPL")
        rows = fetch_fundamentals(["AAPL"])
        assert len(rows) == 1
        row = rows[0]
        expected_keys = {
            "Ticker", "Price", "Fwd P/E", "TTM EPS", "Fwd EPS",
            "Next Earnings", "Est. EPS", "Ex-Div", "Div Amt", "Analyst Tgt",
        }
        assert expected_keys == set(row.keys())

    @patch("fundamental_scanner.yf.Ticker")
    def test_ticker_value_in_row(self, mock_ticker_cls):
        mock_ticker_cls.return_value = self._make_ticker_mock("BLK", price=800.0)
        rows = fetch_fundamentals(["BLK"])
        assert rows[0]["Ticker"] == "BLK"
        assert rows[0]["Price"] == "800.00"

    @patch("fundamental_scanner.yf.Ticker")
    def test_multiple_tickers(self, mock_ticker_cls):
        mock_ticker_cls.return_value = self._make_ticker_mock("X")
        rows = fetch_fundamentals(["A", "B", "C"])
        assert len(rows) == 3
        assert [r["Ticker"] for r in rows] == ["A", "B", "C"]

    @patch("fundamental_scanner.yf.Ticker")
    def test_exception_yields_err_row(self, mock_ticker_cls):
        """A failing ticker produces an ERR row, not an exception."""
        mock_ticker_cls.side_effect = RuntimeError("network error")
        rows = fetch_fundamentals(["BAD"])
        assert len(rows) == 1
        assert rows[0]["Price"] == "ERR"
        assert rows[0]["Ticker"] == "BAD"

    @patch("fundamental_scanner.yf.Ticker")
    def test_missing_info_fields_become_dashes(self, mock_ticker_cls):
        t = MagicMock()
        t.info = {}   # no fields at all
        t.earnings_dates = None
        mock_ticker_cls.return_value = t
        rows = fetch_fundamentals(["EMPTY"])
        row = rows[0]
        assert row["Price"] == "—"
        assert row["Fwd P/E"] == "—"
        assert row["Next Earnings"] == "—"

    @patch("fundamental_scanner.yf.Ticker")
    def test_empty_ticker_list(self, mock_ticker_cls):
        rows = fetch_fundamentals([])
        assert rows == []
        mock_ticker_cls.assert_not_called()


# =============================================================================
# print_fundamentals
# =============================================================================

class TestPrintFundamentals:
    """print_fundamentals produces a readable fixed-width table."""

    def _capture(self, rows):
        buf = io.StringIO()
        with patch("builtins.print", lambda *a, **kw: buf.write(" ".join(str(x) for x in a) + "\n")):
            print_fundamentals(rows)
        return buf.getvalue()

    def test_empty_rows_prints_nothing(self):
        out = self._capture([])
        assert out == ""

    def test_header_present(self):
        row = {
            "Ticker": "BLK", "Price": "800.00", "Fwd P/E": "20.0",
            "TTM EPS": "40.00", "Fwd EPS": "42.00",
            "Next Earnings": "2026-09-01", "Est. EPS": "$10.00",
            "Ex-Div": "2026-07-01", "Div Amt": "$5.00", "Analyst Tgt": "900.00",
        }
        out = self._capture([row])
        assert "Ticker" in out
        assert "Fwd P/E" in out
        assert "Analyst Tgt" in out

    def test_ticker_appears_in_output(self):
        row = {
            "Ticker": "TXN", "Price": "150.00", "Fwd P/E": "18.0",
            "TTM EPS": "8.00", "Fwd EPS": "9.00",
            "Next Earnings": "2026-10-15", "Est. EPS": "$9.00",
            "Ex-Div": "—", "Div Amt": "—", "Analyst Tgt": "175.00",
        }
        out = self._capture([row])
        assert "TXN" in out

    def test_missing_field_shows_dash(self):
        """Row missing a key still renders without error, showing '—'."""
        row = {"Ticker": "X"}   # all other keys absent
        out = self._capture([row])
        assert "—" in out
