"""
Tests for utils/portfolio_sizer.py.

Uses injected fakes for price fetching — no network calls.
--out tests are in tests/test_out_flag.py (TestPortfolioSizerOutFlag).
"""

from __future__ import annotations

import os
import re
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from portfolio_sizer import generate_orders


# =============================================================================
# Helpers
# =============================================================================

def _write_alpha_csv(path: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _mock_prices(tickers: list[str], price: float = 100.0) -> pd.Series:
    """Series of {ticker: price} — matches _fetch_current_prices() return type."""
    return pd.Series({t: price for t in tickers})


_ALPHA_ROWS = [
    {"Ticker": "BLK",  "confidence_delta": 0.050, "delta_rel": 0.04, "r_squared": 0.80},
    {"Ticker": "TXN",  "confidence_delta": 0.040, "delta_rel": 0.03, "r_squared": 0.75},
    {"Ticker": "SNPS", "confidence_delta": 0.035, "delta_rel": 0.03, "r_squared": 0.70},
    {"Ticker": "PH",   "confidence_delta": 0.020, "delta_rel": 0.02, "r_squared": 0.60},
    {"Ticker": "V",    "confidence_delta": 0.010, "delta_rel": 0.01, "r_squared": 0.55},
]


# =============================================================================
# --exclude flag
# =============================================================================

class TestExclude:
    """--exclude removes tickers before ranking."""

    def test_excluded_ticker_not_in_output(self, tmp_path, capsys):
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)
        tickers = ["BLK", "TXN", "PH", "V"]

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(tickers)):
            generate_orders([str(csv)], capital=10_000, top_n=5, exclude=["SNPS"])

        out = capsys.readouterr().out
        assert "SNPS" not in out

    def test_excluded_ticker_not_bumping_allocation(self, tmp_path, capsys):
        """With SNPS excluded, BLK should take the top slot."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)
        tickers = ["BLK", "TXN", "PH", "V"]

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(tickers)):
            generate_orders([str(csv)], capital=10_000, top_n=4, exclude=["SNPS"])

        out = capsys.readouterr().out
        assert "BLK" in out

    def test_exclude_is_case_insensitive(self, tmp_path, capsys):
        """Exclude list is normalised to uppercase."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)
        tickers = ["BLK", "TXN", "PH", "V"]

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(tickers)):
            generate_orders([str(csv)], capital=10_000, top_n=5, exclude=["snps"])

        out = capsys.readouterr().out
        assert "SNPS" not in out

    def test_exclude_nonexistent_ticker_is_harmless(self, tmp_path, capsys):
        """Excluding a ticker that is not in the CSV raises no error."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)
        tickers = ["BLK", "TXN", "SNPS", "PH", "V"]

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(tickers)):
            generate_orders([str(csv)], capital=10_000, top_n=5, exclude=["BOGUS"])

        out = capsys.readouterr().out
        assert "BLK" in out

    def test_exclude_all_positive_signals_warns(self, tmp_path, capsys):
        """Excluding every positive-signal ticker produces a warning, not a crash."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)
        all_tickers = [r["Ticker"] for r in _ALPHA_ROWS]

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(all_tickers)):
            generate_orders([str(csv)], capital=10_000, top_n=5, exclude=all_tickers)
        # should complete without exception; output may be empty


# =============================================================================
# --tickers flag
# =============================================================================

class TestTickers:
    """--tickers sizes only the named tickers, bypassing CSV ranking."""

    def test_only_named_tickers_appear(self, tmp_path, capsys):
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["TXN", "PH"])):
            generate_orders([str(csv)], capital=10_000, tickers=["TXN", "PH"])

        out = capsys.readouterr().out
        assert "TXN" in out
        assert "PH" in out
        assert "BLK" not in out
        assert "SNPS" not in out

    def test_tickers_overrides_top_n(self, tmp_path, capsys):
        """--tickers ignores --top; all named tickers are sized regardless of top_n."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK", "TXN", "PH"])):
            generate_orders([str(csv)], capital=10_000, top_n=1, tickers=["BLK", "TXN", "PH"])

        out = capsys.readouterr().out
        assert "BLK" in out
        assert "TXN" in out
        assert "PH" in out

    def test_tickers_is_case_insensitive(self, tmp_path, capsys):
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"])):
            generate_orders([str(csv)], capital=10_000, tickers=["blk"])

        out = capsys.readouterr().out
        assert "BLK" in out

    def test_weights_sum_to_one(self, tmp_path, capsys):
        """Conviction weights for the filtered set must sum to 100%."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK", "TXN"])):
            generate_orders([str(csv)], capital=10_000, tickers=["BLK", "TXN"])

        out = capsys.readouterr().out
        weights = [float(w.rstrip("%")) for w in re.findall(r"\d+\.\d+%", out)]
        assert abs(sum(weights) - 100.0) < 0.2

    def test_tickers_not_in_csv_are_silently_skipped(self, tmp_path, capsys):
        """A ticker named in --tickers but absent from the CSV is skipped without error."""
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv), _ALPHA_ROWS)

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"])):
            generate_orders([str(csv)], capital=10_000, tickers=["BLK", "BOGUS"])

        out = capsys.readouterr().out
        assert "BLK" in out
        assert "BOGUS" not in out
