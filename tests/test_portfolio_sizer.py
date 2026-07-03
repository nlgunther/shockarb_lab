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

from portfolio_sizer import (
    generate_orders,
    mark_positions,
    _load_held_tickers,
    _LOG_COLUMNS,
)


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


# =============================================================================
# --positions / _load_held_tickers
# =============================================================================

_POSITIONS_HEADER = [
    "Symbol", "Description", "Qty (Quantity)", "Price", "Price Chng $",
    "Price Chng %", "Mkt Val", "Day Chng $", "Day Chng %", "Cost Basis",
    "Gain $", "Gain %", "Reinvest?", "Reinvest Cap?", "Asset Type",
]

_POSITIONS_ROWS = [
    {"Symbol": "ADI", "Qty (Quantity)": "9", "Cost Basis": "$3,482.19", "Asset Type": "Equity"},
    {"Symbol": "CRM", "Qty (Quantity)": "9", "Cost Basis": "$1,395.81", "Asset Type": "Equity"},
    {"Symbol": "VOO", "Qty (Quantity)": "245", "Cost Basis": "$41,372.03",
     "Asset Type": "ETFs & Closed End Funds"},
]


def _write_positions_csv(path: str, rows: list[dict]) -> None:
    """
    Fixture mimicking the brokerage export: an account-title line, a blank
    line, then the real header — the shape _load_held_tickers() must skip
    past via skiprows=2.
    """
    with open(path, "w") as f:
        f.write('"Positions for account Individual ...696 as of 10:08 AM ET, 2026/07/01"\n')
        f.write("\n")
        f.write(",".join(f'"{h}"' for h in _POSITIONS_HEADER) + ",\n")
        for r in rows:
            vals = [str(r.get(h, "")) for h in _POSITIONS_HEADER]
            f.write(",".join(f'"{v}"' for v in vals) + ",\n")


class TestLoadHeldTickers:
    """_load_held_tickers() parses the brokerage positions export."""

    def test_keeps_only_equity_rows(self, tmp_path):
        """VOO (an ETF row) is excluded even though it's in known_tickers."""
        csv = tmp_path / "positions.csv"
        _write_positions_csv(str(csv), _POSITIONS_ROWS)
        held = _load_held_tickers(str(csv), {"ADI", "CRM", "VOO"})
        assert set(held.keys()) == {"ADI", "CRM"}

    def test_shares_and_cost_basis_parsed_correctly(self, tmp_path):
        """Dollar/comma-formatted Cost Basis is converted to a per-share float."""
        csv = tmp_path / "positions.csv"
        _write_positions_csv(str(csv), _POSITIONS_ROWS)
        held = _load_held_tickers(str(csv), {"ADI", "CRM"})
        assert held["ADI"]["shares"] == 9.0
        assert abs(held["ADI"]["cost_basis"] - 386.91) < 0.01
        assert abs(held["CRM"]["cost_basis"] - 155.09) < 0.01

    def test_ticker_not_in_known_set_is_excluded(self, tmp_path):
        """A held Equity ticker ShockArb didn't score today is dropped."""
        csv = tmp_path / "positions.csv"
        _write_positions_csv(str(csv), _POSITIONS_ROWS)
        held = _load_held_tickers(str(csv), {"ADI"})
        assert set(held.keys()) == {"ADI"}

    def test_no_matching_tickers_returns_empty_dict(self, tmp_path):
        csv = tmp_path / "positions.csv"
        _write_positions_csv(str(csv), _POSITIONS_ROWS)
        held = _load_held_tickers(str(csv), {"BOGUS"})
        assert held == {}


class TestMarkPositions:
    """mark_positions() reports ShockArb fair value (price * (1+delta_rel)),
    not an analyst target, for currently-held tickers."""

    def test_fair_price_uses_delta_rel_not_analyst_target(self, tmp_path, capsys):
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)  # BLK delta_rel=0.04, confidence_delta=0.050
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "BLK", "Qty (Quantity)": "4", "Cost Basis": "$3,930.40", "Asset Type": "Equity"},
        ])

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"], price=1000.0)), \
             patch("portfolio_sizer._LOG_PATH", str(tmp_path / "log.csv")):
            mark_positions(str(positions), [str(alpha)], out=str(tmp_path / "mark.csv"))

        out = capsys.readouterr().out
        assert "BLK" in out
        df = pd.read_csv(tmp_path / "mark.csv")
        assert abs(df.iloc[0]["fair_price"] - 1040.0) < 0.01  # 1000 * (1 + 0.04)

    def test_gap_pct_equals_delta_rel(self, tmp_path):
        """The reported gap is exactly delta_rel — this is the whole point of
        --positions: an analyst-independent number."""
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "TXN", "Qty (Quantity)": "10", "Cost Basis": "$1,000.00", "Asset Type": "Equity"},
        ])

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["TXN"], price=100.0)), \
             patch("portfolio_sizer._LOG_PATH", str(tmp_path / "log.csv")):
            mark_positions(str(positions), [str(alpha)], out=str(tmp_path / "mark.csv"))

        df = pd.read_csv(tmp_path / "mark.csv")
        assert abs(df.iloc[0]["delta_rel"] - 0.03) < 1e-9   # TXN's delta_rel in _ALPHA_ROWS

    def test_no_out_suppresses_mark_file(self, tmp_path):
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "BLK", "Qty (Quantity)": "4", "Cost Basis": "$3,930.40", "Asset Type": "Equity"},
        ])
        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"], price=1000.0)):
            mark_positions(str(positions), [str(alpha)], out=None)
        assert not (tmp_path / "mark.csv").exists()

    def test_no_shockarb_scored_holdings_warns_without_crashing(self, tmp_path):
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "AAPL", "Qty (Quantity)": "1", "Cost Basis": "$100.00", "Asset Type": "Equity"},
        ])
        mark_positions(str(positions), [str(alpha)])  # AAPL isn't in _ALPHA_ROWS — should not raise


# =============================================================================
# --execute durable log
# =============================================================================

class TestExecuteLog:
    """--execute appends to the durable, never-overwritten position log."""

    def test_execute_false_writes_no_log(self, tmp_path):
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        log_path = tmp_path / "log.csv"
        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"])), \
             patch("portfolio_sizer._LOG_PATH", str(log_path)):
            generate_orders([str(alpha)], capital=10_000, tickers=["BLK"], out=None, execute=False)
        assert not log_path.exists()

    def test_ticket_execute_logs_event_ticket_with_no_cost_basis(self, tmp_path):
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        log_path = tmp_path / "log.csv"
        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"])), \
             patch("portfolio_sizer._LOG_PATH", str(log_path)):
            generate_orders([str(alpha)], capital=10_000, tickers=["BLK"], out=None, execute=True)

        log = pd.read_csv(log_path)
        assert list(log.columns) == _LOG_COLUMNS
        assert log.iloc[0]["event"] == "ticket"
        assert log.iloc[0]["ticker"] == "BLK"
        assert pd.isna(log.iloc[0]["cost_basis"])

    def test_mark_execute_logs_event_mark_with_real_cost_basis(self, tmp_path):
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "BLK", "Qty (Quantity)": "4", "Cost Basis": "$3,930.40", "Asset Type": "Equity"},
        ])
        log_path = tmp_path / "log.csv"
        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"], price=1000.0)), \
             patch("portfolio_sizer._LOG_PATH", str(log_path)):
            mark_positions(str(positions), [str(alpha)], out=None, execute=True)

        log = pd.read_csv(log_path)
        assert log.iloc[0]["event"] == "mark"
        assert abs(log.iloc[0]["cost_basis"] - 982.60) < 0.01

    def test_log_is_append_only_across_repeated_runs(self, tmp_path):
        """A second --execute run adds rows; it never overwrites the first."""
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "BLK", "Qty (Quantity)": "4", "Cost Basis": "$3,930.40", "Asset Type": "Equity"},
        ])
        log_path = tmp_path / "log.csv"

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"], price=1000.0)), \
             patch("portfolio_sizer._LOG_PATH", str(log_path)):
            mark_positions(str(positions), [str(alpha)], out=None, execute=True)
            mark_positions(str(positions), [str(alpha)], out=None, execute=True)

        log = pd.read_csv(log_path)
        assert len(log) == 2

    def test_ticket_and_mark_rows_share_one_rectangular_schema(self, tmp_path):
        """Mixing event types in one log must not produce a ragged CSV."""
        alpha = tmp_path / "alpha.csv"
        _write_alpha_csv(str(alpha), _ALPHA_ROWS)
        positions = tmp_path / "positions.csv"
        _write_positions_csv(str(positions), [
            {"Symbol": "BLK", "Qty (Quantity)": "4", "Cost Basis": "$3,930.40", "Asset Type": "Equity"},
        ])
        log_path = tmp_path / "log.csv"

        with patch("portfolio_sizer._fetch_current_prices", return_value=_mock_prices(["BLK"], price=1000.0)), \
             patch("portfolio_sizer._LOG_PATH", str(log_path)):
            generate_orders([str(alpha)], capital=10_000, tickers=["BLK"], out=None, execute=True)
            mark_positions(str(positions), [str(alpha)], out=None, execute=True)

        log = pd.read_csv(log_path)
        assert list(log.columns) == _LOG_COLUMNS
        assert list(log["event"]) == ["ticket", "mark"]
