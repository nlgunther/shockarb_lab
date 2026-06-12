"""
Tests for intraday price/% change — informational/display-only feature.

intraday_price/intraday_chg_pct mirror marketfit's _fetch_intraday_current
pattern: a single batch yfinance call fetches live current prices for
candidate tickers, and intraday_chg_pct = (intraday_price - price) / price
is computed against the cached close. Network call, off by default
(compute_intraday=False / --intraday flag).

Coverage
--------
  TestFetchIntradayPrices  — features._fetch_intraday_prices() against mocked yfinance
  TestExtractAllIntraday   — features.extract_all(compute_intraday=...) wiring
  TestStockVerdictIntraday — rules.StockVerdict intraday fields + markdown formatting
  TestReportIntradayColumn — report.py table header includes Intraday
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from stockfit import features, rules, report
from stockfit.features import _fetch_intraday_prices
from stockfit.rules import StockVerdict


# =============================================================================
# Helpers
# =============================================================================

def _make_features(ticker="ETN", intraday_price=None, intraday_chg_pct=None) -> dict:
    """Minimal feature dict with intraday fields, mirroring extract_all() output."""
    return {
        "ticker": ticker,
        "r_squared": 0.70,
        "confidence_delta": 0.030,
        "delta_rel": 0.04,
        "actual_return": -0.05,
        "expected_rel": -0.01,
        "residual_vol": 0.25,
        "price": 396.0,
        "analyst_target": 452.0,
        "analyst_upside": (452.0 - 396.0) / 396.0,
        "fwd_pe": 25.0,
        "ttm_eps": 10.0,
        "fwd_eps": 15.0,
        "ex_div": "2026-05-07",
        "div_amt": 4.40,
        "next_earnings": None,
        "news_headlines": [],
        "target_below_price": False,
        "earnings_imminent": False,
        "rvol": None,
        "rvol_window": None,
        "intraday_price": intraday_price,
        "intraday_chg_pct": intraday_chg_pct,
    }


# =============================================================================
# TestFetchIntradayPrices
# =============================================================================

class TestFetchIntradayPrices:
    """features._fetch_intraday_prices() against mocked yfinance — no real network."""

    def test_no_tickers_returns_empty(self):
        assert _fetch_intraday_prices([]) == {}

    def test_multi_ticker_batch(self, monkeypatch):
        import pandas as pd

        df = pd.DataFrame({
            ("Close", "ETN"): [410.20],
            ("Close", "HON"): [225.10],
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        fake_yf = SimpleNamespace(download=lambda *a, **k: df)
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        prices = _fetch_intraday_prices(["ETN", "HON"])
        assert prices == {"ETN": pytest.approx(410.20), "HON": pytest.approx(225.10)}

    def test_single_ticker(self, monkeypatch):
        import pandas as pd

        df = pd.DataFrame({"Close": [410.20]})
        fake_yf = SimpleNamespace(download=lambda *a, **k: df)
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        prices = _fetch_intraday_prices(["ETN"])
        assert prices == {"ETN": pytest.approx(410.20)}

    def test_empty_dataframe_returns_empty(self, monkeypatch):
        import pandas as pd

        fake_yf = SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        assert _fetch_intraday_prices(["ETN"]) == {}

    def test_exception_returns_empty(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        fake_yf = SimpleNamespace(download=boom)
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        assert _fetch_intraday_prices(["ETN"]) == {}


# =============================================================================
# TestExtractAllIntraday
# =============================================================================

class TestExtractAllIntraday:
    """extract_all(compute_intraday=...) wiring — default off, no network when off."""

    def _write_minimal_inputs(self, tmp_path):
        scores = tmp_path / "live_alpha_us.csv"
        scores.write_text(
            ",actual_return,expected_rel,expected_abs,delta_rel,delta_abs,r_squared,residual_vol,confidence_delta\n"
            "ETN,-0.05,-0.01,0,0.04,0,0.693,0.25,0.028\n", encoding="utf-8")
        fund = tmp_path / "fundamentals.csv"
        fund.write_text(
            "Ticker,Price,Fwd P/E,TTM EPS,Fwd EPS,Next Earnings,Est. EPS,Ex-Div,Div Amt,Analyst Tgt\n"
            "ETN,396.0,25.0,10.0,15.0,—,—,2026-05-07,$4.40,452.0\n", encoding="utf-8")
        return str(scores), str(fund), str(tmp_path / "missing_news.txt")

    def test_compute_intraday_false_by_default(self, tmp_path):
        scores, fund, news = self._write_minimal_inputs(tmp_path)
        result = features.extract_all(scores, fund, news)
        assert result[0]["intraday_price"] is None
        assert result[0]["intraday_chg_pct"] is None

    def test_compute_intraday_true_populates_fields(self, tmp_path, monkeypatch):
        scores, fund, news = self._write_minimal_inputs(tmp_path)

        monkeypatch.setattr(features, "_fetch_intraday_prices", lambda tickers: {"ETN": 415.80})

        result = features.extract_all(scores, fund, news, compute_intraday=True)
        assert result[0]["intraday_price"] == pytest.approx(415.80)
        assert result[0]["intraday_chg_pct"] == pytest.approx((415.80 - 396.0) / 396.0)

    def test_compute_intraday_true_missing_quote_leaves_none(self, tmp_path, monkeypatch):
        scores, fund, news = self._write_minimal_inputs(tmp_path)

        monkeypatch.setattr(features, "_fetch_intraday_prices", lambda tickers: {})

        result = features.extract_all(scores, fund, news, compute_intraday=True)
        assert result[0]["intraday_price"] is None
        assert result[0]["intraday_chg_pct"] is None


# =============================================================================
# TestStockVerdictIntraday
# =============================================================================

class TestStockVerdictIntraday:
    """rules.evaluate_all() passes intraday fields through to StockVerdict."""

    def test_intraday_passed_through(self):
        f = _make_features("ETN", intraday_price=415.80, intraday_chg_pct=0.05)
        v = rules.evaluate_all([f])[0]
        assert v.intraday_price == pytest.approx(415.80)
        assert v.intraday_chg_pct == pytest.approx(0.05)

    def test_intraday_defaults_to_none(self):
        f = _make_features("ETN")
        v = rules.evaluate_all([f])[0]
        assert v.intraday_price is None
        assert v.intraday_chg_pct is None

    def test_markdown_row_shows_intraday_when_present(self):
        v = StockVerdict(
            ticker="ETN", tier="INCLUDE", reason="ok",
            r_squared=0.70, confidence_delta=0.03, analyst_upside=0.10,
            price=396.0, analyst_target=452.0, fwd_pe=25.0,
            news_headlines=[], cluster=None,
            intraday_price=415.80, intraday_chg_pct=0.05,
        )
        assert "+5.00%" in v.as_markdown_row()

    def test_markdown_row_shows_dash_when_absent(self):
        v = StockVerdict(
            ticker="ETN", tier="INCLUDE", reason="ok",
            r_squared=0.70, confidence_delta=0.03, analyst_upside=0.10,
            price=396.0, analyst_target=452.0, fwd_pe=25.0,
            news_headlines=[], cluster=None,
        )
        row = v.as_markdown_row()
        # Last column should be the dash placeholder
        assert row.rstrip().endswith("| — |")


# =============================================================================
# TestReportIntradayColumn
# =============================================================================

class TestReportIntradayColumn:
    """report.py table includes the Intraday column."""

    def test_table_header_has_intraday(self):
        f = _make_features("ETN", intraday_price=415.80, intraday_chg_pct=0.05)
        v = rules.evaluate_all([f])
        md = report.build(v)
        assert "Intraday" in md
        assert "+5.00%" in md

    def test_table_header_shows_dash_when_no_intraday(self):
        f = _make_features("ETN")
        v = rules.evaluate_all([f])
        md = report.build(v)
        assert "Intraday" in md
