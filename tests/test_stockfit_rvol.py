"""
Tests for RVOL (relative volume) — informational/display-only feature.

RVOL = most recent cached day's volume / trailing average volume, computed
from the local DataStore parquet cache (no network calls). It does not
affect scoring, ranking, or the PCA factor model — see docs/KT.md
("RVOL (relative volume) display").

Coverage
--------
  TestComputeRvol      — features._compute_rvol() against a fake DataStore
  TestExtractAllRvol   — features.extract_all(compute_rvol=...) wiring
  TestStockVerdictRvol — rules.StockVerdict rvol fields + markdown formatting
  TestReportRvolColumn — report.py table header includes RVOL
  TestStickyRvolCli    — stockfit.cli sticky on/off + resolution order
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from stockfit import features, rules, report
from stockfit.features import _compute_rvol, RVOL_MIN_WINDOW, RVOL_MAX_WINDOW
from stockfit.rules import StockVerdict


# =============================================================================
# Helpers
# =============================================================================

class _FakeStore:
    """Minimal DataStore double — returns a fixed Volume series for any ticker."""

    def __init__(self, volumes: list[float] | None):
        self.volumes = volumes

    def fetch_daily_ohlcv(self, tickers, start, end):
        if self.volumes is None:
            return pd.DataFrame()
        ticker = tickers[0]
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=len(self.volumes))
        df = pd.DataFrame({"Volume": self.volumes}, index=idx)
        df.columns = pd.MultiIndex.from_tuples([("Volume", ticker)])
        return df


class _FakeStoreLowercase:
    """DataStore double mimicking the real cached parquet schema: lowercase
    column names ('volume', 'adj_close', etc.) — see HIL_todo.md 2026-06-11."""

    def __init__(self, volumes: list[float]):
        self.volumes = volumes

    def fetch_daily_ohlcv(self, tickers, start, end):
        ticker = tickers[0]
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=len(self.volumes))
        df = pd.DataFrame({"volume": self.volumes}, index=idx)
        df.columns = pd.MultiIndex.from_tuples([("volume", ticker)])
        return df


def _make_features(ticker="ETN", rvol=None, rvol_window=None) -> dict:
    """Minimal feature dict with rvol fields, mirroring features.extract_all() output."""
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
        "rvol": rvol,
        "rvol_window": rvol_window,
    }


# =============================================================================
# TestComputeRvol
# =============================================================================

class TestComputeRvol:
    """features._compute_rvol() against a fake DataStore — no network, no real cache."""

    def test_empty_dataframe_returns_none(self):
        store = _FakeStore(volumes=None)
        rvol, window = _compute_rvol("ETN", store)
        assert rvol is None and window is None

    def test_insufficient_history_returns_none(self):
        """Fewer than min_window + 1 cached days → (None, None)."""
        store = _FakeStore(volumes=[100.0] * RVOL_MIN_WINDOW)  # exactly min_window, need +1
        rvol, window = _compute_rvol("ETN", store)
        assert rvol is None and window is None

    def test_ratio_computed_correctly(self):
        """20 days at 100 + today at 300 → rvol=3.0, window=20 (capped at max)."""
        volumes = [100.0] * RVOL_MAX_WINDOW + [300.0]
        store = _FakeStore(volumes=volumes)
        rvol, window = _compute_rvol("ETN", store)
        assert rvol == pytest.approx(3.0)
        assert window == RVOL_MAX_WINDOW

    def test_dynamic_window_below_max(self):
        """Only 9 trailing days cached + today → window=9 (between min and max)."""
        n_trailing = RVOL_MIN_WINDOW + 4
        volumes = [50.0] * n_trailing + [100.0]
        store = _FakeStore(volumes=volumes)
        rvol, window = _compute_rvol("ETN", store)
        assert window == n_trailing
        assert rvol == pytest.approx(2.0)

    def test_window_capped_at_max_with_long_history(self):
        """40 days of cached history → window still capped at RVOL_MAX_WINDOW."""
        volumes = [100.0] * 40 + [200.0]
        store = _FakeStore(volumes=volumes)
        rvol, window = _compute_rvol("ETN", store)
        assert window == RVOL_MAX_WINDOW

    def test_zero_average_returns_none(self):
        volumes = [0.0] * (RVOL_MIN_WINDOW + 1) + [100.0]
        store = _FakeStore(volumes=volumes)
        rvol, window = _compute_rvol("ETN", store)
        assert rvol is None and window is None

    def test_lowercase_volume_column_supported(self):
        """Real cached parquet files use lowercase 'volume' (see HIL_todo.md
        2026-06-11) — _compute_rvol must accept it, not just 'Volume'."""
        volumes = [100.0] * RVOL_MAX_WINDOW + [300.0]
        store = _FakeStoreLowercase(volumes=volumes)
        rvol, window = _compute_rvol("ETN", store)
        assert rvol == pytest.approx(3.0)
        assert window == RVOL_MAX_WINDOW


# =============================================================================
# TestExtractAllRvol
# =============================================================================

class TestExtractAllRvol:
    """extract_all(compute_rvol=...) wiring — default off, no network when off."""

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

    def test_compute_rvol_false_by_default(self, tmp_path):
        scores, fund, news = self._write_minimal_inputs(tmp_path)
        result = features.extract_all(scores, fund, news)
        assert result[0]["rvol"] is None
        assert result[0]["rvol_window"] is None

    def test_compute_rvol_true_uses_injected_store(self, tmp_path, monkeypatch):
        scores, fund, news = self._write_minimal_inputs(tmp_path)

        volumes = [100.0] * RVOL_MAX_WINDOW + [250.0]
        fake_module = SimpleNamespace(DataStore=lambda data_dir: _FakeStore(volumes))
        monkeypatch.setitem(sys.modules, "shockarb.store", fake_module)

        result = features.extract_all(scores, fund, news, compute_rvol=True)
        assert result[0]["rvol"] == pytest.approx(2.5)
        assert result[0]["rvol_window"] == RVOL_MAX_WINDOW


# =============================================================================
# TestStockVerdictRvol
# =============================================================================

class TestStockVerdictRvol:
    """rules.evaluate_all() passes rvol/rvol_window through to StockVerdict."""

    def test_rvol_passed_through(self):
        f = _make_features("ETN", rvol=2.3, rvol_window=10)
        v = rules.evaluate_all([f])[0]
        assert v.rvol == pytest.approx(2.3)
        assert v.rvol_window == 10

    def test_rvol_defaults_to_none(self):
        f = _make_features("ETN")
        v = rules.evaluate_all([f])[0]
        assert v.rvol is None
        assert v.rvol_window is None

    def test_markdown_row_shows_rvol_when_present(self):
        v = StockVerdict(
            ticker="ETN", tier="INCLUDE", reason="ok",
            r_squared=0.70, confidence_delta=0.03, analyst_upside=0.10,
            price=396.0, analyst_target=452.0, fwd_pe=25.0,
            news_headlines=[], cluster=None, rvol=2.3, rvol_window=10,
        )
        assert "2.3x (10d)" in v.as_markdown_row()

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
# TestReportRvolColumn
# =============================================================================

class TestReportRvolColumn:
    """report.py table includes the RVOL column."""

    def test_table_header_has_rvol(self):
        f = _make_features("ETN", rvol=2.3, rvol_window=10)
        v = rules.evaluate_all([f])
        md = report.build(v)
        assert "RVOL" in md
        assert "2.3x (10d)" in md

    def test_table_header_shows_dash_when_no_rvol(self):
        f = _make_features("ETN")
        v = rules.evaluate_all([f])
        md = report.build(v)
        assert "RVOL" in md


# =============================================================================
# TestStickyRvolCli
# =============================================================================

class TestStickyRvolCli:
    """stockfit.cli sticky RVOL setting — resolution order: flag > sticky > default off."""

    def setup_method(self):
        from stockfit import cli
        self.cli = cli

    def _sticky_path(self, tmp_path, monkeypatch):
        path = tmp_path / ".stockfit_rvol"
        monkeypatch.setattr(self.cli, "STOCKFIT_RVOL_FILE", path)
        return path

    def test_get_sticky_rvol_none_when_missing(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        assert self.cli._get_sticky_rvol() is None

    def test_set_and_get_sticky_rvol_on(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        self.cli._set_sticky_rvol(True)
        assert self.cli._get_sticky_rvol() is True

    def test_set_and_get_sticky_rvol_off(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        self.cli._set_sticky_rvol(False)
        assert self.cli._get_sticky_rvol() is False

    def test_resolve_rvol_default_off_when_no_sticky(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        args = SimpleNamespace(rvol=False, no_rvol=False)
        assert self.cli._resolve_rvol(args) is False

    def test_resolve_rvol_uses_sticky_when_set(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        self.cli._set_sticky_rvol(True)
        args = SimpleNamespace(rvol=False, no_rvol=False)
        assert self.cli._resolve_rvol(args) is True

    def test_resolve_rvol_flag_overrides_sticky_off(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        self.cli._set_sticky_rvol(False)
        args = SimpleNamespace(rvol=True, no_rvol=False)
        assert self.cli._resolve_rvol(args) is True

    def test_resolve_no_rvol_flag_overrides_sticky_on(self, tmp_path, monkeypatch):
        self._sticky_path(tmp_path, monkeypatch)
        self.cli._set_sticky_rvol(True)
        args = SimpleNamespace(rvol=False, no_rvol=True)
        assert self.cli._resolve_rvol(args) is False
