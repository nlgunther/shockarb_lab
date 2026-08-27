"""
Tests for utils/market_data.py's last-known-value fallback.

Root cause (2026-08-18, HIL_todo.md MARKET-REPORT-PARTIAL-FETCH-GAP):
several unrelated tickers (QQQ, IWM, XLI, XLU, XLB, HYG, GLD, ^HSI) rendered
blank in the same market report because DataCoordinator batches tickers by
cache-gap span, and a single bad ticker in the batch failed the whole group.
Two-layer fix: coordinator.py retries individually on batch failure (see
tests/test_coordinator_phase1.py::TestDownloadAndCommitRetry); this file
covers the second line of defense — market_data.py falling back to the
last cached value (marked stale) rather than a bare blank when a ticker's
fetch still comes back empty.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import pandas as pd
import pytest

from market_data import _last_known_value, _row_ok, _row_error


class FakeReadStore:
    """Minimal store double — only needs .read(key, start, end)."""

    def __init__(self, frames: dict[str, pd.DataFrame] | None = None, raise_on_read: bool = False):
        self._frames = frames or {}
        self._raise = raise_on_read

    def read(self, key: str, start: str, end: str):
        if self._raise:
            raise RuntimeError("simulated disk error")
        return self._frames.get(key)


def _adj_close_df(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"adj_close": values}, index=pd.to_datetime(dates))


class TestLastKnownValue:

    def test_returns_last_two_rows_when_history_available(self):
        store = FakeReadStore({
            "daily/QQQ": _adj_close_df(
                ["2026-08-12", "2026-08-13", "2026-08-14"], [500.0, 505.0, 510.0]
            )
        })
        result = _last_known_value(store, "QQQ")
        assert result == {"close": 510.0, "prev": 505.0, "last_date": "2026-08-14"}

    def test_returns_none_when_ticker_has_no_history(self):
        store = FakeReadStore({})
        assert _last_known_value(store, "QQQ") is None

    def test_returns_none_when_fewer_than_two_rows(self):
        store = FakeReadStore({"daily/QQQ": _adj_close_df(["2026-08-14"], [510.0])})
        assert _last_known_value(store, "QQQ") is None

    def test_returns_none_on_store_read_exception(self):
        """A broken store must not crash the fallback — just no last-known value."""
        store = FakeReadStore(raise_on_read=True)
        assert _last_known_value(store, "QQQ") is None

    def test_handles_ticker_named_column_not_adj_close(self):
        """Some store formats use the ticker itself as the column name."""
        df = pd.DataFrame({"QQQ": [500.0, 505.0]}, index=pd.to_datetime(["2026-08-13", "2026-08-14"]))
        store = FakeReadStore({"daily/QQQ": df})
        result = _last_known_value(store, "QQQ")
        assert result["close"] == 505.0
        assert result["prev"] == 500.0

    def test_dropna_ignores_gaps_in_history(self):
        """NaN rows (e.g. a holiday placeholder) shouldn't count toward the 2-row minimum."""
        df = _adj_close_df(["2026-08-12", "2026-08-13", "2026-08-14"], [500.0, float("nan"), 510.0])
        store = FakeReadStore({"daily/QQQ": df})
        result = _last_known_value(store, "QQQ")
        assert result == {"close": 510.0, "prev": 500.0, "last_date": "2026-08-14"}


class TestRowStaleFlag:

    def test_row_ok_defaults_not_stale(self):
        row = _row_ok("QQQ", "Nasdaq 100", "us_broad", close=510.0, prev=505.0, last_date="2026-08-18")
        assert row["stale"] is False
        assert row["status"] == "ok"

    def test_row_ok_can_be_marked_stale(self):
        row = _row_ok("QQQ", "Nasdaq 100", "us_broad", close=510.0, prev=505.0,
                       last_date="2026-08-14", stale=True)
        assert row["stale"] is True
        assert row["last_date"] == "2026-08-14"
        # Stale rows still carry real numbers — downstream code (rules,
        # features, report) must NOT treat them as missing.
        assert row["close"] == 510.0
        assert row["status"] == "ok"

    def test_row_error_has_stale_false(self):
        row = _row_error("XYZ", "Unknown", "us_broad")
        assert row["stale"] is False
        assert row["status"] == "error"
