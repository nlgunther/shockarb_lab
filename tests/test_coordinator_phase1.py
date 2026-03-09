"""
Tests for datamgr Phase 1: DataRequest, Frequency, DataCoordinator.

All tests use injected fakes — no real network calls, no shockarb imports.
The fake store implements only the methods the coordinator calls in Phase 1:
fetch_daily() and fetch_intraday().
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from datamgr.coordinator import DataCoordinator
from datamgr.interfaces import DataStore
from datamgr.requests import DataRequest, Frequency


# =============================================================================
# Fakes
# =============================================================================

def _make_daily_df(tickers, start, end):
    idx = pd.bdate_range(start=start, end=end)
    return pd.DataFrame(100.0, index=idx, columns=tickers) if not idx.empty else pd.DataFrame()


def _make_intraday_df(tickers):
    idx = pd.date_range("2026-03-07 09:30", periods=8, freq="15min", tz="America/New_York")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers])
    return pd.DataFrame(100.0, index=idx, columns=cols)


class FakeStore(DataStore):
    """Minimal DataStore fake for Phase 1 coordinator tests."""

    def __init__(self):
        self.daily_calls    = []
        self.intraday_calls = []

    # Required by DataStore ABC — not called in Phase 1 dispatch
    def read(self, key, start, end): return None
    def write(self, key, df, meta): pass
    def coverage(self, key): return None
    def sweep(self, retention, before): return []

    # Called by coordinator._dispatch_daily
    def fetch_daily(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        self.daily_calls.append({"tickers": list(tickers), "start": start, "end": end})
        return _make_daily_df(tickers, start, end)

    # Called by coordinator._dispatch_intraday
    def fetch_intraday(self, tickers: List[str], trade_date=None) -> pd.DataFrame:
        self.intraday_calls.append({"tickers": list(tickers), "trade_date": trade_date})
        return _make_intraday_df(tickers)


@pytest.fixture
def fake_store():
    return FakeStore()


@pytest.fixture
def coordinator(fake_store):
    return DataCoordinator(fake_store)


def _daily_req(tickers, requester="test", start="2022-02-10", end="2022-03-31"):
    return DataRequest(
        tickers   = tuple(tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = requester,
    )


def _intraday_req(tickers, requester="test.intraday", trade_date="2026-03-07"):
    return DataRequest(
        tickers    = tuple(tickers),
        start      = trade_date,
        end        = trade_date,
        frequency  = Frequency.INTRADAY_15M,
        retention  = "ephemeral",
        requester  = requester,
        trade_date = trade_date,
    )


# =============================================================================
# Frequency
# =============================================================================

class TestFrequency:

    def test_valid_constants_pass(self):
        assert Frequency.validate(Frequency.DAILY) == "daily"
        assert Frequency.validate(Frequency.INTRADAY_15M) == "15m"
        assert Frequency.validate(Frequency.INTRADAY_1M) == "1m"

    def test_typo_raises_immediately(self):
        with pytest.raises(ValueError, match="Unknown frequency"):
            Frequency.validate("dayly")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            Frequency.validate("")

    def test_constants_are_strings(self):
        assert isinstance(Frequency.DAILY, str)
        assert isinstance(Frequency.INTRADAY_15M, str)


# =============================================================================
# DataRequest
# =============================================================================

class TestDataRequest:

    def test_valid_construction(self):
        req = _daily_req(["VOO", "TLT"])
        assert req.tickers == ("VOO", "TLT")
        assert req.frequency == Frequency.DAILY
        assert req.retention == "permanent"

    def test_frozen_immutable(self):
        req = _daily_req(["VOO"])
        with pytest.raises((AttributeError, TypeError)):
            req.tickers = ("TLT",)

    def test_hashable(self):
        req = _daily_req(["VOO"])
        d = {req: "value"}
        assert d[req] == "value"

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="Unknown frequency"):
            DataRequest(
                tickers=("VOO",), start="2022-01-01", end="2022-02-01",
                frequency="DAILY",  # wrong — should be "daily"
                retention="permanent", requester="test",
            )

    def test_invalid_retention_raises(self):
        with pytest.raises(ValueError, match="retention"):
            DataRequest(
                tickers=("VOO",), start="2022-01-01", end="2022-02-01",
                frequency=Frequency.DAILY,
                retention="temporary",  # not valid
                requester="test",
            )

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers"):
            DataRequest(
                tickers=(), start="2022-01-01", end="2022-02-01",
                frequency=Frequency.DAILY,
                retention="permanent", requester="test",
            )

    def test_empty_requester_raises(self):
        with pytest.raises(ValueError, match="requester"):
            DataRequest(
                tickers=("VOO",), start="2022-01-01", end="2022-02-01",
                frequency=Frequency.DAILY,
                retention="permanent", requester="",
            )

    def test_intraday_request_with_trade_date(self):
        req = _intraday_req(["VOO"])
        assert req.trade_date == "2026-03-07"
        assert req.frequency == Frequency.INTRADAY_15M
        assert req.retention == "ephemeral"


# =============================================================================
# DataCoordinator — Phase 1
# =============================================================================

class TestCoordinatorPhase1:

    def test_fulfill_empty_returns_empty_dict(self, coordinator):
        result = coordinator.fulfill()
        assert result == {}

    def test_single_daily_request_dispatched(self, coordinator, fake_store):
        coordinator.register(_daily_req(["VOO", "TLT"], "pipeline.etf"))
        results = coordinator.fulfill()

        assert "pipeline.etf" in results
        assert not results["pipeline.etf"].empty
        assert len(fake_store.daily_calls) == 1
        assert set(fake_store.daily_calls[0]["tickers"]) == {"VOO", "TLT"}

    def test_daily_result_has_correct_tickers(self, coordinator):
        coordinator.register(_daily_req(["VOO", "TLT", "VGT"], "pipeline.etf"))
        results = coordinator.fulfill()
        df = results["pipeline.etf"]
        assert set(df.columns) == {"VOO", "TLT", "VGT"}

    def test_two_daily_requests_both_dispatched(self, coordinator, fake_store):
        coordinator.register(_daily_req(["VOO", "TLT"], "pipeline.etf"))
        coordinator.register(_daily_req(["MSFT", "AAPL"], "pipeline.stock"))
        results = coordinator.fulfill()

        assert "pipeline.etf" in results
        assert "pipeline.stock" in results
        assert len(fake_store.daily_calls) == 2

    def test_intraday_request_dispatched(self, coordinator, fake_store):
        coordinator.register(_intraday_req(["VOO", "TLT"], "scanner.intraday"))
        results = coordinator.fulfill()

        assert "scanner.intraday" in results
        assert not results["scanner.intraday"].empty
        assert len(fake_store.intraday_calls) == 1

    def test_mixed_daily_and_intraday(self, coordinator, fake_store):
        coordinator.register(_daily_req(["VOO"], "pipeline.etf"))
        coordinator.register(_intraday_req(["VOO"], "scanner.intraday"))
        results = coordinator.fulfill()

        assert len(results) == 2
        assert len(fake_store.daily_calls) == 1
        assert len(fake_store.intraday_calls) == 1

    def test_clear_empties_registry(self, coordinator, fake_store):
        coordinator.register(_daily_req(["VOO"], "pipeline.etf"))
        coordinator.clear()
        results = coordinator.fulfill()

        assert results == {}
        assert len(fake_store.daily_calls) == 0

    def test_fulfill_idempotent_after_clear(self, coordinator, fake_store):
        coordinator.register(_daily_req(["VOO"], "pipeline.etf"))
        coordinator.fulfill()
        coordinator.clear()
        coordinator.register(_daily_req(["TLT"], "pipeline.etf2"))
        results = coordinator.fulfill()

        assert "pipeline.etf2" in results
        assert len(fake_store.daily_calls) == 2   # one per fulfill

    def test_empty_store_result_returns_empty_df(self, coordinator):
        """If the store returns empty, coordinator returns empty DataFrame gracefully."""
        class EmptyStore(FakeStore):
            def fetch_daily(self, tickers, start, end):
                return pd.DataFrame()

        c = DataCoordinator(EmptyStore())
        c.register(_daily_req(["VOO"], "pipeline.etf"))
        results = c.fulfill()
        assert results["pipeline.etf"].empty

    def test_store_exception_returns_empty_df(self, coordinator):
        """If the store raises, coordinator catches it and returns empty DataFrame."""
        class BrokenStore(FakeStore):
            def fetch_daily(self, tickers, start, end):
                raise RuntimeError("disk error")

        c = DataCoordinator(BrokenStore())
        c.register(_daily_req(["VOO"], "pipeline.etf"))
        results = c.fulfill()
        assert results["pipeline.etf"].empty

    def test_register_invalid_frequency_raises_immediately(self, coordinator):
        """Validation fires at DataRequest construction, before register()."""
        with pytest.raises(ValueError, match="Unknown frequency"):
            DataRequest(
                tickers=("VOO",), start="2022-01-01", end="2022-02-01",
                frequency="bad_freq",
                retention="permanent", requester="test",
            )
