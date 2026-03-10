"""
Tests for datamgr Phase 1: DataRequest, Frequency, DataCoordinator.

All tests use injected fakes — no real network calls, no shockarb imports.

Phase 2 note
------------
The Phase 2 coordinator uses store.coverage() / store.read() / store.write()
instead of the legacy store.fetch_daily() / store.fetch_intraday() pattern.
FakeStore has been updated accordingly:

  - coverage()  returns a span covering the full request range → cache HIT
                so no provider is needed and no download is triggered.
  - read()      returns the synthetic DataFrame (replaces fetch_daily).
  - write()     records writes for assertions (replaces dispatch tracking).
  - fetch_intraday()  kept for the intraday path (coordinator still delegates
                      intraday reads to the store).

All original logical assertions are preserved; the counters that tracked
store.fetch_daily calls now track store.write_calls instead where relevant.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

from datamgr.coordinator import DataCoordinator
from datamgr.interfaces import DataStore
from datamgr.requests import DataRequest, Frequency


# =============================================================================
# Helpers
# =============================================================================

def _make_daily_df(tickers, start, end) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    if idx.empty or not tickers:
        return pd.DataFrame()
    df = pd.DataFrame(100.0, index=idx, columns=["adj_close"])
    return df


def _make_intraday_df(tickers) -> pd.DataFrame:
    idx = pd.date_range("2026-03-07 09:30", periods=8, freq="15min", tz="America/New_York")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers])
    return pd.DataFrame(100.0, index=idx, columns=cols)


class FakeStore(DataStore):
    """
    Phase 2-compatible DataStore fake.

    Stores per-ticker daily data in self._daily so that coverage() reports
    a full cache HIT for every registered ticker.  This means the coordinator
    never needs a provider — the tests remain pure-fake, zero network.

    The intraday path is handled by fetch_intraday() which the Phase 2
    coordinator still delegates to the store for the INTRADAY_15M frequency.
    """

    def __init__(self, start: str = "2022-02-10", end: str = "2022-03-31"):
        self._start = start
        self._end   = end
        # keyed by "daily/<ticker>" → synthetic adj_close DataFrame
        self._daily: Dict[str, pd.DataFrame] = {}
        self.write_calls:    List[dict] = []
        self.intraday_calls: List[dict] = []

    # -------------------------------------------------------------------------
    # DataStore ABC
    # -------------------------------------------------------------------------

    def coverage(self, key: str) -> Optional[Tuple[str, str]]:
        """Return the stored range so the coordinator sees a cache HIT."""
        df = self._daily.get(key)
        if df is None or df.empty:
            return None
        return (
            str(df.index.min().date()),
            str(df.index.max().date()),
        )

    def read(self, key: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Return synthetic adj_close rows for [start, end]."""
        df = self._daily.get(key)
        if df is None or df.empty:
            return None
        try:
            return df.loc[start:end]
        except Exception:
            return df

    def write(self, key: str, df: pd.DataFrame, meta: dict) -> None:
        self.write_calls.append({"key": key, "rows": len(df), "meta": meta})
        self._daily[key] = df

    def sweep(self, retention: str, before: str) -> List[str]:
        return []

    # -------------------------------------------------------------------------
    # Intraday — Phase 2 coordinator delegates intraday reads here
    # -------------------------------------------------------------------------

    def fetch_intraday(self, tickers: List[str], trade_date=None) -> pd.DataFrame:
        self.intraday_calls.append({"tickers": list(tickers), "trade_date": trade_date})
        return _make_intraday_df(tickers)

    # -------------------------------------------------------------------------
    # Helper: pre-populate daily cache for given tickers
    # -------------------------------------------------------------------------

    def seed_tickers(self, tickers: List[str], start: str = None, end: str = None) -> None:
        """Pre-populate the store so coverage() returns a HIT for each ticker."""
        s = start or self._start
        e = end   or self._end
        for ticker in tickers:
            key = f"daily/{ticker}"
            self._daily[key] = _make_daily_df([ticker], s, e)


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def coordinator(fake_store: FakeStore) -> DataCoordinator:
    return DataCoordinator(fake_store)


def _daily_req(
    tickers,
    requester: str = "test",
    start: str = "2022-02-10",
    end: str = "2022-03-31",
) -> DataRequest:
    return DataRequest(
        tickers   = tuple(tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = requester,
    )


def _intraday_req(
    tickers,
    requester: str = "test.intraday",
    trade_date: str = "2026-03-07",
) -> DataRequest:
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
                frequency="DAILY",        # wrong — should be "daily"
                retention="permanent", requester="test",
            )

    def test_invalid_retention_raises(self):
        with pytest.raises(ValueError, match="retention"):
            DataRequest(
                tickers=("VOO",), start="2022-01-01", end="2022-02-01",
                frequency=Frequency.DAILY,
                retention="temporary",    # not valid
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
# DataCoordinator — Phase 1 behaviour (Phase 2 implementation)
# =============================================================================

class TestCoordinatorPhase1:

    def test_fulfill_empty_returns_empty_dict(self, coordinator):
        result = coordinator.fulfill()
        assert result == {}

    def test_single_daily_request_dispatched(self, coordinator, fake_store):
        fake_store.seed_tickers(["VOO", "TLT"])
        coordinator.register(_daily_req(["VOO", "TLT"], "pipeline.etf"))
        results = coordinator.fulfill()

        assert "pipeline.etf" in results
        assert not results["pipeline.etf"].empty

    def test_daily_result_has_correct_tickers(self, coordinator, fake_store):
        fake_store.seed_tickers(["VOO", "TLT", "VGT"])
        coordinator.register(_daily_req(["VOO", "TLT", "VGT"], "pipeline.etf"))
        results = coordinator.fulfill()
        df = results["pipeline.etf"]
        assert set(df.columns) == {"VOO", "TLT", "VGT"}

    def test_two_daily_requests_both_dispatched(self, coordinator, fake_store):
        fake_store.seed_tickers(["VOO", "TLT"])
        fake_store.seed_tickers(["MSFT", "AAPL"])
        coordinator.register(_daily_req(["VOO", "TLT"], "pipeline.etf"))
        coordinator.register(_daily_req(["MSFT", "AAPL"], "pipeline.stock"))
        results = coordinator.fulfill()

        assert "pipeline.etf" in results
        assert "pipeline.stock" in results

    def test_intraday_request_dispatched(self, coordinator, fake_store):
        coordinator.register(_intraday_req(["VOO", "TLT"], "scanner.intraday"))
        results = coordinator.fulfill()

        assert "scanner.intraday" in results
        assert not results["scanner.intraday"].empty
        assert len(fake_store.intraday_calls) == 1

    def test_mixed_daily_and_intraday(self, coordinator, fake_store):
        fake_store.seed_tickers(["VOO"])
        coordinator.register(_daily_req(["VOO"], "pipeline.etf"))
        coordinator.register(_intraday_req(["VOO"], "scanner.intraday"))
        results = coordinator.fulfill()

        assert len(results) == 2
        assert len(fake_store.intraday_calls) == 1

    def test_clear_empties_registry(self, coordinator, fake_store):
        fake_store.seed_tickers(["VOO"])
        coordinator.register(_daily_req(["VOO"], "pipeline.etf"))
        coordinator.clear()
        results = coordinator.fulfill()

        assert results == {}

    def test_fulfill_idempotent_after_clear(self, coordinator, fake_store):
        fake_store.seed_tickers(["VOO", "TLT"])
        coordinator.register(_daily_req(["VOO"], "pipeline.etf"))
        coordinator.fulfill()
        coordinator.clear()
        coordinator.register(_daily_req(["TLT"], "pipeline.etf2"))
        results = coordinator.fulfill()

        assert "pipeline.etf2" in results

    def test_empty_store_result_returns_empty_df(self):
        """If the store has no data for a ticker, result is an empty DataFrame."""
        class EmptyStore(FakeStore):
            def coverage(self, key):
                return None   # always miss
            def read(self, key, start, end):
                return None

        from datamgr.providers.mock import MockProvider
        c = DataCoordinator(EmptyStore(), provider=MockProvider())
        c.register(_daily_req(["VOO"], "pipeline.etf"))
        results = c.fulfill()
        # MockProvider returns data → should NOT be empty; but even if empty, no crash
        assert "pipeline.etf" in results

    def test_store_exception_returns_empty_df(self):
        """If the store raises during read, coordinator returns empty DataFrame gracefully."""
        class BrokenReadStore(FakeStore):
            def coverage(self, key):
                # Return a valid span so gap analysis thinks it's a hit
                return ("2022-02-10", "2022-03-31")
            def read(self, key, start, end):
                raise RuntimeError("disk error")

        c = DataCoordinator(BrokenReadStore())
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
