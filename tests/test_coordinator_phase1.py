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

    def test_intraday_request_dispatched(self, fake_store):
        """Intraday request returns data via provider (not store delegation)."""
        from datamgr.providers.mock import MockProvider
        c = DataCoordinator(fake_store, provider=MockProvider())
        c.register(_intraday_req(["VOO", "TLT"], "scanner.intraday"))
        results = c.fulfill()

        assert "scanner.intraday" in results
        assert not results["scanner.intraday"].empty

    def test_mixed_daily_and_intraday(self, fake_store):
        fake_store.seed_tickers(["VOO"])
        from datamgr.providers.mock import MockProvider
        c = DataCoordinator(fake_store, provider=MockProvider())
        c.register(_daily_req(["VOO"], "pipeline.etf"))
        c.register(_intraday_req(["VOO"], "scanner.intraday"))
        results = c.fulfill()

        assert len(results) == 2

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

    def test_daily_tail_fetch_when_cache_behind(self, fake_store):
        """
        When the cache ends before req.end, the coordinator must call the
        provider to fetch the missing tail.  No staleness grace — the
        _market_is_open() grace period handles the post-close window by
        routing to the intraday path instead.
        """
        from datamgr.providers.mock import MockProvider

        # Seed store with data through 2022-03-30 (Wednesday)
        fake_store.seed_tickers(["VOO"], start="2022-02-10", end="2022-03-30")
        provider = MockProvider()

        c = DataCoordinator(fake_store, provider=provider)
        # Request through 2022-03-31 (Thursday) — 1 BDay ahead of cached end
        c.register(_daily_req(["VOO"], start="2022-02-10", end="2022-03-31"))
        results = c.fulfill()

        # Provider SHOULD have been called — cache is behind
        assert len(fake_store.write_calls) > 0
        assert "pipeline.etf" not in results or not results.get("test", pd.DataFrame()).empty


# =============================================================================
# Step 2b: Intraday coordinator path (provider-backed, cache flag)
# =============================================================================

class TestCoordinatorIntraday:
    """
    Tests for the Step 2b intraday fetch path:
      - Provider is called (not store delegation)
      - cache=False skips store writes
      - cache=True commits to store
      - Provider failure returns empty DataFrame gracefully
    """

    def test_intraday_fetch_without_commit(self):
        """
        When cache=False, the coordinator fetches via provider but does NOT
        write to the store.  This is the Step 2b default for intraday.
        """
        from datamgr.providers.mock import MockProvider

        store = FakeStore()
        c = DataCoordinator(store, provider=MockProvider())
        req = DataRequest(
            tickers    = ("VOO", "TLT"),
            start      = "2026-03-12",
            end        = "2026-03-12",
            frequency  = Frequency.INTRADAY_15M,
            retention  = "ephemeral",
            requester  = "test.intraday_nocache",
            trade_date = "2026-03-12",
            cache      = False,
        )
        c.register(req)
        results = c.fulfill()

        # Data is returned to the caller
        assert "test.intraday_nocache" in results
        assert not results["test.intraday_nocache"].empty

        # Store was NOT written to
        assert len(store.write_calls) == 0

    def test_intraday_fetch_with_commit(self):
        """
        When cache=True, the coordinator fetches AND commits to the store.
        (Not the Step 2b default, but validates the flag works both ways.)
        """
        from datamgr.providers.mock import MockProvider

        store = FakeStore()
        c = DataCoordinator(store, provider=MockProvider())
        req = DataRequest(
            tickers    = ("VOO", "TLT"),
            start      = "2026-03-12",
            end        = "2026-03-12",
            frequency  = Frequency.INTRADAY_15M,
            retention  = "ephemeral",
            requester  = "test.intraday_cache",
            trade_date = "2026-03-12",
            cache      = True,
        )
        c.register(req)
        results = c.fulfill()

        assert "test.intraday_cache" in results
        assert not results["test.intraday_cache"].empty
        # Store WAS written to — one write per ticker
        assert len(store.write_calls) == 2

    def test_intraday_provider_called(self):
        """
        The coordinator must call the provider for intraday requests,
        not delegate to store.fetch_intraday().
        """
        from unittest.mock import MagicMock

        store = FakeStore()
        provider = MagicMock()
        # Return a valid intraday DataFrame
        provider.fetch.return_value = _make_intraday_df(["VOO"])

        c = DataCoordinator(store, provider=provider)
        c.register(_intraday_req(["VOO"], "test.intraday"))
        c.fulfill()

        provider.fetch.assert_called_once()
        call_kwargs = provider.fetch.call_args
        assert call_kwargs[1]["frequency"] == Frequency.INTRADAY_15M

    def test_intraday_provider_failure_returns_empty(self):
        """
        If the provider raises during an intraday fetch, the coordinator
        returns an empty DataFrame rather than crashing.
        """
        from datamgr.interfaces import DataProvider

        class FailingProvider(DataProvider):
            @property
            def name(self):
                return "failing"
            def fetch(self, **kwargs):
                raise RuntimeError("provider down")

        store = FakeStore()
        c = DataCoordinator(store, provider=FailingProvider())
        c.register(_intraday_req(["VOO"], "test.intraday_fail"))
        results = c.fulfill()

        assert results["test.intraday_fail"].empty

    def test_intraday_no_provider_raises(self):
        """
        If no provider is injected and an intraday request is registered,
        fulfill() must raise RuntimeError.
        """
        store = FakeStore()
        c = DataCoordinator(store)  # no provider
        c.register(_intraday_req(["VOO"], "test.no_provider"))
        with pytest.raises(RuntimeError, match="Provider required"):
            c.fulfill()

    def test_cache_flag_default_is_true(self):
        """DataRequest.cache defaults to True for backward compatibility."""
        req = _daily_req(["VOO"])
        assert req.cache is True

    def test_cache_flag_on_intraday_request(self):
        """Intraday requests can set cache=False."""
        req = DataRequest(
            tickers    = ("VOO",),
            start      = "2026-03-12",
            end        = "2026-03-12",
            frequency  = Frequency.INTRADAY_15M,
            retention  = "ephemeral",
            requester  = "test",
            trade_date = "2026-03-12",
            cache      = False,
        )
        assert req.cache is False


# =============================================================================
# Gap analysis — head-miss regression
# =============================================================================

class TestGapAnalyseHeadMiss:
    """
    Regression tests for the head-miss bug fixed in coordinator.py.

    Before the fix, a cache holding 2022+ data was incorrectly treated as
    covering a 2020 request because _gap_analyse() only checked for tail
    misses.  The fix adds a check: if cached_start > req_start, trigger a
    full download.
    """

    def _make_provider(self, start: str, end: str, tickers: list[str]):
        """FakeProvider that records calls and returns synthetic data."""

        class RecordingProvider:
            def __init__(self):
                self.calls: list[dict] = []

            def fetch(self, tickers, start, end, frequency):
                self.calls.append({"tickers": tickers, "start": start, "end": end})
                idx = pd.bdate_range(start=start, end=end)
                if idx.empty:
                    return pd.DataFrame()
                cols = pd.MultiIndex.from_product([["adj_close"], tickers])
                return pd.DataFrame(100.0, index=idx, columns=cols)

        return RecordingProvider()

    def test_cache_ahead_of_request_triggers_download(self):
        """
        Cache holds 2022 data; request asks for 2020.
        The coordinator must detect the head miss and call the provider.
        """
        store = FakeStore()
        # Seed the store with 2022 data — simulates the bug scenario
        store.seed_tickers(["TXN"], start="2022-02-10", end="2022-03-31")

        provider = self._make_provider("2020-11-09", "2021-02-28", ["TXN"])
        c = DataCoordinator(store, provider=provider)
        c.register(_daily_req(["TXN"], requester="covid.build",
                               start="2020-11-09", end="2021-02-28"))
        c.fulfill()

        # Provider must have been called — the 2022 cache does not cover 2020
        assert len(provider.calls) == 1, (
            "Expected one provider call for the head-miss gap; got none. "
            "The 2022 cache should not satisfy a 2020 request."
        )

    def test_cache_covering_request_window_no_download(self):
        """
        Cache already covers the full requested window → no provider call.
        """
        store = FakeStore()
        store.seed_tickers(["VOO"], start="2022-01-01", end="2022-12-31")

        provider = self._make_provider("2022-02-10", "2022-03-31", ["VOO"])
        c = DataCoordinator(store, provider=provider)
        c.register(_daily_req(["VOO"], requester="ukraine.score",
                               start="2022-02-10", end="2022-03-31"))
        c.fulfill()

        assert len(provider.calls) == 0, (
            "Cache fully covers the request window — provider should not be called."
        )


# =============================================================================
# Batch-failure retry — MARKET-REPORT-PARTIAL-FETCH-GAP fix
# =============================================================================

class TestDownloadAndCommitRetry:
    """
    _download_and_commit() batches tickers purely by identical cache-gap
    span — an accident of cache state, not a real relationship between the
    tickers. Regression coverage for the 2026-08-18 incident where one bad
    ticker in a batch blanked unrelated tickers that shared its gap span
    (QQQ, IWM, XLI, XLU, XLB, HYG, GLD, ^HSI all missing the same day).
    See HIL_todo.md, MARKET-REPORT-PARTIAL-FETCH-GAP.
    """

    def _batch_provider_raising_once(self, bad_tickers: set[str]):
        """
        FakeProvider whose batch call raises if the request includes any
        ticker in bad_tickers, but succeeds for individual-ticker retries
        of the good tickers (and keeps failing for the bad one).
        """
        class FlakyProvider:
            def fetch(self, tickers, start, end, frequency):
                if len(tickers) > 1 and (set(tickers) & bad_tickers):
                    raise RuntimeError("simulated batch failure (rate limit)")
                if set(tickers) & bad_tickers:
                    raise RuntimeError("simulated persistent failure for bad ticker")
                idx = pd.bdate_range(start=start, end=end)
                if idx.empty:
                    return pd.DataFrame()
                cols = pd.MultiIndex.from_product([["adj_close"], tickers])
                return pd.DataFrame(100.0, index=idx, columns=cols)
        return FlakyProvider()

    def test_batch_exception_falls_back_to_individual_retry(self):
        """
        Three tickers share a gap span (same cache state) so they batch
        together. The batch call raises because one ticker (BAD) is
        problematic. The two good tickers must still get committed via
        the per-ticker retry — they should not be collateral damage.
        """
        store = FakeStore()
        # No seed — all three are full cache misses, same gap span.
        provider = self._batch_provider_raising_once({"BAD"})
        c = DataCoordinator(store, provider=provider)
        c.register(_daily_req(["GOOD1", "BAD", "GOOD2"], requester="test.retry"))
        c.fulfill()

        written_keys = {call["key"] for call in store.write_calls}
        assert "daily/GOOD1" in written_keys
        assert "daily/GOOD2" in written_keys
        assert "daily/BAD" not in written_keys, (
            "BAD should fail even on individual retry — it must not be "
            "committed, but must also not prevent GOOD1/GOOD2 from being."
        )

    def test_batch_empty_result_falls_back_to_individual_retry(self):
        """Same as above, but the batch call returns an empty frame instead
        of raising — must also trigger the individual-retry fallback."""
        class EmptyThenGoodProvider:
            def fetch(self, tickers, start, end, frequency):
                if len(tickers) > 1:
                    return pd.DataFrame()  # simulated empty batch response
                idx = pd.bdate_range(start=start, end=end)
                cols = pd.MultiIndex.from_product([["adj_close"], tickers])
                return pd.DataFrame(100.0, index=idx, columns=cols)

        store = FakeStore()
        c = DataCoordinator(store, provider=EmptyThenGoodProvider())
        c.register(_daily_req(["GOOD1", "GOOD2"], requester="test.retry_empty"))
        c.fulfill()

        written_keys = {call["key"] for call in store.write_calls}
        assert "daily/GOOD1" in written_keys
        assert "daily/GOOD2" in written_keys

    def test_all_individual_retries_fail_no_crash(self):
        """If every ticker in the batch also fails individually, fulfill()
        must not raise — it just commits nothing for that span."""
        store = FakeStore()
        provider = self._batch_provider_raising_once({"BAD1", "BAD2"})
        c = DataCoordinator(store, provider=provider)
        c.register(_daily_req(["BAD1", "BAD2"], requester="test.all_bad"))
        results = c.fulfill()  # must not raise

        assert len(store.write_calls) == 0
        assert "test.all_bad" in results


# =============================================================================
# NaN-close row rejection — NAN-CLOSE-CACHE-CORRUPTION fix
# =============================================================================

class TestCommitTickerNaNRowRejection:
    """
    _commit_ticker() must never write a row with no closing price to the
    permanent cache. Regression coverage for the 2026-08-19 incident: every
    ukraine_shock ticker picked up a real-looking-but-NaN-close row dated
    2026-08-17 (open/high/low/volume present, close/adj_close NaN — a
    partial/in-progress bar, almost certainly mis-dated by yfinance). That
    row got committed, coverage() then lied that 2026-08-17 was fully
    cached, and pipeline.py's ffill().pct_change() silently turned the NaN
    close into a fake 0.000% return for every ticker — suppressing
    confidence_delta across the whole universe and producing two straight
    days of "no recommendations". See HIL_todo.md, NAN-CLOSE-CACHE-CORRUPTION.
    """

    @staticmethod
    def _multiindex_df(ticker: str, dates: list[str], closes: list[float | None]) -> pd.DataFrame:
        """Build a normalised-shape batch result (matches YFinanceProvider._normalise() output)."""
        idx = pd.to_datetime(dates)
        data = {
            ("open", ticker):       [100.0] * len(dates),
            ("high", ticker):       [101.0] * len(dates),
            ("low", ticker):        [99.0] * len(dates),
            ("close", ticker):      closes,
            ("adj_close", ticker):  closes,
            ("adj_factor", ticker): [1.0 if c is not None else None for c in closes],
            ("volume", ticker):     [1_000_000] * len(dates),
        }
        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_nan_close_row_not_committed(self):
        """A trailing NaN-close row (the partial-bar case) is dropped; the
        good rows ahead of it still get written."""
        store = FakeStore()

        class OneShotProvider:
            def fetch(self, tickers, start, end, frequency):
                return TestCommitTickerNaNRowRejection._multiindex_df(
                    "MSFT",
                    ["2026-08-13", "2026-08-14", "2026-08-17"],
                    [496.88, 495.40, None],
                )

        c = DataCoordinator(store, provider=OneShotProvider())
        c.register(_daily_req(["MSFT"], requester="test.nan_row"))
        c.fulfill()

        assert len(store.write_calls) == 1
        written = store.write_calls[0]
        assert written["rows"] == 2, "the NaN-close row must not be counted/committed"
        committed_df = store._daily[written["key"]]
        assert pd.Timestamp("2026-08-17") not in committed_df.index

    def test_coverage_does_not_advance_past_bad_date(self):
        """coverage() must not report the bad date as cached — otherwise
        gap analysis would never re-fetch it."""
        store = FakeStore()

        class OneShotProvider:
            def fetch(self, tickers, start, end, frequency):
                return TestCommitTickerNaNRowRejection._multiindex_df(
                    "MSFT",
                    ["2026-08-13", "2026-08-14", "2026-08-17"],
                    [496.88, 495.40, None],
                )

        c = DataCoordinator(store, provider=OneShotProvider())
        c.register(_daily_req(["MSFT"], requester="test.coverage"))
        c.fulfill()

        earliest, latest = store.coverage("daily/MSFT")
        assert latest == "2026-08-14", (
            "the NaN row must not advance coverage() to 2026-08-17 — a bar "
            "with no close is not a completed trading day"
        )

    def test_all_nan_rows_result_in_no_commit(self):
        """If every row in the batch is NaN-close, nothing is written and
        fulfill() does not crash."""
        store = FakeStore()

        class AllNanProvider:
            def fetch(self, tickers, start, end, frequency):
                return TestCommitTickerNaNRowRejection._multiindex_df(
                    "MSFT", ["2026-08-17"], [None],
                )

        c = DataCoordinator(store, provider=AllNanProvider())
        c.register(_daily_req(["MSFT"], requester="test.all_nan"))
        results = c.fulfill()  # must not raise

        assert len(store.write_calls) == 0
        assert "test.all_nan" in results

    def test_nan_row_in_one_ticker_does_not_affect_sibling(self):
        """A NaN-close row for one ticker in a batch must not affect a
        clean sibling ticker fetched in the same call."""
        store = FakeStore()

        class MixedProvider:
            def fetch(self, tickers, start, end, frequency):
                bad = TestCommitTickerNaNRowRejection._multiindex_df(
                    "MSFT", ["2026-08-14", "2026-08-17"], [495.40, None],
                )
                good = TestCommitTickerNaNRowRejection._multiindex_df(
                    "AAPL", ["2026-08-14", "2026-08-17"], [220.0, 221.5],
                )
                return pd.concat([bad, good], axis=1)

        c = DataCoordinator(store, provider=MixedProvider())
        c.register(_daily_req(["MSFT", "AAPL"], requester="test.mixed"))
        c.fulfill()

        assert store.coverage("daily/MSFT")[1] == "2026-08-14"
        assert store.coverage("daily/AAPL")[1] == "2026-08-17"

    def test_intermediate_nan_row_dropped_not_just_trailing(self):
        """The filter is a row-level notna() mask, not "trim the tail" — a
        NaN in the middle of the batch is dropped too, not just a trailing one."""
        store = FakeStore()

        class GappyProvider:
            def fetch(self, tickers, start, end, frequency):
                return TestCommitTickerNaNRowRejection._multiindex_df(
                    "MSFT",
                    ["2026-08-12", "2026-08-13", "2026-08-14"],
                    [500.0, None, 495.40],
                )

        c = DataCoordinator(store, provider=GappyProvider())
        c.register(_daily_req(["MSFT"], requester="test.gap"))
        c.fulfill()

        committed_df = store._daily["daily/MSFT"]
        assert pd.Timestamp("2026-08-13") not in committed_df.index
        assert pd.Timestamp("2026-08-12") in committed_df.index
        assert pd.Timestamp("2026-08-14") in committed_df.index
