"""
Unit tests for shockarb.cache — CacheManager.

All network calls are replaced with a deterministic mock downloader
injected via CacheManager's constructor parameter. These tests are
fully offline and safe to run in CI with no external dependencies.

Coverage areas
--------------
  TestCacheManagerInit        — directory creation, custom downloader wiring
  TestFetchOHLCV              — fresh download, cache hit, ticker merge
  TestExtractAdjClose         — MultiIndex extraction, Close fallback
  TestMissingBusinessDates    — static method date-gap logic
  TestBackupAndMetadata       — backup file creation, metadata JSON contents
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from shockarb.cache import CacheManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Construct a minimal MultiIndex OHLCV DataFrame matching yfinance output."""
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_tuples([(f, t) for f in fields for t in tickers])
    data = np.random.rand(len(dates), len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _mock_dl(tickers, start, end, **kwargs):
    tickers = tickers if isinstance(tickers, list) else [tickers]
    dates = pd.bdate_range(start=start, end=end)
    return _make_ohlcv(tickers, dates) if len(dates) > 0 else pd.DataFrame()


@pytest.fixture
def cache_root():
    d = tempfile.mkdtemp(prefix="shockarb_cache_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mgr(cache_root):
    return CacheManager(
        cache_dir=os.path.join(cache_root, "cache"),
        backup_dir=os.path.join(cache_root, "backups"),
        downloader=_mock_dl,
    )


# =============================================================================
# Initialisation
# =============================================================================

class TestCacheManagerInit:

    def test_creates_cache_dir_on_construction(self, cache_root):
        target = os.path.join(cache_root, "nested", "cache")
        CacheManager(cache_dir=target, downloader=_mock_dl)
        assert os.path.isdir(target)

    def test_custom_downloader_is_called(self, cache_root):
        sentinel = MagicMock(return_value=pd.DataFrame())
        mgr = CacheManager(cache_dir=os.path.join(cache_root, "c"), downloader=sentinel)
        mgr.fetch_ohlcv(["SPY"], "2022-01-03", "2022-01-10", "test")
        assert sentinel.called


# =============================================================================
# fetch_ohlcv
# =============================================================================

class TestFetchOHLCV:

    def test_fresh_download_returns_nonempty(self, mgr):
        result = mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "etf")
        assert result is not None
        assert not result.empty

    def test_cache_hit_skips_download(self, mgr):
        mgr.fetch_ohlcv(["VOO"], "2022-02-10", "2022-03-31", "etf")
        # Replace downloader with a sentinel that returns empty (would fail the call)
        mgr._downloader = MagicMock(return_value=pd.DataFrame())
        mgr.fetch_ohlcv(["VOO"], "2022-02-10", "2022-03-31", "etf")
        assert not mgr._downloader.called

    def test_new_ticker_added_triggers_merge(self, mgr):
        mgr.fetch_ohlcv(["VOO"], "2022-02-10", "2022-03-31", "etf")
        result = mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "etf")
        tickers = set(result.columns.get_level_values(1))
        assert {"VOO", "TLT"}.issubset(tickers)

    def test_result_is_multiindex(self, mgr):
        result = mgr.fetch_ohlcv(["VOO"], "2022-02-10", "2022-03-31", "etf")
        assert isinstance(result.columns, pd.MultiIndex)

    def test_empty_downloader_returns_none_or_empty(self, cache_root):
        mgr = CacheManager(
            cache_dir=os.path.join(cache_root, "empty"),
            downloader=lambda *a, **k: pd.DataFrame(),
        )
        result = mgr.fetch_ohlcv(["SPY"], "2022-02-10", "2022-03-31", "etf")
        assert result is None or (result is not None and result.empty)


# =============================================================================
# extract_adj_close
# =============================================================================

class TestExtractAdjClose:

    def test_extracts_to_flat_dataframe(self, mgr):
        ohlcv = mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "etf")
        prices = mgr.extract_adj_close(ohlcv)
        assert not isinstance(prices.columns, pd.MultiIndex)
        assert "VOO" in prices.columns
        assert "TLT" in prices.columns

    def test_empty_input_returns_empty(self, mgr):
        assert mgr.extract_adj_close(pd.DataFrame()).empty

    def test_non_multiindex_input_passthrough(self, mgr):
        plain = pd.DataFrame({"A": [1.0, 2.0]})
        result = mgr.extract_adj_close(plain)
        pd.testing.assert_frame_equal(result, plain)

    def test_fallback_to_close_when_adj_close_absent(self, mgr):
        dates = pd.bdate_range("2022-02-10", periods=5)
        cols = pd.MultiIndex.from_tuples([("Close", "VOO"), ("Volume", "VOO")])
        df = pd.DataFrame(np.random.rand(5, 2), index=dates, columns=cols)
        result = mgr.extract_adj_close(df)
        assert "VOO" in result.columns


# =============================================================================
# _missing_business_dates (static helper)
# =============================================================================

class TestMissingBusinessDates:

    def test_empty_cache_returns_full_requested_range(self):
        result = CacheManager._missing_business_dates(
            pd.DatetimeIndex([]), "2022-02-10", "2022-03-31"
        )
        assert len(result) > 0

    def test_full_coverage_returns_empty(self):
        dates = pd.bdate_range("2022-02-10", "2022-03-31")
        result = CacheManager._missing_business_dates(dates, "2022-02-15", "2022-03-25")
        assert len(result) == 0

    def test_prior_gap_is_detected(self):
        dates = pd.bdate_range("2022-02-20", "2022-03-31")
        result = CacheManager._missing_business_dates(dates, "2022-02-10", "2022-03-31")
        assert len(result) > 0
        assert result.min() < pd.Timestamp("2022-02-20")


# =============================================================================
# Backup and metadata
# =============================================================================

class TestBackupAndMetadata:

    def test_metadata_json_created_after_fetch(self, mgr, cache_root):
        mgr.fetch_ohlcv(["VOO"], "2022-02-10", "2022-03-31", "meta_test")
        meta_path = os.path.join(cache_root, "cache", "cache_metadata.json")
        assert os.path.exists(meta_path)

    def test_metadata_contains_expected_fields(self, mgr, cache_root):
        mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "meta_test")
        meta_path = os.path.join(cache_root, "cache", "cache_metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        entry = next(iter(meta.values()))
        assert "tickers" in entry
        assert "n_rows" in entry
        assert "date_range" in entry

    def test_backup_created_when_cache_mutated(self, mgr, cache_root):
        mgr.fetch_ohlcv(["VOO"], "2022-02-10", "2022-03-31", "bkp_test")
        mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "bkp_test")
        backup_dir = os.path.join(cache_root, "backups")
        assert os.path.isdir(backup_dir)
        backups = [f for f in os.listdir(backup_dir) if f.endswith(".parquet")]
        assert len(backups) >= 1

    def test_get_cache_info_returns_empty_before_any_fetch(self, mgr):
        assert mgr.get_cache_info() == {}
