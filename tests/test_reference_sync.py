"""
Tests for shockarb.reference_sync — syncing NYSE/NASDAQ reference CSVs from
LondonMarket/Global-Stock-Symbols.

Coverage
--------
  TestSyncOne   — _sync_one() merge rule: update existing, add new, leave
                  local-only rows untouched, clear cache for changed symbols
  TestNetworkFailures — graceful no-op on fetch/parse/read errors
  TestClearCacheEntries — shockarb.names.clear_cache_entries()
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shockarb.names import clear_cache_entries
from shockarb.reference_sync import _sync_one, sync_reference_data

_LOCAL_CSV = (
    "Symbol,Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry\n"
    "V,Visa Inc.,$213.015,6.155,2.975%,448734907151,United States,,1424746,Consumer Discretionary,Business Services\n"
    "ZZZ,Zzz Old Name,$1.00,0.0,0.0%,1000,United States,2000,100,Old Sector,Old Industry\n"
)


def _fake_response(text: str, status: int = 200):
    resp = SimpleNamespace(text=text, status_code=status)
    resp.raise_for_status = (lambda: None) if status == 200 else (lambda: (_ for _ in ()).throw(RuntimeError("HTTP error")))
    return resp


# =============================================================================
# TestSyncOne
# =============================================================================

class TestSyncOne:
    def _write_local(self, tmp_path, content=_LOCAL_CSV):
        local = tmp_path / "nyse_1668526574444.csv"
        local.write_text(content, encoding="utf-8")
        cache = tmp_path / "ticker_reference_cache.json"
        cache.write_text(json.dumps({
            "V":   {"Name": "Visa Inc.", "Industry": "Business Services"},
            "ZZZ": {"Name": "Zzz Old Name", "Industry": "Old Industry"},
        }), encoding="utf-8")
        return str(local), str(cache)

    def test_updates_existing_row(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)

        # Remote has a changed Sector/Industry for V, ZZZ untouched.
        remote_csv = (
            "Symbol,Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry\n"
            "V,Visa Inc.,$214.00,1.0,0.5%,449000000000,United States,,1500000,Financials,Credit Services\n"
            "ZZZ,Zzz Old Name,$1.00,0.0,0.0%,1000,United States,2000,100,Old Sector,Old Industry\n"
        )
        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response(remote_csv))

        result = _sync_one(str(tmp_path), "nyse_1668526574444.csv", cache_path, timeout=5)

        assert result.updated == 1
        assert result.added == 0
        assert result.total == 2

        new_content = open(local_path, encoding="utf-8").read()
        assert "Financials,Credit Services" in new_content
        assert "Consumer Discretionary,Business Services" not in new_content

        cache = json.loads(open(cache_path, encoding="utf-8").read())
        assert "V" not in cache          # cleared — will be re-resolved
        assert "ZZZ" in cache            # unchanged row, cache entry kept

    def test_adds_new_symbol(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)

        remote_csv = _LOCAL_CSV + (
            "NEW,New Co.,$5.00,0.1,2.0%,5000,United States,2026,200,Tech,Software\n"
        )
        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response(remote_csv))

        result = _sync_one(str(tmp_path), "nyse_1668526574444.csv", cache_path, timeout=5)

        assert result.added == 1
        assert result.updated == 0
        assert result.total == 3
        assert "NEW,New Co." in open(local_path, encoding="utf-8").read()

    def test_local_only_symbol_untouched(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)

        # Remote omits ZZZ entirely.
        remote_csv = (
            "Symbol,Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry\n"
            "V,Visa Inc.,$213.015,6.155,2.975%,448734907151,United States,,1424746,Consumer Discretionary,Business Services\n"
        )
        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response(remote_csv))

        result = _sync_one(str(tmp_path), "nyse_1668526574444.csv", cache_path, timeout=5)

        assert result.updated == 0
        assert result.added == 0
        assert result.total == 2
        assert "ZZZ,Zzz Old Name" in open(local_path, encoding="utf-8").read()

    def test_no_changes_does_not_rewrite_or_clear_cache(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)
        before = os.path.getmtime(local_path)

        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response(_LOCAL_CSV))

        result = _sync_one(str(tmp_path), "nyse_1668526574444.csv", cache_path, timeout=5)

        assert result.updated == 0
        assert result.added == 0
        assert os.path.getmtime(local_path) == before

        cache = json.loads(open(cache_path, encoding="utf-8").read())
        assert "V" in cache and "ZZZ" in cache


# =============================================================================
# TestNetworkFailures
# =============================================================================

class TestNetworkFailures:
    def _write_local(self, tmp_path):
        local = tmp_path / "nyse_1668526574444.csv"
        local.write_text(_LOCAL_CSV, encoding="utf-8")
        cache = tmp_path / "ticker_reference_cache.json"
        cache.write_text("{}", encoding="utf-8")
        return str(local), str(cache)

    def test_fetch_error_returns_none(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)

        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr("shockarb.reference_sync.requests.get", boom)

        result = _sync_one(str(tmp_path), "nyse_1668526574444.csv", cache_path, timeout=5)
        assert result is None
        # Local file untouched.
        assert open(local_path, encoding="utf-8").read() == _LOCAL_CSV

    def test_http_error_returns_none(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)

        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response("", status=500))

        result = _sync_one(str(tmp_path), "nyse_1668526574444.csv", cache_path, timeout=5)
        assert result is None

    def test_missing_local_file_returns_none(self, tmp_path, monkeypatch):
        cache_path = str(tmp_path / "ticker_reference_cache.json")
        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response(_LOCAL_CSV))

        result = _sync_one(str(tmp_path), "does_not_exist.csv", cache_path, timeout=5)
        assert result is None

    def test_sync_reference_data_skips_failed_files(self, tmp_path, monkeypatch):
        local_path, cache_path = self._write_local(tmp_path)

        monkeypatch.setattr("shockarb.reference_sync.requests.get",
                             lambda *a, **k: _fake_response("", status=500))

        stats = sync_reference_data(
            data_dir=str(tmp_path),
            files=["nyse_1668526574444.csv"],
            cache_path=cache_path,
        )
        assert stats == {}


# =============================================================================
# TestClearCacheEntries
# =============================================================================

class TestClearCacheEntries:
    def test_removes_listed_tickers(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({
            "V": {"Name": "Visa Inc.", "Industry": "Business Services"},
            "MSFT": {"Name": "Microsoft Corporation", "Industry": "Software"},
        }), encoding="utf-8")

        removed = clear_cache_entries(str(cache_path), ["V", "NOTPRESENT"])

        assert removed == 1
        remaining = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "V" not in remaining
        assert "MSFT" in remaining

    def test_missing_cache_file_returns_zero(self, tmp_path):
        removed = clear_cache_entries(str(tmp_path / "nope.json"), ["V"])
        assert removed == 0

    def test_no_op_when_nothing_to_remove(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"V": {"Name": "Visa Inc.", "Industry": "x"}}), encoding="utf-8")
        before = cache_path.read_text(encoding="utf-8")

        removed = clear_cache_entries(str(cache_path), ["MSFT"])

        assert removed == 0
        assert cache_path.read_text(encoding="utf-8") == before
