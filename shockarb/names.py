"""
shockarb.names — Ticker-to-company name and industry resolver.

Resolves ticker symbols to {"Name": ..., "Industry": ...} by searching a
waterfall of local reference files (CSV or Parquet), with a JSON disk cache
so each file is parsed at most once per process and each ticker looked up at
most once across runs.

Lookup order for each ticker
-----------------------------
  1. In-memory cache (populated from disk at construction).
     Accepted only if both Name and Industry are present and non-empty.
  2. Reference files in the order supplied (file_paths).
     First match wins; result is written to the in-memory cache.
  3. Fallback: Name = ticker symbol, Industry = "Unknown".
     Fallbacks are NOT written to the disk cache so a ticker absent today
     will be re-checked against the reference files on the next run
     (useful when new listings are added to the exchange CSVs).

Supported reference file formats
---------------------------------
  .csv     — must contain columns: Symbol, Name, Industry
  .parquet — must contain columns: Symbol, Name, Industry

Usage
-----
    from shockarb.names import TickerReferenceResolver

    resolver = TickerReferenceResolver(
        file_paths = ["data/nyse_1668526574444.csv",
                      "data/nasdaq_1668526380140.csv"],
        cache_path = "data/ticker_reference_cache.json",
    )

    # Resolve a batch — returns dict[ticker, {"Name":..., "Industry":...}]
    ref = resolver.get_reference(["AAPL", "EW", "CRM", "VOO"])
    print(ref["EW"]["Name"])          # Edwards Lifesciences
    print(ref["VOO"]["Industry"])     # ETF / Index Fund (if in cache)

    # Call again with overlapping tickers — files are NOT re-read
    ref2 = resolver.get_reference(["EW", "MSFT"])
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
from loguru import logger

__all__ = ["TickerReferenceResolver"]

# Sentinel Industry value used when a ticker is not found in any reference
# file.  Distinct from "Unknown" (which comes from NaN in the source CSV)
# so callers can tell the difference between "found but no industry listed"
# and "not found at all".
_FALLBACK_INDUSTRY = "Unknown"

# A cache entry is considered complete only if both fields are present and
# non-empty.  Incomplete entries are re-queried against the reference files.
def _is_complete(entry: dict) -> bool:
    return bool(entry.get("Name")) and bool(entry.get("Industry"))


class TickerReferenceResolver:
    """
    Map ticker symbols to company names and industries.

    Holds two layers of in-memory state so repeated calls within a process
    are cheap:
      _cache       — disk-backed JSON; persisted between runs.
      _loaded_dfs  — reference DataFrames; lives only for this process.

    Parameters
    ----------
    file_paths : list of str
        Reference files to search, in priority order (first match wins).
        Supports .csv and .parquet.  Each must contain Symbol, Name, Industry.
    cache_path : str
        Path to the JSON disk cache.  Created automatically if absent.
    """

    def __init__(
        self,
        file_paths: list[str],
        cache_path: str,
    ) -> None:
        self.file_paths = file_paths
        self.cache_path = cache_path
        self._cache: dict[str, dict]              = self._load_cache()
        self._loaded_dfs: dict[str, Optional[pd.DataFrame]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_reference(self, tickers: list[str]) -> dict[str, dict]:
        """
        Resolve a batch of tickers to {"Name": str, "Industry": str}.

        Always returns an entry for every input ticker.  Tickers not found
        anywhere fall back to {"Name": ticker, "Industry": "Unknown"};
        these fallbacks are not cached so they are retried next run.

        Parameters
        ----------
        tickers : ticker symbols to resolve (duplicates handled gracefully)

        Returns
        -------
        dict[ticker, {"Name": str, "Industry": str}]
        """
        result: dict[str, dict] = {}
        misses: list[str] = []

        # Step 1 — in-memory cache (populated from disk at construction).
        for ticker in dict.fromkeys(tickers):   # preserves order, drops dupes
            entry = self._cache.get(ticker, {})
            if _is_complete(entry):
                result[ticker] = entry
            else:
                misses.append(ticker)

        if not misses:
            return result

        # Step 2 — waterfall search through reference files.
        remaining = set(misses)
        for path in self.file_paths:
            if not remaining:
                break
            df = self._load_reference(path)
            if df is None:
                continue
            found = remaining & set(df.index)
            for ticker in found:
                entry = {
                    "Name":     str(df.at[ticker, "Name"]),
                    "Industry": str(df.at[ticker, "Industry"]),
                }
                self._cache[ticker] = entry
                result[ticker] = entry
            remaining -= found

        # Step 3 — fallback for tickers not found anywhere.
        # Not written to cache: absence from reference files may be temporary.
        for ticker in remaining:
            logger.debug(f"[TickerResolver] Not found in any reference: {ticker}")
            result[ticker] = {"Name": ticker, "Industry": _FALLBACK_INDUSTRY}

        self._persist_cache()
        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_reference(self, path: str) -> Optional[pd.DataFrame]:
        """
        Return the reference DataFrame for *path*, loading it on first access.

        Parsed DataFrames are memoised in _loaded_dfs so each file is read
        at most once per process.  Missing files are memoised as None so the
        filesystem is not re-checked on subsequent calls.
        """
        if path in self._loaded_dfs:
            return self._loaded_dfs[path]   # None means "already tried, not found"

        if not os.path.exists(path):
            logger.warning(f"[TickerResolver] Reference file not found: {path}")
            self._loaded_dfs[path] = None
            return None

        logger.info(f"[TickerResolver] Loading reference: {path}")
        try:
            cols = ["Symbol", "Name", "Industry"]
            df = (
                pd.read_csv(path,     usecols=cols, dtype=str) if path.endswith(".csv")     else
                pd.read_parquet(path, columns=cols)            if path.endswith(".parquet") else
                (_ for _ in ()).throw(ValueError(f"Unsupported format: {path}"))
            )
            df["Industry"] = df["Industry"].fillna(_FALLBACK_INDUSTRY)
            # drop_duplicates prevents .at[] returning a Series for repeated symbols
            df = df.drop_duplicates(subset=["Symbol"]).set_index("Symbol")
            logger.debug(f"[TickerResolver] Loaded {len(df):,} symbols from {os.path.basename(path)}")
        except Exception as exc:
            logger.warning(f"[TickerResolver] Could not read {path}: {exc}")
            df = None

        self._loaded_dfs[path] = df
        return df

    def _load_cache(self) -> dict:
        """Load the JSON disk cache; return empty dict on any failure."""
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning(f"[TickerResolver] Cache unreadable ({exc}) — starting fresh.")
            return {}

    def _persist_cache(self) -> None:
        """Write the in-memory cache back to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            logger.warning(f"[TickerResolver] Could not save cache: {exc}")
