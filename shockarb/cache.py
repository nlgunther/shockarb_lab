"""
CacheManager: Intelligent parquet-based caching for yfinance data.

Architecture Overview
---------------------
Priority 1: Basic smart caching (OHLCV) with auto-extraction of 'Adj Close'.
Priority 2: Ticker merging (Append new tickers to existing cached date ranges).
Priority 3: Date extension (Append new dates to the existing ticker universe).
Priority 4: Backup & recovery (Timestamped atomic backups before mutation).
Priority 5: Metadata tracking (JSON sidecar for cache inspection).

Usage
-----
    from shockarb.cache import CacheManager
    
    mgr = CacheManager(cache_dir="data/cache", backup_dir="data/backups")
    
    # Downloads only what isn't already cached
    ohlcv = mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "us_etf")
    adj_close = mgr.extract_adj_close(ohlcv)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import yfinance as yf
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
BACKUP_MAX_AGE_DAYS = 7


class CacheManager:
    """Manages a local parquet cache of full yfinance OHLCV data."""

    def __init__(self, cache_dir: str, backup_dir: Optional[str] = None, downloader=None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.backup_dir = Path(backup_dir) if backup_dir else self.cache_dir.parent / "backups"
        self._downloader = downloader if downloader is not None else yf.download

    # =========================================================================
    # Public API
    # =========================================================================

    def fetch_ohlcv(
        self,
        tickers: List[str],
        start: str,
        end: str,
        cache_name: str,
    ) -> pd.DataFrame:
        """
        Return full OHLCV data, intelligently downloading only missing data.
        """
        cache_path = self._cache_path(cache_name)
        existing = self._load_cache(cache_path)

        if existing is not None:
            cached_tickers = self._cached_tickers(existing)
            missing_tickers = set(tickers) - cached_tickers
            missing_dates = self._missing_business_dates(existing.index, start, end)

            # 1. Perfect Hit
            if not missing_tickers and missing_dates.empty:
                logger.info(f"[Cache HIT] {cache_name}: {len(tickers)} tickers")
                return self._slice(existing, tickers, start, end)

            # 2. Partial Miss (Sequence the updates in memory)
            cache_modified = False

            if not missing_dates.empty:
                logger.info(f"[Cache MISS] {cache_name}: Extending date range.")
                existing = self._extend_dates_in_memory(existing, missing_dates)
                cache_modified = True

            if existing is not None and missing_tickers:
                logger.info(f"[Cache MISS] {cache_name}: Merging {len(missing_tickers)} new tickers.")
                existing = self._merge_tickers_in_memory(existing, missing_tickers)
                cache_modified = True

            # 3. Save exactly once if modified
            if cache_modified and existing is not None:
                self._p4_backup(cache_path)
                self._save_cache(existing, cache_path, cache_name)
                return self._slice(existing, tickers, start, end)

        # 4. Total Miss (or cache corruption fallback)
        logger.info(f"[Cache MISS] {cache_name}: Full download initialized.")
        raw = self._download(tickers, start, end)
        
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = self._normalize_columns(raw)
        self._save_cache(raw, cache_path, cache_name)
        return self._slice(raw, tickers, start, end)

    def extract_adj_close(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Extract 'Adj Close' prices from a full OHLCV MultiIndex DataFrame."""
        if ohlcv.empty:
            return pd.DataFrame()

        if not isinstance(ohlcv.columns, pd.MultiIndex):
            return ohlcv

        if "Adj Close" in ohlcv.columns.get_level_values(0):
            return ohlcv["Adj Close"]
        elif "Close" in ohlcv.columns.get_level_values(0):
            logger.warning("No 'Adj Close' in cache; falling back to 'Close'")
            return ohlcv["Close"]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in cached fields.")

    # =========================================================================
    # Core Data Patching (In-Memory)
    # =========================================================================

    def _merge_tickers_in_memory(
        self, existing: pd.DataFrame, missing_tickers: Set[str]
    ) -> Optional[pd.DataFrame]:
        """Downloads missing tickers for the existing date range and merges them."""
        start_dt = str(existing.index.min().date())
        end_dt = str(existing.index.max().date())

        new_data = self._download(list(missing_tickers), start_dt, end_dt)
        if new_data is None or new_data.empty:
            return None

        return self._merge_ohlcv(existing, self._normalize_columns(new_data))

    def _extend_dates_in_memory(
        self, existing: pd.DataFrame, missing_dates: pd.DatetimeIndex
    ) -> Optional[pd.DataFrame]:
        """Downloads missing dates for the existing tickers and appends them."""
        ext_start = str(missing_dates.min().date())
        ext_end = str(missing_dates.max().date() + pd.offsets.BDay(1))
        
        new_data = self._download(list(self._cached_tickers(existing)), ext_start, ext_end)
        if new_data is None or new_data.empty:
            return None

        new_data = self._normalize_columns(new_data)
        merged = pd.concat([existing, new_data], axis=0)
        
        # Deduplicate on index and sort
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return self._normalize_columns(merged)

    # =========================================================================
    # Internal I/O & Validation
    # =========================================================================

    def _cache_path(self, cache_name: str) -> Path:
        return self.cache_dir / f"{cache_name}_ohlcv.parquet"

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return df if not df.empty else None
        except Exception as exc:
            logger.warning(f"Cache read failed ({path}): {exc} — will re-download")
            return None

    def _save_cache(self, df: pd.DataFrame, path: Path, cache_name: str) -> None:
        """Atomically write DataFrame to parquet using temp files to prevent corruption."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".parquet")
        
        try:
            os.close(tmp_fd)
            df.to_parquet(tmp_path)
            shutil.move(tmp_path, path)
            logger.info(f"Cache saved: {path.name} ({len(df)} rows)")
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.error(f"Cache save failed: {exc}")
            raise

        self._p5_update_metadata(df, path, cache_name)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or isinstance(df.columns, pd.MultiIndex):
            return df.sort_index(axis=1) if not df.empty else df
            
        df.columns = pd.MultiIndex.from_tuples([("Close", col) for col in df.columns])
        return df.sort_index(axis=1)

    @staticmethod
    def _cached_tickers(df: pd.DataFrame) -> Set[str]:
        return set(df.columns.get_level_values(1).unique() if isinstance(df.columns, pd.MultiIndex) else df.columns)

    @staticmethod
    def _missing_business_dates(cached_dates: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
        """Safely calculates required date extensions using modern pandas `.union()`."""
        if cached_dates.empty:
            return pd.bdate_range(start=start, end=end, inclusive="left")

        cache_start, cache_end = cached_dates.min(), cached_dates.max()
        req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)
        
        missing = []
        if req_start < cache_start:
            prior = pd.bdate_range(start=req_start, end=cache_start, inclusive="left")
            if not prior.empty: missing.append(prior)

        if req_end >= pd.Timestamp.now().normalize() - pd.Timedelta(days=30):
            after = pd.bdate_range(start=cache_end, end=req_end, inclusive="right")
            if len(after) >= 2: missing.append(after)

        if not missing:
            return pd.DatetimeIndex([])
            
        # Safely combine multiple indices without using deprecated .append()
        idx = missing[0]
        for m in missing[1:]:
            idx = idx.union(m)
        return idx

    @staticmethod
    def _slice(df: pd.DataFrame, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        result = df.loc[start:end]
        if isinstance(result.columns, pd.MultiIndex):
            wanted = [t for t in tickers if t in result.columns.get_level_values(1)]
            result = result.loc[:, result.columns.get_level_values(1).isin(wanted)]
        return result

    def _download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        logger.info(f"API Download: {len(tickers)} tickers, {start} → {end}")
        try:
            raw = self._downloader(tickers, start=start, end=end, auto_adjust=False, progress=False)
            return raw if not raw.empty else None
        except Exception as exc:
            logger.error(f"yfinance exception: {exc}")
            return None

    @staticmethod
    def _merge_ohlcv(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        merged = pd.concat([existing, new_data], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
        return merged.sort_index(axis=1).sort_index(axis=0)

    # =========================================================================
    # Backup & Metadata (P4 & P5)
    # =========================================================================

    def _p4_backup(self, cache_path: Path) -> None:
        if not cache_path.exists():
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{cache_path.stem}_{ts}{cache_path.suffix}"
        
        shutil.copy2(cache_path, backup_path)
        
        # Prune old backups
        cutoff = datetime.now() - timedelta(days=BACKUP_MAX_AGE_DAYS)
        for backup_file in self.backup_dir.glob("*.parquet"):
            try:
                parts = backup_file.stem.rsplit("_", 2)
                file_dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
                if file_dt < cutoff:
                    backup_file.unlink()
            except (ValueError, IndexError):
                pass

    def _p5_update_metadata(self, df: pd.DataFrame, cache_path: Path, cache_name: str) -> None:
        meta_path = self.cache_dir / "cache_metadata.json"
        metadata = self.get_cache_info()

        tickers = sorted(df.columns.get_level_values(1).unique().tolist()) if isinstance(df.columns, pd.MultiIndex) else list(df.columns)
        fields = sorted(df.columns.get_level_values(0).unique().tolist()) if isinstance(df.columns, pd.MultiIndex) else []

        metadata[cache_path.name] = {
            "cache_name": cache_name,
            "tickers": tickers,
            "fields": fields,
            "n_tickers": len(tickers),
            "n_rows": len(df),
            "date_range": [str(df.index.min().date()), str(df.index.max().date())] if not df.empty else [],
            "last_updated": datetime.now().isoformat(),
            "yfinance_version": getattr(yf, "__version__", "unknown"),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_cache_info(self) -> Dict:
        meta_path = self.cache_dir / "cache_metadata.json"
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}