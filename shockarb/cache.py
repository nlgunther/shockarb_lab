"""
CacheManager: Intelligent parquet-based caching for yfinance data.

Architecture Overview
---------------------
Priority 1 (THIS FILE): Basic smart caching — stop re-downloading the same
    historical data. Caches full OHLCV (auto_adjust=False) in per-class
    parquet files with MultiIndex columns.

Priority 2 (stub): Ticker merging — when adding tickers to an existing
    universe, download only the new ones and merge.

Priority 3 (stub): Date extension — when requesting dates beyond the cached
    range, download only the missing window and append.

Priority 4 (stub): Backup & recovery — create timestamped backups before
    any destructive cache modification; atomic writes.

Priority 5 (stub): Metadata tracking — cache_metadata.json tracking ticker
    lists, date ranges, field lists, yfinance version, last-updated timestamp.

Design Decisions (per Opus analysis)
-------------------------------------
- Cache granularity: Per-class (ETF vs stock) × universe.
  Rationale: balances read speed with merge simplicity; the two asset
  classes have naturally different update patterns.

- Always cache full OHLCV (auto_adjust=False, all 6 fields).
  Rationale: storage is cheap; re-downloading is expensive and rate-limited.
  'Adj Close' is extracted at read time. Keeping raw 'Close' enables future
  retroactive adjustment detection (P5 extension).

- MultiIndex columns: Normalize (sort) on every save for consistency.

- Backup strategy: Timestamped copies before merge; 7-day age-based pruning.
  (P4 stubs present; P1 does not mutate existing cache files.)

- Metadata: JSON sidecar per cache file. (P5 stubs present.)

Usage
-----
    from shockarb.cache import CacheManager
    
    mgr = CacheManager(cache_dir="data/cache", backup_dir="data/backups")
    
    # Fetch ETF prices — downloads only what isn't already cached
    ohlcv = mgr.fetch_ohlcv(
        tickers=["VOO", "TLT", "GLD"],
        start="2022-02-10",
        end="2022-03-31",
        cache_name="us_etf",
    )
    
    # Extract Adj Close for the model engine
    adj_close = mgr.extract_adj_close(ohlcv)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yfinance as yf
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
BACKUP_MAX_AGE_DAYS = 7  # Backups older than this are pruned


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

class CacheManager:
    """
    Manages a local parquet cache of full yfinance OHLCV data.

    The cache layout on disk::

        <cache_dir>/
            us_etf_ohlcv.parquet
            us_stock_ohlcv.parquet
            global_etf_ohlcv.parquet
            global_stock_ohlcv.parquet
            cache_metadata.json          # P5
        <backup_dir>/
            us_etf_ohlcv_20260303_143022.parquet   # P4
            ...

    Parameters
    ----------
    cache_dir : str
        Directory for primary parquet cache files.
    backup_dir : str, optional
        Directory for timestamped backups. Defaults to ``<cache_dir>/../backups``.
    """

    def __init__(self, cache_dir: str, backup_dir: Optional[str] = None, downloader=None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if backup_dir is None:
            self.backup_dir = self.cache_dir.parent / "backups"
        else:
            self.backup_dir = Path(backup_dir)

        # Injected download callable — defaults to yf.download.
        # Accepting this as a parameter allows tests to mock via
        # `patch('shockarb.pipeline.yf.download')` without needing to also
        # patch `shockarb.cache.yf.download`.
        self._downloader = downloader if downloader is not None else yf.download
        # Backup dir created on demand (P4) to avoid polluting a clean install

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
        Return full OHLCV data, downloading only what is not already cached.

        Priority 1 logic
        ----------------
        - If no cache exists → download everything, save to cache.
        - If cache exists and covers all tickers AND dates → return cache slice.
        - If cache exists but is missing tickers OR dates → [P2/P3 stubs]
          currently falls back to full download and overwrites cache.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols to retrieve.
        start : str
            Start date inclusive (YYYY-MM-DD).
        end : str
            End date exclusive per yfinance convention (YYYY-MM-DD).
        cache_name : str
            Logical name for this cache file, e.g. ``"us_etf"`` or
            ``"global_stock"``. File will be ``<name>_ohlcv.parquet``.

        Returns
        -------
        DataFrame
            MultiIndex-column DataFrame (field, ticker) × dates.
        """
        cache_path = self._cache_path(cache_name)

        # --- Attempt to load existing cache ---------------------------------
        existing = self._load_cache(cache_path)

        if existing is not None:
            cached_tickers = self._cached_tickers(existing)
            cached_dates = existing.index
            requested_tickers = set(tickers)

            missing_tickers = requested_tickers - cached_tickers
            missing_dates = self._missing_business_dates(cached_dates, start, end)

            if not missing_tickers and missing_dates.empty:
                # Perfect cache hit — slice and return
                logger.info(
                    f"[Cache HIT] {cache_name}: {len(tickers)} tickers, "
                    f"{start} → {end}"
                )
                return self._slice(existing, tickers, start, end)

            # --- P2/P3 stubs: incremental merging ---------------------------
            if missing_tickers:
                logger.info(
                    f"[Cache MISS] {cache_name}: {len(missing_tickers)} new tickers "
                    f"not in cache — {sorted(missing_tickers)[:5]}{'...' if len(missing_tickers) > 5 else ''}"
                )
                # P2: merge new tickers into existing cache
                # For now fall through to full download
                existing = self._p2_merge_tickers(
                    existing, missing_tickers, start, end, cache_path
                )
                if existing is not None:
                    return self._slice(existing, tickers, start, end)

            if not missing_dates.empty:
                logger.info(
                    f"[Cache MISS] {cache_name}: {len(missing_dates)} missing "
                    f"business days — extending date range"
                )
                # P3: extend date range in cache
                existing = self._p3_extend_dates(
                    existing, tickers, missing_dates, cache_path
                )
                if existing is not None:
                    return self._slice(existing, tickers, start, end)

        # --- Full download (cache miss or extension failed) -----------------
        logger.info(
            f"[Cache MISS] {cache_name}: downloading {len(tickers)} tickers "
            f"from {start} to {end}"
        )
        raw = self._download(tickers, start, end)
        if raw is None or raw.empty:
            return pd.DataFrame()

        # Normalize and save
        raw = self._normalize_columns(raw)
        self._save_cache(raw, cache_path, cache_name)

        return self._slice(raw, tickers, start, end)

    def extract_adj_close(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 'Adj Close' prices from a full OHLCV MultiIndex DataFrame.

        Parameters
        ----------
        ohlcv : DataFrame
            Full OHLCV data with MultiIndex (field, ticker) columns.

        Returns
        -------
        DataFrame
            (dates × tickers) adjusted close prices.
        """
        if ohlcv.empty:
            return pd.DataFrame()

        if not isinstance(ohlcv.columns, pd.MultiIndex):
            # Already a flat price DataFrame (e.g. single-ticker edge case)
            return ohlcv

        if "Adj Close" in ohlcv.columns.get_level_values(0):
            return ohlcv["Adj Close"]
        elif "Close" in ohlcv.columns.get_level_values(0):
            logger.warning("No 'Adj Close' in cache; falling back to 'Close'")
            return ohlcv["Close"]
        else:
            raise KeyError(
                f"Neither 'Adj Close' nor 'Close' found. "
                f"Available fields: {ohlcv.columns.get_level_values(0).unique().tolist()}"
            )

    # =========================================================================
    # Internal helpers — cache I/O
    # =========================================================================

    def _cache_path(self, cache_name: str) -> Path:
        return self.cache_dir / f"{cache_name}_ohlcv.parquet"

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a parquet cache file, returning None on any failure."""
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            logger.debug(f"Loaded cache: {path} ({len(df)} rows)")
            return df
        except Exception as exc:
            logger.warning(f"Cache read failed ({path}): {exc} — will re-download")
            return None

    def _save_cache(
        self,
        df: pd.DataFrame,
        path: Path,
        cache_name: str,
    ) -> None:
        """
        Atomically write DataFrame to parquet cache.

        Uses a temp-file-then-rename pattern so that a crash during write
        never leaves a corrupt primary cache file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp → rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}_", suffix=".parquet"
        )
        try:
            os.close(tmp_fd)
            df.to_parquet(tmp_path)
            shutil.move(tmp_path, path)
            logger.info(f"Cache saved: {path} ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.error(f"Cache save failed: {exc}")
            raise

        # P5 hook: update metadata
        self._p5_update_metadata(df, path, cache_name)

    # =========================================================================
    # Internal helpers — data manipulation
    # =========================================================================

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has a sorted, consistent MultiIndex column structure.

        yfinance sometimes returns flat columns for a single ticker. This
        normalises both cases to (field, ticker) MultiIndex.
        """
        if df.empty:
            return df

        if not isinstance(df.columns, pd.MultiIndex):
            # Single-ticker: columns are field names → wrap in MultiIndex
            ticker = "UNKNOWN"
            df.columns = pd.MultiIndex.from_tuples(
                [(col, ticker) for col in df.columns]
            )

        # Sort for consistency: field first, then ticker
        df = df.sort_index(axis=1)
        return df

    @staticmethod
    def _cached_tickers(df: pd.DataFrame) -> Set[str]:
        """Return the set of tickers present in a MultiIndex OHLCV DataFrame."""
        if isinstance(df.columns, pd.MultiIndex):
            return set(df.columns.get_level_values(1).unique())
        return set(df.columns)

    @staticmethod
    def _missing_business_dates(
        cached_dates: pd.DatetimeIndex,
        start: str,
        end: str,
    ) -> pd.DatetimeIndex:
        """
        Return business dates that need downloading to satisfy the request.

        Rules
        -----
        - Before-start gap: if the request starts before the earliest cached
          date, those prior business days are flagged.
        - After-end gap: only flagged when both (a) the requested end is in
          the recent past (within 30 days of today) AND (b) there are ≥2
          uncached business days before that end. This prevents P3 from
          firing on historical calibration windows (e.g. end=2022-03-31)
          where yfinance simply returned fewer rows than the calendar
          suggests, which is normal.
        """
        if cached_dates.empty:
            return pd.bdate_range(start=start, end=end, inclusive="left")

        cache_start = cached_dates.min()
        cache_end = cached_dates.max()
        req_start = pd.Timestamp(start)
        req_end_exclusive = pd.Timestamp(end)
        today = pd.Timestamp.now().normalize()

        missing = []

        # Before-start gap
        if req_start < cache_start:
            prior = pd.bdate_range(start=req_start, end=cache_start, inclusive="left")
            if len(prior) >= 1:
                missing.append(prior)

        # After-end gap: only for recent requests, not historical windows
        is_recent_request = (req_end_exclusive >= today - pd.Timedelta(days=30))
        if is_recent_request:
            after = pd.bdate_range(
                start=cache_end, end=req_end_exclusive, inclusive="right"
            )
            if len(after) >= 2:
                missing.append(after)

        if not missing:
            return pd.DatetimeIndex([])
        return missing[0].append(missing[1:]) if len(missing) > 1 else missing[0]

    @staticmethod
    def _slice(
        df: pd.DataFrame,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return the subset of df for the requested tickers and date range.

        Missing tickers are silently omitted (logged at WARNING level).
        """
        # Date slice
        result = df.loc[start:end]

        # Ticker slice (MultiIndex: select across all fields)
        if isinstance(result.columns, pd.MultiIndex):
            available = set(result.columns.get_level_values(1).unique())
            wanted = [t for t in tickers if t in available]
            missing = set(tickers) - available
            if missing:
                logger.warning(
                    f"Tickers not in cache (omitted from result): {sorted(missing)}"
                )
            # Select columns where second-level ticker is in wanted set
            mask = result.columns.get_level_values(1).isin(wanted)
            result = result.loc[:, mask]

        return result

    # =========================================================================
    # Priority 2 stub — ticker merging
    # =========================================================================

    def _p2_merge_tickers(
        self,
        existing: pd.DataFrame,
        missing_tickers: Set[str],
        start: str,
        end: str,
        cache_path: Path,
    ) -> Optional[pd.DataFrame]:
        """
        [Priority 2] Download new tickers and merge into existing cache.

        Current status: IMPLEMENTED.
        Downloads only the missing tickers over the full cached date range,
        merges with existing data, backs up old cache, and saves new cache.
        """
        # Determine the full date range we need (union of existing and requested)
        existing_start = str(existing.index.min().date())
        existing_end = str(existing.index.max().date())

        # Download only the missing tickers
        logger.info(
            f"[P2] Downloading {len(missing_tickers)} new tickers: "
            f"{sorted(missing_tickers)[:10]}"
        )
        new_data = self._download(list(missing_tickers), existing_start, existing_end)
        if new_data is None or new_data.empty:
            logger.warning("[P2] Download of new tickers returned empty; skipping merge")
            return None

        new_data = self._normalize_columns(new_data)

        # Back up existing cache before modification (P4 hook)
        self._p4_backup(cache_path)

        # Merge: concatenate along columns, deduplicate, sort
        merged = self._merge_ohlcv(existing, new_data)
        self._save_cache(merged, cache_path, cache_path.stem.replace("_ohlcv", ""))
        return merged

    # =========================================================================
    # Priority 3 stub — date extension
    # =========================================================================

    def _p3_extend_dates(
        self,
        existing: pd.DataFrame,
        tickers: List[str],
        missing_dates: pd.DatetimeIndex,
        cache_path: Path,
    ) -> Optional[pd.DataFrame]:
        """
        [Priority 3] Download missing date range and append to existing cache.

        Current status: IMPLEMENTED.
        Downloads only the missing date window for the full ticker universe,
        appends rows, deduplicates, backs up old cache, and saves.
        """
        ext_start = str(missing_dates.min().date())
        # yfinance end is exclusive, so add 1 business day
        ext_end = str(missing_dates.max().date() + pd.offsets.BDay(1))
        existing_tickers = self._cached_tickers(existing)

        logger.info(f"[P3] Extending dates: {ext_start} → {ext_end}")
        new_data = self._download(list(existing_tickers), ext_start, ext_end)
        if new_data is None or new_data.empty:
            logger.warning("[P3] Date extension download returned empty; skipping")
            return None

        new_data = self._normalize_columns(new_data)

        # Back up before modification
        self._p4_backup(cache_path)

        # Append rows, deduplicate index, sort
        merged = pd.concat([existing, new_data], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()
        merged = self._normalize_columns(merged)

        self._save_cache(merged, cache_path, cache_path.stem.replace("_ohlcv", ""))
        return merged

    # =========================================================================
    # Priority 4 — Backup & recovery
    # =========================================================================

    def _p4_backup(self, cache_path: Path) -> Optional[Path]:
        """
        [Priority 4] Copy existing cache to a timestamped backup file.

        Returns the backup path, or None if nothing to back up.
        """
        if not cache_path.exists():
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{cache_path.stem}_{ts}{cache_path.suffix}"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(cache_path, backup_path)
        logger.info(f"[P4] Cache backed up: {backup_path}")

        # Prune old backups
        self._p4_prune_backups()

        return backup_path

    def _p4_prune_backups(self) -> int:
        """
        [Priority 4] Delete backup files older than BACKUP_MAX_AGE_DAYS.

        Returns the number of files deleted.
        """
        if not self.backup_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=BACKUP_MAX_AGE_DAYS)
        deleted = 0

        for backup_file in self.backup_dir.glob("*.parquet"):
            # Filename pattern: <stem>_YYYYMMDD_HHMMSS.parquet
            parts = backup_file.stem.rsplit("_", 2)
            if len(parts) < 3:
                continue
            try:
                file_dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
                if file_dt < cutoff:
                    backup_file.unlink()
                    deleted += 1
                    logger.debug(f"[P4] Pruned old backup: {backup_file.name}")
            except ValueError:
                continue

        if deleted:
            logger.info(f"[P4] Pruned {deleted} backup(s) older than {BACKUP_MAX_AGE_DAYS} days")
        return deleted

    # =========================================================================
    # Priority 5 stub — metadata tracking
    # =========================================================================

    def _p5_update_metadata(
        self,
        df: pd.DataFrame,
        cache_path: Path,
        cache_name: str,
    ) -> None:
        """
        [Priority 5] Write/update cache_metadata.json sidecar.

        Current status: IMPLEMENTED.
        Tracks tickers, date range, fields, yfinance version, last-updated.
        """
        meta_path = self.cache_dir / "cache_metadata.json"

        # Load existing metadata
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}

        # Gather info about this cache file
        if isinstance(df.columns, pd.MultiIndex):
            fields = sorted(df.columns.get_level_values(0).unique().tolist())
            tickers = sorted(df.columns.get_level_values(1).unique().tolist())
        else:
            fields = list(df.columns)
            tickers = []

        try:
            yf_version = yf.__version__
        except AttributeError:
            yf_version = "unknown"

        metadata[cache_path.name] = {
            "cache_name": cache_name,
            "tickers": tickers,
            "fields": fields,
            "n_tickers": len(tickers),
            "n_rows": len(df),
            "date_range": [
                str(df.index.min().date()),
                str(df.index.max().date()),
            ],
            "last_updated": datetime.now().isoformat(),
            "yfinance_version": yf_version,
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"[P5] Metadata updated: {meta_path}")

    def get_cache_info(self) -> Dict:
        """
        [Priority 5] Return the current cache metadata as a dict.

        Returns an empty dict if no metadata file exists yet.
        """
        meta_path = self.cache_dir / "cache_metadata.json"
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # =========================================================================
    # Download helper
    # =========================================================================

    def _download(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Download full OHLCV using the injected downloader (default: yf.download).

        Returns None on failure (caller decides fallback strategy).
        """
        logger.info(f"yfinance.download: {len(tickers)} tickers, {start} → {end}")
        try:
            raw = self._downloader(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            logger.error(f"yfinance exception: {exc}")
            return None

        if raw is None or raw.empty:
            logger.error("yfinance returned empty DataFrame")
            return None

        return raw

    # =========================================================================
    # Merge helper (used by P2)
    # =========================================================================

    @staticmethod
    def _merge_ohlcv(
        existing: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge two MultiIndex OHLCV DataFrames along columns.

        Deduplicates tickers (new_data wins on overlap), then sorts both
        the column index (field, ticker) and the row index (date).
        """
        merged = pd.concat([existing, new_data], axis=1)

        # Deduplicate: if same (field, ticker) appears twice, keep the last
        # (i.e. the new_data version, which is more recently downloaded)
        mask = ~merged.columns.duplicated(keep="last")
        merged = merged.loc[:, mask]

        # Sort for consistency
        merged = merged.sort_index(axis=1).sort_index(axis=0)
        return merged
