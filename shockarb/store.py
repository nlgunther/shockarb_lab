"""
DataStore — unified price data manager for ShockArb.

Replaces the ad-hoc CacheManager with a design that has three explicit tiers:

  calibration  Permanent daily OHLCV per ticker.  One parquet file per ticker
               in prices/daily/.  Incremental tail-fetch: only downloads the
               missing dates after the last cached row.  Never deleted.

  intraday     15-minute bars for the current trading day.  One parquet file
               per ticker per calendar date in prices/intraday/.  Appended on
               every call during the day (deduplication on timestamp index).
               Deleted by sweep() when >= 2 calendar days stale.

  scoring      The live alpha CSV written by the scanner.  Not managed here;
               listed in the manifest for audit purposes only.

Everything that touches disk is recorded in manifest.json — a single JSON
file at the data root.  The manifest is the authoritative record of what
exists, what it contains, and when it should be deleted.  sweep() reads the
manifest to decide what to prune; it never scans the filesystem directly.

Design decisions
----------------
* Per-ticker files (not per-universe).  A 60-ticker universe stores 60 small
  parquets rather than one large one.  This makes incremental tail-fetches a
  targeted single-file write rather than a full-universe rewrite.

* Append-on-write for intraday.  Multiple calls during the day accumulate
  bars; the index is deduplicated on each write so re-downloading the same
  interval is harmless.

* Atomic writes.  Every parquet write goes through a temp file + rename so a
  crash mid-write never corrupts the cache.

* Manifest-driven sweep.  Deletion decisions are made purely from manifest
  metadata — no filesystem glob, no date parsing from filenames.

* Backward-compatible fetch_ohlcv shim.  The existing pipeline.py calls
  CacheManager.fetch_ohlcv().  DataStore exposes the same signature so it
  can be swapped in with a one-line change.

Usage
-----
    from shockarb.store import DataStore

    store = DataStore("data")

    # Calibration data (permanent)
    prices = store.fetch_daily(["VOO", "TLT"], "2022-02-10", "2022-03-31")

    # Intraday tape (appended, ephemeral)
    bars = store.fetch_intraday(["VOO", "TLT", "AMAT"])

    # Delete intraday files >= 2 calendar days stale
    store.sweep()

    # Inspect the manifest
    store.print_manifest()
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import yfinance as yf
from loguru import logger


# =============================================================================
# Constants
# =============================================================================

INTRADAY_INTERVAL  = "15m"      # yfinance interval string for intraday bars
INTRADAY_PERIOD    = "2d"       # look-back window — enough for prior close row
SWEEP_STALE_DAYS   = 2          # delete intraday files older than this many days
DAILY_PERIOD       = "5d"       # look-back for live close fetch


# =============================================================================
# DataStore
# =============================================================================

class DataStore:
    """
    Unified price data manager.  All yfinance downloads go through here.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory — same value used by ExecutionConfig.data_dir.
        Sub-directories (prices/daily, prices/intraday) are created on first use.
    downloader : callable, optional
        Injected yfinance.download replacement — used by tests to avoid
        real network calls.  Must match the yf.download signature.

    Directory layout created under data_dir
    ----------------------------------------
        prices/
            daily/          one parquet per ticker  (permanent calibration data)
            intraday/       one parquet per ticker per date  (ephemeral)
        manifest.json       provenance log for every managed file
    """

    def __init__(
        self,
        data_dir: str | Path,
        downloader=None,
    ) -> None:
        self.root        = Path(data_dir)
        self._daily_dir  = self.root / "prices" / "daily"
        self._intra_dir  = self.root / "prices" / "intraday"
        self._daily_dir.mkdir(parents=True, exist_ok=True)
        self._intra_dir.mkdir(parents=True, exist_ok=True)
        self._dl         = downloader or yf.download
        self._manifest   = self._load_manifest()

    # =========================================================================
    # Public API — calibration (permanent daily OHLCV)
    # =========================================================================

    def fetch_daily(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return daily Adj Close prices for *tickers* over [start, end].

        Downloads only the missing tail for each ticker; never re-fetches data
        already on disk.  Returns a (dates × tickers) DataFrame of Adj Close.

        Parameters
        ----------
        tickers : list of str
        start, end : str   YYYY-MM-DD  (end exclusive, yfinance convention)

        Returns
        -------
        DataFrame  (dates × tickers)  Adj Close prices.
        """
        frames: Dict[str, pd.Series] = {}

        for ticker in tickers:
            series = self._fetch_daily_one(ticker, start, end)
            if series is not None and not series.empty:
                frames[ticker] = series

        missing = set(tickers) - set(frames)
        if missing:
            logger.warning(f"No daily data for: {sorted(missing)}")

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        return result.loc[start:end]

    def fetch_ohlcv(
        self,
        tickers: List[str],
        start: str,
        end: str,
        cache_name: str = "",          # accepted for backward-compat, not used
    ) -> pd.DataFrame:
        """
        Backward-compatible shim matching the CacheManager.fetch_ohlcv() signature.

        Returns a real MultiIndex OHLCV DataFrame (Open, High, Low, Close,
        Adj Close, Volume) × tickers, identical in structure to what the
        existing pipeline.py expects.  extract_adj_close() works unchanged.

        Parameters
        ----------
        tickers : list of str
        start, end : str   YYYY-MM-DD
        cache_name : str   ignored — kept for drop-in compatibility

        Returns
        -------
        DataFrame with MultiIndex columns (field, ticker).
        """
        # Ensure daily parquets are populated first
        self.fetch_daily(tickers, start, end)
        return self.fetch_daily_ohlcv(tickers, start, end)

    def fetch_daily_ohlcv(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return full OHLCV data for *tickers* as a MultiIndex DataFrame.

        Columns are (field, ticker) where field ∈ {Open, High, Low, Close,
        Adj Close, Volume}.  Use this when you need Volume or OHLC ranges,
        for example to compute VWAP or to assess whether a dislocation is
        accompanied by unusual volume.

        Unlike fetch_daily(), this method reads from the already-cached
        per-ticker parquets.  Call fetch_daily() first if the cache might
        be stale — fetch_ohlcv() does this automatically.

        Parameters
        ----------
        tickers : list of str
        start, end : str   YYYY-MM-DD

        Returns
        -------
        DataFrame  MultiIndex columns (field, ticker), DatetimeIndex rows.
        """
        frames: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            path = self._daily_path(ticker)
            if not path.exists():
                logger.warning(f"No daily cache for {ticker} — run fetch_daily() first")
                continue
            try:
                df = pd.read_parquet(path).loc[start:end]
                if not df.empty:
                    frames[ticker] = df
            except Exception as exc:
                logger.warning(f"Could not read daily OHLCV for {ticker}: {exc}")

        if not frames:
            return pd.DataFrame()

        # Build MultiIndex: (field, ticker)
        pieces = []
        for ticker, df in frames.items():
            df = df.copy()
            df.columns = pd.MultiIndex.from_tuples(
                [(col, ticker) for col in df.columns]
            )
            pieces.append(df)

        result = pd.concat(pieces, axis=1).sort_index(axis=1).sort_index(axis=0)
        return result

    def extract_adj_close(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Adj Close from a MultiIndex OHLCV frame.  Matches
        CacheManager.extract_adj_close() for drop-in compatibility.
        """
        if ohlcv.empty:
            return pd.DataFrame()
        if not isinstance(ohlcv.columns, pd.MultiIndex):
            return ohlcv
        level0 = ohlcv.columns.get_level_values(0)
        if "Adj Close" in level0:
            return ohlcv["Adj Close"]
        if "Close" in level0:
            logger.warning("No 'Adj Close'; falling back to 'Close'")
            return ohlcv["Close"]
        raise KeyError("Neither 'Adj Close' nor 'Close' found.")

    # =========================================================================
    # Public API — intraday (ephemeral 15-minute bars)
    # =========================================================================

    def fetch_intraday(
        self,
        tickers: List[str],
        trade_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch and append today's 15-minute bars for *tickers*.

        Downloads the last 2 days at 15-minute resolution, extracts today's
        bars (plus yesterday's final bar for prior-close reference), and
        appends them to the per-ticker intraday parquets.  Duplicate timestamps
        are deduplicated on write — calling this twice during the day is safe.

        Parameters
        ----------
        tickers : list of str
        trade_date : date, optional
            The trading date being captured.  Defaults to today.
            Override in tests to avoid date-sensitive logic.

        Returns
        -------
        DataFrame
            All bars stored for *trade_date* across all tickers, as a
            MultiIndex (field, ticker) DataFrame sorted by timestamp.
            Columns include Open, High, Low, Close, Adj Close, Volume.
        """
        trade_date = trade_date or date.today()
        logger.info(
            f"Fetching intraday bars for {len(tickers)} tickers "
            f"({trade_date}, {INTRADAY_INTERVAL})"
        )

        raw = self._download_intraday(tickers)
        if raw is None or raw.empty:
            logger.warning("Intraday download returned no data.")
            return pd.DataFrame()

        # Normalise to MultiIndex
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_tuples(
                [(col, tickers[0]) for col in raw.columns]
            )

        # Append each ticker's bars to its per-date parquet
        for ticker in tickers:
            ticker_cols = [c for c in raw.columns if c[1] == ticker]
            if not ticker_cols:
                logger.debug(f"No intraday data for {ticker}")
                continue
            ticker_df = raw[ticker_cols].copy()
            ticker_df.columns = pd.MultiIndex.from_tuples(ticker_cols)
            self._append_intraday(ticker, trade_date, ticker_df)

        return self._load_intraday(tickers, trade_date)

    # =========================================================================
    # Public API — sweep (delete stale intraday files)
    # =========================================================================

    def sweep(self, reference_date: Optional[date] = None) -> List[str]:
        """
        Delete intraday files that are >= SWEEP_STALE_DAYS calendar days old.

        Uses the manifest exclusively — never globs the filesystem.  After
        deletion the manifest entry is removed and saved.

        Parameters
        ----------
        reference_date : date, optional
            Date to measure staleness against.  Defaults to today.
            Override in tests.

        Returns
        -------
        list of str
            Paths of files that were deleted.
        """
        reference_date = reference_date or date.today()
        cutoff         = reference_date - timedelta(days=SWEEP_STALE_DAYS)
        deleted        = []

        stale_keys = [
            key for key, entry in self._manifest.items()
            if entry.get("tier") == "intraday"
            and date.fromisoformat(entry["trade_date"]) <= cutoff
        ]

        for key in stale_keys:
            entry = self._manifest[key]
            path  = Path(entry["path"])
            if path.exists():
                path.unlink()
                logger.info(f"Sweep: deleted {path.name}  ({entry['trade_date']})")
            else:
                logger.debug(f"Sweep: already gone — {path.name}")
            deleted.append(str(path))
            del self._manifest[key]

        if deleted:
            self._save_manifest()
            logger.success(f"Sweep complete: {len(deleted)} file(s) deleted.")
        else:
            logger.info("Sweep: nothing to delete.")

        return deleted

    # =========================================================================
    # Public API — manifest inspection
    # =========================================================================

    def print_manifest(self) -> None:
        """Print a human-readable summary of the manifest to the terminal."""
        if not self._manifest:
            print("Manifest is empty.")
            return

        daily    = {k: v for k, v in self._manifest.items() if v.get("tier") == "daily"}
        intraday = {k: v for k, v in self._manifest.items() if v.get("tier") == "intraday"}

        print(f"\n{'='*70}")
        print(f"  SHOCKARB DATA MANIFEST  ({len(self._manifest)} entries)")
        print(f"{'='*70}")
        print(f"\n  CALIBRATION (permanent) — {len(daily)} tickers")
        for key, e in sorted(daily.items()):
            print(f"    {e['ticker']:<8}  {e['date_range'][0]} → {e['date_range'][1]}"
                  f"  ({e['rows']} rows)  updated {e['last_updated'][:10]}")

        print(f"\n  INTRADAY (ephemeral) — {len(intraday)} file(s)")
        for key, e in sorted(intraday.items()):
            print(f"    {e['ticker']:<8}  {e['trade_date']}"
                  f"  ({e['rows']} rows)  updated {e['last_updated'][:10]}")
        print()

    # =========================================================================
    # Internal — daily per-ticker fetch
    # =========================================================================

    def _daily_path(self, ticker: str) -> Path:
        return self._daily_dir / f"{ticker}.parquet"

    def _fetch_daily_one(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> Optional[pd.Series]:
        """
        Return Adj Close series for one ticker, fetching only the missing tail.

        The parquet stores the full OHLCV DataFrame (Open, High, Low, Close,
        Adj Close, Volume) so that callers such as fetch_daily_ohlcv() and
        fetch_ohlcv() can access Volume and other fields without a re-download.
        This method extracts and returns only the Adj Close column so that
        fetch_daily() continues to return a clean (dates × tickers) price matrix
        as expected by the factor engine.

        Cache layout: prices/daily/{ticker}.parquet
            Index   : DatetimeIndex (trading days)
            Columns : ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        """
        path    = self._daily_path(ticker)
        req_end = pd.Timestamp(end)

        if path.exists():
            try:
                cached = pd.read_parquet(path)
                last   = cached.index.max()

                # Full hit — requested range already covered
                if last >= req_end - pd.tseries.offsets.BDay(1):
                    logger.debug(f"[Daily HIT] {ticker}")
                    return cached["Adj Close"] if "Adj Close" in cached.columns else cached.iloc[:, 0]

                # Tail miss — download only the missing dates
                tail_start = (last + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
                logger.info(f"[Daily TAIL] {ticker}: fetching {tail_start} → {end}")
                new_df = self._download_daily_one(ticker, tail_start, end)
                if new_df is not None and not new_df.empty:
                    merged = pd.concat([cached, new_df])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                    self._write_atomic(merged, path)
                    self._manifest_update_daily(ticker, merged)
                    self._save_manifest()
                stored = pd.read_parquet(path)
                return stored["Adj Close"] if "Adj Close" in stored.columns else stored.iloc[:, 0]

            except Exception as exc:
                logger.warning(f"Cache read failed for {ticker}: {exc} — re-downloading")

        # Total miss
        logger.info(f"[Daily MISS] {ticker}: full download {start} → {end}")
        ohlcv = self._download_daily_one(ticker, start, end)
        if ohlcv is None or ohlcv.empty:
            return None

        self._write_atomic(ohlcv, path)
        self._manifest_update_daily(ticker, ohlcv)
        self._save_manifest()
        return ohlcv["Adj Close"] if "Adj Close" in ohlcv.columns else ohlcv.iloc[:, 0]

    def _download_daily_one(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Download full OHLCV for a single ticker; return DataFrame or None.

        Returns a flat DataFrame with columns:
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        indexed by trading day.  The MultiIndex returned by yfinance is
        flattened here so callers never see it.
        """
        try:
            raw = self._dl(
                [ticker], start=start, end=end,
                auto_adjust=False, progress=False,
            )
            if raw is None or raw.empty:
                return None

            # yfinance returns a MultiIndex (field, ticker) for batch downloads
            if isinstance(raw.columns, pd.MultiIndex):
                # Flatten: keep all fields, drop the ticker level
                raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1)                       else raw.droplevel(1, axis=1)

            # Keep only recognised OHLCV fields; drop any yfinance extras
            keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                    if c in raw.columns]
            df = raw[keep].dropna(how="all")
            df.index.name = "Date"
            return df if not df.empty else None

        except Exception as exc:
            logger.error(f"Download failed for {ticker}: {exc}")
            return None

    # =========================================================================
    # Internal — intraday per-ticker append
    # =========================================================================

    def _intra_path(self, ticker: str, trade_date: date) -> Path:
        return self._intra_dir / f"{ticker}_{trade_date.isoformat()}.parquet"

    def _download_intraday(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Download 2-day 15-minute bars for all tickers in one API call."""
        try:
            raw = self._dl(
                tickers,
                period=INTRADAY_PERIOD,
                interval=INTRADAY_INTERVAL,
                auto_adjust=False,
                progress=False,
            )
            return raw if not raw.empty else None
        except Exception as exc:
            logger.error(f"Intraday download failed: {exc}")
            return None

    def _append_intraday(
        self,
        ticker: str,
        trade_date: date,
        new_bars: pd.DataFrame,
    ) -> None:
        """
        Append new bars to the per-ticker intraday parquet, deduplicating on
        the DatetimeIndex.  Creates the file if it doesn't exist yet.
        """
        path = self._intra_path(ticker, trade_date)

        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, new_bars])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            except Exception as exc:
                logger.warning(f"Intraday read failed for {ticker}: {exc} — overwriting")
                combined = new_bars
        else:
            combined = new_bars

        self._write_atomic(combined, path)
        self._manifest_update_intraday(ticker, trade_date, combined)
        self._save_manifest()
        logger.debug(
            f"[Intraday] {ticker} {trade_date}: {len(combined)} bars stored"
        )

    def _load_intraday(
        self,
        tickers: List[str],
        trade_date: date,
    ) -> pd.DataFrame:
        """Load and concatenate all per-ticker intraday parquets for trade_date."""
        frames = []
        for ticker in tickers:
            path = self._intra_path(ticker, trade_date)
            if path.exists():
                try:
                    frames.append(pd.read_parquet(path))
                except Exception as exc:
                    logger.warning(f"Could not load intraday {ticker}: {exc}")
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, axis=1)
        return result.sort_index()

    # =========================================================================
    # Internal — atomic write
    # =========================================================================

    def _write_atomic(self, df: pd.DataFrame, path: Path) -> None:
        """Write df to path via a temp file + rename to prevent corruption."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}_", suffix=".parquet"
        )
        try:
            os.close(fd)
            df.to_parquet(tmp)
            shutil.move(tmp, path)
        except Exception as exc:
            if os.path.exists(tmp):
                os.unlink(tmp)
            logger.error(f"Atomic write failed → {path.name}: {exc}")
            raise

    # =========================================================================
    # Internal — manifest
    # =========================================================================

    @property
    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _load_manifest(self) -> Dict:
        try:
            with open(self._manifest_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_manifest(self) -> None:
        fd, tmp = tempfile.mkstemp(
            dir=self.root, prefix=".manifest_", suffix=".json"
        )
        try:
            os.close(fd)
            with open(tmp, "w") as f:
                json.dump(self._manifest, f, indent=2, sort_keys=True)
            shutil.move(tmp, self._manifest_path)
        except Exception as exc:
            if os.path.exists(tmp):
                os.unlink(tmp)
            logger.error(f"Manifest save failed: {exc}")
            raise

    def _manifest_update_daily(self, ticker: str, data) -> None:
        fields = list(data.columns) if isinstance(data, pd.DataFrame) else ["Adj Close"]
        key    = f"daily/{ticker}"
        self._manifest[key] = {
            "tier":         "daily",
            "ticker":       ticker,
            "path":         str(self._daily_path(ticker)),
            "fields":       fields,
            "date_range":   [
                str(data.index.min().date()),
                str(data.index.max().date()),
            ],
            "rows":         len(data),
            "last_updated": datetime.now().isoformat(),
        }

    def _manifest_update_intraday(
        self,
        ticker: str,
        trade_date: date,
        df: pd.DataFrame,
    ) -> None:
        key = f"intraday/{ticker}/{trade_date.isoformat()}"
        self._manifest[key] = {
            "tier":         "intraday",
            "ticker":       ticker,
            "trade_date":   trade_date.isoformat(),
            "path":         str(self._intra_path(ticker, trade_date)),
            "rows":         len(df),
            "last_updated": datetime.now().isoformat(),
        }
