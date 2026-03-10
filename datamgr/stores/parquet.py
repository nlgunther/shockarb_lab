"""
datamgr.stores.parquet — ParquetStore: DataStore implementation.

Phase 1: thin adapter wrapping the existing shockarb.store.DataStore.

Why an adapter rather than rewriting store.py now?
----------------------------------------------------
store.py has 35 passing tests.  Rewriting it to implement DataStore directly
is Phase 2-4 work.  For Phase 1, ParquetStore wraps the existing class and
satisfies the DataStore ABC so the coordinator can call it via the interface.

The adapter translates between the DataStore ABC contract and shockarb's
existing method signatures::

    ABC.read()      → pd.read_parquet on the inner store's path helpers
    ABC.write()     → store._write_atomic() + store._manifest_update_daily()
    ABC.coverage()  → reads store._manifest directly (no filesystem access)
    ABC.sweep()     → store.sweep()

Phase 2/3 upgrade path
-----------------------
When store.py is refactored to implement DataStore directly (Phase 2),
this adapter is deleted and the coordinator is pointed at the new class.
No other files change — the coordinator only knows the DataStore ABC.

Note on imports
---------------
This file imports from shockarb — it is the ONE permitted crossing point
of the datamgr/shockarb boundary.  All other datamgr files are import-clean.
The dependency arrow is: shockarb.pipeline → datamgr; datamgr.stores.parquet
→ shockarb.store (adapter seam only).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from loguru import logger

from datamgr.interfaces import DataStore


class ParquetStore(DataStore):
    """
    DataStore adapter over shockarb.store.DataStore.

    Translates the DataStore ABC into the inner store's method calls so the
    coordinator never imports shockarb.store directly.  All read/write
    operations are logged at DEBUG level using the inherited _log() helper —
    enable with ``--log-level DEBUG`` to trace exact parquet paths and row
    counts during a build or score run.

    Parameters
    ----------
    inner : shockarb.store.DataStore
        The existing store instance.  Typed as Any internally to keep
        datamgr import-clean.  At runtime this is always a
        shockarb.store.DataStore instance constructed in pipeline._coordinator().
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    # =========================================================================
    # DataStore ABC implementation
    # =========================================================================

    def read(
        self,
        key: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Read stored data for *key* over [start, end].

        Derives the filesystem path from the inner store's path helpers and
        reads the parquet file directly.  Returns None (never raises) on any
        failure — missing file, corrupt parquet, or unrecognised key format.

        Key format
        ----------
        "daily/{ticker}"
            Reads data/prices/daily/{ticker}.parquet, sliced to [start, end].
        "intraday/{ticker}/{trade_date}"
            Reads data/prices/intraday/{ticker}_{trade_date}.parquet.
            No date slice applied — returns the full file.

        Parameters
        ----------
        key : str
            Store key.
        start : str
            Inclusive start date YYYY-MM-DD.
        end : str
            Inclusive end date YYYY-MM-DD.

        Returns
        -------
        pd.DataFrame or None
            Sliced DataFrame, or None if the file is missing or unreadable.
        """
        parts = key.split("/")

        if parts[0] == "daily" and len(parts) == 2:
            ticker = parts[1]
            path = self._inner._daily_path(ticker)
            if not path.exists():
                logger.debug(self._log("read", "MISS", path))
                return None
            try:
                df = pd.read_parquet(path).loc[start:end]
                logger.debug(self._log("read", "OK", path, len(df)))
                return df
            except Exception as exc:
                logger.warning(self._log("read", "FAIL", path))
                logger.warning(f"[ParquetStore] read error: {exc}")
                return None

        if parts[0] == "intraday" and len(parts) == 3:
            ticker = parts[1]
            from datetime import date
            trade_date = date.fromisoformat(parts[2])
            path = self._inner._intra_path(ticker, trade_date)
            if not path.exists():
                logger.debug(self._log("read", "MISS", path))
                return None
            try:
                df = pd.read_parquet(path)
                logger.debug(self._log("read", "OK", path, len(df)))
                return df
            except Exception as exc:
                logger.warning(self._log("read", "FAIL", path))
                logger.warning(f"[ParquetStore] read error: {exc}")
                return None

        logger.warning(f"[ParquetStore] Unrecognised key format: {key!r}")
        return None

    def write(
        self,
        key: str,
        df: pd.DataFrame,
        meta: dict,
    ) -> None:
        """
        Atomically write *df* and update the manifest for *key*.

        Delegates to the inner store's atomic write helper and the
        appropriate manifest update method, then flushes the manifest
        to disk.

        Phase 1: uses _write_atomic() + _manifest_update_daily().
        Phase 2: this becomes the WAL-backed _commit() path, inserting
                 a pending.json entry before write and clearing it after.

        Parameters
        ----------
        key : str
            "daily/{ticker}" or "intraday/{ticker}/{trade_date}".
        df : pd.DataFrame
            Merged data to write (existing cache + new rows, deduped).
        meta : dict
            Manifest metadata: {"ticker": str, "frequency": str,
            "retention": str}.

        Raises
        ------
        ValueError
            If the key format is not recognised.
        """
        parts = key.split("/")

        if parts[0] == "daily" and len(parts) == 2:
            ticker = parts[1]
            path = self._inner._daily_path(ticker)
            self._inner._write_atomic(df, path)
            self._inner._manifest_update_daily(ticker, df)
            self._inner._save_manifest()
            logger.debug(self._log("write", "OK", path, len(df)))
            return

        if parts[0] == "intraday" and len(parts) == 3:
            ticker = parts[1]
            from datetime import date
            trade_date = date.fromisoformat(parts[2])
            path = self._inner._intra_path(ticker, trade_date)
            self._inner._write_atomic(df, path)
            self._inner._manifest_update_intraday(ticker, trade_date, df)
            self._inner._save_manifest()
            logger.debug(self._log("write", "OK", path, len(df)))
            return

        raise ValueError(f"[ParquetStore] Unrecognised key format for write: {key!r}")

    def coverage(
        self,
        key: str,
    ) -> Optional[tuple]:
        """
        Return (earliest_date, latest_date) for *key*, or None.

        Reads directly from the in-memory manifest — no filesystem access.
        Used by DataCoordinator._gap_analyse() to decide whether a download
        is needed and how much of the date range is already stored.

        Warning: stale manifest
        -----------------------
        If manifest.json contains an entry that no longer matches the parquet
        file on disk (e.g. after manual deletion, a crash mid-write, or a
        partial build), coverage() will report a false cache hit.

        Symptom: ``build()`` logs "Aligned on 0 common trading days" even
        though tickers appear to be registered correctly.

        Fix: delete ``data/manifest.json`` and run ``build`` again.

        Parameters
        ----------
        key : str
            Store key, e.g. "daily/VOO".

        Returns
        -------
        tuple of (str, str) or None
            (earliest_date, latest_date) as "YYYY-MM-DD" strings, or None.
        """
        entry = self._inner._manifest.get(key)
        if entry is None:
            return None
        date_range = entry.get("date_range")
        if not date_range or len(date_range) != 2:
            return None
        return (date_range[0], date_range[1])

    def sweep(
        self,
        retention: str,
        before: str,
    ) -> List[str]:
        """
        Delete ephemeral (intraday) assets older than *before*.

        Delegates to the inner store's sweep() method.  Only the
        "ephemeral" retention tier is currently supported — permanent/daily
        data is never swept.

        Parameters
        ----------
        retention : str
            Tier to sweep.  Only "ephemeral" triggers a deletion; any other
            value logs a debug message and returns [].
        before : str
            YYYY-MM-DD cutoff.  Files with trade_date on or before this
            date are deleted.

        Returns
        -------
        list of str
            Keys deleted from the manifest, or [] if nothing to sweep.
        """
        if retention != "ephemeral":
            logger.debug(
                f"[ParquetStore] sweep() called with retention={retention!r}; "
                "only 'ephemeral' is currently supported."
            )
            return []

        from datetime import date
        reference_date = date.fromisoformat(before)
        return self._inner.sweep(reference_date=reference_date)

    # =========================================================================
    # Pass-through helpers used by coordinator dispatch
    # =========================================================================

    def fetch_daily(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Pass-through to inner store's fetch_daily().

        Used by legacy callers that have not yet migrated to the coordinator
        path.  New callers should register a DataRequest with
        Frequency.DAILY and call coordinator.fulfill() instead.
        """
        return self._inner.fetch_daily(tickers, start, end)

    def fetch_intraday(
        self,
        tickers: List[str],
        trade_date,
    ) -> pd.DataFrame:
        """
        Pass-through to inner store's fetch_intraday().

        Called by DataCoordinator._read_intraday() for INTRADAY_15M requests.
        The coordinator bypasses gap analysis for intraday data and delegates
        directly to this method.

        Parameters
        ----------
        tickers : list of str
        trade_date : datetime.date
            The trading date to retrieve bars for.

        Returns
        -------
        pd.DataFrame
            Raw intraday bars as stored by the inner store.
        """
        return self._inner.fetch_intraday(tickers, trade_date=trade_date)
