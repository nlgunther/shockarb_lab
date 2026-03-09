"""
datamgr.stores.parquet — ParquetStore: DataStore implementation.

Phase 1: thin adapter wrapping the existing shockarb.store.DataStore.

Why an adapter rather than rewriting store.py now?
----------------------------------------------------
store.py has 35 passing tests.  Rewriting it to implement DataStore directly
is Phase 2-4 work.  For Phase 1, ParquetStore wraps the existing class and
satisfies the DataStore ABC so the coordinator can call it via the interface.

The adapter translates between the DataStore ABC contract and shockarb's
existing method signatures:
    ABC.read()      → store.fetch_daily() / store.fetch_intraday()
    ABC.write()     → store._write_atomic() + store._manifest_update_daily()
    ABC.coverage()  → reads store._manifest directly
    ABC.sweep()     → store.sweep()

Phase 2/3 upgrade path
-----------------------
When store.py is refactored to implement DataStore directly (Phase 2),
this adapter is deleted and the coordinator is pointed at the new class.
No other files change.

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

    Parameters
    ----------
    inner : shockarb.store.DataStore
        The existing store instance.  Injected so the coordinator never
        imports shockarb directly.
    """

    def __init__(self, inner) -> None:
        # inner is typed as Any to keep datamgr import-clean.
        # At runtime this will be a shockarb.store.DataStore instance.
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
        Read stored data for *key* over [start, end).

        Key format:
          "daily/{ticker}"                 → fetch_daily for that ticker
          "intraday/{ticker}/{trade_date}" → load_intraday for that ticker/date

        Returns None if the key is not in the manifest or the file is missing.
        """
        parts = key.split("/")

        if parts[0] == "daily" and len(parts) == 2:
            ticker = parts[1]
            path = self._inner._daily_path(ticker)
            if not path.exists():
                return None
            try:
                return pd.read_parquet(path).loc[start:end]
            except Exception as exc:
                logger.warning(f"[ParquetStore] read failed for {key}: {exc}")
                return None

        if parts[0] == "intraday" and len(parts) == 3:
            ticker = parts[1]
            from datetime import date
            trade_date = date.fromisoformat(parts[2])
            path = self._inner._intra_path(ticker, trade_date)
            if not path.exists():
                return None
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning(f"[ParquetStore] read failed for {key}: {exc}")
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
        Atomically write *df* and update the manifest.

        Phase 1: delegates to the inner store's atomic write + manifest helpers.
        Phase 2: this becomes the WAL-backed _commit() path.
        """
        parts = key.split("/")

        if parts[0] == "daily" and len(parts) == 2:
            ticker = parts[1]
            path = self._inner._daily_path(ticker)
            self._inner._write_atomic(df, path)
            self._inner._manifest_update_daily(ticker, df)
            self._inner._save_manifest()
            return

        if parts[0] == "intraday" and len(parts) == 3:
            ticker = parts[1]
            from datetime import date
            trade_date = date.fromisoformat(parts[2])
            path = self._inner._intra_path(ticker, trade_date)
            self._inner._write_atomic(df, path)
            self._inner._manifest_update_intraday(ticker, trade_date, df)
            self._inner._save_manifest()
            return

        raise ValueError(f"[ParquetStore] Unrecognised key format for write: {key!r}")

    def coverage(
        self,
        key: str,
    ) -> Optional[tuple[str, str]]:
        """
        Return (earliest_date, latest_date) for *key*, or None.

        Reads directly from the manifest — no filesystem access.
        Used by the coordinator for gap analysis (Phase 2).
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
        Delegate to inner store's sweep().

        Phase 1: translates the ABC signature to the existing sweep() call.
        retention="ephemeral" maps to intraday tier.
        before is a YYYY-MM-DD string.
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
        """Pass-through to inner store's fetch_daily()."""
        return self._inner.fetch_daily(tickers, start, end)

    def fetch_intraday(
        self,
        tickers: List[str],
        trade_date,
    ) -> pd.DataFrame:
        """Pass-through to inner store's fetch_intraday()."""
        return self._inner.fetch_intraday(tickers, trade_date=trade_date)
