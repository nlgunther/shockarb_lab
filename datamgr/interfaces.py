"""
datamgr.interfaces — Abstract base classes for the data management layer.

DataProvider  — pure I/O.  One class per data source.
DataStore     — pure persistence.  Reads/writes parquet + manifest.

Neither class knows about the other.  The coordinator wires them together.

Zero imports from shockarb or any application code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class ProviderError(Exception):
    """Raised by DataProvider.fetch() on unrecoverable failure."""


class DataProvider(ABC):
    """
    Pure I/O interface.  Fetches raw OHLCV data from one external source.

    Implementations: YFinanceProvider, MockProvider.

    Contract
    --------
    fetch() returns a flat (dates × tickers) DataFrame with ticker names
    as columns and adj_close prices as values.  This is the shape that
    DataCoordinator._read_daily() expects — a single adj_close value per
    ticker per date, suitable for direct use in FactorModel.

    YFinanceProvider normalises the yfinance "Adj Close" column name to
    "adj_close" (snake_case) at the provider boundary.  Callers downstream
    always use "adj_close".

    All tickers with no data are absent from the result — never all-NaN
    columns.  Raise ProviderError on unrecoverable failure; the coordinator
    catches Exception broadly in _download_and_commit() so a provider
    failure never crashes a run, it just leaves a gap in the cache.
    """

    @abstractmethod
    def fetch(
        self,
        tickers  : List[str],
        start    : str,        # YYYY-MM-DD inclusive
        end      : str,        # YYYY-MM-DD exclusive (yfinance convention)
        frequency: str,        # Frequency constant
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for *tickers* over [start, end).

        Parameters
        ----------
        tickers : list of str
            Tickers to fetch.  One provider call for the full batch.
        start : str
            Start date YYYY-MM-DD, inclusive.
        end : str
            End date YYYY-MM-DD, exclusive (yfinance convention).
        frequency : str
            Use Frequency constants.  Controls whether daily bars or
            intraday bars are fetched.

        Returns
        -------
        pd.DataFrame
            Flat (dates × tickers) DataFrame.
            Columns: ticker names.
            Values:  adj_close prices (float64).
            Index:   DatetimeIndex.

        Raises
        ------
        ProviderError
            On unrecoverable failure (network error, invalid ticker, etc.).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'yfinance'."""


class DataStore(ABC):
    """
    Pure persistence interface with shared logging helpers.

    Reads and writes parquet files, manages the manifest, and runs sweep().
    Has no knowledge of where data came from or what it is for.

    Implementations: ParquetStore (wrapping shockarb.store.DataStore).

    Logging helpers
    ---------------
    Two concrete methods provide consistent, columnar DEBUG-level log output
    to all subclasses without repeating formatting code.

    _fmt(cls_name, operation, status, path, rows) -> str
        Pure static formatter — usable in tests without instantiation.

    _log(operation, status, path, rows) -> str
        Instance wrapper that injects self.__class__.__name__ automatically.
        Use this in all subclass read/write methods.

    Usage in subclasses::

        logger.debug(self._log("read",  "MISS", path))
        logger.debug(self._log("read",  "OK",   path, len(df)))
        logger.debug(self._log("write", "OK",   path, len(df)))
        logger.warning(self._log("read", "FAIL", path))

    Sample output (visible with --log-level DEBUG)::

        [ParquetStore] read  MISS  data\\prices\\daily\\VOO.parquet
        [ParquetStore] read  OK    data\\prices\\daily\\VOO.parquet (34 rows)
        [ParquetStore] write OK    data\\prices\\daily\\VOO.parquet (1050 rows)

    To enable DEBUG output for a single run::

        python -m shockarb --log-level DEBUG build --universe us
        python -m shockarb --log-level DEBUG score --universe us

    Note: --log-level is a global flag and must appear before the subcommand.
    """

    @staticmethod
    def _fmt(
        cls_name : str,
        operation: str,
        status   : str,
        path,
        rows     : int = None,
    ) -> str:
        """
        Format a structured log message for a store operation.

        Parameters
        ----------
        cls_name : str
            Name of the concrete store class (e.g. "ParquetStore").
            Injected automatically by _log(); pass explicitly in tests.
        operation : str
            "read" or "write".
        status : str
            "OK", "MISS", or "FAIL".
        path : path-like
            File path being operated on.
        rows : int, optional
            Row count to append.  Omitted when not applicable (e.g. MISS).

        Returns
        -------
        str
            Fixed-width columnar log string.
        """
        row_str = f" ({rows} rows)" if rows is not None else ""
        return f"[{cls_name}] {operation:<5} {status:<4}  {path}{row_str}"

    def _log(
        self,
        operation: str,
        status   : str,
        path,
        rows     : int = None,
    ) -> str:
        """
        Format a log message for this store instance.

        Convenience wrapper around _fmt() that injects
        self.__class__.__name__ automatically so subclasses never need
        to repeat the class name.

        Parameters
        ----------
        operation : str
            "read" or "write".
        status : str
            "OK", "MISS", or "FAIL".
        path : path-like
            File path being operated on.
        rows : int, optional
            Row count to append.

        Returns
        -------
        str
            Formatted log string ready to pass to logger.debug() or
            logger.warning().
        """
        return self._fmt(self.__class__.__name__, operation, status, path, rows)

    @abstractmethod
    def read(
        self,
        key  : str,
        start: str,
        end  : str,
    ) -> Optional[pd.DataFrame]:
        """
        Return stored DataFrame for *key* over [start, end], or None.

        Never raises — return None on any failure (missing file, corrupt
        parquet, key not in manifest).  The coordinator handles None
        gracefully.

        Key format
        ----------
        "daily/{TICKER}"
            e.g. "daily/VOO" — reads from data/prices/daily/VOO.parquet.
        "intraday/{TICKER}/{DATE}"
            e.g. "intraday/VOO/2026-03-07" — reads from
            data/prices/intraday/VOO_2026-03-07.parquet.

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
        """

    @abstractmethod
    def write(
        self,
        key : str,
        df  : pd.DataFrame,
        meta: dict,
    ) -> None:
        """
        Atomically write *df* and update the manifest for *key*.

        Parameters
        ----------
        key : str
            Store key in "daily/{TICKER}" or "intraday/{TICKER}/{DATE}"
            format.
        df : pd.DataFrame
            Data to write.
        meta : dict
            Metadata for the manifest entry (ticker, frequency, retention).

        Raises
        ------
        ValueError
            If the key format is not recognised.
        """

    @abstractmethod
    def coverage(
        self,
        key: str,
    ) -> Optional[tuple]:
        """
        Return (earliest_date, latest_date) for *key*, or None if not cached.

        Reads from the in-memory manifest only — no filesystem access.
        Used by DataCoordinator._gap_analyse() to decide whether a download
        is needed and how much of the range is already stored.

        Parameters
        ----------
        key : str
            Store key e.g. "daily/VOO".

        Returns
        -------
        tuple of (str, str) or None
            (earliest_date, latest_date) as "YYYY-MM-DD" strings, or None
            if the key is not in the manifest or has no valid date_range.

        Warning
        -------
        A stale manifest.json can cause coverage() to report a false cache
        hit — returning a date range that no longer matches the parquet file
        on disk (e.g. after manual deletion or a crash mid-write).
        Symptom: build() reports "Aligned on 0 common trading days".
        Fix: delete data/manifest.json and rebuild.
        """

    @abstractmethod
    def sweep(
        self,
        retention: str,
        before   : str,
    ) -> List[str]:
        """
        Delete assets of *retention* tier older than *before*.

        Manifest-driven — never globs the filesystem.  Only deletes
        ephemeral/intraday files; never touches the permanent/daily tier.

        Parameters
        ----------
        retention : str
            Tier to sweep, e.g. "ephemeral".
        before : str
            YYYY-MM-DD cutoff date.  Files with trade_date <= cutoff
            are deleted.

        Returns
        -------
        list of str
            Keys that were deleted from the manifest.
        """
