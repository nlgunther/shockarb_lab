"""
datamgr.interfaces — Abstract base classes for the data management layer.

Two ABCs live here and nothing else:

  DataProvider  — pure I/O contract.  One concrete implementation per external
                  data source (YFinanceProvider, PolygonProvider, etc.).

  DataStore     — pure persistence contract.  One concrete implementation for
                  each storage backend (ParquetStore).

Design constraints (must not be relaxed):
  - No imports from shockarb or any application code.
  - No concrete behaviour — interfaces.py is a contract document.
  - DataProvider and DataStore never import each other.
  - Field names are normalised to snake_case at the provider boundary;
    callers always use adj_close, never "Adj Close".

Column contract for DataProvider.fetch()
-----------------------------------------
Returns a MultiIndex DataFrame:
    columns : (field, ticker)   field ∈ {open, high, low, close,
                                          adj_close, adj_factor, volume}
    index   : DatetimeIndex (trading days for daily; timestamps for intraday)

The adj_close contract: pct_change() on adj_close yields total returns
(dividends and splits correctly reflected in adjacent-row ratios).
adj_factor = Adj Close / Close at download time; adj_close = close * adj_factor.
close is write-once — never mutated after first storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


# =============================================================================
# DataProvider
# =============================================================================

class DataProvider(ABC):
    """
    Pure I/O.  Fetches raw data from one external source.

    Implementations must normalise all field names to snake_case before
    returning — the caller should never see "Adj Close" or "High".
    Tickers with no data are absent from the result; never all-NaN columns.

    Raise ProviderError on unrecoverable failure (bad credentials, hard 404).
    Log and drop on per-ticker soft failures (delisted tickers in a batch).
    """

    @abstractmethod
    def fetch(
        self,
        tickers: List[str],
        start: str,          # YYYY-MM-DD, inclusive
        end: str,            # YYYY-MM-DD, exclusive
        frequency: str,      # Frequency constant
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for *tickers* over [start, end).

        Parameters
        ----------
        tickers : list of str
        start   : str   YYYY-MM-DD inclusive
        end     : str   YYYY-MM-DD exclusive (yfinance convention)
        frequency : str  Use Frequency constants, e.g. Frequency.DAILY

        Returns
        -------
        DataFrame
            MultiIndex columns (field, ticker).
            Fields: open, high, low, close, adj_close, adj_factor, volume.
            Index: DatetimeIndex.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging and manifest provenance, e.g. 'yfinance'."""


class ProviderError(Exception):
    """Raised by DataProvider.fetch() on unrecoverable failure."""


# =============================================================================
# DataStore
# =============================================================================

class DataStore(ABC):
    """
    Pure persistence.  Reads and writes data assets.

    Implementations manage parquet files, atomic writes, the manifest, WAL,
    and sweep().  They have no knowledge of where data came from or how it
    will be used.

    Key invariants enforced by every implementation:
      - All writes are atomic (temp file + rename).
      - adj_close == close * adj_factor is asserted on every daily write.
      - close is never mutated after first write.
      - sweep() is manifest-driven — never globs the filesystem.
    """

    @abstractmethod
    def read(
        self,
        key: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Read stored data for *key* over [start, end).

        Parameters
        ----------
        key   : str   Storage key, e.g. "daily/VOO" or "intraday/VOO/2026-03-07"
        start : str   YYYY-MM-DD
        end   : str   YYYY-MM-DD

        Returns
        -------
        DataFrame or None
            None if the key does not exist.
        """

    @abstractmethod
    def write(
        self,
        key: str,
        df: pd.DataFrame,
        meta: dict,
    ) -> None:
        """
        Atomically persist *df* under *key*, updating the manifest with *meta*.

        Parameters
        ----------
        key  : str
        df   : DataFrame
        meta : dict   Provenance fields merged into the manifest entry.
        """

    @abstractmethod
    def coverage(
        self,
        key: str,
    ) -> Optional[tuple[str, str]]:
        """
        Return (earliest_date, latest_date) for *key*, or None if not stored.

        Used by the coordinator for gap analysis.  Dates are YYYY-MM-DD strings.
        """

    @abstractmethod
    def sweep(
        self,
        retention: str,
        before: str,
    ) -> List[str]:
        """
        Delete stored assets of given *retention* tier older than *before*.

        Parameters
        ----------
        retention : str   e.g. "ephemeral" to delete intraday files
        before    : str   YYYY-MM-DD cutoff; assets with date < before are deleted

        Returns
        -------
        list of str
            Keys of deleted assets.
        """
