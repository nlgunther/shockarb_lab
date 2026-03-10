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
    Pure I/O interface.  Fetches raw data from one external source.

    Implementations: YFinanceProvider, MockProvider.
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
        Return MultiIndex DataFrame: columns (field, ticker), DatetimeIndex rows.

        Fields (normalised, snake_case):
            open, high, low, close, adj_close, adj_factor, volume

        CONTRACT:
            adj_close = close * adj_factor
            pct_change() on adj_close yields total returns.

        Tickers with no data are absent — never all-NaN columns.
        Raises ProviderError on unrecoverable failure.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'yfinance'."""


class DataStore(ABC):
    """
    Pure persistence interface.  Reads/writes parquet files, manages manifest.

    Implementations: ParquetStore (wrapping shockarb.store.DataStore).
    """

    @abstractmethod
    def read(
        self,
        key  : str,
        start: str,
        end  : str,
    ) -> Optional[pd.DataFrame]:
        """
        Return stored DataFrame for *key* over [start, end], or None.

        Key format:
            "daily/{TICKER}"                 e.g. "daily/VOO"
            "intraday/{TICKER}/{DATE}"       e.g. "intraday/VOO/2026-03-07"
        """

    @abstractmethod
    def write(
        self,
        key : str,
        df  : pd.DataFrame,
        meta: dict,
    ) -> None:
        """Atomically write *df* and update the manifest for *key*."""

    @abstractmethod
    def coverage(
        self,
        key: str,
    ) -> Optional[tuple]:
        """
        Return (earliest_date, latest_date) for *key*, or None if not cached.

        Reads from manifest only — no filesystem access.
        """

    @abstractmethod
    def sweep(
        self,
        retention: str,
        before   : str,
    ) -> List[str]:
        """
        Delete assets of *retention* tier older than *before* (YYYY-MM-DD).
        Returns list of deleted keys.
        Never touches the permanent/daily tier.
        """
