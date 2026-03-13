"""
datamgr.providers.yfinance — YFinanceProvider: DataProvider backed by yfinance.

Normalises yfinance output to the DataProvider contract:
  - Column names lowercased to snake_case (Adj Close → adj_close)
  - adj_factor = Adj Close / Close computed at download time
  - adj_close  = close * adj_factor (re-derived, not ratio-multiplied)
  - MultiIndex columns: (field, ticker)

This is the only file in datamgr that imports yfinance.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from loguru import logger

from datamgr.interfaces import DataProvider, ProviderError
from datamgr.requests import Frequency

# yfinance is imported lazily inside __init__ so that test environments that
# patch yf.download at the pipeline level still work.  The provider itself
# holds a reference to the downloader callable (defaulting to yf.download)
# so tests can inject a fake via YFinanceProvider(downloader=fake_dl).


class YFinanceProvider(DataProvider):
    """
    DataProvider backed by yfinance.

    Parameters
    ----------
    downloader : callable, optional
        Replacement for yf.download — injected by tests.
        Must match the yf.download(tickers, ...) signature.
    """

    def __init__(self, downloader=None) -> None:
        if downloader is not None:
            self._dl = downloader
        else:
            import yfinance as yf
            self._dl = yf.download

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        frequency: str,
    ) -> pd.DataFrame:
        """
        Download OHLCV data and return a normalised MultiIndex DataFrame.

        Columns: (field, ticker) where field in
            {open, high, low, close, adj_close, adj_factor, volume}

        Parameters
        ----------
        tickers   : list of str
        start, end: str  YYYY-MM-DD
        frequency : str  Frequency constant

        Returns
        -------
        pd.DataFrame with MultiIndex columns, DatetimeIndex rows.

        Raises
        ------
        ProviderError  on unrecoverable download failure.
        """
        interval = self._frequency_to_interval(frequency)

        try:
            if frequency == Frequency.DAILY:
                # yfinance's end parameter is EXCLUSIVE — end="2026-03-12"
                # returns data through 2026-03-11.  Bump by 1 day so the
                # caller's intended end date is included in the result.
                from datetime import date as _date, timedelta
                yf_end = (
                    _date.fromisoformat(end) + timedelta(days=1)
                ).isoformat()
                raw = self._dl(
                    tickers,
                    start       = start,
                    end         = yf_end,
                    interval    = interval,
                    auto_adjust = False,
                    progress    = False,
                )
            else:
                # Intraday: yfinance rejects start/end for sub-daily intervals.
                # period="1d" always covers the full current session.
                raw = self._dl(
                    tickers,
                    period      = "1d",
                    interval    = interval,
                    auto_adjust = False,
                    progress    = False,
                )
        except Exception as exc:
            raise ProviderError(f"yfinance download failed: {exc}") from exc

        if raw is None or raw.empty:
            return pd.DataFrame()

        return self._normalise(raw, tickers)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _frequency_to_interval(frequency: str) -> str:
        mapping = {
            Frequency.DAILY:        "1d",
            Frequency.INTRADAY_15M: "15m",
            Frequency.INTRADAY_1M:  "1m",
        }
        interval = mapping.get(frequency)
        if interval is None:
            raise ProviderError(f"Unsupported frequency: {frequency!r}")
        return interval

    @staticmethod
    def _normalise(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Normalise a yfinance MultiIndex result to the DataProvider contract.

        yfinance returns (field_TitleCase, ticker) MultiIndex.
        We normalise to (field_snake_case, ticker) and add adj_factor.
        """
        if not isinstance(raw.columns, pd.MultiIndex):
            # Single ticker — yfinance may return flat columns
            raw.columns = pd.MultiIndex.from_tuples(
                [(col, tickers[0]) for col in raw.columns]
            )

        # Rename field level to snake_case
        rename_map = {
            "Open":      "open",
            "High":      "high",
            "Low":       "low",
            "Close":     "close",
            "Adj Close": "adj_close",
            "Volume":    "volume",
        }
        raw.columns = pd.MultiIndex.from_tuples(
            [(rename_map.get(f, f.lower()), t) for f, t in raw.columns]
        )

        # Compute adj_factor = adj_close / close; add to DataFrame
        pieces = []
        for ticker in raw.columns.get_level_values(1).unique():
            try:
                ticker_df = raw.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                continue

            if "close" in ticker_df.columns and "adj_close" in ticker_df.columns:
                close               = ticker_df["close"].replace(0, float("nan"))
                ticker_df["adj_factor"] = ticker_df["adj_close"] / close
                # Re-derive adj_close from close * adj_factor (contract)
                ticker_df["adj_close"]  = ticker_df["close"] * ticker_df["adj_factor"]
            else:
                ticker_df["adj_factor"] = 1.0

            ticker_df.columns = pd.MultiIndex.from_tuples(
                [(col, ticker) for col in ticker_df.columns]
            )
            pieces.append(ticker_df)

        if not pieces:
            return pd.DataFrame()

        result = pd.concat(pieces, axis=1).sort_index(axis=1).sort_index(axis=0)
        result.index.name = "Date"
        return result
