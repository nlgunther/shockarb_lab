"""
datamgr.providers.mock — MockProvider for unit tests.

Generates deterministic synthetic OHLCV data without any network calls.
Inject this instead of YFinanceProvider in all tests.

Fields returned use snake_case (the normalised contract):
    open, high, low, close, adj_close, adj_factor, volume

Usage
-----
    from datamgr.providers.mock import MockProvider
    from datamgr.requests import Frequency

    provider = MockProvider()
    df = provider.fetch(["VOO", "TLT"], "2022-02-10", "2022-03-31", Frequency.DAILY)
    assert ("adj_close", "VOO") in df.columns
"""

from __future__ import annotations

from typing import List

import pandas as pd

from datamgr.interfaces import DataProvider
from datamgr.requests import Frequency


class MockProvider(DataProvider):
    """
    Deterministic OHLCV provider for tests.

    All prices are 100.0; adj_factor is 1.0 (no corporate actions by default).
    Override fetch() on the instance for specific test scenarios:

        provider = MockProvider()
        provider.fetch = lambda tickers, start, end, frequency: my_custom_df

    Parameters
    ----------
    call_log : list, optional
        If provided, each fetch() call appends its arguments for assertion.
    adj_factor_override : float
        Default adj_factor for all rows.  Change to test restatement logic.
    """

    def __init__(
        self,
        call_log: List[dict] | None = None,
        adj_factor_override: float = 1.0,
    ) -> None:
        self._call_log = call_log if call_log is not None else []
        self._adj_factor = adj_factor_override

    @property
    def name(self) -> str:
        return "mock"

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        frequency: str,
    ) -> pd.DataFrame:
        """
        Return a synthetic MultiIndex DataFrame with all values = 100.0.

        adj_close = close * adj_factor = 100.0 * adj_factor_override.
        """
        self._call_log.append({
            "tickers":   list(tickers),
            "start":     start,
            "end":       end,
            "frequency": frequency,
        })

        if frequency == Frequency.DAILY:
            idx = pd.bdate_range(start=start, end=end)
        else:
            # Intraday: 26 bars of 15-minute data anchored to start date
            idx = pd.date_range(
                f"{start} 09:30",
                periods=26,
                freq="15min",
                tz="America/New_York",
            )

        if idx.empty or not tickers:
            return pd.DataFrame()

        fields = ["open", "high", "low", "close", "adj_close", "adj_factor", "volume"]
        cols = pd.MultiIndex.from_product([fields, tickers])
        df = pd.DataFrame(100.0, index=idx, columns=cols)

        # Set adj_factor and re-derive adj_close correctly
        for ticker in tickers:
            df[("adj_factor", ticker)] = self._adj_factor
            df[("adj_close", ticker)] = df[("close", ticker)] * self._adj_factor

        return df
