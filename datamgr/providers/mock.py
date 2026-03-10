"""
datamgr.providers.mock — MockProvider for unit tests.

Generates deterministic synthetic OHLCV data without any network calls.
Inject this instead of YFinanceProvider in all tests that exercise the
coordinator, pipeline, or any code that registers DataRequests.

Output shape
------------
MockProvider.fetch() returns a flat (dates × tickers) DataFrame with ticker
names as columns and a constant value of 1.0 as prices.  This is the same
shape that YFinanceProvider produces after normalisation, and the shape that
DataCoordinator._read_daily() expects.

    columns: ["VOO", "TLT", "VGT", ...]
    index:   DatetimeIndex (business days for DAILY, 15-min bars for intraday)
    values:  1.0 (constant — no corporate actions, no drift)

Usage
-----
    from datamgr.providers.mock import MockProvider
    from datamgr.requests import Frequency

    provider = MockProvider()
    df = provider.fetch(["VOO", "TLT"], "2022-02-10", "2022-03-31", Frequency.DAILY)
    assert "VOO" in df.columns
    assert df["VOO"].iloc[0] == 1.0

Injecting into the coordinator
-------------------------------
    store    = InMemoryStore()   # from tests/helpers.py
    provider = MockProvider()
    coord    = DataCoordinator(store, provider=provider)

Overriding for specific test scenarios
---------------------------------------
    provider = MockProvider()
    provider.fetch = lambda tickers, start, end, frequency: my_custom_df

Logging provider calls for assertion
--------------------------------------
    call_log = []
    provider = MockProvider(call_log=call_log)
    coord.fulfill()
    assert call_log[0]["tickers"] == ["VOO", "TLT"]
"""

from __future__ import annotations

from typing import List

import pandas as pd

from datamgr.interfaces import DataProvider
from datamgr.requests import Frequency


class MockProvider(DataProvider):
    """
    Deterministic OHLCV provider for unit tests.

    All prices are 1.0 and adj_factor is 1.0 (no corporate actions by
    default).  The constant price series means pct_change() always returns
    0.0 after the first row — factor model tests should seed InMemoryStore
    with realistic return data rather than relying on MockProvider for
    return variance.

    Parameters
    ----------
    call_log : list of dict, optional
        If provided, each fetch() call appends a dict of its arguments::

            {"tickers": [...], "start": "...", "end": "...", "frequency": "..."}

        Useful for asserting that the coordinator made the expected provider
        calls without a real network round-trip.
    adj_factor_override : float
        adj_factor value for all rows.  Change to test restatement logic in
        _commit_ticker() (e.g. adj_factor_override=0.5 simulates a 2:1 split).
        Currently stored for reference; the flat DataFrame output does not
        include a separate adj_factor column.
    """

    def __init__(
        self,
        call_log: List[dict] | None = None,
        adj_factor_override: float = 1.0,
    ) -> None:
        self._call_log   = call_log if call_log is not None else []
        self._adj_factor = adj_factor_override

    @property
    def name(self) -> str:
        """Provider identifier, always "mock"."""
        return "mock"

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        frequency: str,
    ) -> pd.DataFrame:
        """
        Return a flat (dates × tickers) DataFrame with all values = 1.0.

        Date index
        ----------
        DAILY        — pd.bdate_range(start, end): business days only.
        INTRADAY_15M — 26 × 15-minute bars from 09:30 ET on *start*
                       (enough for one full trading session).

        Parameters
        ----------
        tickers : list of str
            Columns in the returned DataFrame.
        start : str
            Start date YYYY-MM-DD, inclusive.
        end : str
            End date YYYY-MM-DD, inclusive for DAILY; ignored for intraday
            (period is always 26 bars from start).
        frequency : str
            Use Frequency constants.  Determines the index type.

        Returns
        -------
        pd.DataFrame
            (dates × tickers) with ticker names as columns and 1.0 values.
            Empty DataFrame if the date range produces no bars or tickers
            is empty.
        """
        self._call_log.append({
            "tickers": tickers, "start": start, "end": end, "frequency": frequency,
        })

        if frequency == Frequency.DAILY:
            idx = pd.bdate_range(start, end)
        else:
            idx = pd.date_range(
                f"{start} 09:30", periods=26, freq="15min", tz="America/New_York"
            )

        if idx.empty or not tickers:
            return pd.DataFrame()

        # Flat (dates × tickers) shape — same as YFinanceProvider output.
        return pd.DataFrame(1.0, index=idx, columns=tickers)
