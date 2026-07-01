"""
refresh_prices.py — top up the daily OHLCV cache via the DataCoordinator.

Gap-analyses the existing parquet cache and downloads only the missing tail,
then writes to data/prices/daily/{TICKER}.parquet — the same store that the
scoring pipeline and price_trend.py read from.

Usage
-----
    python utils/refresh_prices.py ETN HON ISRG
    python utils/refresh_prices.py ETN HON ISRG --days 30

    # No tickers → refresh every ticker in live_alpha_us.csv
    python utils/refresh_prices.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from shockarb.store import DataStore as _InnerStore
from datamgr.coordinator import DataCoordinator
from datamgr.stores.parquet import ParquetStore
from datamgr.providers.yfinance import YFinanceProvider
from datamgr.requests import DataRequest, Frequency
from paths import DATA, LIVE_ALPHA_US


def _load_tickers(path: Path) -> list[str]:
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return [row["Ticker"] for row in csv.DictReader(f) if row.get("Ticker")]


def refresh(tickers: list[str], days: int) -> None:
    """
    Ensure *days* calendar days of daily OHLCV history are cached for each ticker.

    The coordinator's gap analysis means only missing dates are downloaded.
    Already-current tickers generate zero network calls.

    Example
    -------
        refresh(["MSFT", "BLK", "ORCL"], days=30)
        # → "MSFT: cache up to date through 2026-06-25"
        # → ...
    """
    start = (date.today() - timedelta(days=days)).isoformat()
    end   = date.today().isoformat()

    inner      = _InnerStore(DATA)
    coordinator = DataCoordinator(ParquetStore(inner), provider=YFinanceProvider())
    coordinator.register(DataRequest(
        tickers   = tuple(tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "refresh_prices",
    ))
    results = coordinator.fulfill()
    prices  = results.get("refresh_prices", pd.DataFrame())

    if prices.empty:
        print("No data returned.")
        return

    for ticker in prices.columns:
        last = prices[ticker].dropna()
        if last.empty:
            print(f"{ticker}: no data returned")
        else:
            print(f"{ticker}: cache up to date through {last.index.max().date()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tickers", nargs="*",
                        help="Tickers to refresh (default: all in live_alpha_us.csv)")
    parser.add_argument("--days", type=int, default=30,
                        help="Calendar days of history to ensure (default: 30)")
    args = parser.parse_args()

    tickers = args.tickers or _load_tickers(LIVE_ALPHA_US)
    if not tickers:
        print("No tickers found.")
        return

    refresh(tickers, args.days)


if __name__ == "__main__":
    main()
