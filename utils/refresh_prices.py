"""
refresh_prices.py — manually top up the daily OHLCV (incl. Volume) cache.

Downloads via yfinance through the existing DataStore, which does an
incremental tail-fetch (only missing days) and writes
data/prices/daily/{TICKER}.parquet — the same cache stockfit's --rvol
reads from.

Usage
-----
    cd utils
    python refresh_prices.py ETN HON ISRG
    python refresh_prices.py ETN HON ISRG --days 30

    # No tickers given -> refresh every ticker in live_alpha_us.csv
    python refresh_prices.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shockarb.store import DataStore
from paths import DATA, LIVE_ALPHA_US


def _tickers_from_scores(path: Path) -> list[str]:
    import csv
    with open(path, encoding="utf-8") as f:
        return [row["Ticker"] for row in csv.DictReader(f) if row.get("Ticker")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*",
                         help="Tickers to refresh (default: all in live_alpha_us.csv)")
    parser.add_argument("--days", type=int, default=30,
                         help="How many calendar days of history to ensure (default: 30)")
    args = parser.parse_args()

    tickers = args.tickers or _tickers_from_scores(LIVE_ALPHA_US)
    if not tickers:
        print("No tickers found.")
        return

    start = (date.today() - timedelta(days=args.days)).isoformat()
    end   = date.today().isoformat()

    store = DataStore(DATA)
    for ticker in tickers:
        prices = store.fetch_daily([ticker], start, end)
        if prices.empty:
            print(f"{ticker}: no data returned")
            continue
        last_date = prices.index.max().date()
        print(f"{ticker}: cache up to date through {last_date}")


if __name__ == "__main__":
    main()
