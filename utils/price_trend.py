"""
price_trend.py -- trailing price history for ShockArb universe tickers.

Fetches adj-close prices via the DataCoordinator (parquet cache + gap-analysis
tail-fetch), so repeated calls within a session cost nothing after the first
download.  The same cache is shared with the ShockArb scoring pipeline.

Usage
-----
    python utils/price_trend.py                        # all tickers in live_alpha_us.csv
    python utils/price_trend.py --tickers MSFT BLK ORCL
    python utils/price_trend.py --days 30
    python utils/price_trend.py --csv                  # save summary  → data/price_trend.csv
    python utils/price_trend.py --daily                # save adj-close matrix → data/price_trend_daily.csv
"""

from __future__ import annotations

import argparse
import csv
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
from paths import DATA, LIVE_ALPHA_US, PRICE_TREND_DAILY, PRICE_TREND_SUMMARY


def _build_coordinator() -> DataCoordinator:
    """Return a DataCoordinator wired to the shared parquet cache."""
    inner = _InnerStore(DATA)
    return DataCoordinator(ParquetStore(inner), provider=YFinanceProvider())


def _load_tickers(path: Path) -> list[str]:
    with open(path, newline="") as f:
        return [row[0] for row in csv.reader(f) if row and row[0] != "Ticker"]


def _bar(pct: float, width: int = 12) -> str:
    """ASCII sparkbar: positive = ▲, negative = ▼."""
    filled = min(abs(int(pct / 2)), width)
    return ("▲ " if pct >= 0 else "▼ ") + "█" * filled


def run(
    tickers:    list[str],
    days:       int,
    save_csv:   bool,
    save_daily: bool,
    coordinator: DataCoordinator | None = None,
) -> None:
    """
    Print a trailing price trend table and optionally save CSV output.

    Uses the DataCoordinator to fetch adj-close prices; data already in the
    parquet cache is served without a network call.

    Parameters
    ----------
    tickers    : list of str   Tickers to include.
    days       : int           Trailing window in trading sessions.
    save_csv   : bool          Write per-ticker summary to PRICE_TREND_SUMMARY.
    save_daily : bool          Write full adj-close matrix to PRICE_TREND_DAILY.
    coordinator : DataCoordinator, optional
                               Injected in tests; defaults to _build_coordinator().

    Example
    -------
        run(["MSFT", "BLK"], days=30, save_csv=False, save_daily=True)
        # Prints 30-day table; writes data/price_trend_daily.csv
    """
    today = date.today()
    # Buffer: request extra days so weekends/holidays don't cut the window short.
    start = (today - timedelta(days=days + 20)).isoformat()
    end   = today.isoformat()

    if coordinator is None:
        coordinator = _build_coordinator()

    coordinator.register(DataRequest(
        tickers   = tuple(tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "price_trend",
    ))
    results = coordinator.fulfill()
    closes  = results.get("price_trend", pd.DataFrame())
    closes  = closes.dropna(how="all").tail(days)

    if closes.empty:
        print("No data returned.", file=sys.stderr)
        sys.exit(1)

    start_prices = closes.iloc[0]
    end_prices   = closes.iloc[-1]
    pct_changes  = ((end_prices - start_prices) / start_prices * 100).sort_values()

    start_date = closes.index[0].date()
    end_date   = closes.index[-1].date()

    print(f"\n{days}-Day Price Trend  ({start_date} → {end_date}, {len(closes)} sessions)\n")
    print(f"{'Ticker':<7} {'Start':>8} {'End':>8} {'Chg%':>7}  Trend")
    print("─" * 52)

    rows = []
    for ticker in pct_changes.index:
        s   = start_prices[ticker]
        e   = end_prices[ticker]
        pct = pct_changes[ticker]
        if pd.isna(s) or pd.isna(e):
            continue
        print(f"{ticker:<7} {s:>8.2f} {e:>8.2f} {pct:>+6.1f}%  {_bar(pct)}")
        rows.append({"Ticker": ticker, "Start": round(s, 2), "End": round(e, 2), "Chg_pct": round(pct, 2)})

    if save_csv:
        with open(PRICE_TREND_SUMMARY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Ticker", "Start", "End", "Chg_pct"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved summary to {PRICE_TREND_SUMMARY}")

    if save_daily:
        closes.round(2).to_csv(PRICE_TREND_DAILY)
        print(f"\nSaved daily adj-close matrix ({len(closes)} sessions × {len(closes.columns)} tickers) to {PRICE_TREND_DAILY}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trailing closing price trend")
    ap.add_argument("--tickers", nargs="+", metavar="T", help="Tickers (default: live_alpha_us.csv)")
    ap.add_argument("--days",    type=int, default=60,   help="Trailing window in trading sessions (default 60)")
    ap.add_argument("--csv",     action="store_true",    help="Save per-ticker summary to data/price_trend.csv")
    ap.add_argument("--daily",   action="store_true",    help="Save adj-close matrix to data/price_trend_daily.csv")
    args = ap.parse_args()

    tickers = args.tickers or _load_tickers(LIVE_ALPHA_US)
    if not tickers:
        print("No tickers found.", file=sys.stderr)
        sys.exit(1)

    run(tickers, args.days, args.csv, args.daily)


if __name__ == "__main__":
    main()
