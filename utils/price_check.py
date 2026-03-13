"""
Price Check — read yesterday's close, today's open, and current price.

Computes the overnight return (open / prev_close - 1) and the intraday
return (current / prev_close - 1) for any set of tickers.

Usage:
    python utils/price_check.py VOO TLT XOM
    python utils/price_check.py VOO --source cache     # use parquet cache for prev close
    python utils/price_check.py VOO --source yfinance   # use yfinance for prev close (default)
    python utils/price_check.py --universe us           # all tickers from a universe config
    python utils/price_check.py VOO --json              # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def get_prev_close_from_cache(tickers: List[str], data_dir: str = "data") -> Dict[str, dict]:
    """Read yesterday's adj_close from the parquet cache."""
    daily_dir = Path(data_dir) / "prices" / "daily"
    results = {}
    for ticker in tickers:
        path = daily_dir / f"{ticker}.parquet"
        if not path.exists():
            results[ticker] = {"prev_close": None, "prev_date": None, "source": "cache", "error": "file not found"}
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                results[ticker] = {"prev_close": None, "prev_date": None, "source": "cache", "error": "empty file"}
                continue
            # Get the adj_close column
            if "adj_close" in df.columns:
                col = "adj_close"
            elif "Adj Close" in df.columns:
                col = "Adj Close"
            else:
                col = df.columns[0]
            last_val = float(df[col].iloc[-1])
            last_date = df.index[-1]
            results[ticker] = {
                "prev_close": last_val,
                "prev_date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date),
                "field": col,
                "source": "cache",
            }
        except Exception as e:
            results[ticker] = {"prev_close": None, "prev_date": None, "source": "cache", "error": str(e)}
    return results


def get_prev_close_from_yfinance(tickers: List[str]) -> Dict[str, dict]:
    """Fetch yesterday's adj_close from yfinance daily bars."""
    import yfinance as yf
    results = {}
    try:
        raw = yf.download(tickers, period="5d", interval="1d",
                          auto_adjust=False, progress=False)
        if raw.empty:
            for t in tickers:
                results[t] = {"prev_close": None, "prev_date": None, "source": "yfinance", "error": "empty response"}
            return results

        # Get adj close
        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" in raw.columns.get_level_values(0):
                prices = raw["Adj Close"]
            else:
                prices = raw["Close"]
        else:
            prices = raw

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        for ticker in tickers:
            if ticker not in prices.columns:
                results[ticker] = {"prev_close": None, "prev_date": None, "source": "yfinance", "error": "ticker not in response"}
                continue
            series = prices[ticker].dropna()
            if len(series) < 1:
                results[ticker] = {"prev_close": None, "prev_date": None, "source": "yfinance", "error": "no data"}
                continue

            # Yesterday = last completed trading day (second to last if today is in the data, else last)
            today = date.today()
            dates = [d.date() for d in series.index]
            if today in dates:
                # Today is in the daily data — prev close is the row before today
                today_idx = dates.index(today)
                if today_idx == 0:
                    results[ticker] = {"prev_close": None, "prev_date": None, "source": "yfinance", "error": "no prev day"}
                    continue
                prev_val = float(series.iloc[today_idx - 1])
                prev_date = str(dates[today_idx - 1])
                today_close = float(series.iloc[today_idx])
                results[ticker] = {
                    "prev_close": prev_val,
                    "prev_date": prev_date,
                    "today_daily_close": today_close,
                    "today_daily_date": str(today),
                    "field": "Adj Close",
                    "source": "yfinance",
                }
            else:
                # Today not in data — last row is the prev close
                prev_val = float(series.iloc[-1])
                prev_date = str(dates[-1])
                results[ticker] = {
                    "prev_close": prev_val,
                    "prev_date": prev_date,
                    "field": "Adj Close",
                    "source": "yfinance",
                }
    except Exception as e:
        for t in tickers:
            results[t] = {"prev_close": None, "prev_date": None, "source": "yfinance", "error": str(e)}
    return results


def get_intraday_prices(tickers: List[str]) -> Dict[str, dict]:
    """Fetch today's open and current price from yfinance 15m bars."""
    import yfinance as yf
    results = {}
    try:
        raw = yf.download(tickers, period="1d", interval="15m",
                          auto_adjust=False, progress=False)
        if raw.empty:
            for t in tickers:
                results[t] = {"open": None, "current": None, "error": "empty response"}
            return results

        # Extract close prices (latest bar = current)
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                closes = raw["Close"]
            else:
                closes = raw.iloc[:, :len(tickers)]
            if "Open" in raw.columns.get_level_values(0):
                opens = raw["Open"]
            else:
                opens = closes
        else:
            closes = raw
            opens = raw

        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=tickers[0])
        if isinstance(opens, pd.Series):
            opens = opens.to_frame(name=tickers[0])

        for ticker in tickers:
            if ticker not in closes.columns:
                results[ticker] = {"open": None, "current": None, "error": "not in response"}
                continue

            close_series = closes[ticker].dropna()
            open_series = opens[ticker].dropna()

            if close_series.empty:
                results[ticker] = {"open": None, "current": None, "error": "no bars"}
                continue

            today_open = float(open_series.iloc[0])
            open_ts = str(open_series.index[0])
            current = float(close_series.iloc[-1])
            current_ts = str(close_series.index[-1])
            n_bars = len(close_series)

            results[ticker] = {
                "open": today_open,
                "open_ts": open_ts,
                "current": current,
                "current_ts": current_ts,
                "n_bars": n_bars,
            }
    except Exception as e:
        for t in tickers:
            results[t] = {"open": None, "current": None, "error": str(e)}
    return results


def print_report(tickers: List[str], prev: Dict, intraday: Dict) -> None:
    """Print a formatted report."""
    print(f"\n  {'Ticker':<8s} {'Prev Close':>12s} {'Prev Date':>12s} "
          f"{'Open':>12s} {'Current':>12s} {'O/C Ret':>10s} {'Curr/C Ret':>10s} "
          f"{'Bars':>5s}")
    print("  " + "─" * 95)

    for ticker in tickers:
        p = prev.get(ticker, {})
        i = intraday.get(ticker, {})

        if p.get("error") or i.get("error"):
            err = p.get("error") or i.get("error")
            print(f"  {ticker:<8s}  ❌ {err}")
            continue

        pc = p.get("prev_close")
        pd_str = p.get("prev_date", "?")
        opn = i.get("open")
        cur = i.get("current")
        bars = i.get("n_bars", 0)

        pc_str = f"{pc:12.2f}" if pc else f"{'N/A':>12s}"
        opn_str = f"{opn:12.2f}" if opn else f"{'N/A':>12s}"
        cur_str = f"{cur:12.2f}" if cur else f"{'N/A':>12s}"

        # Overnight return: open / prev_close - 1
        if pc and opn:
            oc_ret = (opn / pc) - 1
            oc_str = f"{oc_ret*100:+.4f}%"
        else:
            oc_str = "N/A"

        # Current return: current / prev_close - 1
        if pc and cur:
            cur_ret = (cur / pc) - 1
            cur_str2 = f"{cur_ret*100:+.4f}%"
        else:
            cur_str2 = "N/A"

        print(f"  {ticker:<8s} {pc_str} {pd_str:>12s} {opn_str} {cur_str} {oc_str:>10s} {cur_str2:>10s} {bars:5d}")

    # Timestamps
    sample = next((i for i in intraday.values() if "open_ts" in i), None)
    if sample:
        print(f"\n  Open bar:    {sample.get('open_ts', '?')}")
        print(f"  Current bar: {sample.get('current_ts', '?')}")

    sample_prev = next((p for p in prev.values() if "prev_date" in p), None)
    if sample_prev:
        print(f"  Prev close:  {sample_prev.get('prev_date', '?')} ({sample_prev.get('source', '?')})")


def main():
    parser = argparse.ArgumentParser(
        description="Check yesterday's close, today's open, and current price."
    )
    parser.add_argument("tickers", nargs="*", help="Ticker symbols")
    parser.add_argument("--universe", default=None,
                        help="Load tickers from a universe config (us/global)")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "cache"],
                        help="Source for prev close (default: yfinance)")
    parser.add_argument("--data-dir", default="data",
                        help="Data directory for cache source")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    # Resolve tickers
    tickers = args.tickers
    if args.universe:
        try:
            from shockarb.config import US_UNIVERSE, GLOBAL_UNIVERSE
            u = {"us": US_UNIVERSE, "global": GLOBAL_UNIVERSE}.get(args.universe.lower())
            if u:
                tickers = list(u.market_etfs) + list(u.individual_stocks)
            else:
                print(f"❌ Unknown universe: {args.universe}")
                sys.exit(1)
        except ImportError:
            print("❌ Could not import shockarb.config — run from shockarb_lab root")
            sys.exit(1)

    if not tickers:
        parser.print_help()
        sys.exit(1)

    import pytz
    now_et = datetime.now(pytz.timezone("America/New_York"))
    print("=" * 100)
    print(f"  PRICE CHECK  |  {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}  |  {len(tickers)} ticker(s)")
    print("=" * 100)

    # Get prev close
    if args.source == "cache":
        prev = get_prev_close_from_cache(tickers, args.data_dir)
    else:
        prev = get_prev_close_from_yfinance(tickers)

    # Get intraday
    intraday = get_intraday_prices(tickers)

    if args.json:
        output = {}
        for t in tickers:
            output[t] = {"prev": prev.get(t, {}), "intraday": intraday.get(t, {})}
        print(json.dumps(output, indent=2))
    else:
        print_report(tickers, prev, intraday)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
