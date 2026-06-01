"""
Surgical fetch: add SYK to the existing backtest price cache.

The backtest cache (backtest_us_static_ohlcv.parquet) was saved without SYK
due to a yfinance rate limit. This script fetches SYK for the same date range
and merges it into the existing parquet in-place. No other tickers are touched.

Usage
-----
    python utils/fetch_syk.py

    # Dry run — show what would be fetched without writing
    python utils/fetch_syk.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


CACHE_PATH = Path("data/cache/backtest_us_static_ohlcv.parquet")
TICKER = "SYK"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without writing")
    parser.add_argument("--cache", default=str(CACHE_PATH),
                        help=f"Path to the cache parquet (default: {CACHE_PATH})")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        print("Run the backtest once first to create it.")
        return

    existing = pd.read_parquet(cache_path)
    if TICKER in existing.columns.get_level_values(1):
        print(f"{TICKER} is already in the cache — nothing to do.")
        return

    start = existing.index[0].strftime("%Y-%m-%d")
    end   = existing.index[-1].strftime("%Y-%m-%d")
    print(f"Fetching {TICKER}  {start} → {end} ...")

    if args.dry_run:
        print("(dry run — not writing)")
        return

    raw = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        print(f"No data returned for {TICKER}. Rate limit may still be active — try again in a minute.")
        return

    # Reindex to match the cache's trading days exactly
    raw = raw.reindex(existing.index)

    # Build MultiIndex columns matching the cache structure
    syk_cols = pd.MultiIndex.from_tuples(
        [(field, TICKER) for field in raw.columns],
        names=existing.columns.names,
    )
    syk_df = raw.copy()
    syk_df.columns = syk_cols

    merged = pd.concat([existing, syk_df], axis=1).sort_index(axis=1)
    merged.to_parquet(cache_path)

    print(f"Done. Cache updated: {cache_path}")
    print(f"  Before: {existing.shape[1]} columns  →  After: {merged.shape[1]} columns")
    print(f"  {TICKER} NaN rows: {syk_df[('Adj Close', TICKER)].isna().sum()} of {len(syk_df)}")


if __name__ == "__main__":
    main()
