"""
load_tickers.py — Load per-ticker parquet files into a single MultiIndex DataFrame.

Usage
-----
    python load_tickers.py VOO TLT GLD
    python load_tickers.py VOO TLT GLD --data-dir data/prices/daily
    python load_tickers.py VOO TLT GLD --start 2022-02-10 --end 2022-03-31

Output
------
    MultiIndex DataFrame with (field, ticker) columns and DatetimeIndex rows.
    Prints shape and head to stdout.
"""

import argparse
import os
import sys

import pandas as pd


def load_tickers(
    tickers: list[str],
    data_dir: str = "data/prices/daily",
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Load per-ticker parquet files into a single MultiIndex DataFrame.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to load.
    data_dir : str
        Directory containing {TICKER}.parquet files.
    start : str, optional
        Inclusive start date YYYY-MM-DD.
    end : str, optional
        Inclusive end date YYYY-MM-DD.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (field, ticker).
        Index: DatetimeIndex.
    """
    frames = {}
    missing = []

    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}.parquet")
        if not os.path.exists(path):
            missing.append(ticker)
            continue
        df = pd.read_parquet(path)
        if start or end:
            df = df.loc[start:end]
        frames[ticker] = df

    if missing:
        print(f"WARNING: no parquet file found for: {missing}", file=sys.stderr)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {data_dir} for {tickers}")

    combined = pd.concat(frames, axis=1)  # MultiIndex: (ticker, field)
    combined.columns = combined.columns.swaplevel(0, 1)  # → (field, ticker)
    combined.sort_index(axis=1, inplace=True)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Load ticker parquet files into a MultiIndex DataFrame.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to load")
    parser.add_argument("--data-dir", default="data/prices/daily", help="Directory containing parquet files")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    df = load_tickers(
        tickers  = [t.upper() for t in args.tickers],
        data_dir = args.data_dir,
        start    = args.start,
        end      = args.end,
    )

    print(f"\nLoaded {len(args.tickers)} ticker(s) — shape: {df.shape}")
    print(f"Fields:  {df.columns.get_level_values(0).unique().tolist()}")
    print(f"Tickers: {df.columns.get_level_values(1).unique().tolist()}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print()
    print(df.head())

    return df


if __name__ == "__main__":
    main()
