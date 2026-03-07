"""
ShockArb Portfolio Sizing Utility.

Reads one or more ShockArb score CSVs, selects the top-N positive signals
by conviction (confidence_delta), and prints a dollar-denominated trade
ticket with allocation weights and take-profit limit prices.

Usage examples
--------------
    # Size $100k across the top 5 US signals
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000

    # Merge US + Global into a single ticket
    python utils/portfolio_sizer.py \
        --csv data/live_alpha_us.csv data/live_alpha_global.csv \
        --capital 50000 --top 8
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yfinance as yf
from loguru import logger


def generate_orders(
    csv_paths: list[str],
    capital: float,
    top_n: int = 5,
) -> None:
    """
    Print a trade ticket for the top-N conviction signals.

    Parameters
    ----------
    csv_paths : list of str
        Paths to ShockArb score CSVs.  Multiple files are merged before ranking.
    capital : float
        Total dollar capital to allocate.
    top_n : int
        Number of positions to take.
    """
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            logger.warning(f"Alpha report not found: {path}")
            continue
        try:
            df = pd.read_csv(path)
            if "Ticker" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Ticker"})
            dfs.append(df)
        except Exception as exc:
            logger.error(f"Failed to read {path}: {exc}")

    if not dfs:
        logger.error("No valid CSVs loaded.")
        return

    master = pd.concat(dfs, ignore_index=True)

    required = {"confidence_delta", "delta_rel"}
    missing = required - set(master.columns)
    if missing:
        logger.error(f"CSV is missing required columns: {missing}")
        logger.error(f"  Available columns: {list(master.columns)}")
        return

    buys = (
        master[master["confidence_delta"] > 0]
        .sort_values("confidence_delta", ascending=False)
        .head(top_n)
    )

    if buys.empty:
        logger.warning("No positive alpha signals found.")
        return

    # Fetch live prices
    tickers = buys["Ticker"].tolist()
    logger.info(f"Fetching live prices for: {tickers}")
    raw = yf.download(tickers, period="1d", progress=False, auto_adjust=False)

    # Resolve price series robustly (MultiIndex or flat)
    if isinstance(raw.columns, pd.MultiIndex):
        price_col = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        current = raw[price_col].iloc[-1]
    else:
        current = raw.iloc[-1]

    # Conviction-weighted allocation
    total_conviction = buys["confidence_delta"].sum()
    buys = buys.copy()
    buys["Weight"]       = buys["confidence_delta"] / total_conviction
    buys["Dollar_Alloc"] = buys["Weight"] * capital

    # Print ticket
    print("\n" + "=" * 100)
    print(f"  SHOCKARB TRADE TICKET  |  Capital: ${capital:,.2f}  |  Positions: {len(buys)}")
    print("=" * 100)
    print(f"  {'TICKER':<8}  {'WEIGHT':>8}  {'ALLOCATION':>14}  {'CURRENT':>10}  {'TARGET':>10}  SHARES")
    print("-" * 100)

    for _, row in buys.iterrows():
        ticker = row["Ticker"]
        if ticker not in current.index or pd.isna(current[ticker]):
            logger.warning(f"No live price for {ticker} — skipping row.")
            continue

        price       = float(current[ticker])
        target      = price * (1 + row["delta_rel"])
        shares      = int(row["Dollar_Alloc"] / price)

        print(
            f"  {ticker:<8}  {row['Weight']:>7.1%}  ${row['Dollar_Alloc']:>13,.2f}"
            f"  ${price:>9.2f}  ${target:>9.2f}  {shares}"
        )

    print("=" * 100)
    print("  EXIT: Place GTC sell-limit orders at the Target price.")
    print()


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a conviction-weighted ShockArb trade ticket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv", nargs="+", default=["./data/live_alpha_us.csv"],
        help="Path(s) to ShockArb score CSV files",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Total capital to allocate in dollars (default: 100000)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top positions (default: 5)",
    )
    args = parser.parse_args()
    generate_orders(args.csv, args.capital, args.top)
