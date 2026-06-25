"""
ShockArb Portfolio Sizing Utility.

Reads one or more ShockArb score CSVs, selects the top-N positive signals
by conviction (confidence_delta), and prints a dollar-denominated trade
ticket with allocation weights and take-profit limit prices.

Output is saved to data/portfolio_sizer.csv by default. Suppress with --no-out.

Usage examples
--------------
    # Size $100k across the top 5 US signals
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000

    # Merge US + Global into a single ticket
    python utils/portfolio_sizer.py \
        --csv data/live_alpha_us.csv data/live_alpha_global.csv \
        --capital 50000 --top 8

    # Exclude specific tickers (output still saved to data/portfolio_sizer.csv by default)
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000 \
        --exclude SNPS BSX

    # Size only specific tickers (bypasses CSV ranking entirely)
    python utils/portfolio_sizer.py --tickers AMAT ADI ETN --capital 10000

    # Suppress file output entirely
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000 --no-out

    # Save to a custom path
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000 \
        --out data/ticket.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger


_DEFAULT_OUT = "./data/portfolio_sizer.csv"


def _check_cwd() -> None:
    """Exit with a clear error if not run from the project root."""
    if not Path("data").is_dir():
        print(
            "\n❌  portfolio_sizer.py must be run from the project root.\n"
            "\n"
            "    Correct usage:\n"
            "        cd <project_root>\n"
            "        python utils\\portfolio_sizer.py --tickers MSFT IDXX --capital 10000\n"
            "\n"
            f"    Current directory: {Path.cwd()}\n"
        )
        sys.exit(1)


def generate_orders(
    csv_paths: list[str],
    capital: float,
    top_n: int = 5,
    exclude: list[str] | None = None,
    out: str | None = _DEFAULT_OUT,
    tickers: list[str] | None = None,
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
        Number of positions to take.  Ignored when tickers is set.
    exclude : list of str, optional
        Tickers to exclude before ranking (e.g. catalyst-driven traps).
        Ignored when tickers is set.
    out : str, optional
        Path to save the ticket CSV.  Defaults to data/portfolio_sizer.csv.
        Pass None to suppress file output.
    tickers : list of str, optional
        If supplied, only these tickers are sized (CSV ranking is bypassed).
        Overrides top_n and exclude.
    """
    exclude = [t.upper() for t in (exclude or [])]
    tickers = [t.upper() for t in (tickers or [])]
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

    if tickers:
        master = master[master["Ticker"].str.upper().isin(tickers)]
    elif exclude:
        master = master[~master["Ticker"].str.upper().isin(exclude)]

    buys = (
        master[master["confidence_delta"] > 0]
        .sort_values("confidence_delta", ascending=False)
        .head(top_n if not tickers else len(master))
    )

    if buys.empty:
        logger.warning("No positive alpha signals found.")
        return

    # Fetch live prices
    ticker_list = buys["Ticker"].tolist()
    logger.info(f"Fetching live prices for: {ticker_list}")
    raw = yf.download(ticker_list, period="1d", progress=False, auto_adjust=False)

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

    rows = []
    for _, row in buys.iterrows():
        ticker = row["Ticker"]
        if ticker not in current.index or pd.isna(current[ticker]):
            logger.warning(f"No live price for {ticker} — skipping row.")
            continue

        price  = float(current[ticker])
        target = price * (1 + row["delta_rel"])
        shares = int(row["Dollar_Alloc"] / price)

        print(
            f"  {ticker:<8}  {row['Weight']:>7.1%}  ${row['Dollar_Alloc']:>13,.2f}"
            f"  ${price:>9.2f}  ${target:>9.2f}  {shares}"
        )
        rows.append({
            "Ticker":           ticker,
            "Weight":           round(row["Weight"], 4),
            "Dollar_Alloc":     round(row["Dollar_Alloc"], 2),
            "Current":          round(price, 2),
            "Target":           round(target, 2),
            "Shares":           shares,
            "confidence_delta": round(row["confidence_delta"], 6),
            "r_squared":        round(row.get("r_squared", float("nan")), 4),
        })

    print("=" * 100)
    print("  EXIT: Place GTC sell-limit orders at the Target price.")
    print()

    if out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.success(f"Ticket saved: {out}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    _check_cwd()
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
        help="Number of top positions (default: 5); ignored when --tickers is set",
    )
    parser.add_argument(
        "--exclude", "-e", nargs="+", default=[],
        help="Tickers to exclude before ranking (e.g. --exclude SNPS BSX); ignored when --tickers is set",
    )
    parser.add_argument(
        "--tickers", "-t", nargs="+", default=[],
        help="Size only these tickers; bypasses CSV ranking, --top, and --exclude (e.g. --tickers AMAT ADI ETN)",
    )
    parser.add_argument(
        "--out", "-o", default=_DEFAULT_OUT,
        help=f"Save ticket to CSV (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--no-out", "-sout", action="store_true",
        help="Suppress CSV output (do not write a file)",
    )
    args = parser.parse_args()
    out = None if args.no_out else args.out
    generate_orders(args.csv, args.capital, args.top, args.exclude, out, args.tickers)
