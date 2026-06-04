"""
ShockArb Pick Evaluator.

Compares entry prices for the top-N signals in a score CSV against current
market prices, reporting P&L per position and overall.

Entry price source (priority order)
-------------------------------------
1. data/trades.csv          — Ticker + Trade columns (case-insensitive headers)
2. --input <path>           — same two-column trades format (any filename except
                              portfolio_sizer.csv)
3. data/portfolio_sizer.csv — Ticker + Current columns (portfolio_sizer output)
4. --input portfolio_sizer.csv — same Ticker + Current format

If no entry price file is found or a ticker is missing from it, that position
is skipped with a warning.

Usage examples
--------------
    # Evaluate top 4 from the default alpha sheet
    python utils/eval_picks.py

    # Use a custom alpha CSV and trades file
    python utils/eval_picks.py --csv data/live_alpha_us.csv --input data/my_trades.csv

    # Evaluate top 8 using a portfolio_sizer ticket
    python utils/eval_picks.py --top 8 --input data/portfolio_sizer.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yfinance as yf
from loguru import logger


_DATA_DIR = "./data"
_DEFAULT_SCORE_CSV = "live_alpha_us.csv"
_DEFAULT_TRADES     = "trades.csv"
_SIZER_FILENAME     = "portfolio_sizer.csv"


# =============================================================================
# Entry price loading
# =============================================================================

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all column names for case-insensitive matching."""
    df.columns = [c.lower() for c in df.columns]
    return df


def _load_trades_file(path: str) -> dict[str, float]:
    """
    Load a trades file (Ticker + Trade columns) → {ticker: entry_price}.
    Returns empty dict on failure.
    """
    try:
        df = _normalise_cols(pd.read_csv(path))
        if "ticker" not in df.columns or "trade" not in df.columns:
            logger.error(f"Trades file {path!r} must have 'Ticker' and 'Trade' columns.")
            return {}
        return {str(row["ticker"]).upper(): float(row["trade"])
                for _, row in df.iterrows()
                if pd.notna(row["trade"])}
    except Exception as exc:
        logger.error(f"Failed to read trades file {path!r}: {exc}")
        return {}


def _load_sizer_file(path: str) -> dict[str, float]:
    """
    Load a portfolio_sizer CSV (Ticker + Current columns) → {ticker: entry_price}.
    Returns empty dict on failure.
    """
    try:
        df = _normalise_cols(pd.read_csv(path))
        if "ticker" not in df.columns or "current" not in df.columns:
            logger.error(f"Sizer file {path!r} must have 'Ticker' and 'Current' columns.")
            return {}
        return {str(row["ticker"]).upper(): float(row["current"])
                for _, row in df.iterrows()
                if pd.notna(row["current"])}
    except Exception as exc:
        logger.error(f"Failed to read sizer file {path!r}: {exc}")
        return {}


def _is_sizer_file(path: str) -> bool:
    return os.path.basename(path).lower() == _SIZER_FILENAME


def load_entry_prices(input_path: str | None) -> dict[str, float]:
    """
    Resolve entry prices using the priority hierarchy.

    Returns dict of {TICKER: entry_price} — may be empty if no source found.
    """
    # Priority 1: data/trades.csv
    default_trades = os.path.join(_DATA_DIR, _DEFAULT_TRADES)
    if os.path.exists(default_trades):
        logger.info(f"Entry prices from: {default_trades}")
        return _load_trades_file(default_trades)

    # Priority 2 & 4: --input flag (trades or sizer, detected by filename)
    if input_path:
        if _is_sizer_file(input_path):
            logger.info(f"Entry prices from sizer file: {input_path}")
            return _load_sizer_file(input_path)
        else:
            logger.info(f"Entry prices from trades file: {input_path}")
            return _load_trades_file(input_path)

    # Priority 3: data/portfolio_sizer.csv
    default_sizer = os.path.join(_DATA_DIR, _SIZER_FILENAME)
    if os.path.exists(default_sizer):
        logger.info(f"Entry prices from: {default_sizer}")
        return _load_sizer_file(default_sizer)

    logger.error(
        "No entry price source found. Provide data/trades.csv, data/portfolio_sizer.csv, "
        "or use --input."
    )
    return {}


# =============================================================================
# Core evaluation
# =============================================================================

def evaluate_picks(
    csv_path: str,
    top_n: int = 4,
    input_path: str | None = None,
) -> None:
    """
    Load the top-N signals from a score CSV, resolve entry prices, fetch
    current prices, and print a P&L summary table.

    Parameters
    ----------
    csv_path   : Path to a ShockArb score CSV (live_alpha_us.csv format).
    top_n      : Number of top signals to evaluate.
    input_path : Optional explicit path to entry price file.
    """
    # Load score CSV
    try:
        scores = pd.read_csv(csv_path)
        if "Ticker" not in scores.columns:
            scores = scores.rename(columns={scores.columns[0]: "Ticker"})
    except Exception as exc:
        logger.error(f"Cannot read score CSV {csv_path!r}: {exc}")
        return

    if "confidence_delta" not in scores.columns:
        logger.error("Score CSV missing 'confidence_delta' column.")
        return

    top = (scores[scores["confidence_delta"] > 0]
           .sort_values("confidence_delta", ascending=False)
           .head(top_n))

    if top.empty:
        logger.warning("No positive-signal tickers found.")
        return

    tickers = top["Ticker"].str.upper().tolist()

    # Resolve entry prices
    entry_prices = load_entry_prices(input_path)
    if not entry_prices:
        return

    # Fetch current prices
    logger.info(f"Fetching current prices for: {tickers}")
    raw = yf.download(tickers, period="1d", progress=False, auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        price_col = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        current_prices = raw[price_col].iloc[-1]
    else:
        current_prices = raw.iloc[-1]

    # Build results
    rows = []
    for ticker in tickers:
        entry = entry_prices.get(ticker)
        if entry is None:
            logger.warning(f"{ticker}: no entry price — skipping.")
            continue

        if ticker not in current_prices.index or pd.isna(current_prices[ticker]):
            logger.warning(f"{ticker}: no current price from yfinance — skipping.")
            continue

        current = float(current_prices[ticker])
        pnl_pct = (current - entry) / entry
        pnl_abs = current - entry
        rows.append({
            "Ticker":  ticker,
            "Entry":   entry,
            "Current": current,
            "P&L $":   pnl_abs,
            "P&L %":   pnl_pct,
        })

    if not rows:
        logger.warning("No positions could be evaluated.")
        return

    # Print table
    width = 72
    print(f"\n{'='*width}")
    print("  SHOCKARB PICK EVALUATOR")
    print(f"{'='*width}")
    print(f"  {'TICKER':<8}  {'ENTRY':>10}  {'CURRENT':>10}  {'P&L $':>10}  {'P&L %':>8}")
    print(f"  {'-'*width}")

    total_pnl = 0.0
    winners, losers = 0, 0
    for r in rows:
        sign = "▲" if r["P&L $"] >= 0 else "▼"
        print(
            f"  {r['Ticker']:<8}  ${r['Entry']:>9.2f}  ${r['Current']:>9.2f}"
            f"  {sign}${abs(r['P&L $']):>8.2f}  {r['P&L %']:>+7.2%}"
        )
        total_pnl += r["P&L $"]
        if r["P&L $"] >= 0:
            winners += 1
        else:
            losers += 1

    avg_pnl_pct = sum(r["P&L %"] for r in rows) / len(rows)
    print(f"  {'-'*width}")
    print(f"  {'SUMMARY':<8}  {'':>10}  {'':>10}  {'':>10}  {avg_pnl_pct:>+7.2%} avg")
    print(f"  {winners}W / {losers}L across {len(rows)} position(s)")
    print(f"{'='*width}\n")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ShockArb pick performance against entry prices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv", "-c",
        default=os.path.join(_DATA_DIR, _DEFAULT_SCORE_CSV),
        help=f"Score CSV to evaluate (default: {_DEFAULT_SCORE_CSV})",
    )
    parser.add_argument(
        "--top", "-n", type=int, default=4,
        help="Number of top signals to evaluate (default: 4)",
    )
    parser.add_argument(
        "--input", "-i", default=None,
        metavar="PATH",
        help=(
            "Entry price file. Use trades.csv format (Ticker + Trade columns) "
            "or name it 'portfolio_sizer.csv' to read Ticker + Current columns."
        ),
    )
    args = parser.parse_args()
    evaluate_picks(args.csv, args.top, args.input)
