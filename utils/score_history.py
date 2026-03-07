"""
ShockArb Historical Backtest Scanner.

Scores the market tape for any past trading date against a saved factor model.
Useful for validating the model against known historical dislocations (e.g.,
Ukraine invasion, Fed pivot days, earnings shocks).

Date snapping: if the requested date falls on a weekend or holiday the scanner
automatically snaps to the nearest prior valid trading day.

Usage examples
--------------
    # Score the day Russia invaded Ukraine
    python utils/score_history.py --universe us --date 2022-02-24

    # Score a specific date with a specific model file
    python utils/score_history.py --universe us --date 2022-03-16 --model data/model_us_20220401.json

    # Show all results, not just top 20
    python utils/score_history.py --universe us --date 2022-02-24 --top 0
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import yfinance as yf
from loguru import logger

from shockarb.config import ExecutionConfig
from shockarb.report import print_scores
import shockarb.pipeline as pipeline


# =============================================================================
# Historical return fetcher
# =============================================================================

def fetch_historical_returns(tickers: list[str], target_date_str: str) -> tuple[pd.Series, pd.Timestamp]:
    """
    Fetch the daily return for each ticker on a specific past date.

    Downloads a ±10-day window so the prior close is always available even
    across long holiday weekends.  Snaps to the nearest prior trading day
    if the requested date is not itself a trading day.

    Parameters
    ----------
    tickers : list of str
    target_date_str : str
        Date in YYYY-MM-DD format.

    Returns
    -------
    (returns_series, actual_date)
    """
    target = pd.to_datetime(target_date_str)
    start  = (target - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end    = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    logger.info(f"Fetching historical window [{start} → {end}] for {len(tickers)} tickers…")
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for the window around {target_date_str}.")

    # Resolve price series (MultiIndex or flat)
    if isinstance(raw.columns, pd.MultiIndex):
        price_col = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[price_col]
    else:
        prices = raw

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Clean: drop dead tickers, fill halts, compute returns
    prices  = prices.dropna(axis=1, how="all").ffill()
    returns = prices.pct_change().dropna(how="all")

    valid = returns.index[returns.index <= target]
    if valid.empty:
        raise ValueError(f"No valid trading data on or before {target_date_str}.")

    actual_date = valid[-1]
    if actual_date != target:
        logger.warning(
            f"{target_date_str} is not a trading day — "
            f"snapped to {actual_date.strftime('%Y-%m-%d')}."
        )

    return returns.loc[actual_date], actual_date


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a historical date against a saved ShockArb model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe", "-u", default="us",
        help="Universe name matching a saved model (e.g. 'us', 'global'). Default: us",
    )
    parser.add_argument(
        "--date", "-d", required=True,
        help="Historical date to score (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Explicit path to a model JSON (optional — uses latest by default)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data directory (default: ./data or $SHOCK_ARB_DATA_DIR)",
    )
    parser.add_argument(
        "--top", "-n", type=int, default=20,
        help="Number of results to display (0 = all). Default: 20",
    )
    args = parser.parse_args()

    exec_cfg = ExecutionConfig(data_dir=args.data_dir, log_to_file=False)

    model_path = args.model or pipeline.find_latest_model(args.universe, exec_cfg)
    if not model_path:
        logger.error(f"No model found for '{args.universe}'. Run: python -m shockarb build --universe {args.universe}")
        sys.exit(1)

    model = pipeline.load_model(model_path)
    etf_tickers   = list(model.etf_returns.columns)
    stock_tickers = list(model.stock_returns.columns)

    try:
        etf_returns,   actual_date = fetch_historical_returns(etf_tickers,   args.date)
        stock_returns, _           = fetch_historical_returns(stock_tickers,  args.date)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    scores = model.score(etf_returns, stock_returns)
    title  = f"{args.universe.upper()} | {actual_date.strftime('%Y-%m-%d')}"
    top_n  = args.top if args.top > 0 else len(scores)
    print_scores(scores, title, top_n=top_n)


if __name__ == "__main__":
    main()
