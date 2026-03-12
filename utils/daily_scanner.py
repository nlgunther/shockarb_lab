"""
ShockArb End-of-Day Scanner.

Loads the saved factor model(s), fetches today's closing returns, scores the
tape, and exports the results to CSV.  Run this once after the 4pm close to
produce the alpha sheets consumed by the other utils.

Output files
------------
    data/live_alpha_us.csv      (if a US model exists)
    data/live_alpha_global.csv  (if a Global model exists)

Usage examples
--------------
    # Scan both universes (default)
    python utils/daily_scanner.py

    # Scan US only
    python utils/daily_scanner.py --universe us

    # Scan with a custom data directory
    python utils/daily_scanner.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yfinance as yf
from loguru import logger

from shockarb.config import ExecutionConfig
import shockarb.pipeline as pipeline


# =============================================================================
# Live return fetcher
# =============================================================================

def fetch_todays_returns(tickers: list[str]) -> pd.Series:
    """
    Fetch today's closing return for each ticker.

    .. deprecated::
        No longer called by run_scanner(). Scheduled for deletion in Step 2a
        (Part A1) once the score_universe() path is confirmed stable.
        Use pipeline.score_universe() instead.

    Downloads the last 5 trading days so the prior close is available even
    after long weekends or single-day holidays.

    Parameters
    ----------
    tickers : list of str

    Returns
    -------
    pd.Series — index = ticker, value = daily return (decimal fraction)
    """
    import warnings
    warnings.warn(
        "fetch_todays_returns() is deprecated and will be removed in Step 2a. "
        "Use pipeline.score_universe() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info(f"Fetching live returns for {len(tickers)} tickers…")
    raw = yf.download(tickers, period="5d", progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError("yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_col = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[price_col]
    else:
        prices = raw

    returns = prices.ffill().pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Return matrix is empty after cleaning.")

    return returns.iloc[-1]


# =============================================================================
# Scanner
# =============================================================================

def run_scanner(universe_names: list[str], exec_cfg: ExecutionConfig) -> None:
    """Score and export one or more universes."""
    # Universe registry — maps CLI name to UniverseConfig.
    # Add new universes here as they are built and saved.
    from shockarb.config import US_UNIVERSE, GLOBAL_UNIVERSE
    UNIVERSES = {
        "us":     US_UNIVERSE,
        "global": GLOBAL_UNIVERSE,
    }

    any_ran = False

    for name in universe_names:
        print(f"\n{'='*80}")
        print(f"  SCANNING: {name.upper()} MODEL")
        print(f"{'='*80}")

        universe = UNIVERSES.get(name)
        if universe is None:
            logger.error(f"Unknown universe {name!r}. Known: {sorted(UNIVERSES)}")
            continue

        model_path = pipeline.find_latest_model(name, exec_cfg)
        if not model_path:
            logger.error(f"No saved model for '{name}'. Run: python -m shockarb build --universe {name}")
            continue

        model = pipeline.load_model(model_path)

        try:
            scores = pipeline.score_universe(universe, model, exec_cfg)
        except ValueError as exc:
            logger.error(f"Failed to score '{name}': {exc}")
            continue

        output_path = os.path.join(exec_cfg.data_dir, f"live_alpha_{name}.csv")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        scores.to_csv(output_path)
        logger.success(f"Alpha sheet saved: {output_path}")
        any_ran = True

    if any_ran:
        print("\n✅ Scan complete.")
        print("   Next steps:")
        print("   • python utils/news_scanner.py              (headlines for top signals)")
        print("   • python utils/portfolio_sizer.py           (size a trade ticket)")
        print("   • python utils/csv_to_md.py data/live_alpha_us.csv  (markdown report)")
    else:
        print("\n⚠️  No universes scanned successfully.")


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-of-day ShockArb scanner — score today's tape and export CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe", "-u", nargs="+", default=["us", "global"],
        help="Universe(s) to scan (default: us global)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data directory (default: ./data or $SHOCK_ARB_DATA_DIR)",
    )
    args = parser.parse_args()

    exec_cfg = ExecutionConfig(data_dir=args.data_dir, log_to_file=False)
    run_scanner(args.universe, exec_cfg)


if __name__ == "__main__":
    main()
