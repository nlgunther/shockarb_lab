"""
ShockArb Walk-Forward Backtest Runner.

Runs the walk-forward backtester over a specified evaluation window and
writes results to CSV.  Prints a summary table to the terminal.

The backtest fits a fresh FactorModel on each day's calibration window,
scores the *next* day's tape, and measures how signals performed at
T+1, T+2, T+3, and T+5 (configurable).  No look-ahead bias.

Usage examples
--------------
    # Basic run: US universe, 2023 evaluation window
    python utils/run_backtest.py --universe us --start 2023-01-01 --end 2023-12-31

    # Tighter thresholds, shorter calibration window
    python utils/run_backtest.py \\
        --universe us \\
        --start 2024-01-01 --end 2024-12-31 \\
        --calib 30 \\
        --min-conf 0.008 \\
        --min-r2 0.60 \\
        --top 3

    # Save trade ledger to CSV
    python utils/run_backtest.py \\
        --universe us --start 2023-01-01 --end 2024-12-31 \\
        --out data/backtest_us_2023_2024.csv

    # Custom holding periods
    python utils/run_backtest.py \\
        --universe us --start 2023-01-01 --end 2023-12-31 \\
        --horizons 1 3 5 10
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib

# Ensure shockarb package is importable without pip install -e .
_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from loguru import logger
from shockarb.backtest import Backtest, BacktestConfig
from shockarb.config import US_UNIVERSE, GLOBAL_UNIVERSE, ExecutionConfig


UNIVERSES = {
    "us":     US_UNIVERSE,
    "global": GLOBAL_UNIVERSE,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShockArb walk-forward backtester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe", "-u", default="us", choices=list(UNIVERSES),
        help="Universe to backtest (default: us)",
    )
    parser.add_argument(
        "--start", "-s", required=True,
        help="Evaluation window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", "-e", required=True,
        help="Evaluation window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--calib", type=int, default=35,
        help="Calibration window length in trading days (default: 35)",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 2, 3, 5],
        metavar="N",
        help="Holding periods in days to evaluate (default: 1 2 3 5)",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.005,
        help="Minimum confidence_delta threshold (default: 0.005)",
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.50,
        help="Minimum R² threshold (default: 0.50)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Max signals per day (default: 5)",
    )
    parser.add_argument(
        "--factors", type=int, default=None,
        help="Override n_components (default: universe setting)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path to save trade ledger CSV (optional)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data directory (default: ./data)",
    )
    args = parser.parse_args()

    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir or os.path.join(os.getcwd(), "data"),
        log_to_file=False,
    )

    cfg = BacktestConfig(
        universe=UNIVERSES[args.universe],
        calib_window=args.calib,
        holding_periods=sorted(args.horizons),
        min_confidence=args.min_conf,
        min_r_squared=args.min_r2,
        eval_start=args.start,
        eval_end=args.end,
        top_n=args.top,
        n_components=args.factors,
    )

    bt = Backtest(cfg, exec_config=exec_cfg)
    results = bt.run()
    results.print_summary()

    if args.out:
        if results.ledger.empty:
            logger.warning("No trades to save.")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            results.ledger.to_csv(args.out)
            logger.success(f"Trade ledger saved: {args.out}")

            # Also save the summary next to the ledger
            summary_path = args.out.replace(".csv", "_summary.csv")
            results.summary.to_csv(summary_path)
            logger.success(f"Summary saved: {summary_path}")

            # And the equity curve
            if not results.equity_curve.empty:
                curve_path = args.out.replace(".csv", "_equity.csv")
                results.equity_curve.to_csv(curve_path, header=True)
                logger.success(f"Equity curve saved: {curve_path}")


if __name__ == "__main__":
    main()
