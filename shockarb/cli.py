#!/usr/bin/env python3
"""
ShockArb Command-Line Interface

Commands
--------
  build   Fit and save a factor model from historical data.
  score   Score live or historical returns against a fitted model.
  export  Generate CSV reports (ETF basis + stock loadings).
  show    Display a saved model's diagnostics and factor structure.

Examples
--------
    # Fit and save the US model
    python -m shockarb build --universe us

    # Score today's live tape (and save raw OHLCV parquet)
    python -m shockarb score --universe us --save-tape

    # Score a specific historical date
    python -m shockarb score --universe us --date 2022-03-01

    # Export CSVs for manual inspection
    python -m shockarb export --universe us

    # Show model diagnostics (add -v for factor loadings)
    python -m shockarb show --universe us -v
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

import shockarb.pipeline as pipeline
from shockarb.config import (
    GLOBAL_UNIVERSE,
    US_UNIVERSE,
    ExecutionConfig,
    UniverseConfig,
)
from shockarb.report import print_model_state, print_scores


# =============================================================================
# Universe registry
# =============================================================================

UNIVERSES: dict[str, UniverseConfig] = {
    "us": US_UNIVERSE,
    "global": GLOBAL_UNIVERSE,
}


def get_universe(name: str) -> UniverseConfig:
    """Look up a universe by name (case-insensitive)."""
    key = name.lower()
    if key not in UNIVERSES:
        raise ValueError(
            f"Unknown universe: '{name}'. Available: {list(UNIVERSES.keys())}"
        )
    return UNIVERSES[key]


# =============================================================================
# Commands
# =============================================================================

def cmd_build(args) -> None:
    """Fit and save a factor model."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir,
        log_to_file=not args.no_log,
    )

    model = pipeline.build(universe, exec_cfg)
    path = pipeline.save_model(model, universe.name, exec_cfg)
    pipeline.export_csvs(model, universe.name, exec_cfg)

    print(f"\n✅ Model saved: {path}")
    print(f"   Factors:           {model.diagnostics.n_factors}")
    print(f"   Variance explained: {model.diagnostics.cumulative_variance:.1%}")
    print(f"   Stocks:            {model.diagnostics.n_stocks}")


def cmd_score(args) -> None:
    """Score returns against a fitted model."""
    import os
    from datetime import date as _date
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir,
        log_to_file=not args.no_log,
    )
    model_path = args.model or pipeline.find_latest_model(universe.name, exec_cfg)
    if not model_path:
        print(f"❌ No model found for '{universe.name}'. Run 'build' first.")
        sys.exit(1)
    model = pipeline.load_model(model_path)
    etf_tickers   = list(model.etf_returns.columns) or list(universe.market_etfs)
    stock_tickers = list(model.stock_returns.columns) or list(universe.individual_stocks)

    if args.date:
        # Historical scoring — tape saving not applicable
        etf_returns, stock_returns = _fetch_historical(
            etf_tickers, stock_tickers, args.date
        )
        title = f"{universe.name.upper()} | {args.date}"
        scores = model.score(etf_returns, stock_returns)
    else:
        # Live scoring — optionally save the raw OHLCV tape first
        if getattr(args, "save_tape", False):
            today_str = _date.today().strftime("%Y%m%d")
            tape_dir  = os.path.join(exec_cfg.data_dir, "tapes")
            tape_path = os.path.join(tape_dir, f"{universe.name}_{today_str}.parquet")
            tape = pipeline.save_live_tape(etf_tickers, stock_tickers, tape_path)
            if tape is not None:
                print(f"\n💾 Tape saved: {tape_path}")
                print(f"   Rows: {len(tape)}  |  Tickers: {tape.shape[1] // len(tape.columns.get_level_values(0).unique())}")
            else:
                print("⚠️  Tape save failed — continuing with live fetch for scoring")

        scores, prov = pipeline.score_universe(universe, model, exec_cfg,
                                        force_daily=args.use_prior_close,
                                        from_open=args.from_open)
        prov.model_file = model_path
        title = f"{universe.name.upper()} | LIVE"
        print(f"\n{prov.summary()}\n")

    print_scores(scores, title, top_n=args.top)
    if args.output:
        scores.to_csv(args.output)
        print(f"\n📁 Saved to: {args.output}")


def cmd_export(args) -> None:
    """Export model factor tables to CSV."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)

    model_path = pipeline.find_latest_model(universe.name, exec_cfg)
    if not model_path:
        print(f"❌ No model found for '{universe.name}'")
        sys.exit(1)

    model = pipeline.load_model(model_path)
    basis_path, loadings_path = pipeline.export_csvs(model, universe.name, exec_cfg)

    print("✅ Exported:")
    print(f"   ETF basis:      {basis_path}")
    print(f"   Stock loadings: {loadings_path}")


def cmd_show(args) -> None:
    """Display model diagnostics and factor structure."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)

    model_path = pipeline.find_latest_model(universe.name, exec_cfg)
    if not model_path:
        print(f"❌ No model found for '{universe.name}'")
        sys.exit(1)

    if args.verbose:
        # Full structural report from the JSON file
        print_model_state(model_path)
    else:
        # Compact diagnostics via loaded model
        model = pipeline.load_model(model_path)
        print(f"\n{'='*60}")
        print(f"  SHOCKARB MODEL: {universe.name.upper()}")
        print(f"{'='*60}")
        print(f"  Source: {model_path}")
        print()
        print(model.diagnostics.summary())
        print()


# =============================================================================
# Helpers
# =============================================================================

def _fetch_historical(
    etf_tickers: list,
    stock_tickers: list,
    date_str: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Fetch closing returns for a historical date, snapping to the nearest
    valid trading day if the requested date falls on a weekend or holiday.
    """
    target = pd.to_datetime(date_str)
    start = (target - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    def get_returns(tickers: list) -> pd.Series:
        raw = yf.download(tickers, start=start, end=end, progress=False)
        # yf returns a Series for a single ticker; normalise to DataFrame
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        returns = prices.dropna(axis=1, how="all").ffill().pct_change().dropna(how="all")

        valid = returns.index[returns.index <= target]
        if valid.empty:
            raise ValueError(f"No trading data on or before {date_str}")

        matched = valid[-1]
        if matched != target:
            logger.warning(
                f"Date {date_str} is not a trading day; "
                f"snapped to {matched.strftime('%Y-%m-%d')}"
            )
        return returns.loc[matched]

    return get_returns(etf_tickers), get_returns(stock_tickers)


# =============================================================================
# Argument parser
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shockarb",
        description="ShockArb Factor Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s build --universe us
  %(prog)s score --universe us
  %(prog)s score --universe us --date 2022-03-01
  %(prog)s show  --universe us -v
        """,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: ./data or $SHOCK_ARB_DATA_DIR)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p = sub.add_parser("build", help="Fit and save a factor model")
    p.add_argument("--universe", "-u", required=True, help="us | global")
    p.add_argument("--no-log", action="store_true", help="Disable file logging")
    p.set_defaults(func=cmd_build)

    # score
    p = sub.add_parser("score", help="Score returns against a fitted model")
    p.add_argument("--from-open", "-O", action="store_true",
                help="Use today's session open as denominator (pure intraday)")
    p.add_argument("--use-prior-close", "-p", action="store_true",
                help="Force daily close-to-close returns")    
    p.add_argument("--universe", "-u", required=True)
    p.add_argument("--date",   "-d", help="Historical date YYYY-MM-DD")
    p.add_argument("--model",  "-m", help="Specific model .json to load")
    p.add_argument("--output", "-o", help="Save score results to CSV")
    p.add_argument("--top",    "-n", type=int, default=20, help="Show top N results")
    p.add_argument(
        "--save-tape", action="store_true",
        help=(
            "Save raw daily OHLCV (ETFs + stocks combined) as parquet before scoring. "
            "Written to data/tapes/{universe}_{YYYYMMDD}.parquet. "
            "Ignored when --date is used (historical data only)."
        ),
    )
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(func=cmd_score)

    # export
    p = sub.add_parser("export", help="Export model to CSVs")
    p.add_argument("--universe", "-u", required=True)
    p.set_defaults(func=cmd_export)

    # show
    p = sub.add_parser("show", help="Display model summary")
    p.add_argument("--universe", "-u", required=True)
    p.add_argument("--verbose", "-v", action="store_true", help="Full factor tables")
    p.set_defaults(func=cmd_show)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Propagate --data-dir via environment so ExecutionConfig picks it up
    # even in code paths that don't thread exec_config explicitly
    if args.data_dir:
        import os
        os.environ["SHOCK_ARB_DATA_DIR"] = args.data_dir

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        sys.exit(130)
    except Exception as exc:
        logger.exception(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
