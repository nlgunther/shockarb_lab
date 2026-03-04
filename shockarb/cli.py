#!/usr/bin/env python3
"""
ShockArb Command-Line Interface

Unified entry point for all ShockArb operations:
  - build:   Fit and save a factor model
  - score:   Score live or historical returns
  - export:  Generate CSV reports
  - show:    Display model state

Examples
--------
    # Build US model
    python -m shockarb build --universe us
    
    # Score today's tape
    python -m shockarb score --universe us
    
    # Score historical date
    python -m shockarb score --universe us --date 2022-03-01
    
    # Export CSVs
    python -m shockarb export --universe us
    
    # Show model summary
    python -m shockarb show --universe us
"""

import argparse
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from shockarb.config import (
    US_UNIVERSE, 
    GLOBAL_UNIVERSE, 
    UniverseConfig, 
    ExecutionConfig
)
from shockarb.pipeline import Pipeline, fetch_live_returns


# =============================================================================
# Universe Registry
# =============================================================================

UNIVERSES = {
    "us": US_UNIVERSE,
    "global": GLOBAL_UNIVERSE,
}


def get_universe(name: str) -> UniverseConfig:
    """Get universe by name, case-insensitive."""
    key = name.lower()
    if key not in UNIVERSES:
        raise ValueError(f"Unknown universe: {name}. Available: {list(UNIVERSES.keys())}")
    return UNIVERSES[key]


# =============================================================================
# Commands
# =============================================================================

def cmd_build(args):
    """Build and save a factor model."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir,
        use_cache=not args.no_cache,
        log_to_file=not args.no_log,
    )
    
    model = Pipeline.build(universe, exec_cfg)
    path = Pipeline.save_model(model, universe.name, exec_cfg)
    
    # Also export CSVs by default
    Pipeline.export_csvs(model, universe.name, exec_cfg)
    
    print(f"\n✅ Model saved: {path}")
    print(f"   Factors: {model.diagnostics.n_factors}")
    print(f"   Variance explained: {model.diagnostics.cumulative_variance:.1%}")
    print(f"   Stocks: {model.diagnostics.n_stocks}")


def cmd_score(args):
    """Score returns against a fitted model."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir,
        log_to_file=not args.no_log,
    )
    
    # Load model
    model_path = args.model or Pipeline.find_latest_model(universe.name, exec_cfg)
    if not model_path:
        print(f"❌ No model found for '{universe.name}'. Run 'build' first.")
        sys.exit(1)
    
    model = Pipeline.load_model(model_path)
    etf_tickers = list(model.etf_returns.columns)
    stock_tickers = list(model.stock_returns.columns)
    
    # Fetch returns
    if args.date:
        # Historical scoring
        etf_returns, stock_returns = _fetch_historical(
            etf_tickers, stock_tickers, args.date
        )
        title = f"{universe.name.upper()} | {args.date}"
    else:
        # Live scoring
        etf_returns = fetch_live_returns(etf_tickers)
        stock_returns = fetch_live_returns(stock_tickers)
        title = f"{universe.name.upper()} | LIVE"
    
    # Score
    scores = model.score(etf_returns, stock_returns)
    
    # Display
    _print_scores(scores, title, args.top)
    
    # Save if requested
    if args.output:
        scores.to_csv(args.output)
        print(f"\n📁 Saved to: {args.output}")


def cmd_export(args):
    """Export model to CSVs."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    
    model_path = Pipeline.find_latest_model(universe.name, exec_cfg)
    if not model_path:
        print(f"❌ No model found for '{universe.name}'")
        sys.exit(1)
    
    model = Pipeline.load_model(model_path)
    basis_path, loadings_path = Pipeline.export_csvs(model, universe.name, exec_cfg)
    
    print(f"✅ Exported:")
    print(f"   ETF basis:      {basis_path}")
    print(f"   Stock loadings: {loadings_path}")


def cmd_show(args):
    """Display model summary."""
    universe = get_universe(args.universe)
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    
    model_path = Pipeline.find_latest_model(universe.name, exec_cfg)
    if not model_path:
        print(f"❌ No model found for '{universe.name}'")
        sys.exit(1)
    
    model = Pipeline.load_model(model_path)
    
    print(f"\n{'='*60}")
    print(f"  SHOCKARB MODEL: {universe.name.upper()}")
    print(f"{'='*60}")
    print(f"  Source: {model_path}")
    print()
    print(model.diagnostics.summary())
    
    if args.verbose:
        print(f"\n{'─'*60}")
        print("  ETF Factor Loadings (top by abs Factor_1)")
        print("─"*60)
        basis = model.etf_basis.copy()
        basis["abs_f1"] = basis["Factor_1"].abs()
        print(basis.nlargest(10, "abs_f1").drop(columns="abs_f1").to_string())
        
        print(f"\n{'─'*60}")
        print("  Stock R² (top 10)")
        print("─"*60)
        r2 = model.diagnostics.stock_r_squared.nlargest(10)
        for ticker, val in r2.items():
            print(f"  {ticker:<8} {val:.3f}")
    
    print()


# =============================================================================
# Helpers
# =============================================================================

def _fetch_historical(
    etf_tickers: list, 
    stock_tickers: list, 
    date_str: str
) -> tuple:
    """Fetch returns for a historical date."""
    import yfinance as yf
    
    target = pd.to_datetime(date_str)
    start = target - pd.Timedelta(days=10)
    end = target + pd.Timedelta(days=2)
    
    def get_returns(tickers):
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )["Close"]
        
        # Handle single ticker (returns Series) vs multiple (returns DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0] if len(tickers) == 1 else "Close")
        
        data = data.dropna(axis=1, how="all").ffill()
        returns = data.pct_change().dropna(how="all")
        
        # Snap to closest valid date
        valid = returns.index[returns.index <= target]
        if valid.empty:
            raise ValueError(f"No trading data on or before {date_str}")
        
        matched = valid[-1]
        if matched != target:
            logger.warning(f"Snapped {date_str} to {matched.strftime('%Y-%m-%d')}")
        
        return returns.loc[matched]
    
    return get_returns(etf_tickers), get_returns(stock_tickers)


def _print_scores(scores: pd.DataFrame, title: str, top_n: int = 20):
    """Pretty-print scoring results."""
    print(f"\n{'='*90}")
    print(f"  ⚡ SHOCKARB SCORES: {title}")
    print(f"{'='*90}")
    
    # Filter actionable (positive delta with decent R²)
    actionable = scores[
        (scores["confidence_delta"] > 0.001) & 
        (scores["r_squared"] > 0.3)
    ].head(top_n)
    
    if actionable.empty:
        print("\n  No actionable signals (confidence_delta > 0.1% & R² > 0.3)")
    else:
        print(f"\n  Top {len(actionable)} actionable signals:\n")
        print(f"  {'Ticker':<8} {'Actual':>10} {'Expected':>10} {'Delta':>10} {'R²':>8} {'Conf.Δ':>10}")
        print("  " + "─"*60)
        
        for ticker, row in actionable.iterrows():
            print(
                f"  {ticker:<8} "
                f"{row['actual_return']:>+9.2%} "
                f"{row['expected_return']:>+9.2%} "
                f"{row['delta']:>+9.2%} "
                f"{row['r_squared']:>7.2f} "
                f"{row['confidence_delta']:>+9.2%}"
            )
    
    # Show worst (potential shorts or avoid)
    worst = scores.nsmallest(5, "confidence_delta")
    if not worst.empty and worst["confidence_delta"].iloc[0] < -0.001:
        print(f"\n  ⚠️  Bottom 5 (outperformed factors - avoid):\n")
        for ticker, row in worst.iterrows():
            print(
                f"  {ticker:<8} "
                f"{row['actual_return']:>+9.2%} "
                f"{row['expected_return']:>+9.2%} "
                f"{row['delta']:>+9.2%}"
            )
    
    print(f"\n{'='*90}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="shockarb",
        description="ShockArb Factor Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s build --universe us          Build and save US model
  %(prog)s score --universe us          Score today's tape
  %(prog)s score --universe us --date 2022-03-01   Historical backtest
  %(prog)s show --universe us -v        Show model details
        """
    )
    
    parser.add_argument(
        "--data-dir", 
        default=None,
        help="Override data directory (default: ./data or $SHOCK_ARB_DATA_DIR)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Build command
    build_p = subparsers.add_parser("build", help="Fit and save a factor model")
    build_p.add_argument("--universe", "-u", required=True, help="Universe: us, global")
    build_p.add_argument("--no-cache", action="store_true", help="Ignore cached data")
    build_p.add_argument("--no-log", action="store_true", help="Disable file logging")
    build_p.set_defaults(func=cmd_build)
    
    # Score command
    score_p = subparsers.add_parser("score", help="Score returns against model")
    score_p.add_argument("--universe", "-u", required=True, help="Universe: us, global")
    score_p.add_argument("--date", "-d", help="Historical date (YYYY-MM-DD)")
    score_p.add_argument("--model", "-m", help="Specific model file to load")
    score_p.add_argument("--output", "-o", help="Save results to CSV")
    score_p.add_argument("--top", "-n", type=int, default=20, help="Show top N results")
    score_p.add_argument("--no-log", action="store_true", help="Disable file logging")
    score_p.set_defaults(func=cmd_score)
    
    # Export command
    export_p = subparsers.add_parser("export", help="Export model to CSVs")
    export_p.add_argument("--universe", "-u", required=True, help="Universe: us, global")
    export_p.set_defaults(func=cmd_export)
    
    # Show command
    show_p = subparsers.add_parser("show", help="Display model summary")
    show_p.add_argument("--universe", "-u", required=True, help="Universe: us, global")
    show_p.add_argument("--verbose", "-v", action="store_true", help="Show details")
    show_p.set_defaults(func=cmd_show)
    
    args = parser.parse_args()
    
    # Handle data_dir
    if args.data_dir:
        import os
        os.environ["SHOCK_ARB_DATA_DIR"] = args.data_dir
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
