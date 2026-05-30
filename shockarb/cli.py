#!/usr/bin/env python3
"""
ShockArb Command-Line Interface

Commands
--------
  build        Fit and save a factor model from historical data.
  score        Score live or historical returns against a fitted model.
  export       Generate CSV reports (ETF basis + stock loadings).
  show         Display a saved model's diagnostics and factor structure.
  set-regime   Set the active regime (sticky across sessions).
  show-regime  Display the current sticky regime.
  list-regimes List all available regimes.
  backtest     Run walk-forward backtest to measure signal decay.

Examples
--------
    # Set and build with a regime
    python -m shockarb set-regime ukraine_shock
    python -m shockarb build

    # One-shot regime override (doesn't change sticky)
    python -m shockarb build --regime gulf_war_recovery

    # Score with regime
    python -m shockarb score
    python -m shockarb score --regime ukraine_shock --save-tape

    # Legacy universe syntax still works (maps to ukraine_shock)
    python -m shockarb build --universe us
"""

from __future__ import annotations

import argparse
import os
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
from shockarb.regimes import HistoricFactorModel, get_regime, list_regimes
from shockarb.report import print_model_state, print_scores


# =============================================================================
# Regime management
# =============================================================================

def _get_sticky_file(exec_config: ExecutionConfig) -> str:
    """Return path to the sticky regime file."""
    return os.path.join(exec_config.data_dir, ".shockarb_regime")


def _get_sticky_regime(exec_config: ExecutionConfig) -> Optional[str]:
    """Read the sticky regime from .shockarb_regime file."""
    sticky_file = _get_sticky_file(exec_config)
    if not os.path.exists(sticky_file):
        return None
    
    with open(sticky_file, "r") as f:
        regime_name = f.read().strip()
    
    return regime_name if regime_name else None


def _set_sticky_regime(regime_name: str, exec_config: ExecutionConfig) -> None:
    """Write the sticky regime to .shockarb_regime file."""
    get_regime(regime_name)  # Will raise ValueError if invalid
    
    sticky_file = _get_sticky_file(exec_config)
    with open(sticky_file, "w") as f:
        f.write(regime_name)


def _resolve_regime(
    args,
    exec_config: ExecutionConfig,
    require: bool = True,
) -> Optional[HistoricFactorModel]:
    """Resolve regime from CLI args or sticky file."""
    regime_name = None
    source = None
    
    # Priority 1: --regime flag
    if hasattr(args, "regime") and args.regime:
        regime_name = args.regime
        source = "--regime flag"
    
    # Priority 2: --universe flag (legacy compatibility)
    elif hasattr(args, "universe") and args.universe:
        universe_map = {
            "us": "ukraine_shock",
            "global": "global_ukraine_shock",
        }
        regime_name = universe_map.get(args.universe.lower())
        if regime_name:
            logger.warning(
                "--universe flag is DEPRECATED. "
                f"Use --regime instead: --regime {regime_name}"
            )
            source = f"--universe {args.universe} (DEPRECATED, mapped to {regime_name})"
        else:
            print(f"❌ Unknown universe: '{args.universe}'.")
            print("   The --universe flag is deprecated.")
            print(f"   Use --regime instead. Available: {', '.join(list_regimes())}")
            sys.exit(1)
    
    # Priority 3: Sticky file
    else:
        regime_name = _get_sticky_regime(exec_config)
        if regime_name:
            source = "sticky file (.shockarb_regime)"
    
    # Handle not found
    if not regime_name:
        if require:
            print("❌ No regime specified.")
            print("   Set a regime: python -m shockarb set-regime <regime_name>")
            print("   Or use:       python -m shockarb build --regime <regime_name>")
            print(f"   Available regimes: {', '.join(list_regimes())}")
            sys.exit(1)
        else:
            return None
    
    # Resolve regime
    try:
        regime = get_regime(regime_name)
        logger.info(f"Regime: {regime.name} (from {source})")
        return regime
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)


# =============================================================================
# Universe registry (legacy compatibility)
# =============================================================================

UNIVERSES: dict[str, UniverseConfig] = {
    "us": US_UNIVERSE,
    "global": GLOBAL_UNIVERSE,
}

def get_universe(name: str) -> UniverseConfig:
    """Look up a universe by name (case-insensitive). DEPRECATED."""
    key = name.lower()
    if key not in UNIVERSES:
        raise ValueError(f"Unknown universe: '{name}'. Available: {list(UNIVERSES.keys())}")
    return UNIVERSES[key]


# =============================================================================
# Commands
# =============================================================================

def cmd_build(args) -> None:
    """Fit and save a factor model."""
    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir,
        log_to_file=not args.no_log,
    )
    
    regime = _resolve_regime(args, exec_cfg, require=True)
    universe = regime.universe
    
    logger.info(f"Building model: {regime.description}")
    model = pipeline.build(universe, exec_cfg, regime=regime)
    path = pipeline.save_model(model, universe.name, exec_cfg, regime=regime)
    pipeline.export_csvs(model, universe.name, exec_cfg)

    print(f"\n✅ Model saved: {path}")
    print(f"   Regime:            {regime.name}")
    print(f"   Factors:           {model.diagnostics.n_factors}")
    print(f"   Variance explained: {model.diagnostics.cumulative_variance:.1%}")
    print(f"   Stocks:            {model.diagnostics.n_stocks}")


def cmd_score(args) -> None:
    """Score returns against a fitted model."""
    from datetime import date as _date
    
    exec_cfg = ExecutionConfig(
        data_dir=args.data_dir,
        log_to_file=not args.no_log,
    )
    
    regime = _resolve_regime(args, exec_cfg, require=True)
    universe = regime.universe
    
    model_path = args.model or pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
    if not model_path:
        print(f"❌ No model found for regime '{regime.name}' / universe '{universe.name}'.")
        print("   Run 'build' first.")
        sys.exit(1)

    model = pipeline.load_model(model_path)
    etf_tickers   = list(model.etf_returns.columns) or list(universe.market_etfs)
    stock_tickers = list(model.stock_returns.columns) or list(universe.individual_stocks)

    if args.date:
        etf_returns, stock_returns = _fetch_historical(etf_tickers, stock_tickers, args.date)
        title = f"{regime.name.upper()} | {args.date}"
        scores = model.score(etf_returns, stock_returns)
    else:
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
        title = f"{regime.name.upper()} | LIVE"
        print(f"\n{prov.summary()}\n")

    print_scores(scores, title, top_n=args.top,
                 min_confidence=args.min_confidence,
                 min_r_squared=args.min_r_squared)
    if args.out:
        scores.to_csv(args.out)
        print(f"\n📁 Saved to: {args.out}")


def cmd_export(args) -> None:
    """Export model factor tables to CSV."""
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    regime = _resolve_regime(args, exec_cfg, require=True)
    universe = regime.universe
    
    model_path = pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
    if not model_path:
        print(f"❌ No model found for regime '{regime.name}'")
        sys.exit(1)

    model = pipeline.load_model(model_path)
    basis_path, loadings_path = pipeline.export_csvs(model, universe.name, exec_cfg)

    print("✅ Exported:")
    print(f"   ETF basis:      {basis_path}")
    print(f"   Stock loadings: {loadings_path}")


def cmd_show(args) -> None:
    """Display model diagnostics and factor structure."""
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    regime = _resolve_regime(args, exec_cfg, require=True)
    universe = regime.universe
    
    model_path = pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
    if not model_path:
        print(f"❌ No model found for regime '{regime.name}'")
        sys.exit(1)

    if args.verbose:
        print_model_state(model_path)
    else:
        model = pipeline.load_model(model_path)
        display_name = universe.name.upper() if len(universe.name) <= 15 else regime.name.upper()
        
        print(f"\n{'='*60}")
        print(f"  SHOCKARB MODEL: {display_name}")
        print(f"{'='*60}")
        print(f"  Regime:  {regime.description}")
        print(f"  Source:  {model_path}\n")
        print(model.diagnostics.summary())
        print()


def cmd_set_regime(args) -> None:
    """Set the active regime (sticky across sessions)."""
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    try:
        regime = get_regime(args.regime_name)
        _set_sticky_regime(args.regime_name, exec_cfg)
        print(f"✅ Active regime set to: {regime.name}")
        print(f"   {regime.description}")
        print(f"\n   This regime will be used by default for all commands.")
        print(f"   Override with --regime flag if needed.")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)


def cmd_show_regime(args) -> None:
    """Display the current sticky regime."""
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    regime_name = _get_sticky_regime(exec_cfg)
    
    if not regime_name:
        print("❌ No regime is currently set.")
        print("   Set one with: python -m shockarb set-regime <regime_name>")
        print(f"   Available: {', '.join(list_regimes())}")
        sys.exit(1)
    
    try:
        regime = get_regime(regime_name)
        print(f"\n✅ Current regime: {regime.name}")
        print(f"   {regime.description}")
        print(f"\n   Period: {regime.universe.start_date} to {regime.universe.end_date}")
        print(f"   Factors: {regime.universe.n_components}")
        if regime.tags: print(f"   Tags: {', '.join(regime.tags)}")
        if regime.supersedes: print(f"   Supersedes: {regime.supersedes}")
        print()
    except ValueError as e:
        print(f"❌ Sticky regime '{regime_name}' is invalid: {e}")
        sys.exit(1)


def cmd_list_regimes(args) -> None:
    """List all available regimes."""
    regimes = [get_regime(name) for name in list_regimes()]
    print("\n" + "="*70)
    print("  AVAILABLE REGIMES")
    print("="*70)
    for regime in regimes:
        print(f"\n  {regime.name}")
        print(f"  {'-' * len(regime.name)}")
        print(f"  {regime.description}")
        print(f"  Period: {regime.universe.start_date} to {regime.universe.end_date}")
        print(f"  Factors: {regime.universe.n_components}")
        if regime.tags: print(f"  Tags: {', '.join(regime.tags)}")
    print("\n" + "="*70 + "\n")


def cmd_add_asset(args) -> None:
    """Add new assets to an existing model and optionally save the result."""
    exec_cfg = ExecutionConfig(data_dir=args.data_dir, log_to_file=not args.no_log)
    regime = _resolve_regime(args, exec_cfg, require=True)
    universe = regime.universe

    model_path = args.model or pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
    if not model_path:
        print(f"❌ No model found for regime '{regime.name}'.\n   Run 'build' first.")
        sys.exit(1)

    model = pipeline.load_model(model_path)
    tickers = args.tickers

    already = [t for t in tickers if t in model.loadings.index]
    if already:
        print(f"⚠️  Already in model (skipped): {already}")
        tickers = [t for t in tickers if t not in already]

    if not tickers:
        print("Nothing new to add.")
        sys.exit(0)

    summary = pipeline.add_assets(tickers, model, universe, exec_cfg)
    if summary.empty:
        print("❌ No tickers could be added (check logs for details).")
        sys.exit(1)

    print(f"\n{'='*60}\n  ADDED ASSETS — {regime.name.upper()}\n{'='*60}")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string() + "\n")

    if args.save:
        path = pipeline.save_model(model, universe.name, exec_cfg, regime=regime)
        pipeline.export_csvs(model, universe.name, exec_cfg)
        print(f"✅ Model saved: {path}\n   Stocks now in model: {model.diagnostics.n_stocks}")
    else:
        print("ℹ️  Model NOT saved (pass --save to persist the change).")


def cmd_remove_asset(args) -> None:
    """Remove assets from an existing model and optionally save the result."""
    exec_cfg = ExecutionConfig(data_dir=args.data_dir, log_to_file=not args.no_log)
    regime = _resolve_regime(args, exec_cfg, require=True)
    universe = regime.universe

    model_path = args.model or pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
    if not model_path:
        print(f"❌ No model found for regime '{regime.name}'.\n   Run 'build' first.")
        sys.exit(1)

    model = pipeline.load_model(model_path)
    missing = [t for t in args.tickers if t not in model.loadings.index]
    if missing:
        print(f"⚠️  Not in model (skipped): {missing}")

    tickers = [t for t in args.tickers if t in model.loadings.index]
    if not tickers:
        print("Nothing to remove.")
        sys.exit(0)

    removed = []
    for ticker in tickers:
        model.remove_asset(ticker)
        removed.append(ticker)

    print(f"\n{'='*60}\n  REMOVED ASSETS — {regime.name.upper()}\n{'='*60}")
    for t in removed:
        print(f"  ✅ {t}")
    print()

    if args.save:
        path = pipeline.save_model(model, universe.name, exec_cfg, regime=regime)
        pipeline.export_csvs(model, universe.name, exec_cfg)
        print(f"✅ Model saved: {path}\n   Stocks now in model: {model.diagnostics.n_stocks}")
    else:
        print("ℹ️  Model NOT saved (pass --save to persist the change).")


def cmd_backtest(args) -> None:
    """Execute the walk-forward backtest."""
    from shockarb.backtest import Backtest, BacktestConfig
    from shockarb.regimes import get_regime
    import shockarb.pipeline as pipeline
    from loguru import logger
    from datetime import date, timedelta
    
    exec_cfg = ExecutionConfig(data_dir=args.data_dir)
    regime_name = _resolve_regime(args, exec_cfg)
    regime = get_regime(regime_name.name)
    
    config = BacktestConfig(
        universe=regime.universe,
        holding_periods=tuple(args.holding_periods),
        top_n=args.top_n,
        min_confidence=args.min_confidence,
        min_r_squared=args.min_r_squared,
        return_type=args.return_type
    )

    static_model = None
    if args.model:
        logger.info(f"Loading static model for out-of-sample backtest: {args.model}")
        static_model = pipeline.load_model(args.model)
        
        # Override dates to fetch recent trailing window
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=args.trailing_window + 20) # +20 buffer for weekends/holidays
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        logger.info(f"Fetching trailing {args.trailing_window} trading days ({start_str} to {end_str})...")
    else:
        logger.info(f"Preparing rolling backtest for historical regime: {regime.name}")
        start_str = regime.universe.start_date
        end_str = regime.universe.end_date

    all_tickers = regime.universe.market_etfs + regime.universe.individual_stocks
    prices = pipeline.fetch_prices(
        tickers=all_tickers,
        start=start_str,
        end=end_str,
        cache_name=f"backtest_{regime.universe.name}_{'static' if static_model else 'rolling'}"
    )
    returns = pipeline.prices_to_returns(prices)
    
    # If static, strictly enforce the trading day count
    if static_model:
        returns = returns.tail(args.trailing_window)
    
    historical_etf_returns = returns[regime.universe.market_etfs]
    historical_stock_returns = returns[regime.universe.individual_stocks]
    
    logger.info("Executing cohort-tracking backtest engine...")
    runner = Backtest(
        config, 
        historical_etf_returns, 
        historical_stock_returns, 
        static_model=static_model  # Pass it in!
    )
    results = runner.run()
    
    if results.summary.empty:
        logger.warning("No trades were generated. Try lowering thresholds.")
        return
        
    mode_str = "STATIC" if static_model else "ROLLING"
    print(f"\n{'='*80}")
    print(f" 📊 BACKTEST SUMMARY: {regime.name.upper()} | MODE: {mode_str} | TYPE: {args.return_type.upper()}")
    print(f"{'='*80}")
    print(results.summary.to_string())
    print(f"{'='*80}")


# =============================================================================
# Helpers
# =============================================================================

def _fetch_historical(etf_tickers: list, stock_tickers: list, date_str: str) -> tuple[pd.Series, pd.Series]:
    """Fetch closing returns for a historical date, snapping to nearest valid trading day."""
    target = pd.to_datetime(date_str)
    start = (target - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    def get_returns(tickers: list) -> pd.Series:
        raw = yf.download(tickers, start=start, end=end, progress=False)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        returns = prices.dropna(axis=1, how="all").ffill().pct_change().dropna(how="all")
        valid = returns.index[returns.index <= target]
        if valid.empty:
            raise ValueError(f"No trading data on or before {date_str}")

        matched = valid[-1]
        if matched != target:
            logger.warning(f"Date {date_str} snapped to {matched.strftime('%Y-%m-%d')}")
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
  # Regime workflow
  %(prog)s set-regime ukraine_shock
  %(prog)s build
  %(prog)s score

  # One-shot regime override
  %(prog)s build --regime gulf_war_recovery
  %(prog)s score --regime ukraine_shock --date 2022-03-01

  # Walk-forward backtesting
  %(prog)s backtest --return-type residual
        """,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data directory (default: ./data or $SHOCK_ARB_DATA_DIR)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p = sub.add_parser("build", help="Fit and save a factor model")
    p.add_argument("--regime", "-r", help="Regime name (overrides sticky)")
    p.add_argument("--universe", "-u", help="[LEGACY] us | global (maps to ukraine_shock)")
    p.add_argument("--no-log", action="store_true", help="Disable file logging")
    p.set_defaults(func=cmd_build)

    # score
    p = sub.add_parser("score", help="Score returns against a fitted model")
    p.add_argument("--regime", "-r", help="Regime name (overrides sticky)")
    p.add_argument("--universe", "-u", help="[LEGACY] us | global (maps to ukraine_shock)")
    p.add_argument("--from-open", "-O", action="store_true", help="Use today's session open as denominator")
    p.add_argument("--use-prior-close", "-p", action="store_true", help="Force daily close-to-close returns")
    p.add_argument("--date",   "-d", help="Historical date YYYY-MM-DD")
    p.add_argument("--model",  "-m", help="Specific model .json to load")
    p.add_argument("--out", "-o", help="Save score results to CSV")
    p.add_argument("--top",           "-n", type=int,   default=20,    help="Show top N results")
    p.add_argument("--min-confidence",      type=float, default=0.001, help="Min confidence_delta to show (default 0.1%%)")
    p.add_argument("--min-r-squared",       type=float, default=0.30,  help="Min R² to show (default 0.30)")
    p.add_argument("--save-tape", action="store_true", help="Save raw daily OHLCV before scoring.")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(func=cmd_score)

    # export
    p = sub.add_parser("export", help="Export model to CSVs")
    p.add_argument("--regime", "-r", help="Regime name (overrides sticky)")
    p.add_argument("--universe", "-u", help="[LEGACY] us | global")
    p.set_defaults(func=cmd_export)

    # show
    p = sub.add_parser("show", help="Display model summary")
    p.add_argument("--regime", "-r", help="Regime name (overrides sticky)")
    p.add_argument("--universe", "-u", help="[LEGACY] us | global")
    p.add_argument("--verbose", "-v", action="store_true", help="Full factor tables")
    p.set_defaults(func=cmd_show)

    # regimes
    p_set = sub.add_parser("set-regime", help="Set active regime (sticky)")
    p_set.add_argument("regime_name")
    p_set.set_defaults(func=cmd_set_regime)

    sub.add_parser("show-regime", help="Display current regime").set_defaults(func=cmd_show_regime)
    sub.add_parser("list-regimes", help="List all available regimes").set_defaults(func=cmd_list_regimes)

    # assets
    p = sub.add_parser("add-asset", help="Add new assets to an existing model")
    p.add_argument("tickers", nargs="+", help="Ticker symbol(s) to add")
    p.add_argument("--regime", "-r", help="Regime name (overrides sticky)")
    p.add_argument("--model",  "-m", help="Specific model .json to load")
    p.add_argument("--save",   "-s", action="store_true", help="Save the updated model")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(func=cmd_add_asset)

    p = sub.add_parser("remove-asset", help="Remove assets from an existing model")
    p.add_argument("tickers", nargs="+", help="Ticker symbol(s) to remove")
    p.add_argument("--regime", "-r", help="Regime name (overrides sticky)")
    p.add_argument("--model",  "-m", help="Specific model .json to load")
    p.add_argument("--save",   "-s", action="store_true", help="Save the updated model")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(func=cmd_remove_asset)

    # backtest
    p = sub.add_parser("backtest", help="Run walk-forward backtest to measure signal decay.")
    p.add_argument("--regime", "-r", type=str, help="Regime name (overrides sticky)")
    p.add_argument("--model", "-m", type=str, help="Specific model .json to load (enables static out-of-sample mode)")
    p.add_argument("--trailing-window", type=int, default=120, help="Days of recent data to test the static model against")
    p.add_argument("--return-type", choices=["raw", "residual", "both"], default="both",
                   help="Calculate gross returns ('raw'), factor-hedged ('residual'), or 'both'.")
    p.add_argument("--holding-periods", nargs="+", type=int, default=[1, 2, 3, 5], 
                   help="Days to hold positions (space separated, e.g., 1 2 3)")
    p.add_argument("--top-n", type=int, default=10, help="Max positions to enter per day")
    p.add_argument("--min-confidence", type=float, default=0.005, help="Minimum entry threshold")
    p.add_argument("--min-r-squared", type=float, default=0.50, help="Minimum model fit quality")
    p.set_defaults(func=cmd_backtest)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.data_dir:
        os.environ["SHOCK_ARB_DATA_DIR"] = args.data_dir

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\u23f9\ufe0f  Interrupted")
        sys.exit(130)
    except Exception as exc:
        logger.exception(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()