"""
Live Intraday Execution Scanner for the ShockArb pipeline.

This script acts as the operational Controller. It connects the data ingestion 
pipeline to the mathematical engine, and feeds the output to the presentation layer.

Workflow:
  1. Parses the CLI to determine the target universe(s) ('us', 'global', or 'all').
  2. Deserializes the most recently built JSON structural model for that universe.
  3. Fetches the live 1-minute intraday tape for all required ETFs and Equities.
  4. Amputates halted/dead tickers and forward-fills liquidity gaps.
  5. Passes the live return vectors to the engine for scoring.
  6. Renders the actionable alpha sheet to the terminal.
  7. Persists the complete mathematical output to a timestamped CSV for auditing.
"""

import time
import argparse
from datetime import datetime
import pandas as pd
import yfinance as yf
from loguru import logger

from src.config import ExecutionConfig, configure_logger
from src.pipeline import Pipeline
from src.report import print_live_alpha

def fetch_intraday_returns(tickers: list) -> pd.Series:
    """
    Fetches the live 1-minute intraday tape and calculates the cumulative return 
    from the opening print to the current minute.
    
    Uses robust data cleaning (dropna and ffill) to prevent a single halted asset 
    from collapsing the entire matrix calculation downstream.
    
    Fetches intraday data with robust error handling, retries, and pandas 2.0+ compliance."""
    
    # For intraday data, 'Close' represents the final price of the specific minute bar

    for attempt in range(max_retries):
        try:
            # Fetch 2 days to ensure we have yesterday's close (fixing Bug C)
            data = yf.download(tickers, period="2d", interval="1m", progress=False)['Close']
            
            if data.empty:
                raise ValueError("yfinance returned an empty DataFrame.")
            
            # Fix Bug B: Remove inplace=True
            data = data.dropna(axis=1, how='all')
            data = data.ffill()
            
            # Calculate return from prior close to current minute
            yesterdays_close = data[data.index.date < data.index[-1].date()].iloc[-1]
            current_price = data.iloc[-1]
            returns = (current_price - yesterdays_close) / yesterdays_close
            
            return returns
            
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            logger.warning(f"Data fetch failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    logger.error("Failed to fetch intraday data after multiple attempts.")
    return None # Handled gracefully by the main loop

def execute_scan(target_prefix: str, base_data_dir: str = None):
    """
    Executes a single pass of the scoring pipeline for a specific model prefix.
    This encapsulates the entire workflow from loading the structural basis
    to publishing the trading signals.
    """
    
    # 1. Configure the state for this specific target
    exec_config = ExecutionConfig(
        log_to_file=True,
        model_prefix=target_prefix
    )
    if base_data_dir:
        exec_config.data_dir = base_data_dir
        
    # 2. Automatically find the most recent chronological JSON for this prefix
    latest_json = Pipeline.find_latest_model(exec_config)
    if not latest_json:
        logger.warning(f"No compiled JSON model found for '{target_prefix}'. Skipping.")
        return
        
    model = Pipeline.load_model(latest_json)
    
    try:
        logger.info(f"Pulling live 1m tape for the '{target_prefix}' universe...")
        
        # 3. Extract strictly the tickers that exist within the fitted structural basis
        etf_tickers = list(model.etf_returns.columns)
        stock_tickers = list(model.stock_returns.columns)
        
        # 4. Fetch the live tape via yfinance API
        live_etfs = fetch_intraday_returns(etf_tickers)
        live_stocks = fetch_intraday_returns(stock_tickers)
        
        # 5. Engine Processing: Calculate absolute and relative mispricing deltas
        score_df = model.score(live_etfs, live_stocks)
        
        # 6. Presentation Layer: Render the actionable sheet to the terminal
        print_live_alpha(score_df, model_name=target_prefix)
        
        # 7. Persistence: Dump the complete execution state to a timestamped CSV
        datestring = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = exec_config.resolve_path(f"live_alpha_{target_prefix}_{datestring}.csv")
        score_df.to_csv(out_file)
        
        logger.success(f"Execution complete for {target_prefix}. Alpha sheet saved to {out_file}")
        
    except Exception as e:
        logger.error(f"Live scoring pipeline failed for {target_prefix} during execution: {e}")

def main():
    # Parse CLI arguments to seamlessly switch between configurations or run both
    parser = argparse.ArgumentParser(description="ShockArb Live Intraday Scanner")
    parser.add_argument(
        '--target', type=str, default='all',
        help="The prefix of the compiled JSON model to load ('us', 'global', or 'all')"
    )
    parser.add_argument(
        '--interval', type=int, default=60,
        help="Seconds to wait between scans. Set to 0 for a single run."
    )
    args = parser.parse_args()

    # Determine which models to run in the current loop
    if args.target.lower() == 'all':
        targets_to_run = ['us', 'global']
    else:
        targets_to_run = [args.target]

    # Set up the base logger once to avoid duplicating file handlers during the loop
    base_exec_config = ExecutionConfig(log_to_file=True, model_prefix="scanner")
    configure_logger(base_exec_config)

    # The Execution Loop
    if args.interval > 0:
        logger.info(f"Scanner armed. Monitoring {targets_to_run} every {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                for target in targets_to_run:
                    execute_scan(target, base_data_dir=base_exec_config.data_dir)
                
                logger.info(f"Scan pass complete. Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.warning("\nLive scanner manually disarmed. Exiting.")
    else:
        # Single execution if interval is set to 0
        for target in targets_to_run:
            execute_scan(target, base_data_dir=base_exec_config.data_dir)

if __name__ == "__main__":
    main()