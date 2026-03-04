"""
Global Portfolio Sizing Utility.
Reads multiple raw ShockArb CSV exports to calculate a unified, dollar-denominated portfolio.
"""

import pandas as pd
import argparse
import os
import yfinance as yf
from loguru import logger

def generate_orders(csv_paths: list, capital: float, top_n: int = 5):
    dfs = []
    
    # 1. Load and concatenate all provided CSVs
    for path in csv_paths:
        if not os.path.exists(path):
            logger.warning(f"Alpha report not found: {path}")
            continue

        try:
            df = pd.read_csv(path)
            if 'Ticker' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'Ticker'})
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read CSV {path}: {e}")
            
    if not dfs:
        logger.error("No valid CSVs loaded.")
        return
        
    master_df = pd.concat(dfs, ignore_index=True)

    # 2. Filter for top positive conviction signals across ALL universes
    if 'confidence_delta' not in master_df.columns or 'delta' not in master_df.columns:
        logger.error("CSV is missing required columns ('confidence_delta', 'delta').")
        return

    buys = master_df[master_df['confidence_delta'] > 0].copy()
    buys = buys.sort_values(by='confidence_delta', ascending=False).head(top_n)

    if buys.empty:
        logger.warning("No positive alpha signals found today.")
        return
    
    # 3. Fetch Current Prices and Calculate Limit Orders
    tickers = buys['Ticker'].tolist()
    logger.info(f"Fetching live prices for {tickers}...")
    current_data = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]

    # Calculate weights and allocations
    total_conviction = buys['confidence_delta'].sum()
    buys['Weight'] = buys['confidence_delta'] / total_conviction
    buys['Dollar_Alloc'] = buys['Weight'] * capital

    # 4. Generate Final Trade Ticket
    print("\n" + "="*95)
    print(f"SHOCKARB UNIFIED TRADE TICKET | CAPITAL: ${capital:,.2f}")
    print(f"{'TICKER':<8} | {'ALLOCATION':<10} | {'DOLLAR AMT':<15} | {'CURRENT':<10} | {'TAKE-PROFIT TARGET'}")
    print("-" * 95)
    
    for _, row in buys.iterrows():
        ticker = row['Ticker']
        current_price = current_data[ticker]
        
        limit_price = current_price * (1 + row['delta'])
        shares = int(row['Dollar_Alloc'] / current_price)
        
        print(f"{ticker:<8} | {row['Weight']:>9.1%} | ${row['Dollar_Alloc']:>13,.2f} | ${current_price:>8.2f} | ${limit_price:>8.2f} ({shares} shares)")
    
    print("="*95)
    print("EXIT LOGIC: Place GTC Sell Limit Orders at the 'Take-Profit Target' level.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept one or more CSV files
    parser.add_argument("--csv", nargs='+', default=["./data/live_alpha_us.csv"], help="Path(s) to the alpha CSV files")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    
    generate_orders(args.csv, args.capital, args.top)