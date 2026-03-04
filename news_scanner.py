"""
ShockArb Catalyst/News Scanner.
Fetches recent headlines for top arbitrage targets or a custom list of tickers.
"""

import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
from loguru import logger

def extract_article_data(article: dict) -> tuple:
    """Defensively extracts headline data across multiple YF JSON formats."""
    if 'title' in article and 'publisher' in article:
        return article.get('title'), article.get('publisher'), article.get('providerPublishTime')
    
    if 'content' in article and isinstance(article['content'], dict):
        content = article['content']
        title = content.get('title', 'Unknown Title')
        publisher = content.get('provider', {}).get('displayName', 'Unknown Publisher')
        pub_time = content.get('pubDate')
        
        if isinstance(pub_time, str):
            try:
                dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                return title, publisher, int(dt.timestamp())
            except ValueError:
                pass
        return title, publisher, pub_time

    return f"Format Unknown. Keys found: {list(article.keys())}", "N/A", None

def scan_news(csv_paths: list = None, top_n: int = 10, explicit_tickers: list = None):
    print(f"\n{'='*95}")
    print(f"📰 CATALYST SCANNER")
    print(f"{'='*95}")
    print("Reviewing for earnings reports, downgrades, or fundamental impairments...\n")

    targets_info = []

    # Route A: User provided a custom list of tickers
    if explicit_tickers:
        for t in explicit_tickers:
            targets_info.append({'ticker': t, 'signal': 'N/A (Manual Input)'})
            
    # Route B: Automatically pull top targets from the CSVs
    else:
        dfs = []
        for path in (csv_paths or []):
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'Ticker' not in df.columns:
                    df = df.rename(columns={df.columns[0]: 'Ticker'})
                dfs.append(df)
                
        if not dfs:
            logger.error("No valid CSVs loaded and no custom tickers provided.")
            return

        master_df = pd.concat(dfs, ignore_index=True)

        if 'confidence_delta' not in master_df.columns:
            logger.error("CSV is missing 'confidence_delta'.")
            return

        buys = master_df[master_df['confidence_delta'] > 0].copy()
        buys = buys.sort_values(by='confidence_delta', ascending=False).head(top_n)

        if buys.empty:
            logger.warning("No targets to scan.")
            return
            
        for _, row in buys.iterrows():
            targets_info.append({
                'ticker': row['Ticker'],
                'signal': f"{row['confidence_delta']:+.2%}"
            })

    # Fetch and print news
    for item in targets_info:
        ticker = item['ticker'].strip()
        print(f"[{ticker:<5}] | Model Signal: {item['signal']}")
        
        try:
            tick = yf.Ticker(ticker)
            news = tick.news
            
            if not news:
                print("  > No recent news found on the Yahoo Finance feed.")
            else:
                for article in news[:3]:
                    title, publisher, timestamp = extract_article_data(article)
                    
                    if isinstance(timestamp, (int, float)):
                        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                    else:
                        date_str = "Unknown Date"
                        
                    print(f"  > {date_str} | {publisher}")
                    print(f"    {title}")
                    
        except Exception as e:
            print(f"  > Error fetching news for {ticker}: {e}")
            
        print("-" * 95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan news headlines for targets.")
    parser.add_argument("--csv", nargs='+', default=[], help="Path(s) to CSV files")
    parser.add_argument("--top", type=int, default=10, help="Number of targets to scan from CSV")
    # Added the --tickers argument to accept manual input
    parser.add_argument("--tickers", nargs='+', help="List of specific tickers to scan (overrides CSV)")
    args = parser.parse_args()
    
    # If no explicit tickers and no CSVs provided, default to the standard US csv
    if not args.tickers and not args.csv:
        args.csv = ["./data/live_alpha_us.csv"]
        
    scan_news(args.csv, args.top, args.tickers)