"""
check_1991.py — Gulf War Recovery regime data validation and permanent caching

ONE-TIME SCRIPT: Downloads and permanently caches all 1991 historical data
so that GULF_WAR_RECOVERY regime never depends on future yfinance availability.

This script tests ticker availability for the Gulf War recovery period
(1991-03-01 to 1991-06-28) and downloads confirmed tickers via datamgr with
permanent retention. Once cached, this data is stored forever in per-ticker
parquet files under data/prices/daily/ and never needs re-downloading.

Usage
-----
    cd C:\\Users\\nlgun\\personal\\nlgcode\\shockarb_lab
    python check_1991.py

Output
------
    1. Summary of ticker availability for 1991-03-01 to 1991-06-28
    2. ALL confirmed tickers downloaded and permanently cached via datamgr
    3. Recommended ticker lists for GULF_WAR_RECOVERY regime definition
    4. Coverage statistics (% of expected trading days with data)

Data Storage
------------
    - Per-ticker parquets: data/prices/daily/{ticker}.parquet
    - Manifest registry:   data/manifest.json
    - Fields stored:       close, adj_close, adj_factor, open, high, low, volume
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

# Ensure imports work from shockarb_lab directory
sys.path.insert(0, str(Path(__file__).parent))

from datamgr.coordinator import DataCoordinator
from datamgr.providers.yfinance import YFinanceProvider
from datamgr.requests import DataRequest, Frequency
from datamgr.stores.parquet import ParquetStore
from shockarb.store import DataStore

# =============================================================================
# Configuration
# =============================================================================

# Gulf War recovery calibration window (ceasefire to normalization)
START_DATE = "1991-03-01"
END_DATE = "1991-06-28"

# Expected trading days in this window (approximate)
# March 1991: 21 days, April: 22 days, May: 22 days, June 1-28: 20 days ≈ 85 days
EXPECTED_TRADING_DAYS = 85

# Coverage threshold: require at least this % of expected days
MIN_COVERAGE_PCT = 0.80  # 80% = ~68 days minimum (relaxed due to 1991 data quality)

# Data directory (same as ExecutionConfig default)
DATA_DIR = os.environ.get("SHOCK_ARB_DATA_DIR", os.path.join(os.getcwd(), "data"))

# Candidate tickers from the regime adaptation plan
# These are organized by category for the report

CANDIDATE_TICKERS = {
    "indexes": [
        "^GSPC",  # S&P 500 index
        "^DJI",   # Dow Jones Industrial Average
        "^IXIC",  # Nasdaq Composite
    ],
    "commodities": [
        "GC=F",   # Gold futures
        "CL=F",   # WTI crude futures
    ],
    "treasuries": [
        "^TYX",   # 30-year Treasury yield (NOTE: yields, not prices)
        "^TNX",   # 10-year Treasury yield
        "^IRX",   # 13-week Treasury bill
    ],
    "energy": [
        "XOM",    # Exxon
        "CVX",    # Chevron (may not exist in 1991 form)
        "MOB",    # Mobil (pre-merger, may be available)
    ],
    "industrials": [
        "GE",     # General Electric
        "BA",     # Boeing
        "MMM",    # 3M
        "CAT",    # Caterpillar
    ],
    "defense": [
        "LMT",    # Lockheed Martin
        "RTX",    # Raytheon (legacy United Technologies)
        "GD",     # General Dynamics
        "NOC",    # Northrop Grumman
    ],
    "technology": [
        "IBM",    # IBM
        "MSFT",   # Microsoft (IPO 1986)
        "INTC",   # Intel
        "TXN",    # Texas Instruments
    ],
    "healthcare": [
        "JNJ",    # Johnson & Johnson
        "MRK",    # Merck
        "PFE",    # Pfizer
        "LLY",    # Eli Lilly
    ],
    "consumer": [
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        "PEP",    # PepsiCo
        "MCD",    # McDonald's
    ],
    "financials": [
        "JPM",    # JPMorgan
        "BAC",    # Bank of America
        "C",      # Citigroup (Citicorp in 1991)
        "WFC",    # Wells Fargo
    ],
    "materials": [
        "DD",     # DuPont
        "DOW",    # Dow Chemical (may not exist in 1991 ticker form)
    ],
}

# =============================================================================
# Core Functions
# =============================================================================

def setup_coordinator() -> DataCoordinator:
    """Initialize DataCoordinator with ParquetStore and YFinanceProvider."""
    inner_store = DataStore(DATA_DIR)
    store = ParquetStore(inner_store)
    provider = YFinanceProvider()
    return DataCoordinator(store, provider)


def check_ticker_availability(
    coordinator: DataCoordinator,
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, Dict]:
    """
    Download data for all tickers with permanent retention and return coverage stats.

    Parameters
    ----------
    coordinator : DataCoordinator
    tickers : list of str
        All candidate tickers to test.
    start, end : str
        Date range in YYYY-MM-DD format.

    Returns
    -------
    dict
        {ticker: {"available": bool, "days": int, "coverage_pct": float, 
                  "first_date": str, "last_date": str, "error": str}}
    """
    logger.info(f"Testing {len(tickers)} tickers for {start} → {end}")
    logger.info(f"Data will be permanently cached in: {DATA_DIR}/prices/daily/")
    
    results = {}
    
    # Register all tickers with permanent retention
    coordinator.register(DataRequest(
        tickers=tuple(tickers),
        start=start,
        end=end,
        frequency=Frequency.DAILY,
        retention="permanent",  # <-- PERMANENT CACHING
        requester="check_1991",
    ))
    
    # Fulfill the request (downloads and caches)
    try:
        data_dict = coordinator.fulfill()
        df = data_dict.get("check_1991", pd.DataFrame())
        
        if df.empty:
            logger.error("No data returned from coordinator.fulfill()")
            for ticker in tickers:
                results[ticker] = {
                    "available": False,
                    "days": 0,
                    "coverage_pct": 0.0,
                    "first_date": "",
                    "last_date": "",
                    "error": "No data returned"
                }
            return results
        
        # Analyze each ticker's coverage
        for ticker in tickers:
            if ticker not in df.columns:
                results[ticker] = {
                    "available": False,
                    "days": 0,
                    "coverage_pct": 0.0,
                    "first_date": "",
                    "last_date": "",
                    "error": "Ticker not in response"
                }
                continue
            
            series = df[ticker].dropna()
            
            if series.empty:
                results[ticker] = {
                    "available": False,
                    "days": 0,
                    "coverage_pct": 0.0,
                    "first_date": "",
                    "last_date": "",
                    "error": "All NaN values"
                }
                continue
            
            n_days = len(series)
            coverage = n_days / EXPECTED_TRADING_DAYS
            
            results[ticker] = {
                "available": coverage >= MIN_COVERAGE_PCT,
                "days": n_days,
                "coverage_pct": coverage,
                "first_date": str(series.index[0].date()),
                "last_date": str(series.index[-1].date()),
                "error": "" if coverage >= MIN_COVERAGE_PCT else f"Coverage {coverage:.1%} < {MIN_COVERAGE_PCT:.1%}"
            }
            
    except Exception as exc:
        logger.error(f"Coordinator.fulfill() failed: {exc}")
        for ticker in tickers:
            results[ticker] = {
                "available": False,
                "days": 0,
                "coverage_pct": 0.0,
                "first_date": "",
                "last_date": "",
                "error": str(exc)
            }
    
    return results


def print_report(results: Dict[str, Dict], categories: Dict[str, List[str]]) -> None:
    """Print formatted availability report organized by category."""
    
    print("\n" + "=" * 80)
    print("GULF WAR RECOVERY (1991-03-01 to 1991-06-28) DATA AVAILABILITY REPORT")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Expected trading days: {EXPECTED_TRADING_DAYS}")
    print(f"Minimum coverage threshold: {MIN_COVERAGE_PCT:.0%}")
    print("\n" + "-" * 80)
    
    total_tested = 0
    total_available = 0
    available_by_category = {}
    
    for category, tickers in categories.items():
        print(f"\n{category.upper()}")
        print("-" * 80)
        
        available = []
        unavailable = []
        
        for ticker in tickers:
            total_tested += 1
            info = results.get(ticker, {})
            
            if info.get("available", False):
                total_available += 1
                available.append(ticker)
                status = "✓ AVAILABLE"
                detail = f"{info['days']:3d} days ({info['coverage_pct']:5.1%})  {info['first_date']} → {info['last_date']}"
            else:
                unavailable.append(ticker)
                status = "✗ MISSING  "
                detail = info.get("error", "Unknown error")
            
            print(f"  {ticker:8s}  {status}  {detail}")
        
        available_by_category[category] = available
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tickers tested:     {total_tested}")
    print(f"Available (>={MIN_COVERAGE_PCT:.0%} coverage): {total_available}")
    print(f"Missing/insufficient:     {total_tested - total_available}")
    print(f"Success rate:             {total_available / total_tested:.1%}")
    
    # Recommended ticker lists
    print("\n" + "=" * 80)
    print("RECOMMENDED TICKER LISTS FOR GULF_WAR_RECOVERY REGIME")
    print("=" * 80)
    
    print("\n# Factor basis ETFs/Indexes (paste into GULF_WAR_RECOVERY regime):")
    print("_GULF_WAR_FACTOR_BASIS = [")
    for category in ["indexes", "commodities", "treasuries"]:
        if available_by_category.get(category):
            print(f"    # {category.capitalize()}")
            for ticker in available_by_category[category]:
                print(f'    "{ticker}",')
    print("]")
    
    print("\n# Individual stocks (paste into GULF_WAR_RECOVERY regime):")
    print("_GULF_WAR_STOCKS = [")
    for category in ["energy", "industrials", "defense", "technology", "healthcare", "consumer", "financials", "materials"]:
        tickers = available_by_category.get(category, [])
        if tickers:
            print(f"    # {category.capitalize()}")
            for ticker in tickers:
                print(f'    "{ticker}",')
    print("]")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review the recommended ticker lists above")
    print("2. Copy them into shockarb/regimes.py GULF_WAR_RECOVERY definition")
    print("3. ALL available tickers are now permanently cached in:")
    print(f"   {DATA_DIR}/prices/daily/{{ticker}}.parquet")
    print("4. Run: python -m shockarb build --regime gulf_war_recovery")
    print("=" * 80 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run the 1991 data availability check."""
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>"
    )
    
    print("\n" + "=" * 80)
    print("Gulf War Recovery (1991) Data Validation & Permanent Caching")
    print("=" * 80)
    print(f"Start: {START_DATE}")
    print(f"End:   {END_DATE}")
    print(f"Data will be cached permanently in: {DATA_DIR}/prices/daily/")
    print("=" * 80 + "\n")
    
    # Flatten all tickers
    all_tickers = []
    for tickers in CANDIDATE_TICKERS.values():
        all_tickers.extend(tickers)
    
    logger.info(f"Testing {len(all_tickers)} candidate tickers")
    
    # Setup coordinator
    coordinator = setup_coordinator()
    
    # Check availability and download with permanent retention
    results = check_ticker_availability(coordinator, all_tickers, START_DATE, END_DATE)
    
    # Print report
    print_report(results, CANDIDATE_TICKERS)


if __name__ == "__main__":
    main()
