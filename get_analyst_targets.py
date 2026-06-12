import argparse
import pandas as pd
import requests
import yfinance as yf
import os
import sys
import time
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from loguru import logger

_REQUEST_TIMEOUT = 10  # seconds

# ==========================================
# 1. PROVIDER INTERFACE & IMPLEMENTATIONS
# ==========================================

class BaseProvider(ABC):
    @abstractmethod
    def fetch_target(self, symbol: str) -> dict:
        pass

class FMPProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable is not set.")
        self.base_url = "https://financialmodelingprep.com/api/v4/price-target-consensus"

    def fetch_target(self, symbol: str) -> dict:
        url = f"{self.base_url}?symbol={symbol}&apikey={self.api_key}"
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                pt_data = data[0]
                return {
                    "Symbol": symbol,
                    "Target_Mean": pt_data.get("targetConsensus"),
                    "Target_Median": pt_data.get("targetMedian"),
                    "Target_High": pt_data.get("targetHigh"),
                    "Target_Low": pt_data.get("targetLow")
                }
            return None

        elif response.status_code in [402, 403]:
            raise PermissionError(f"HTTP {response.status_code} on {symbol}: {response.text}")
        else:
            logger.warning(f"FMP failed to fetch {symbol}: HTTP {response.status_code}")
            return None

class YFinanceProvider(BaseProvider):
    def fetch_target(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            mean_target = info.get('targetMeanPrice')
            
            if mean_target: 
                return {
                    "Symbol": symbol,
                    "Target_Mean": mean_target,
                    "Target_Median": info.get('targetMedianPrice'),
                    "Target_High": info.get('targetHighPrice'),
                    "Target_Low": info.get('targetLowPrice')
                }
            return None
        except Exception as e:
            logger.warning(f"yfinance failed to fetch {symbol}: {e}")
            return None

class AlphaVantageProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("AV_API_KEY")
        if not self.api_key:
            raise ValueError("AV_API_KEY environment variable is not set.")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_target(self, symbol: str) -> dict:
        params = {"function": "EARNINGS", "symbol": symbol, "apikey": self.api_key}
        response = requests.get(self.base_url, params=params, timeout=_REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if "Information" in data and "rate limit" in data["Information"].lower():
                raise PermissionError(f"Alpha Vantage rate limit hit on {symbol} (Max 25/day or 5/min on free tier).")
                
            quarterly_earnings = data.get("quarterlyEarnings", [])
            if quarterly_earnings:
                latest = quarterly_earnings[0]
                return {
                    "Symbol": symbol,
                    "Estimated_EPS": latest.get("estimatedEPS"),
                    "Reported_EPS": latest.get("reportedEPS"),
                    "Surprise": latest.get("surprise"),
                    "Surprise_Pct": latest.get("surprisePercentage")
                }
            return None
        else:
            logger.warning(f"Alpha Vantage failed to fetch {symbol}: HTTP {response.status_code}")
            return None

class FinnhubProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY environment variable is not set.")
        self.base_url = "https://finnhub.io/api/v1/stock/price-target"

    def fetch_target(self, symbol: str) -> dict:
        params = {"symbol": symbol, "token": self.api_key}
        response = requests.get(self.base_url, params=params, timeout=_REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if data and "targetMean" in data:
                time.sleep(1.1)  # Respect the 60 calls/min limit
                return {
                    "Symbol": symbol,
                    "Target_Mean": data.get("targetMean"),
                    "Target_Median": data.get("targetMedian"),
                    "Target_High": data.get("targetHigh"),
                    "Target_Low": data.get("targetLow")
                }
            return None
        elif response.status_code == 429:
            raise PermissionError(f"Finnhub rate limit hit on {symbol}. Try slowing down the loop.")
        else:
            logger.warning(f"Finnhub failed to fetch {symbol}: HTTP {response.status_code}")
            return None

class FinvizProvider(BaseProvider):
    def fetch_target(self, symbol: str) -> dict:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        
        try:
            response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                target_td = soup.find('td', string='Target Price')

                if target_td:
                    target_price_str = target_td.find_next_sibling('td').text.strip()
                    if target_price_str != '-':
                        time.sleep(1)  # Be kind to their servers
                        return {
                            "Symbol": symbol,
                            "Target_Consensus": float(target_price_str)
                        }
            logger.info(f"No Target Price found on Finviz for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Finviz scraping failed for {symbol}: {e}")
            return None

# ==========================================
# 2. MAIN EXECUTION LOGIC
# ==========================================

def get_provider(provider_name: str) -> BaseProvider:
    providers = {
        "fmp": FMPProvider,
        "yfinance": YFinanceProvider,
        "alpha_advantage": AlphaVantageProvider,
        "finnhub": FinnhubProvider,
        "finviz": FinvizProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider '{provider_name}'.")
    return providers[provider_name]()

def _load_tickers(args: argparse.Namespace) -> list[str]:
    """
    Return the ticker list for this run.

    Either from --tickers directly, or from column `args.column` of the
    CSV at `args.file`. Raises on a bad CSV path/column so main() can
    report the error and exit.
    """
    if args.tickers:
        return [t.strip().upper() for t in args.tickers]

    filepath = args.file if args.file.endswith('.csv') else f"{args.file}.csv"
    df = pd.read_csv(filepath)
    tickers = df.iloc[:, args.column].dropna().unique().tolist()
    logger.info(f"Loaded {len(tickers)} unique tickers from '{filepath}' (column {args.column})")
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Fetch consensus analyst data for a list of tickers.")

    ticker_source = parser.add_mutually_exclusive_group()
    ticker_source.add_argument(
        "--tickers", "-t", type=str, nargs="+",
        help="One or more ticker symbols to fetch directly, e.g. --tickers AAPL MSFT GOOGL.",
    )
    ticker_source.add_argument(
        "--file", "-f", type=str, default="fundamentals.csv",
        help="Path to input CSV (used if --tickers is not given).",
    )
    parser.add_argument("--column", "-c", type=int, default=0, help="Zero-indexed column containing tickers.")

    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["yfinance", "fmp", "alpha_advantage", "finnhub", "finviz"],
        default="finviz",
        help="Data provider to use. Defaults to 'finviz' — the only provider that needs no API key.",
    )

    args = parser.parse_args()

    try:
        tickers = _load_tickers(args)
    except Exception as e:
        print(f"[Error] Failed to load tickers: {e}")
        sys.exit(1)

    try:
        provider = get_provider(args.provider)
        logger.info(f"Initialized provider: {args.provider.upper()}")
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    results = []
    logger.info(f"Fetching data for {len(tickers)} ticker(s)...")

    for symbol in tickers:
        try:
            data = provider.fetch_target(symbol)
            if data:
                results.append(data)
            else:
                logger.info(f"No data found for {symbol}")
        except PermissionError as e:
            print(f"\n[CRITICAL ERROR] {e}")
            print("Stopping execution to prevent further API blocks.")
            break
        except Exception as e:
            logger.warning(f"Unexpected error on {symbol}: {e}")

    if results:
        out_df = pd.DataFrame(results)
        print(out_df.to_string(index=False))
        output_file = f"{args.provider}_analyst_data.csv"
        out_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved data for {len(results)} tickers to '{output_file}'.")
    else:
        print("\nNo analyst data was retrieved.")

if __name__ == "__main__":
    main()