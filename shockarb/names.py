import os
import json
import pandas as pd
from loguru import logger

class TickerReferenceResolver:
    """
    Extensible plugin to map ticker symbols to company names and industries.
    Searches a list of local reference files sequentially. Stops searching 
    for a ticker once a match is found and caches the result.
    """
    def __init__(self, 
                 file_paths: list = None,
                 cache_path: str = "./data/ticker_reference_cache.json"):
        
        # Default waterfall sequence: NYSE first, then NASDAQ
        self.file_paths = file_paths or [
            "./data/nyse.csv", 
            "./data/nasdaq.csv"
        ]
        self.cache_path = cache_path
        
        self._cache = self._load_cache()
        self._loaded_dfs = {}  # In-memory cache for the reference DataFrames

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Cache file unreadable ({e}) — starting fresh.")
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=4)

    def _get_reference_df(self, path: str) -> pd.DataFrame:
        """Loads a reference file into memory if not already loaded."""
        if path in self._loaded_dfs:
            return self._loaded_dfs[path]
        
        if not os.path.exists(path):
            logger.warning(f"Reference file not found: {path}")
            self._loaded_dfs[path] = None
            return None
            
        logger.info(f"Loading reference file into memory: {path}")
        if path.endswith('.csv'):
            df = pd.read_csv(path, usecols=['Symbol', 'Name', 'Industry'])
        elif path.endswith('.parquet'):
            df = pd.read_parquet(path, columns=['Symbol', 'Name', 'Industry'])
        else:
            raise ValueError(f"Unsupported file format for {path}")
            
        df['Industry'] = df['Industry'].fillna('Unknown')
        # Drop duplicates to prevent pandas from returning a Series instead of a scalar later
        df = df.drop_duplicates(subset=['Symbol']).set_index('Symbol')
        self._loaded_dfs[path] = df
        return df

    def get_reference(self, tickers: list) -> dict:
        """
        Returns a dictionary mapping: 
        {'TICKER': {'Name': 'Company Name', 'Industry': 'Industry Name'}}
        """
        result = {}
        missing_tickers = set()

        # 1. Check the local JSON cache first
        for ticker in tickers:
            if ticker in self._cache:
                result[ticker] = self._cache[ticker]
            else:
                missing_tickers.add(ticker)

        if not missing_tickers:
            return result

        # 2. Sequential Waterfall Search for missing tickers
        for path in self.file_paths:
            if not missing_tickers:
                break  # Early exit: all tickers have been found
                
            df = self._get_reference_df(path)
            if df is None:
                continue
                
            # Find which missing tickers exist in this specific file
            found_tickers = missing_tickers.intersection(df.index)
            
            for ticker in found_tickers:
                entry = {
                    "Name": str(df.loc[ticker, 'Name']),
                    "Industry": str(df.loc[ticker, 'Industry'])
                }
                self._cache[ticker] = entry
                result[ticker] = entry
                missing_tickers.remove(ticker)  # Amputate from the search queue

        # 3. Handle any tickers that were never found in any file
        for ticker in missing_tickers:
            entry = {"Name": ticker, "Industry": "ETF / Unknown"}
            self._cache[ticker] = entry
            result[ticker] = entry

        # Update the disk cache with the new findings
        self._save_cache()

        return result