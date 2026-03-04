"""
Data pipeline and model persistence - handles all I/O so the engine stays pure.

This module provides:
  - Market data fetching (yfinance with caching)
  - Return computation with robust missing data handling
  - Model save/load (JSON for portability and inspection)
  - CSV export for manual analysis

Design Principle: All file system interaction, API calls, and caching live here.
You can swap data sources (Bloomberg, Snowflake) without touching the engine.

Example
-------
    from shockarb import Pipeline, US_UNIVERSE, ExecutionConfig
    
    # Build and fit
    model = Pipeline.build(US_UNIVERSE)
    
    # Save for later
    Pipeline.save_model(model, US_UNIVERSE.name)
    
    # Load saved model
    model = Pipeline.load_model("models/us_20260303_120000.json")
"""

from __future__ import annotations
import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from shockarb.config import UniverseConfig, ExecutionConfig
from shockarb.engine import FactorModel
from shockarb.cache import CacheManager


# =============================================================================
# Pipeline Class
# =============================================================================

class Pipeline:
    """
    Orchestrates data fetching, model fitting, and persistence.
    
    This is a namespace class (all static/class methods) rather than an
    instantiated object, since it has no meaningful state.
    """
    
    # Default execution config used when none specified
    _default_exec: Optional[ExecutionConfig] = None
    
    @classmethod
    def _get_exec_config(cls, exec_config: Optional[ExecutionConfig] = None) -> ExecutionConfig:
        """Get execution config, creating default if needed."""
        if exec_config is not None:
            exec_config.configure_logger()
            return exec_config
        
        if cls._default_exec is None:
            cls._default_exec = ExecutionConfig()
            cls._default_exec.configure_logger()
        return cls._default_exec

    # =========================================================================
    # Data Fetching
    # =========================================================================

    @staticmethod
    def _get_cache_manager(exec_cfg: ExecutionConfig) -> CacheManager:
        """Return a CacheManager rooted in exec_cfg's data directory."""
        cache_dir = os.path.join(exec_cfg.data_dir, "cache")
        backup_dir = os.path.join(exec_cfg.data_dir, "backups")
        # Pass yf.download from this module so that patches on
        # `shockarb.pipeline.yf.download` are correctly intercepted.
        return CacheManager(cache_dir=cache_dir, backup_dir=backup_dir, downloader=yf.download)

    @staticmethod
    def fetch_prices(
        tickers: List[str],
        start: str,
        end: str,
        cache_path: Optional[str] = None,
        cache_name: Optional[str] = None,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> pd.DataFrame:
        """
        Download adjusted close prices from yfinance with OHLCV parquet caching.

        The full OHLCV download is cached internally; only 'Adj Close' is
        returned by this method. This means a second call with the same
        tickers and dates costs zero yfinance API calls.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols to download.
        start : str
            Start date (YYYY-MM-DD), inclusive.
        end : str
            End date (YYYY-MM-DD), exclusive per yfinance convention.
        cache_name : str
            Logical name for the cache file (e.g. ``"us_etf"``).
        exec_config : ExecutionConfig, optional
            Controls caching and paths.

        Returns
        -------
        DataFrame
            (dates × tickers) adjusted close prices.

        Notes
        -----
        If yfinance fails and the cache is also unavailable, returns synthetic
        crisis data for testing. This is logged at ERROR level.
        """
        from shockarb.pipeline import Pipeline  # local to avoid circular import
        exec_cfg = Pipeline._get_exec_config(exec_config)

        # Backwards-compatibility: derive cache_name and cache dir from legacy cache_path
        if cache_name is None:
            if cache_path is not None:
                p = Path(cache_path)
                cache_name = p.stem
                if cache_name.endswith("_ohlcv"):
                    cache_name = cache_name[:-6]
                # Use the legacy path's directory as the cache dir so the
                # file lands where the test / caller expects it
                cache_dir = str(p.parent)
                backup_dir = str(p.parent.parent / "backups")
                mgr = CacheManager(
                    cache_dir=cache_dir,
                    backup_dir=backup_dir,
                    downloader=yf.download,
                )
            else:
                cache_name = "prices"
                mgr = Pipeline._get_cache_manager(exec_cfg)
        else:
            mgr = Pipeline._get_cache_manager(exec_cfg)

        ohlcv = mgr.fetch_ohlcv(tickers, start, end, cache_name)

        if ohlcv is None or ohlcv.empty:
            logger.error(
                "CacheManager returned no data. "
                "*** FALLING BACK TO SYNTHETIC DATA ***"
            )
            return Pipeline._synthetic_prices(tickers, start, end)

        prices = mgr.extract_adj_close(ohlcv)

        # Report coverage
        received = set(prices.columns)
        missing_tickers = set(tickers) - received
        if missing_tickers:
            logger.warning(f"{len(missing_tickers)} tickers missing: {sorted(missing_tickers)}")

        return prices

    @staticmethod
    def _synthetic_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Generate synthetic prices mimicking crisis dynamics.
        
        *** FOR TESTING ONLY. Not real market data. ***
        
        Stylized facts embedded (Russia-Ukraine Feb-Mar 2022):
          - Broad equity: down ~5%
          - Energy: up ~15%
          - Gold: up ~6%
          - Defense: up ~8%
        """
        rng = np.random.RandomState(2022)
        dates = pd.bdate_range(start=start, end=end)
        T = len(dates)
        
        # Drift by sector (daily %)
        drift_map = {
            "VOO": -0.0014, "VYM": -0.0010, "VEU": -0.0020,
            "VDE": +0.0043, "TLT": +0.0006, "GLD": +0.0017,
            "USO": +0.0043, "ITA": +0.0023,
            "LMT": +0.0025, "RTX": +0.0020, "NOC": +0.0022,
            "CVX": +0.0025, "MSFT": -0.0020, "V": -0.0010,
        }
        default_drift = -0.0014
        default_vol = 0.012
        
        # Correlated market factor
        market_factor = rng.normal(0, 0.010, T)
        
        prices = {}
        for ticker in tickers:
            drift = drift_map.get(ticker, default_drift)
            beta = -0.3 if ticker in ("TLT", "GLD", "USO") else 0.7
            returns = beta * market_factor + rng.normal(drift, default_vol, T)
            prices[ticker] = 100 * np.cumprod(1 + returns)  # Start at 100 = synthetic flag
        
        logger.error(
            f"SYNTHETIC DATA: {T} days × {len(tickers)} tickers. "
            "All prices start at 100. Delete cache and retry when network available."
        )
        
        return pd.DataFrame(prices, index=dates)

    @staticmethod
    def prices_to_returns(
        prices: pd.DataFrame, 
        min_coverage: float = 0.8
    ) -> pd.DataFrame:
        """
        Convert price matrix to daily returns with robust missing data handling.
        
        Strategy:
          1. Drop tickers with <min_coverage data (recently IPO'd, etc.)
          2. Forward-fill small gaps (foreign holiday misalignment)
          3. Compute percentage returns
          4. Fill remaining isolated NaNs with 0.0 (preferable to losing entire days)
        
        Parameters
        ----------
        prices : DataFrame
            (T × N) adjusted close prices.
        
        min_coverage : float
            Minimum fraction of non-NaN prices to keep a ticker. Default 0.8.
        
        Returns
        -------
        DataFrame
            (T-1 × N') returns. N' ≤ N if low-coverage tickers dropped.
        """
        # Drop tickers with insufficient data
        coverage = prices.notna().mean()
        good = coverage[coverage >= min_coverage].index
        dropped = set(prices.columns) - set(good)
        
        if dropped:
            logger.warning(f"Dropped {len(dropped)} tickers (<{min_coverage:.0%} coverage): {sorted(dropped)}")
        
        prices = prices[good].ffill()
        returns = prices.pct_change().iloc[1:].dropna(how="all")
        
        # Fill isolated NaNs
        n_filled = returns.isna().sum().sum()
        if n_filled > 0:
            logger.info(f"Filled {n_filled} isolated NaN returns with 0.0")
            returns = returns.fillna(0.0)
        
        return returns

    # =========================================================================
    # Model Building
    # =========================================================================

    @classmethod
    def build(
        cls,
        universe: UniverseConfig,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> FactorModel:
        """
        Full pipeline: fetch data → compute returns → fit model.
        
        Parameters
        ----------
        universe : UniverseConfig
            What to analyze (tickers, dates, n_components).
        
        exec_config : ExecutionConfig, optional
            How to run (caching, paths). Uses default if not specified.
        
        Returns
        -------
        FactorModel
            Fitted model ready for scoring.
        
        Example
        -------
            from shockarb import Pipeline, US_UNIVERSE
            model = Pipeline.build(US_UNIVERSE)
        """
        exec_cfg = cls._get_exec_config(exec_config)
        logger.info(f"Building model: {universe.name}")
        
        # Fetch prices via CacheManager (full OHLCV cached; Adj Close returned)
        etf_prices = cls.fetch_prices(
            universe.market_etfs, 
            universe.start_date, 
            universe.end_date,
            cache_name=f"{universe.name}_etf",
            exec_config=exec_cfg,
        )
        stock_prices = cls.fetch_prices(
            universe.individual_stocks,
            universe.start_date,
            universe.end_date,
            cache_name=f"{universe.name}_stock",
            exec_config=exec_cfg,
        )
        
        # Convert to returns
        etf_returns = cls.prices_to_returns(etf_prices)
        stock_returns = cls.prices_to_returns(stock_prices)
        
        # Align on common dates
        common = etf_returns.index.intersection(stock_returns.index)
        logger.info(f"Aligned on {len(common)} common trading days")
        
        etf_returns = etf_returns.loc[common]
        stock_returns = stock_returns.loc[common]
        
        # Fit
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=universe.n_components)
        
        return model

    # =========================================================================
    # Persistence
    # =========================================================================

    @classmethod
    def save_model(
        cls,
        model: FactorModel,
        name: str,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> str:
        """
        Persist a fitted model to disk as JSON.
        
        JSON is used instead of pickle for:
          - Human readability and inspection
          - Version stability (pickle breaks across class changes)
          - Security (pickle can execute arbitrary code)
        
        Parameters
        ----------
        model : FactorModel
            Fitted model to save.
        
        name : str
            Model name (e.g., "us", "global"). Used in filename.
        
        exec_config : ExecutionConfig, optional
            Controls data directory.
        
        Returns
        -------
        str
            Path to saved file.
        """
        exec_cfg = cls._get_exec_config(exec_config)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        path = exec_cfg.resolve_path(filename)
        
        payload = model.to_dict()
        payload["metadata"]["name"] = name
        payload["metadata"]["created_at"] = datetime.now().isoformat()
        
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        
        logger.success(f"Model saved: {path}")
        return path

    @classmethod
    def load_model(cls, path: str) -> FactorModel:
        """
        Load a saved model from JSON.
        
        Parameters
        ----------
        path : str
            Path to .json model file.
        
        Returns
        -------
        FactorModel
            Reconstructed fitted model ready for scoring.
        """
        with open(path) as f:
            d = json.load(f)
        
        model = FactorModel.from_dict(d)
        logger.success(f"Model loaded: {path}")
        return model

    @classmethod
    def find_latest_model(
        cls, 
        name: str,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> Optional[str]:
        """
        Find the most recent saved model for a given name.
        
        Parameters
        ----------
        name : str
            Model name (e.g., "us", "global").
        
        exec_config : ExecutionConfig, optional
            Controls data directory.
        
        Returns
        -------
        str or None
            Path to latest model, or None if not found.
        """
        exec_cfg = cls._get_exec_config(exec_config)
        pattern = exec_cfg.resolve_path(f"{name}_*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            logger.warning(f"No saved models found: {pattern}")
            return None
        
        return files[-1]

    # =========================================================================
    # Export
    # =========================================================================

    @classmethod
    def export_csvs(
        cls,
        model: FactorModel,
        name: str,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> Tuple[str, str]:
        """
        Export human-readable CSVs of factor basis and stock loadings.
        
        Returns
        -------
        tuple of str
            (etf_basis_path, stock_loadings_path)
        """
        exec_cfg = cls._get_exec_config(exec_config)
        
        # ETF basis
        basis_path = exec_cfg.resolve_path(f"{name}_etf_basis.csv")
        model.etf_basis.sort_index().to_csv(basis_path, float_format="%.6f")
        
        # Stock loadings with diagnostics
        loadings_path = exec_cfg.resolve_path(f"{name}_stock_loadings.csv")
        output = model.loadings.copy()
        output["R_squared"] = model.diagnostics.stock_r_squared
        output["Residual_Vol"] = model.diagnostics.residual_vol
        output.sort_values("R_squared", ascending=False).to_csv(
            loadings_path, float_format="%.6f"
        )
        
        logger.info(f"Exported CSVs: {basis_path}, {loadings_path}")
        return basis_path, loadings_path


# =============================================================================
# Live Data Fetching Utilities
# =============================================================================

def fetch_live_returns(tickers: List[str], period: str = "5d") -> pd.Series:
    """
    Fetch recent market data and return the most recent day's returns.
    
    Uses 5-day window to safely compute returns across weekends/holidays.
    
    Parameters
    ----------
    tickers : List[str]
        Ticker symbols.
    
    period : str
        yfinance period string. Default "5d".
    
    Returns
    -------
    Series
        Today's returns indexed by ticker.
    
    Raises
    ------
    ValueError
        If yfinance returns no data.
    """
    logger.info(f"Fetching live data for {len(tickers)} tickers...")
    
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=False)
    
    if raw.empty:
        raise ValueError("yfinance returned no data")
    
    # Extract Adj Close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.get("Adj Close", raw.get("Close"))
    else:
        prices = raw
    
    if prices is None or prices.empty:
        raise ValueError("No price data in yfinance response")
    
    returns = prices.ffill().pct_change().dropna(how="all")
    
    if returns.empty:
        raise ValueError("Return computation produced empty result")
    
    return returns.iloc[-1]


def fetch_intraday_returns(
    tickers: List[str], 
    max_retries: int = 3
) -> Optional[pd.Series]:
    """
    Fetch intraday returns (open to current) with retry logic.
    
    Parameters
    ----------
    tickers : List[str]
        Ticker symbols.
    
    max_retries : int
        Number of retry attempts on failure.
    
    Returns
    -------
    Series or None
        Intraday returns, or None if all retries fail.
    """
    import time
    
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers, 
                period="2d", 
                interval="1m", 
                progress=False
            )
            
            if data.empty:
                raise ValueError("Empty response")
            
            # Get Close prices
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"]
            else:
                prices = data
            
            prices = prices.dropna(axis=1, how="all").ffill()
            
            # Yesterday's close vs current
            yesterday_close = prices[prices.index.date < prices.index[-1].date()].iloc[-1]
            current = prices.iloc[-1]
            
            returns = (current - yesterday_close) / yesterday_close
            return returns
            
        except Exception as e:
            wait = 5 * (attempt + 1)
            logger.warning(f"Intraday fetch failed (attempt {attempt+1}): {e}. Retry in {wait}s...")
            time.sleep(wait)
    
    logger.error("Intraday fetch failed after all retries")
    return None
