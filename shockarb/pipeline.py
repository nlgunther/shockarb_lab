"""
Data pipeline and model persistence - handles all I/O so the engine stays pure.

This module provides:
  - Market data fetching (yfinance with CacheManager integration)
  - Return computation with robust missing data handling
  - Live & Intraday data fetching for execution scanning
  - Model save/load/export (JSON/CSV)

Design Principle: All file system interaction, API calls, and caching live here.
You can swap data sources without touching the core mathematical engine.

Example
-------
    from shockarb import Pipeline, US_UNIVERSE

    model = Pipeline.build(US_UNIVERSE)
    Pipeline.save_model(model, "us")
    live_returns = Pipeline.fetch_intraday_returns(US_UNIVERSE.individual_stocks)
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


class Pipeline:
    """
    Unified orchestrator for data fetching, model fitting, and persistence.

    Designed as a namespace facade (all class/static methods) rather than an
    instantiated object, since it has no meaningful state of its own.
    """

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
    # Historical Data Fetching (Cached)
    # =========================================================================

    @staticmethod
    def _get_cache_manager(exec_cfg: ExecutionConfig) -> CacheManager:
        """Return a CacheManager rooted in exec_cfg's data directory."""
        cache_dir = os.path.join(exec_cfg.data_dir, "cache")
        backup_dir = os.path.join(exec_cfg.data_dir, "backups")
        # Pass yf.download from this module so that patches on
        # `shockarb.pipeline.yf.download` are correctly intercepted by tests.
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

        The full OHLCV download is cached internally via CacheManager; only
        'Adj Close' is returned. A second call with the same tickers and dates
        costs zero yfinance API calls.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols to download.
        start : str
            Start date (YYYY-MM-DD), inclusive.
        end : str
            End date (YYYY-MM-DD), exclusive per yfinance convention.
        cache_path : str, optional
            Legacy parameter. When provided, its stem is used as ``cache_name``
            and its directory is used as the cache root. Kept for backwards
            compatibility with existing call sites and tests.
        cache_name : str, optional
            Logical name for the cache file (e.g. ``"us_etf"``). Ignored when
            ``cache_path`` is supplied.
        exec_config : ExecutionConfig, optional
            Controls caching and paths. Ignored when ``cache_path`` is supplied.

        Returns
        -------
        DataFrame
            (dates x tickers) adjusted close prices.

        Notes
        -----
        If yfinance fails and the cache is also unavailable, returns synthetic
        crisis data for testing. This is logged at ERROR level.
        """
        exec_cfg = Pipeline._get_exec_config(exec_config)

        # Backwards-compatibility: when a legacy cache_path is supplied, derive
        # cache_name from its stem and use its directory as the cache root so
        # the file lands exactly where the caller expects it.
        if cache_path is not None:
            p = Path(cache_path)
            resolved_name = p.stem
            if resolved_name.endswith("_ohlcv"):
                resolved_name = resolved_name[:-6]
            mgr = CacheManager(
                cache_dir=str(p.parent),
                backup_dir=str(p.parent.parent / "backups"),
                downloader=yf.download,
            )
        else:
            resolved_name = cache_name if cache_name is not None else "prices"
            mgr = Pipeline._get_cache_manager(exec_cfg)

        ohlcv = mgr.fetch_ohlcv(tickers, start, end, resolved_name)

        if ohlcv is None or ohlcv.empty:
            logger.error(
                "CacheManager returned no data. "
                "*** FALLING BACK TO SYNTHETIC DATA ***"
            )
            return Pipeline._synthetic_prices(tickers, start, end)

        prices = mgr.extract_adj_close(ohlcv)

        missing_tickers = set(tickers) - set(prices.columns)
        if missing_tickers:
            logger.warning(
                f"{len(missing_tickers)} requested tickers entirely missing: "
                f"{sorted(missing_tickers)}"
            )

        return prices

    @staticmethod
    def prices_to_returns(
        prices: pd.DataFrame,
        min_coverage: float = 0.8,
    ) -> pd.DataFrame:
        """
        Convert price matrix to daily returns with robust missing data handling.

        Strategy
        --------
        1. Drop tickers with <min_coverage data (recently IPO'd, delisted, etc.)
        2. Forward-fill small gaps (foreign holiday misalignment)
        3. Compute percentage returns
        4. Fill remaining isolated NaNs with 0.0 (preferable to losing entire days)

        Parameters
        ----------
        prices : DataFrame
            (T x N) adjusted close prices.
        min_coverage : float
            Minimum fraction of non-NaN prices to keep a ticker. Default 0.8.

        Returns
        -------
        DataFrame
            (T-1 x N') returns. N' <= N if low-coverage tickers were dropped.
        """
        coverage = prices.notna().mean()
        good = coverage[coverage >= min_coverage].index

        if dropped := set(prices.columns) - set(good):
            logger.warning(
                f"Dropped {len(dropped)} tickers (<{min_coverage:.0%} coverage): "
                f"{sorted(dropped)}"
            )

        prices = prices[good].ffill()
        returns = prices.pct_change().iloc[1:].dropna(how="all")

        if n_filled := returns.isna().sum().sum():
            logger.debug(f"Filled {n_filled} isolated NaN returns with 0.0")
            returns = returns.fillna(0.0)

        return returns

    # =========================================================================
    # Live / Intraday Fetching (Uncached -- for execution scanning)
    # =========================================================================

    @staticmethod
    def fetch_live_returns(tickers: List[str], period: str = "5d") -> pd.Series:
        """
        Fetch recent market data and return the most recent day's returns.

        Uses a 5-day window to safely compute returns across weekends/holidays.

        Parameters
        ----------
        tickers : list of str
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
        logger.info(f"Fetching live closing data for {len(tickers)} tickers...")
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=False)

        if raw.empty:
            raise ValueError("yfinance returned no data")

        prices = raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

        if prices is None or prices.empty:
            raise ValueError("No price data in yfinance response")

        returns = prices.ffill().pct_change().dropna(how="all")

        if returns.empty:
            raise ValueError("Return computation produced empty result")

        return returns.iloc[-1]

    @staticmethod
    def fetch_intraday_returns(
        tickers: List[str],
        max_retries: int = 3,
    ) -> Optional[pd.Series]:
        """
        Fetch intraday returns (prior close to current) with retry logic.

        Parameters
        ----------
        tickers : list of str
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
                    progress=False,
                )

                if data.empty:
                    raise ValueError("Empty response")

                prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
                prices = prices.dropna(axis=1, how="all").ffill()

                yesterday_close = prices[prices.index.date < prices.index[-1].date()].iloc[-1]
                current = prices.iloc[-1]

                return (current - yesterday_close) / yesterday_close

            except Exception as e:
                wait = 5 * (attempt + 1)
                logger.warning(
                    f"Intraday fetch failed (attempt {attempt+1}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error("Intraday fetch failed after all retries.")
        return None

    # =========================================================================
    # Build & Persistence
    # =========================================================================

    @classmethod
    def build(
        cls,
        universe: UniverseConfig,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> FactorModel:
        """
        Full pipeline: fetch data -> compute returns -> fit model.

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
        """
        exec_cfg = cls._get_exec_config(exec_config)
        logger.info(f"Building model: {universe.name}")

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

        etf_returns = cls.prices_to_returns(etf_prices)
        stock_returns = cls.prices_to_returns(stock_prices)

        common = etf_returns.index.intersection(stock_returns.index)
        logger.info(f"Aligned on {len(common)} common trading days")

        model = FactorModel(etf_returns.loc[common], stock_returns.loc[common])
        model.fit(n_components=universe.n_components)
        return model

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
            Model name (e.g., "us", "global"). Used in the filename.
        exec_config : ExecutionConfig, optional
            Controls data directory.

        Returns
        -------
        str
            Path to the saved file.
        """
        exec_cfg = cls._get_exec_config(exec_config)
        path = exec_cfg.resolve_path(
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        payload = model.to_dict()
        payload["metadata"].update({
            "name": name,
            "created_at": datetime.now().isoformat(),
        })

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
            model = FactorModel.from_dict(json.load(f))
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
            Path to the latest model, or None if not found.
        """
        exec_cfg = cls._get_exec_config(exec_config)
        files = sorted(glob.glob(exec_cfg.resolve_path(f"{name}_*.json")))

        if not files:
            logger.warning(f"No saved models found matching: {name}_*.json")
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
        Export human-readable CSVs of the macro factor basis and stock loadings.

        Returns
        -------
        tuple of str
            (etf_basis_path, stock_loadings_path)
        """
        exec_cfg = cls._get_exec_config(exec_config)

        # ETF basis vectors
        basis_path = exec_cfg.resolve_path(f"{name}_etf_basis.csv")
        model.etf_basis.sort_index().to_csv(basis_path, float_format="%.6f")

        # Stock loadings with confidence diagnostics
        loadings_path = exec_cfg.resolve_path(f"{name}_stock_loadings.csv")
        output = model.loadings.copy()

        if model.diagnostics:
            output["R_squared"] = model.diagnostics.stock_r_squared
            output["Residual_Vol"] = model.diagnostics.residual_vol
            output = output.sort_values("R_squared", ascending=False)

        output.to_csv(loadings_path, float_format="%.6f")

        logger.info(f"Exported CSVs: {basis_path}, {loadings_path}")
        return basis_path, loadings_path

    # =========================================================================
    # Failsafe / Testing
    # =========================================================================

    @staticmethod
    def _synthetic_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Generate synthetic prices mimicking crisis dynamics.

        *** FOR TESTING / FALLBACK ONLY. Not real market data. ***

        Stylized facts embedded (Russia-Ukraine Feb-Mar 2022):
          - Broad equity: down ~5%
          - Energy: up ~15%
          - Gold: up ~6%
          - Defense: up ~8%
        """
        rng = np.random.RandomState(2022)
        dates = pd.bdate_range(start=start, end=end)
        T = len(dates)

        drift_map = {
            "VOO": -0.0014, "VYM": -0.0010, "VEU": -0.0020,
            "VDE": +0.0043, "TLT": +0.0006, "GLD": +0.0017,
            "USO": +0.0043, "ITA": +0.0023,
            "LMT": +0.0025, "RTX": +0.0020, "NOC": +0.0022,
            "CVX": +0.0025, "MSFT": -0.0020, "V": -0.0010,
        }
        market_factor = rng.normal(0, 0.010, T)

        prices = {}
        for t in tickers:
            beta = -0.3 if t in ("TLT", "GLD", "USO") else 0.7
            returns = beta * market_factor + rng.normal(drift_map.get(t, -0.0014), 0.012, T)
            prices[t] = 100 * np.cumprod(1 + returns)  # Start at 100 = synthetic flag

        logger.error(
            f"SYNTHETIC DATA: {T} days x {len(tickers)} tickers. "
            "All prices start at 100. Delete cache and retry when network available."
        )

        return pd.DataFrame(prices, index=dates)


# =============================================================================
# Module-level aliases for backwards compatibility
# (cli.py and any external scripts that import these directly)
# =============================================================================

fetch_live_returns = Pipeline.fetch_live_returns
fetch_intraday_returns = Pipeline.fetch_intraday_returns
