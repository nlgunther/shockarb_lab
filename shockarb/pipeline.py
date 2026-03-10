"""
Data pipeline — all I/O so the engine stays pure.

Functions
---------
  fetch_prices            Download + cache adjusted close prices via CacheManager.
  prices_to_returns       Convert prices to returns with robust NaN handling.
  fetch_live_returns      Latest closing-day returns via coordinator (cached),
                          or direct yfinance during market hours (hot fix).
  fetch_intraday_returns  Intraday returns since prior close (uncached, with retry).
  save_live_tape          15-minute OHLCV tape for ETFs + stocks, 3 rows, parquet.
  save_intraday_tape      1-minute tape, 3 vital rows (or full), parquet.
  build                   Full pipeline: register DataRequests → fulfill → fit FactorModel.
  save_model              Persist a fitted model as JSON.
  load_model              Load a saved model from JSON.
  find_latest_model       Find the most-recently saved model by name prefix.
  export_csvs             Write ETF basis and stock loadings to CSV.

Design principle
----------------
All filesystem interaction, API calls, and caching live here.
The engine (FactorModel) is pure: no I/O, no filesystem, no yfinance.
Swap out the data source (e.g., Bloomberg instead of yfinance) by
modifying this module only — the engine is untouched.

datamgr integration
-------------------
build() and fetch_live_returns() use the datamgr coordinator rather than
CacheManager directly.  The coordinator entry point is _coordinator(), which
is the single point where shockarb hands off to datamgr::

    pipeline._coordinator()
      → ParquetStore(ShockArbStore(exec_cfg.data_dir))
      → DataCoordinator(store, provider=YFinanceProvider(...))

Tests patch _coordinator() via patch.object(pipeline, '_coordinator', ...)
to inject an InMemoryStore + MockProvider without touching the filesystem.
Use side_effect= (not return_value=) when patching so that each build()
call receives a fresh coordinator instance with an empty request registry.

Intraday hot fix
----------------
fetch_live_returns() bypasses the coordinator during NYSE market hours.
When _market_is_open() returns True (or force_live=True is passed),
_fetch_live_direct() is called instead, fetching 5-minute bars from
yfinance directly and computing returns against the prior day's close.

This is a temporary fix tracked on branch intraday_hotfix.  The proper
fix (Phases 1-3) moves this logic into YFinanceProvider.fetch() with full
coordinator gap-analysis and intraday store caching.

Example
-------
    import shockarb.pipeline as pipeline
    from shockarb.config import US_UNIVERSE

    model = pipeline.build(US_UNIVERSE)
    pipeline.save_model(model, "us")

    live_etfs   = pipeline.fetch_live_returns(list(model.etf_returns.columns))
    live_stocks = pipeline.fetch_live_returns(list(model.stock_returns.columns))
    scores = model.score(live_etfs, live_stocks)
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from shockarb.cache import CacheManager
from shockarb.config import ExecutionConfig, UniverseConfig
from shockarb.engine import FactorModel

from datamgr.coordinator import DataCoordinator
from datamgr.providers.yfinance import YFinanceProvider
from datamgr.requests import DataRequest, Frequency
from datamgr.stores.parquet import ParquetStore

# =============================================================================
# Internal helpers
# =============================================================================

def _default_exec() -> ExecutionConfig:
    """Return a default ExecutionConfig with logger configured."""
    cfg = ExecutionConfig()
    cfg.configure_logger()
    return cfg


def _exec(exec_config: Optional[ExecutionConfig]) -> ExecutionConfig:
    """
    Return *exec_config* if provided (and configure its logger), else a default.

    All public functions accept an optional ExecutionConfig so callers can
    control data_dir and log_level without a global.  This helper collapses
    the None check to one line per function.
    """
    if exec_config is not None:
        exec_config.configure_logger()
        return exec_config
    return _default_exec()


def _cache_manager(exec_cfg: ExecutionConfig) -> CacheManager:
    """
    Build a CacheManager rooted in exec_cfg's data directory.

    Used only by fetch_prices() (the legacy CacheManager path).  New callers
    should use the coordinator path via build() instead.

    The downloader= parameter is passed as the module-level yf.download symbol
    so that test patches on shockarb.pipeline.yf.download are intercepted.
    """
    return CacheManager(
        cache_dir=os.path.join(exec_cfg.data_dir, "cache"),
        backup_dir=os.path.join(exec_cfg.data_dir, "backups"),
        downloader=yf.download,
    )


# =============================================================================
# datamgr coordinator
# =============================================================================

def _coordinator(exec_cfg: ExecutionConfig) -> DataCoordinator:
    """
    Build a fresh DataCoordinator wired to ParquetStore + YFinanceProvider.

    This is the single point where shockarb hands off to datamgr.  All
    coordinator construction is centralised here so tests can patch a single
    symbol to inject test doubles::

        with patch.object(pipeline, '_coordinator', side_effect=_fresh_coord):
            result = pipeline.build(US_UNIVERSE)

    Use side_effect= (not return_value=) so that each call to build() or
    fetch_live_returns() gets a fresh coordinator instance with an empty
    request registry.  Reusing a single instance across calls causes
    previously registered requests to be re-fulfilled on every call.

    Parameters
    ----------
    exec_cfg : ExecutionConfig
        Provides data_dir for ParquetStore path resolution.

    Returns
    -------
    DataCoordinator
        Wired to disk (ParquetStore) and yfinance (YFinanceProvider).
    """
    from shockarb.store import DataStore as ShockArbStore
    inner    = ShockArbStore(exec_cfg.data_dir)
    store    = ParquetStore(inner)
    provider = YFinanceProvider(downloader=yf.download)
    return DataCoordinator(store, provider=provider)


# =============================================================================
# Historical data (cached)
# =============================================================================

def fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    cache_name: str = "prices",
    exec_config: Optional[ExecutionConfig] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices, using a local parquet cache.

    Legacy CacheManager path — retained for backward compatibility.  New
    callers should use build() which uses the coordinator + ParquetStore.

    Only hits the network for tickers or date ranges not already cached.
    A second call with identical arguments costs zero API calls.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date YYYY-MM-DD, inclusive.
    end : str
        End date YYYY-MM-DD, exclusive (yfinance convention).
    cache_name : str
        Logical cache key (e.g. "us_etf").  Each unique name gets its own
        parquet file in data/cache/.  Keep ETF and stock universes separate
        to avoid key collisions.
    exec_config : ExecutionConfig, optional
        Controls data directory and logging.

    Returns
    -------
    pd.DataFrame
        (dates × tickers) adjusted close prices.
        Falls back to synthetic crisis prices if yfinance fails entirely —
        logged at ERROR level.  Synthetic prices all start at 100 and are
        detectable programmatically.
    """
    exec_cfg = _exec(exec_config)
    mgr = _cache_manager(exec_cfg)
    ohlcv = mgr.fetch_ohlcv(tickers, start, end, cache_name)

    if ohlcv is None or ohlcv.empty:
        logger.error(
            "CacheManager returned no data. "
            "*** FALLING BACK TO SYNTHETIC DATA — NOT REAL PRICES ***"
        )
        return _synthetic_prices(tickers, start, end)

    prices = mgr.extract_adj_close(ohlcv)

    missing = set(tickers) - set(prices.columns)
    if missing:
        logger.warning(
            f"{len(missing)} tickers entirely missing from download: {sorted(missing)}"
        )

    return prices


def prices_to_returns(
    prices: pd.DataFrame,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """
    Convert a price matrix to daily returns with robust NaN handling.

    Three-step strategy:
      1. Drop tickers with < min_coverage non-NaN rows (delisted, recent IPOs).
      2. Forward-fill remaining gaps (foreign holiday misalignment).
      3. Fill isolated post-pct_change NaNs with 0.0 (beats losing entire days).

    Parameters
    ----------
    prices : pd.DataFrame
        (T × N) adjusted close prices.
    min_coverage : float
        Minimum fraction of non-NaN prices to retain a ticker.  Default 0.8.
        Tickers below this threshold are dropped and logged at WARNING level.

    Returns
    -------
    pd.DataFrame
        (T−1 × N') daily returns.  N' ≤ N if low-coverage tickers were dropped.
    """
    coverage = prices.notna().mean()
    good = coverage[coverage >= min_coverage].index

    if dropped := set(prices.columns) - set(good):
        logger.warning(
            f"Dropped {len(dropped)} tickers (<{min_coverage:.0%} coverage): "
            f"{sorted(dropped)}"
        )

    returns = prices[good].ffill().pct_change().iloc[1:].dropna(how="all")

    if n_filled := returns.isna().sum().sum():
        logger.debug(f"Filled {n_filled} isolated NaN returns with 0.0")
        returns = returns.fillna(0.0)

    return returns


# =============================================================================
# Live / intraday data (uncached)
# =============================================================================

# HOT_FIX: bypasses coordinator during market hours — replace with
# Phase 1-3 intraday coordinator path before merging to main.

def _market_is_open() -> bool:
    """
    Return True if NYSE is currently in its regular trading session.

    Uses America/New_York timezone.  Returns False on weekends regardless
    of time.  Regular session: 09:30 – 16:00 ET inclusive.

    Used by fetch_live_returns() to decide whether to call _fetch_live_direct()
    (intraday hot fix path) or the coordinator cache path (after close).
    """
    import pytz
    from datetime import datetime
    now = datetime.now(pytz.timezone("America/New_York"))
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def _fetch_live_direct(tickers: List[str]) -> pd.Series:
    """
    Fetch the most recent intraday return directly from yfinance.

    Bypasses the coordinator and ParquetStore entirely.  Downloads a 5-day
    window of 5-minute bars, then computes the return from the prior day's
    last bar to the latest available bar today.

    HOT_FIX — this logic moves into YFinanceProvider.fetch() in the proper
    intraday fix (Phases 1-3 on branch intraday_hotfix):
      Phase 1 — YFinanceProvider handles INTRADAY_15M with interval="5m"
      Phase 2 — ParquetStore intraday read/write with append-dedup
      Phase 3 — fetch_live_returns registers INTRADAY_15M + DAILY requests

    Parameters
    ----------
    tickers : list of str
        Tickers to fetch.

    Returns
    -------
    pd.Series
        Return from prior day's close to latest 5-minute bar, indexed by ticker.

    Raises
    ------
    ValueError
        If yfinance returns no data, or if there is insufficient history
        to compute a return (fewer than two distinct trading dates in the
        5-day window).
    """
    import yfinance as yf
    import pytz
    from datetime import datetime

    now = datetime.now(pytz.timezone("America/New_York"))
    logger.info(
        f"Market is open — fetching intraday 5m bars for {len(tickers)} tickers "
        f"(as of {now.strftime('%H:%M:%S')} ET)…"
    )

    raw = yf.download(tickers, period="5d", interval="5m",
                      progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError("fetch_live_returns: yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
    else:
        prices = raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(axis=1, how="all").ffill()

    dates = prices.index.normalize().unique()
    if len(dates) < 2:
        raise ValueError("fetch_live_returns: not enough history for return computation.")

    prev_date  = dates[-2]
    today_date = dates[-1]
    prev_close = prices[prices.index.normalize() == prev_date].iloc[-1]
    latest     = prices[prices.index.normalize() == today_date].iloc[-1]
    latest_ts  = prices[prices.index.normalize() == today_date].index[-1]

    returns = (latest / prev_close) - 1

    logger.info(
        f"Intraday returns computed: "
        f"prev close {prev_date.strftime('%Y-%m-%d')}, "
        f"latest bar {latest_ts.strftime('%H:%M')} ET — "
        f"{len(returns)} tickers"
    )
    skipped = returns[returns.isna()].index.tolist()
    if skipped:
        logger.warning(f"{len(skipped)} ticker(s) had no intraday data: {skipped}")
        returns = returns.dropna()

    return returns


def fetch_live_returns(
    tickers: List[str],
    period: str = "5d",
    exec_config: Optional[ExecutionConfig] = None,
    force_live: bool = False,
) -> pd.Series:
    """
    Fetch the most recent day's closing returns.

    Two paths depending on market hours:

    During market hours (or force_live=True)
        Calls _fetch_live_direct() — bypasses the coordinator and fetches
        5-minute bars from yfinance to reflect intraday prices.  No caching.
        HOT_FIX: this path is temporary; see _fetch_live_direct() docstring.

    After market close
        Uses the datamgr coordinator with a 5-day daily lookback.  Results
        are cached by the coordinator — calling this multiple times on the
        same day does not trigger redundant downloads.

    Test patching
    -------------
    To test the empty-response error path without a live yfinance call::

        with patch.object(pipeline, "_market_is_open", return_value=True), \\
             patch.object(pipeline, "_fetch_live_direct",
                          side_effect=ValueError("fetch_live_returns: coordinator returned no data.")):
            with pytest.raises(ValueError, match="no data"):
                pipeline.fetch_live_returns(["AAPL"])

    Parameters
    ----------
    tickers : list of str
        Tickers to fetch returns for.
    period : str
        Look-back window as a yfinance period string.  Default "5d" — enough
        to bridge weekends and single-day holidays.  Supported values: "1d",
        "5d", "1mo", "3mo".
    exec_config : ExecutionConfig, optional
        Controls data directory and logging.  Uses a process-wide default
        if omitted.
    force_live : bool
        Force the intraday hot-fix path regardless of market hours.
        Useful for manual testing of the intraday return computation.

    Returns
    -------
    pd.Series
        Most recent day's returns, indexed by ticker.

    Raises
    ------
    ValueError
        If no data can be retrieved at all, if all tickers return empty
        data, or if the return computation produces an empty result.
    """
    from datetime import date, timedelta
    exec_cfg = _exec(exec_config)
    logger.info(f"Fetching live closing data for {len(tickers)} tickers…")

    # HOT_FIX: bypass coordinator during market hours
    if force_live or _market_is_open():
        return _fetch_live_direct(tickers)

    # After close — use coordinator cache
    period_days = {"1d": 4, "5d": 7, "1mo": 35, "3mo": 95}
    lookback    = period_days.get(period, 7)
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")

    coordinator = _coordinator(exec_cfg)
    coordinator.register(DataRequest(
        tickers   = tuple(tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "live_returns",
    ))
    results = coordinator.fulfill()
    prices  = results.get("live_returns", pd.DataFrame())

    if prices is None or prices.empty:
        raise ValueError("fetch_live_returns: coordinator returned no data.")

    bad = prices.columns[prices.isna().all()].tolist()
    if bad:
        logger.warning(
            f"{len(bad)} ticker(s) returned no data and will be skipped: {bad}. "
            "Check for delistings or ticker changes."
        )
        prices = prices.drop(columns=bad)

    if prices.empty:
        raise ValueError("fetch_live_returns: all tickers returned empty data.")

    returns = prices.ffill().pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("fetch_live_returns: return computation produced empty result")

    return returns.iloc[-1]


def save_live_tape(
    etf_tickers: List[str],
    stock_tickers: List[str],
    path: str,
) -> Optional[pd.DataFrame]:
    """
    Download 15-minute OHLCV for ETFs and stocks combined and save as parquet.

    Fetches a 2-day, 15-minute-interval window then slices to exactly three
    structurally vital rows — the minimum data needed to compute both the
    prior-close return and the intraday move:

      Row 0 — yesterday's last 15-minute bar  (prior close reference)
      Row 1 — today's first 15-minute bar     (opening print)
      Row 2 — today's most recent 15-minute bar (last print; equals today's
               closing bar if called after the bell)

    The full MultiIndex column structure from yfinance (``auto_adjust=False``)
    is preserved: ``(Adj Close, ticker)``, ``(Close, ticker)``,
    ``(High, ticker)``, ``(Low, ticker)``, ``(Open, ticker)``,
    ``(Volume, ticker)``.

    Parameters
    ----------
    etf_tickers : list of str
    stock_tickers : list of str
    path : str
        Destination parquet path.  Parent directory is created if needed.

    Returns
    -------
    pd.DataFrame or None
        The saved 3-row tape, or None on failure.

    Notes
    -----
    yfinance returns at most 60 days of 15-minute history.  The ``period="2d"``
    window is enough to guarantee yesterday's bars are present regardless of
    weekends or single-day holidays.

    To reload and inspect::

        df = pd.read_parquet("data/tapes/us_20240315.parquet")
        print(df.index)                                   # 3 timestamps
        print(df.columns.get_level_values(0).unique())    # OHLCV fields
        print(df.columns.get_level_values(1).unique())    # tickers
    """
    all_tickers = etf_tickers + stock_tickers
    logger.info(
        f"Fetching 15m OHLCV tape for {len(etf_tickers)} ETFs + "
        f"{len(stock_tickers)} stocks…"
    )

    try:
        raw = yf.download(
            all_tickers,
            period="2d",
            interval="15m",
            progress=False,
            auto_adjust=False,
        )
        if raw.empty:
            logger.error("save_live_tape: yfinance returned empty data")
            return None

        # Ensure MultiIndex columns even for a single ticker
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_tuples(
                [(col, all_tickers[0]) for col in raw.columns]
            )

        tape = _minimal_tape(raw)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tape.to_parquet(path)
        logger.info(
            f"Tape saved → {os.path.basename(path)} "
            f"({len(tape)} rows × {tape.shape[1] // len(raw.columns.get_level_values(0).unique())} tickers)"
        )
        return tape

    except Exception as exc:
        logger.error(f"save_live_tape failed: {exc}")
        return None


def fetch_intraday_returns(
    tickers: List[str],
    max_retries: int = 3,
    playback_path: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Fetch intraday returns (prior close to current minute) with retry logic.

    Parameters
    ----------
    tickers : list of str
    max_retries : int
        Retry attempts on transient yfinance failures.  Default 3.
        Each retry waits 5 * attempt_number seconds before retrying.
    playback_path : str, optional
        Path to a previously saved parquet tape (from save_intraday_tape()).
        When provided, returns are computed from the file rather than live
        data.  Intended for offline replay and debugging.  If the file does
        not exist, falls through to the live fetch path.

    Returns
    -------
    pd.Series or None
        Intraday return from prior close, or None after all retries fail.
    """
    import time

    # --- Offline playback ---
    if playback_path and os.path.exists(playback_path):
        logger.info(f"Playback mode: loading tape from {os.path.basename(playback_path)}")
        try:
            data = pd.read_parquet(playback_path)
            return _intraday_return_from_tape(data)
        except Exception as exc:
            logger.error(f"Failed to load playback file: {exc}")
            return None

    # --- Live fetch with retry ---
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers, period="2d", interval="1m", progress=False, auto_adjust=False
            )
            if data.empty:
                raise ValueError("Empty response from yfinance")
            return _intraday_return_from_tape(data)

        except Exception as exc:
            wait = 5 * (attempt + 1)
            logger.warning(
                f"Intraday fetch failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {wait}s…"
            )
            time.sleep(wait)

    logger.error("fetch_intraday_returns: all retries exhausted.")
    return None


def save_intraday_tape(
    tickers: List[str],
    path: str,
    full_tape: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Download a 1-minute tape and persist it for later playback.

    By default stores only the three structurally vital rows (yesterday's
    last close, today's open, today's current print) to keep files small.
    Pass full_tape=True to store all ~600 rows.

    Parameters
    ----------
    tickers : list of str
    path : str
        Destination parquet path.  Directory is created if needed.
    full_tape : bool
        If True, store the complete 2-day 1-minute tape rather than the
        3-row minimal slice.

    Returns
    -------
    pd.DataFrame or None
        The saved tape, or None on failure.
    """
    try:
        data = yf.download(
            tickers, period="2d", interval="1m", progress=False, auto_adjust=False
        )
        if data.empty:
            logger.error("save_intraday_tape: yfinance returned empty data")
            return None

        tape = data if full_tape else _minimal_tape(data)
        tape.to_parquet(path)
        logger.debug(f"Saved {'full' if full_tape else 'minimal'} tape → {os.path.basename(path)}")
        return tape

    except Exception as exc:
        logger.error(f"save_intraday_tape failed: {exc}")
        return None


def _intraday_return_from_tape(data: pd.DataFrame) -> pd.Series:
    """
    Compute (current − prior_close) / prior_close from a 1-minute tape.

    Parameters
    ----------
    data : pd.DataFrame
        Raw yfinance 1-minute tape.  MultiIndex columns if multi-ticker.

    Returns
    -------
    pd.Series
        Return from yesterday's close to current bar, indexed by ticker.
    """
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices = prices.dropna(axis=1, how="all").ffill()

    yesterday_close = prices[prices.index.date < prices.index[-1].date()].iloc[-1]
    current = prices.iloc[-1]
    return (current - yesterday_close) / yesterday_close


def _minimal_tape(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a full intraday tape to the three structurally vital rows.

    Rows selected:
      - Yesterday's last bar  (prior close reference)
      - Today's first bar     (opening print)
      - Today's current bar   (latest print)

    These three rows are sufficient to compute the prior-close return and
    the intraday move for any ticker.  Duplicate indices (e.g. if current ==
    first bar at open) are de-duplicated.

    Parameters
    ----------
    data : pd.DataFrame
        Full intraday tape from yfinance (1-minute or 15-minute bars).

    Returns
    -------
    pd.DataFrame
        2–3 row DataFrame.
    """
    tape = data.ffill()
    dates = tape.index.normalize().unique()
    if len(dates) < 2:
        return tape.iloc[[0, -1]]

    yesterday, today = dates[-2], dates[-1]
    y_close = tape[tape.index.normalize() == yesterday].index[-1]
    t_open  = tape[tape.index.normalize() == today].index[0]
    current = tape.index[-1]

    return tape.loc[pd.Index([y_close, t_open, current]).unique()]


# =============================================================================
# Model lifecycle
# =============================================================================

def build(
    universe: UniverseConfig,
    exec_config: Optional[ExecutionConfig] = None,
) -> FactorModel:
    """
    Full pipeline: register DataRequests → coordinator.fulfill() → fit model.

    Registers two DataRequests (ETFs and stocks) with the coordinator, fulfills
    them (gap-analysed downloads + cache), converts prices to returns, aligns
    the two return matrices on common trading days, and fits the FactorModel.

    Stale manifest warning
    ----------------------
    If build() logs "Aligned on 0 common trading days" despite a successful
    data fetch, the manifest.json is likely stale — its date_range entries
    cover the wrong window so coverage() reports false cache hits.
    Fix: delete data/manifest.json and run build again.

    Parameters
    ----------
    universe : UniverseConfig
        Defines what to analyze: tickers, date window, n_components.
    exec_config : ExecutionConfig, optional
        Controls caching, paths, and logging.  Uses a process-wide default
        if omitted.

    Returns
    -------
    FactorModel
        Fitted and ready for score().
    """
    exec_cfg = _exec(exec_config)
    logger.info(f"Building model: {universe.name}")

    coordinator = _coordinator(exec_cfg)
    coordinator.register(DataRequest(
        tickers   = tuple(universe.market_etfs),
        start     = universe.start_date,
        end       = universe.end_date,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = f"{universe.name}.etf",
    ))
    coordinator.register(DataRequest(
        tickers   = tuple(universe.individual_stocks),
        start     = universe.start_date,
        end       = universe.end_date,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = f"{universe.name}.stock",
    ))
    results       = coordinator.fulfill()
    etf_returns   = prices_to_returns(results.get(f"{universe.name}.etf",   pd.DataFrame()))
    stock_returns = prices_to_returns(results.get(f"{universe.name}.stock", pd.DataFrame()))
    common = etf_returns.index.intersection(stock_returns.index)
    logger.info(f"Aligned on {len(common)} common trading days")

    model = FactorModel(etf_returns.loc[common], stock_returns.loc[common])
    model.fit(n_components=universe.n_components)
    return model


def save_model(
    model: FactorModel,
    name: str,
    exec_config: Optional[ExecutionConfig] = None,
) -> str:
    """
    Persist a fitted model to JSON.

    Timestamps the filename so multiple saves do not overwrite each other.
    Use find_latest_model() to retrieve the most recent file by name prefix.

    Parameters
    ----------
    model : FactorModel
        A fitted model (model.fit() must have been called).
    name : str
        Short identifier, e.g. "us" or "global".  Used as the filename prefix
        and stored in the JSON metadata.
    exec_config : ExecutionConfig, optional

    Returns
    -------
    str
        Absolute path to the saved JSON file.
    """
    exec_cfg = _exec(exec_config)
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


def load_model(path: str) -> FactorModel:
    """
    Load a previously saved model from JSON.

    Parameters
    ----------
    path : str
        Path to a .json file produced by save_model().

    Returns
    -------
    FactorModel
        Ready to call score() immediately.
    """
    with open(path) as f:
        model = FactorModel.from_dict(json.load(f))
    logger.success(f"Model loaded: {path}")
    return model


def find_latest_model(
    name: str,
    exec_config: Optional[ExecutionConfig] = None,
) -> Optional[str]:
    """
    Find the most recently saved model file for *name*.

    Globs data_dir for {name}_*.json and returns the lexicographically last
    match.  Because filenames embed a timestamp (YYYYMMDD_HHMMSS), the last
    file alphabetically is always the most recent.

    Parameters
    ----------
    name : str
        Model name prefix, e.g. "us".
    exec_config : ExecutionConfig, optional

    Returns
    -------
    str or None
        Path to the latest file, or None if none found.
    """
    exec_cfg = _exec(exec_config)
    files = sorted(glob.glob(exec_cfg.resolve_path(f"{name}_*.json")))

    if not files:
        logger.warning(f"No saved models found matching: {name}_*.json")
        return None

    return files[-1]


def export_csvs(
    model: FactorModel,
    name: str,
    exec_config: Optional[ExecutionConfig] = None,
) -> Tuple[str, str]:
    """
    Write ETF factor basis and stock loadings to human-readable CSV files.

    If diagnostics are available (always the case after build()), stock
    loadings are augmented with R² and residual volatility columns and sorted
    by R² descending so the most trustworthy factor signals are at the top.

    Parameters
    ----------
    model : FactorModel
        Fitted model with diagnostics populated.
    name : str
        Prefix for output filenames, e.g. "us" → "us_etf_basis.csv",
        "us_stock_loadings.csv".
    exec_config : ExecutionConfig, optional

    Returns
    -------
    tuple of (str, str)
        (etf_basis_path, stock_loadings_path) — absolute paths to the
        two CSV files written.
    """
    exec_cfg = _exec(exec_config)

    basis_path = exec_cfg.resolve_path(f"{name}_etf_basis.csv")
    model.etf_basis.sort_index().to_csv(basis_path, float_format="%.6f")

    loadings_path = exec_cfg.resolve_path(f"{name}_stock_loadings.csv")
    output = model.loadings.copy()

    if model.diagnostics:
        output["R_squared"]   = model.diagnostics.stock_r_squared
        output["Residual_Vol"] = model.diagnostics.residual_vol
        output = output.sort_values("R_squared", ascending=False)

    output.to_csv(loadings_path, float_format="%.6f")

    logger.info(f"Exported CSVs → {basis_path}, {loadings_path}")
    return basis_path, loadings_path


# =============================================================================
# Synthetic fallback (testing / network outage)
# =============================================================================

# Approximate daily drift rates observed during Russia-Ukraine Feb–Mar 2022.
# Tickers not listed here default to a mild negative drift (-0.14% / day).
_CRISIS_DRIFT = {
    "VOO": -0.0014, "VYM": -0.0010, "VEU": -0.0020,
    "VDE": +0.0043, "TLT": +0.0006, "GLD": +0.0017,
    "USO": +0.0043, "ITA": +0.0023,
    "LMT": +0.0025, "RTX": +0.0020, "NOC": +0.0022,
    "CVX": +0.0025, "MSFT": -0.0020, "V": -0.0010,
}

# Tickers with negative market beta (safe-haven assets)
_SAFE_HAVENS = {"TLT", "GLD", "USO"}


def _synthetic_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Generate reproducible synthetic prices that approximate crisis dynamics.

    FOR TESTING AND NETWORK-OUTAGE FALLBACK ONLY.

    All series start at 100 — a deliberate flag so callers can detect
    synthetic data programmatically (real prices very rarely all start at 100).

    Stylised facts embedded (Russia-Ukraine, Feb–Mar 2022):
      - Broad equity:   −5%
      - Energy / defence: +10–15%
      - Gold / bonds:   +5%

    Parameters
    ----------
    tickers : list of str
    start, end : str
        YYYY-MM-DD range for the synthetic index.

    Returns
    -------
    pd.DataFrame
        (dates × tickers) prices anchored at 100.
    """
    logger.error(
        f"SYNTHETIC DATA: {start} → {end}, {len(tickers)} tickers. "
        "All prices anchored at 100. Check network / cache before using results."
    )

    rng = np.random.RandomState(2022)
    dates = pd.bdate_range(start=start, end=end)
    T = len(dates)
    market = rng.normal(0, 0.010, T)

    prices = {}
    for t in tickers:
        beta = -0.3 if t in _SAFE_HAVENS else 0.7
        rets = beta * market + rng.normal(_CRISIS_DRIFT.get(t, -0.0014), 0.012, T)
        prices[t] = 100 * np.cumprod(1 + rets)

    return pd.DataFrame(prices, index=dates)


def fetch_live_prices(
    tickers: List[str],
    period: str = "5d",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch recent OHLCV prices and optionally save them as parquet.

    Returns the raw multi-level OHLCV DataFrame from yfinance — one row per
    trading day, columns are a MultiIndex of (field, ticker).  This is the
    data before pct_change, suitable for auditing, replaying, or feeding
    back into prices_to_returns().

    Parameters
    ----------
    tickers : list of str
    period : str
        yfinance period string.  Default "5d" (enough to compute one return).
    save_path : str, optional
        If provided, the OHLCV DataFrame is written to this path as parquet
        before being returned.  The directory is created if it does not exist.
        File naming convention used by the CLI::

            {data_dir}/prices_{name}_{YYYYMMDD_HHMMSS}.parquet

    Returns
    -------
    pd.DataFrame
        Raw OHLCV MultiIndex DataFrame (dates × (field, ticker)).

    Raises
    ------
    ValueError
        If yfinance returns no data.

    Example
    -------
        ohlcv   = fetch_live_prices(["VOO", "TLT"], save_path="data/prices_etf.parquet")
        prices  = CacheManager.extract_adj_close(ohlcv)
        returns = prices_to_returns(prices)
    """
    logger.info(f"Fetching live OHLCV for {len(tickers)} tickers (period={period})…")
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError("yfinance returned no data for fetch_live_prices")

    # Normalise to MultiIndex even for a single ticker
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, tickers[0]) for col in raw.columns]
        )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        raw.to_parquet(save_path)
        logger.info(f"Prices saved → {os.path.basename(save_path)}")

    return raw
