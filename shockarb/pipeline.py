"""
Data pipeline — all I/O so the engine stays pure.

Functions
---------
  fetch_prices        Download + cache adjusted close prices via CacheManager.
  prices_to_returns   Convert prices to returns with robust NaN handling.
  fetch_live_returns  Latest closing-day returns (uncached, for scoring).
  score_universe      Score a universe via a single shared coordinator instance.
                      Primary entry point for the score CLI and daily scanner.
  fetch_intraday_returns  Intraday returns since prior close (uncached, with retry).
  save_live_tape      Daily OHLCV for ETFs + stocks combined, sliced to 2 rows, saved as parquet.
  save_intraday_tape  1-minute tape sliced to 3 vital rows, saved as parquet.
  build               Full pipeline: fetch → returns → fit FactorModel.
  save_model          Persist a fitted model as JSON.
  load_model          Load a saved model from JSON.
  find_latest_model   Find the most-recently saved model by name.
  export_csvs         Write ETF basis and stock loadings to CSV.

Design principle
----------------
All filesystem interaction, API calls, and caching live here.
Swap out the data source (e.g., Bloomberg instead of yfinance) by
modifying this module only — the engine is untouched.

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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
# ScoreProvenance — attached to every scored output
# =============================================================================

@dataclass
class ScoreProvenance:
    """
    Full provenance record for a score_universe() run.

    Every scored output (CSV, CLI table, markdown report) should include
    this metadata so the operator can reconstruct the computation from
    first principles.

    Attributes
    ----------
    timestamp_utc : str
        ISO timestamp (UTC) when scoring completed.
    timestamp_et : str
        ISO timestamp (America/New_York) when scoring completed.
    universe : str
        Universe name, e.g. "us" or "global".
    model_file : str
        Not set by score_universe (caller sets this). Path to model JSON.
    path : str
        "daily" or "intraday" — which code path was taken.
    provider : str
        Data provider used, e.g. "yfinance".
    n_etfs : int
        Number of ETF tickers scored.
    n_stocks : int
        Number of stock tickers scored.
    return_formula : str
        Human-readable formula, e.g.
        "adj_close @ 2026-03-12 15:45 ET / adj_close @ 2026-03-11 (daily cache) - 1"
    numerator_field : str
        Field used as numerator, e.g. "adj_close".
    numerator_timestamp : str
        Timestamp of the numerator observation (latest bar or daily close).
    denominator_field : str
        Field used as denominator, e.g. "adj_close".
    denominator_timestamp : str
        Timestamp of the denominator observation (prev close date).
    interval : str
        Fetch interval, e.g. "15m" or "1d".
    fetch_period : str
        yfinance period string used, e.g. "1d" or "5d".
    n_intraday_bars : int
        Number of intraday bars returned (0 for daily path).
    sample_tickers : Dict[str, Dict[str, Any]]
        A few example tickers with their raw numerator, denominator, and
        computed return — so the operator can hand-verify.
    """
    timestamp_utc:          str  = ""
    timestamp_et:           str  = ""
    universe:               str  = ""
    model_file:             str  = ""
    path:                   str  = ""
    provider:               str  = "yfinance"
    n_etfs:                 int  = 0
    n_stocks:               int  = 0
    return_formula:         str  = ""
    numerator_field:        str  = ""
    numerator_timestamp:    str  = ""
    denominator_field:      str  = ""
    denominator_timestamp:  str  = ""
    interval:               str  = ""
    fetch_period:           str  = ""
    n_intraday_bars:        int  = 0
    sample_tickers:         Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON / CSV header embedding."""
        return {
            "timestamp_utc":         self.timestamp_utc,
            "timestamp_et":          self.timestamp_et,
            "universe":              self.universe,
            "model_file":            self.model_file,
            "path":                  self.path,
            "provider":              self.provider,
            "n_etfs":                self.n_etfs,
            "n_stocks":              self.n_stocks,
            "return_formula":        self.return_formula,
            "numerator_field":       self.numerator_field,
            "numerator_timestamp":   self.numerator_timestamp,
            "denominator_field":     self.denominator_field,
            "denominator_timestamp": self.denominator_timestamp,
            "interval":              self.interval,
            "fetch_period":          self.fetch_period,
            "n_intraday_bars":       self.n_intraday_bars,
            "sample_tickers":        self.sample_tickers,
        }

    def summary(self) -> str:
        """Human-readable summary for CLI / log output."""
        lines = [
            f"  Provenance",
            f"  ──────────────────────────────────────────────",
            f"  Scored at:    {self.timestamp_et}",
            f"  Universe:     {self.universe}",
            f"  Path:         {self.path}",
            f"  Provider:     {self.provider}",
            f"  Tickers:      {self.n_etfs} ETFs, {self.n_stocks} stocks",
            f"  Interval:     {self.interval}",
            f"  Formula:      {self.return_formula}",
            f"  Numerator:    {self.numerator_field} @ {self.numerator_timestamp}",
            f"  Denominator:  {self.denominator_field} @ {self.denominator_timestamp}",
        ]
        if self.n_intraday_bars:
            lines.append(f"  Intraday bars: {self.n_intraday_bars}")
        if self.model_file:
            lines.append(f"  Model:        {self.model_file}")
        if self.sample_tickers:
            lines.append(f"  Sample verification:")
            for ticker, vals in self.sample_tickers.items():
                lines.append(
                    f"    {ticker}: {vals['numerator']:.6f} / {vals['denominator']:.6f} "
                    f"- 1 = {vals['return']:.6f} ({vals['return']*100:+.4f}%)"
                )
        return "\n".join(lines)


def _now_et():
    """Current datetime in America/New_York."""
    import pytz
    return datetime.now(pytz.timezone("America/New_York"))


def _sample_tickers(
    numerators: pd.Series,
    denominators: pd.Series,
    returns: pd.Series,
    n: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Pick up to n tickers with the largest absolute return for verification."""
    top = returns.abs().nlargest(min(n, len(returns)))
    samples = {}
    for ticker in top.index:
        samples[ticker] = {
            "numerator":   float(numerators.get(ticker, float("nan"))),
            "denominator": float(denominators.get(ticker, float("nan"))),
            "return":      float(returns.get(ticker, float("nan"))),
        }
    return samples

# =============================================================================
# Internal helpers
# =============================================================================

def _default_exec() -> ExecutionConfig:
    """Return a default ExecutionConfig with logger configured."""
    cfg = ExecutionConfig()
    cfg.configure_logger()
    return cfg


def _exec(exec_config: Optional[ExecutionConfig]) -> ExecutionConfig:
    """Return *exec_config* if provided (and configure its logger), else a default."""
    if exec_config is not None:
        exec_config.configure_logger()
        return exec_config
    return _default_exec()


def _cache_manager(exec_cfg: ExecutionConfig) -> CacheManager:
    """Build a CacheManager rooted in exec_cfg's data directory."""
    return CacheManager(
        cache_dir=os.path.join(exec_cfg.data_dir, "cache"),
        backup_dir=os.path.join(exec_cfg.data_dir, "backups"),
        # Pass the module-level symbol so test patches on
        # shockarb.pipeline.yf.download are intercepted correctly.
        downloader=yf.download,
    )


# =============================================================================
# datamgr coordinator
# =============================================================================

def _coordinator(exec_cfg: ExecutionConfig) -> DataCoordinator:
    """
    Build a DataCoordinator wired to ParquetStore + YFinanceProvider.

    This is the single point where shockarb hands off to datamgr.
    Tests replace this via patch.object(pipeline, '_coordinator', ...).
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
        parquet file, so keep ETF and stock universes separate.
    exec_config : ExecutionConfig, optional
        Controls data directory and logging.  Uses a process-wide default if omitted.

    Returns
    -------
    DataFrame
        (dates × tickers) adjusted close prices.
        Falls back to synthetic crisis prices if yfinance fails entirely —
        this is logged at ERROR level and is only intended for testing.
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
    prices : DataFrame
        (T × N) adjusted close prices.
    min_coverage : float
        Minimum fraction of non-NaN prices to retain a ticker. Default 0.8.

    Returns
    -------
    DataFrame
        (T−1 × N') daily returns. N' ≤ N if low-coverage tickers were dropped.
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

def _market_is_open() -> bool:
    """
    True if NYSE is in its regular session or the post-close grace window.

    The grace period (market close 4:00 PM → 5:00 PM ET) exists because
    yfinance does not publish today's final daily bar until ~30-60 minutes
    after the bell.  During this window the intraday 15m path is used
    instead, which has data available immediately.
    """
    import pytz
    from datetime import datetime
    now = datetime.now(pytz.timezone("America/New_York"))
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    # Grace period: intraday path stays active until 5:00 PM ET so we
    # don't fall back to stale daily data while yfinance finalises today's bar.
    market_close = now.replace(hour=17, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def _intraday_returns_from_frame(
    raw: pd.DataFrame, prev_adj_close: pd.Series,
) -> Tuple[pd.Series, str, str, pd.Series, int]:
    """
    Compute intraday returns from a 15m bar DataFrame and yesterday's adj_close.

    Parameters
    ----------
    raw : DataFrame
        MultiIndex (field, ticker) intraday bars from the provider.
    prev_adj_close : Series
        Yesterday's adjusted close, indexed by ticker.

    Returns
    -------
    returns : Series
        (latest_15m_close / prev_adj_close) - 1, indexed by ticker.
    field_used : str
        Which field was used as the numerator (e.g. "adj_close").
    latest_bar_ts : str
        ISO timestamp of the latest bar used.
    latest_values : Series
        Raw numerator values (latest close per ticker).
    n_bars : int
        Number of intraday bars in the raw data.
    """
    # Extract the close prices from the MultiIndex result
    field_used = "unknown"
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "adj_close" in level0:
            prices = raw["adj_close"]
            field_used = "adj_close"
        elif "Adj Close" in level0:
            prices = raw["Adj Close"]
            field_used = "Adj Close"
        elif "close" in level0:
            prices = raw["close"]
            field_used = "close"
        else:
            prices = raw["Close"]
            field_used = "Close"
    else:
        prices = raw
        field_used = "Close"

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    n_bars = len(prices)
    latest = prices.ffill().iloc[-1]
    latest_bar_ts = str(prices.index[-1]) if len(prices) > 0 else "unknown"

    # Align tickers — only compute returns where we have both sides
    common = latest.index.intersection(prev_adj_close.index)
    if common.empty:
        return pd.Series(dtype=float), field_used, latest_bar_ts, latest, n_bars

    returns = (latest[common] / prev_adj_close[common]) - 1
    return returns, field_used, latest_bar_ts, latest[common], n_bars


def _open_prices_from_frame(raw: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Extract today's opening price per ticker from the first 15m bar.

    Parameters
    ----------
    raw : DataFrame
        MultiIndex (field, ticker) intraday bars from the provider.

    Returns
    -------
    opens : Series
        Open price from the first bar, indexed by ticker.
    open_bar_ts : str
        ISO timestamp of the first bar.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "open" in level0:
            opens = raw["open"]
        elif "Open" in level0:
            opens = raw["Open"]
        else:
            # Fallback to close of first bar
            if "adj_close" in level0:
                opens = raw["adj_close"]
            elif "Close" in level0:
                opens = raw["Close"]
            else:
                opens = raw.iloc[:, :raw.columns.get_level_values(1).nunique()]
    else:
        opens = raw

    if isinstance(opens, pd.Series):
        opens = opens.to_frame()

    open_bar_ts = str(opens.index[0]) if len(opens) > 0 else "unknown"
    first_bar_opens = opens.iloc[0]

    return first_bar_opens, open_bar_ts


def fetch_live_returns(
    tickers: List[str],
    period: str = "5d",
    exec_config: Optional[ExecutionConfig] = None,
    force_live: bool = False,
) -> pd.Series:
    """
    Fetch the most recent day's closing returns via the datamgr coordinator.
    Uses a 5-day look-back window so the return computation is safe across
    weekends and holidays.  Results are cached by the coordinator — calling
    this multiple times on the same day does not trigger redundant downloads.

    During market hours (or when force_live=True), bypasses the coordinator
    and fetches directly from yfinance so intraday prices are reflected.

    Parameters
    ----------
    tickers : list of str
    period : str
        Look-back window as a yfinance period string.  Default "5d".
    exec_config : ExecutionConfig, optional
    force_live : bool
        Force direct yfinance fetch regardless of market hours.
    Returns
    -------
    Series
        Most recent day's returns, indexed by ticker.
    Raises
    ------
    ValueError
        If no data can be retrieved at all.
    """
    from datetime import date, timedelta
    exec_cfg = _exec(exec_config)
    logger.info(f"Fetching live closing data for {len(tickers)} tickers…")

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


def score_universe(
    universe: UniverseConfig,
    model: "FactorModel",
    exec_config: Optional[ExecutionConfig] = None,
    force_daily: bool = False,
    from_open: bool = False,
) -> Tuple["pd.DataFrame", ScoreProvenance]:
    """
    Score a universe using a single shared coordinator instance.

    Three modes, selected by flags and market state:

      Path A — Market closed (daily), or forced via force_daily:
        Two daily DataRequests (ETF + stock), one coordinator, one fulfill().
        Returns are computed from adj_close price series via pct_change().

      Path B — Market open (intraday), default:
        Returns = latest_15m_close / yesterday's_daily_close - 1.
        Captures the overnight gap plus today's move.

      Path B with from_open:
        Returns = latest_15m_close / today's_first_15m_open - 1.
        Pure intraday move since the bell, stripping the overnight gap.

    Parameters
    ----------
    universe : UniverseConfig
    model : FactorModel
        Already-loaded model. Must be fitted.
    exec_config : ExecutionConfig, optional
    force_daily : bool
        Force Path A regardless of market state.  CLI: --use-prior-close / -p.
    from_open : bool
        Use today's session open as denominator instead of yesterday's close.
        CLI: --from-open / -o.  Ignored when market is closed.

    Returns
    -------
    (DataFrame, ScoreProvenance)

    Raises
    ------
    ValueError
        If the coordinator returns empty data for either leg.
    """
    from datetime import date, timedelta

    exec_cfg = _exec(exec_config)

    # Ticker source: prefer live columns from the model (available right after
    # build()); fall back to universe config (correct after load_model()).
    etf_tickers   = (list(model.etf_returns.columns)
                     if model.etf_returns is not None and not model.etf_returns.empty
                     else list(universe.market_etfs))
    stock_tickers = (list(model.stock_returns.columns)
                     if model.stock_returns is not None and not model.stock_returns.empty
                     else list(universe.individual_stocks))

    logger.info(
        f"score_universe({universe.name!r}): "
        f"{len(etf_tickers)} ETFs, {len(stock_tickers)} stocks"
    )

    coordinator = _coordinator(exec_cfg)
    prov = ScoreProvenance(
        universe = universe.name,
        provider = "yfinance",
        n_etfs   = len(etf_tickers),
        n_stocks = len(stock_tickers),
    )

    # Early-session warning: within 30 minutes of the open, intraday bars
    # are thin (1-2 bars).  Suggest --use-prior-close if not already set.
    if not force_daily and _market_is_open():
        now_et = _now_et()
        minutes_since_open = (
            (now_et.hour - 9) * 60 + (now_et.minute - 30)
        )
        if 0 <= minutes_since_open < 30:
            logger.warning(
                f"⚠️  Only {minutes_since_open} minutes since market open — "
                f"intraday data is thin ({minutes_since_open // 15 + 1} bar(s)). "
                f"Consider using --use-prior-close / -p for yesterday's "
                f"full-day returns."
            )

    use_intraday = _market_is_open() and not force_daily

    if force_daily and _market_is_open():
        logger.info(
            "Market is open but --use-prior-close is set — "
            "using daily close-to-close returns."
        )

    if use_intraday:
        # -----------------------------------------------------------------
        # Path B — intraday: single request, all tickers combined, no cache
        # -----------------------------------------------------------------
        today = date.today().strftime("%Y-%m-%d")
        all_tickers = etf_tickers + stock_tickers

        if from_open:
            logger.info("Market is open — using intraday path (--from-open: today's open as denominator).")
            prov.path = "intraday (from open)"
        else:
            logger.info("Market is open — using intraday coordinator path.")
            prov.path = "intraday"
        prov.interval     = "15m"
        prov.fetch_period = "1d"

        # Daily cache is needed for the default mode (prev close denominator).
        # Skip it when --from-open since the denominator comes from the 15m bars.
        if not from_open:
            # Request through YESTERDAY — not today.
            # TODO Step 2c: run-log check + backfill if nightly run was missed.
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            start_daily = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            coordinator.register(DataRequest(
                tickers   = tuple(all_tickers),
                start     = start_daily,
                end       = yesterday,
                frequency = Frequency.DAILY,
                retention = "permanent",
                requester = f"{universe.name}.prev_close",
            ))

        # Intraday fetch — all tickers in one DataRequest, cache=False.
        coordinator.register(DataRequest(
            tickers    = tuple(all_tickers),
            start      = today,
            end        = today,
            frequency  = Frequency.INTRADAY_15M,
            retention  = "ephemeral",
            requester  = f"{universe.name}.intraday",
            trade_date = today,
            cache      = False,
        ))

        results = coordinator.fulfill()

        # Intraday bars (needed for both modes)
        intraday_raw = results.get(f"{universe.name}.intraday", pd.DataFrame())
        if intraday_raw is None or intraday_raw.empty:
            raise ValueError(
                f"score_universe({universe.name!r}): coordinator returned no intraday data."
            )

        if from_open:
            # --from-open: denominator = today's session open (first 15m bar)
            denominator, open_bar_ts = _open_prices_from_frame(intraday_raw)
            denom_label = "open"
            denom_timestamp = open_bar_ts + " (first 15m bar open)"
        else:
            # Default: denominator = yesterday's daily close
            prev_prices = results.get(f"{universe.name}.prev_close", pd.DataFrame())
            if prev_prices is None or prev_prices.empty:
                raise ValueError(
                    f"score_universe({universe.name!r}): no daily data for prev close."
                )
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_ts = pd.Timestamp(yesterday)
            prev_prices = prev_prices[prev_prices.index <= yesterday_ts]
            if prev_prices.empty:
                raise ValueError(
                    f"score_universe({universe.name!r}): no daily data on or before "
                    f"{yesterday} for prev close."
                )
            denominator = prev_prices.ffill().iloc[-1]
            prev_close_date = str(prev_prices.index[-1].date())
            denom_label = "adj_close"
            denom_timestamp = prev_close_date + " (daily cache, last row)"

        # Compute returns: latest_15m_close / denominator - 1
        all_returns, field_used, latest_bar_ts, latest_values, n_bars = \
            _intraday_returns_from_frame(intraday_raw, denominator)

        # Populate provenance
        now_et = _now_et()
        prov.timestamp_utc        = datetime.utcnow().isoformat() + "Z"
        prov.timestamp_et         = now_et.isoformat()
        prov.numerator_field      = field_used
        prov.numerator_timestamp  = latest_bar_ts
        prov.denominator_field    = denom_label
        prov.denominator_timestamp = denom_timestamp
        prov.n_intraday_bars      = n_bars
        prov.return_formula       = (
            f"{field_used} @ {latest_bar_ts} / "
            f"{denom_label} @ {denom_timestamp} - 1"
        )
        prov.sample_tickers = _sample_tickers(
            latest_values, denominator, all_returns,
        )

        etf_returns   = all_returns.reindex(etf_tickers).dropna()
        stock_returns = all_returns.reindex(stock_tickers).dropna()

        if etf_returns.empty:
            raise ValueError(
                f"score_universe({universe.name!r}): no intraday returns for ETFs."
            )
        if stock_returns.empty:
            raise ValueError(
                f"score_universe({universe.name!r}): no intraday returns for stocks."
            )

        scores = model.score(etf_returns, stock_returns)
        return scores, prov

    # -----------------------------------------------------------------
    # Path A — daily (after close): single coordinator, single fulfill()
    # -----------------------------------------------------------------
    prov.path         = "daily (forced via --use-prior-close)" if force_daily else "daily"
    prov.interval     = "1d"
    prov.fetch_period = "5d"

    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    coordinator.register(DataRequest(
        tickers   = tuple(etf_tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = f"{universe.name}.live_etf",
    ))
    coordinator.register(DataRequest(
        tickers   = tuple(stock_tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = f"{universe.name}.live_stock",
    ))
    results = coordinator.fulfill()

    etf_prices   = results.get(f"{universe.name}.live_etf",   pd.DataFrame())
    stock_prices = results.get(f"{universe.name}.live_stock", pd.DataFrame())

    if etf_prices is None or etf_prices.empty:
        raise ValueError(
            f"score_universe({universe.name!r}): coordinator returned no ETF data."
        )
    if stock_prices is None or stock_prices.empty:
        raise ValueError(
            f"score_universe({universe.name!r}): coordinator returned no stock data."
        )

    etf_returns   = prices_to_returns(etf_prices).iloc[-1]
    stock_returns = prices_to_returns(stock_prices).iloc[-1]

    # Daily provenance: numerator is today's adj_close, denominator is yesterday's
    all_prices = pd.concat([etf_prices, stock_prices], axis=1)
    if len(all_prices) >= 2:
        today_close     = all_prices.ffill().iloc[-1]
        yesterday_close = all_prices.ffill().iloc[-2]
        today_date      = str(all_prices.index[-1].date())
        yesterday_date  = str(all_prices.index[-2].date())
        all_returns_combined = pd.concat([etf_returns, stock_returns])
        prov.sample_tickers = _sample_tickers(
            today_close, yesterday_close, all_returns_combined,
        )
    else:
        today_date     = "unknown"
        yesterday_date = "unknown"

    now_et = _now_et()
    prov.timestamp_utc         = datetime.utcnow().isoformat() + "Z"
    prov.timestamp_et          = now_et.isoformat()
    prov.numerator_field       = "adj_close"
    prov.numerator_timestamp   = today_date + " (daily close)"
    prov.denominator_field     = "adj_close"
    prov.denominator_timestamp = yesterday_date + " (daily close)"
    prov.return_formula        = (
        f"adj_close @ {today_date} / adj_close @ {yesterday_date} - 1 "
        f"(via pct_change on ffilled daily series)"
    )

    scores = model.score(etf_returns, stock_returns)
    return scores, prov


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
    is preserved:  ``(Adj Close, ticker)``, ``(Close, ticker)``,
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
    DataFrame or None
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
    playback_path : str, optional
        Path to a previously saved parquet tape (from save_intraday_tape()).
        When provided, returns are computed from the file rather than live data.
        Intended for offline replay and debugging.

    Returns
    -------
    Series or None
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

    By default stores only the three structurally vital rows (yesterday's last
    close, today's open, today's current print) to keep files small.
    Pass full_tape=True to store all ~600 rows.

    Parameters
    ----------
    tickers : list of str
    path : str
        Destination parquet path.
    full_tape : bool
        If True, store the complete 2-day 1-minute tape.

    Returns
    -------
    DataFrame or None
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
    """Compute (current − prior_close) / prior_close from a 1-minute tape."""
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices = prices.dropna(axis=1, how="all").ffill()

    yesterday_close = prices[prices.index.date < prices.index[-1].date()].iloc[-1]
    current = prices.iloc[-1]
    return (current - yesterday_close) / yesterday_close


def _minimal_tape(data: pd.DataFrame) -> pd.DataFrame:
    """Reduce a full 1-minute tape to the three structurally vital rows."""
    tape = data.ffill()
    dates = tape.index.normalize().unique()
    if len(dates) < 2:
        return tape.iloc[[0, -1]]

    yesterday, today = dates[-2], dates[-1]
    y_close = tape[tape.index.normalize() == yesterday].index[-1]
    t_open = tape[tape.index.normalize() == today].index[0]
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
    Full pipeline: fetch → compute returns → fit model.

    Parameters
    ----------
    universe : UniverseConfig
        Defines what to analyze: tickers, date window, n_components.
    exec_config : ExecutionConfig, optional
        Controls caching, paths, and logging.

    Returns
    -------
    FactorModel
        Fitted and ready for scoring.
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

    The filename embeds a timestamp so multiple saves don't overwrite each
    other.  Use find_latest_model() to retrieve the most recent one.

    Parameters
    ----------
    model : FactorModel
    name : str
        Short identifier, e.g. "us" or "global".
    exec_config : ExecutionConfig, optional

    Returns
    -------
    str
        Absolute path to the saved file.
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
    loadings are augmented with R² and residual volatility columns and
    sorted by R² descending so the most trustworthy signals are at the top.

    Returns
    -------
    tuple of str
        (etf_basis_path, stock_loadings_path)
    """
    exec_cfg = _exec(exec_config)

    basis_path = exec_cfg.resolve_path(f"{name}_etf_basis.csv")
    model.etf_basis.sort_index().to_csv(basis_path, float_format="%.6f")

    loadings_path = exec_cfg.resolve_path(f"{name}_stock_loadings.csv")
    output = model.loadings.copy()

    if model.diagnostics:
        output["R_squared"] = model.diagnostics.stock_r_squared
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

    *** FOR TESTING AND NETWORK-OUTAGE FALLBACK ONLY. ***
    All series start at 100 — a deliberate flag so callers can detect
    synthetic data programmatically (real prices very rarely start at 100).

    Stylised facts embedded (Russia-Ukraine, Feb–Mar 2022):
      - Broad equity:  −5%
      - Energy / defence:  +10–15%
      - Gold / bonds:   +5%
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
    data *before* pct_change, suitable for auditing, replaying, or feeding
    back into prices_to_returns().

    Parameters
    ----------
    tickers : list of str
    period : str
        yfinance period string.  Default "5d" (enough to compute one return).
    save_path : str, optional
        If provided, the OHLCV DataFrame is written to this path as parquet
        before being returned.  The directory is created if it does not exist.
        File naming convention used by the CLI:
          {data_dir}/prices_{name}_{YYYYMMDD_HHMMSS}.parquet

    Returns
    -------
    DataFrame
        Raw OHLCV MultiIndex DataFrame (dates × (field, ticker)).

    Raises
    ------
    ValueError
        If yfinance returns no data.

    Example
    -------
        ohlcv = fetch_live_prices(["VOO", "TLT"], save_path="data/prices_etf.parquet")
        prices = CacheManager.extract_adj_close(ohlcv)   # or mgr.extract_adj_close()
        returns = prices_to_returns(prices)
    """
    logger.info(f"Fetching live OHLCV for {len(tickers)} tickers (period={period})…")
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError("yfinance returned no data for fetch_live_prices")

    # Normalise to MultiIndex even for a single ticker
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_tuples(
            [("Close", raw.columns[0])] if len(tickers) == 1
            else [("Close", t) for t in raw.columns]
        )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        raw.to_parquet(save_path)
        logger.info(f"Prices saved → {os.path.basename(save_path)}")

    return raw
