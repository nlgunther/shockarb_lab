"""
ShockArb Market Data Fetcher.

Pulls a market snapshot and writes it to data/market_snapshot.json.

Daily mode (default): fetches yesterday's close + today's close via the
DataCoordinator (parquet cache shared with `shockarb score`). Only downloads
what isn't already cached. VIX is fetched directly — it has no Adj Close.

Intraday mode (--intraday): uses the cached prev_close for the baseline, then
makes a single live yf.download() call for current prices. Nothing is cached
for intraday — prices are too ephemeral. The report clearly labels the baseline
date so the reader knows what the percentage change is measured against.

Usage
-----
    python utils/market_data.py                    # daily → data/market_snapshot.json
    python utils/market_data.py --intraday         # live prices vs prev close
    python utils/market_data.py --out data/snap.json

Schedule (optional)
-------------------
    # Windows Task Scheduler: run at 4:15pm ET daily
    # Or add to your shockarb post-close workflow

Environment
-----------
    SHOCK_ARB_DATA_DIR   Override data directory (default: ./data)
                         See docs/ENVIRONMENT_VARIABLES.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import yfinance as yf
from loguru import logger

# Resolve project root so datamgr / shockarb imports work regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datamgr.coordinator import DataCoordinator          # noqa: E402
from datamgr.providers.yfinance import YFinanceProvider  # noqa: E402
from datamgr.requests import DataRequest, Frequency      # noqa: E402
from datamgr.stores.parquet import ParquetStore          # noqa: E402


from paths import MARKET_SNAPSHOT as _MARKET_SNAPSHOT_PATH  # noqa: E402
_DEFAULT_OUT = str(_MARKET_SNAPSHOT_PATH)

# Days of history to request — enough to survive long weekends / holidays.
_HISTORY_DAYS = 10

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

_US_BROAD = {
    "SPY":  ("S&P 500",       "us_broad"),
    "QQQ":  ("Nasdaq 100",    "us_broad"),
    "IWM":  ("Russell 2000",  "us_broad"),
    "DIA":  ("Dow Jones",     "us_broad"),
}

_US_SECTORS = {
    "XLK":  ("Tech",              "us_sector"),
    "XLF":  ("Financials",        "us_sector"),
    "XLE":  ("Energy",            "us_sector"),
    "XLV":  ("Health Care",       "us_sector"),
    "XLI":  ("Industrials",       "us_sector"),
    "XLY":  ("Consumer Disc.",    "us_sector"),
    "XLP":  ("Consumer Staples",  "us_sector"),
    "XLU":  ("Utilities",         "us_sector"),
    "XLRE": ("Real Estate",       "us_sector"),
    "XLB":  ("Materials",         "us_sector"),
    "XLC":  ("Comm. Services",    "us_sector"),
}

_BONDS = {
    "TLT":  ("20yr Treasury",    "bond"),
    "IEF":  ("7-10yr Treasury",  "bond"),
    "HYG":  ("High Yield",       "bond"),
    "LQD":  ("Inv. Grade Corp.", "bond"),
}

_RISK_CACHEABLE = {
    "GLD": ("Gold", "risk"),
    "USO": ("Oil",  "risk"),
}

# VIX is an index with no Adj Close — always fetched directly, never via coordinator.
_VIX_TICKER = "^VIX"
_VIX_META   = ("VIX", "risk")

_OVERSEAS = {
    "^FTSE":     ("FTSE 100 (London)",    "overseas"),
    "^GDAXI":    ("DAX (Frankfurt)",      "overseas"),
    "^FCHI":     ("CAC 40 (Paris)",       "overseas"),
    "^STOXX50E": ("Euro Stoxx 50",        "overseas"),
    "^N225":     ("Nikkei 225 (Tokyo)",   "overseas"),
    "^HSI":      ("Hang Seng (HK)",       "overseas"),
    "000001.SS": ("Shanghai Composite",   "overseas"),
    "^BSESN":    ("BSE Sensex (Mumbai)",  "overseas"),
    "^AXJO":     ("ASX 200 (Sydney)",     "overseas"),
    "^BVSP":     ("Bovespa (São Paulo)",  "overseas"),
}

# Tickers routed through the DataCoordinator (have reliable Adj Close).
_COORDINATOR_TICKERS: dict[str, tuple[str, str]] = {
    **_US_BROAD,
    **_US_SECTORS,
    **_BONDS,
    **_RISK_CACHEABLE,
    **_OVERSEAS,
}

ALL_TICKERS: dict[str, tuple[str, str]] = {
    **_COORDINATOR_TICKERS,
    _VIX_TICKER: _VIX_META,
}


# ---------------------------------------------------------------------------
# Coordinator wiring
# ---------------------------------------------------------------------------

def _build_coordinator(data_dir: str) -> DataCoordinator:
    """
    Build a DataCoordinator backed by the shared ShockArb parquet cache.

    Mirrors pipeline._coordinator() so market_data reads from the same cache
    that `shockarb score` writes — no duplicate downloads for shared tickers.
    """
    from shockarb.store import DataStore as ShockArbStore  # local import — avoids circular on startup
    inner = ShockArbStore(data_dir)
    store = ParquetStore(inner)
    return DataCoordinator(store, provider=YFinanceProvider(downloader=yf.download))


# ---------------------------------------------------------------------------
# Per-ticker row builders
# ---------------------------------------------------------------------------

def _row_ok(ticker: str, label: str, group: str,
            close: float, prev: float, last_date: str) -> dict:
    return {
        "ticker":    ticker,
        "label":     label,
        "group":     group,
        "close":     round(close, 4),
        "prev":      round(prev, 4),
        "chg_pct":   round((close - prev) / prev * 100, 4),
        "last_date": last_date,
        "status":    "ok",
    }


def _row_error(ticker: str, label: str, group: str) -> dict:
    return {
        "ticker": ticker, "label": label, "group": group,
        "close": None, "prev": None, "chg_pct": None,
        "last_date": None, "status": "error",
    }


# ---------------------------------------------------------------------------
# VIX — direct yfinance (no Adj Close in the coordinator contract)
# ---------------------------------------------------------------------------

def _fetch_vix(current_price: Optional[float] = None) -> dict:
    """
    Fetch VIX via yf.Ticker directly.

    In intraday mode the caller supplies current_price from a live download;
    we still need the prev close from history. In daily mode both values come
    from history.
    """
    label, group = _VIX_META
    try:
        hist = yf.Ticker(_VIX_TICKER).history(period="5d")
        if len(hist) < 2:
            raise ValueError("insufficient history")
        prev      = float(hist["Close"].iloc[-2])
        close     = current_price if current_price is not None else float(hist["Close"].iloc[-1])
        last_date = hist.index[-1].strftime("%Y-%m-%d")
        return _row_ok(_VIX_TICKER, label, group, close, prev, last_date)
    except Exception as exc:
        logger.warning(f"^VIX: {exc}")
        return _row_error(_VIX_TICKER, label, group)


# ---------------------------------------------------------------------------
# Daily fetch via coordinator
# ---------------------------------------------------------------------------

def _fetch_daily(data_dir: str) -> tuple[dict[str, dict], str]:
    """
    Fetch latest two daily closes for all coordinator tickers.

    Returns
    -------
    prices : dict  ticker → {close, prev, last_date}
    baseline_date : str  YYYY-MM-DD date of the prev_close row
    """
    start = (date.today() - timedelta(days=_HISTORY_DAYS)).isoformat()
    end   = (date.today() + timedelta(days=1)).isoformat()

    coordinator = _build_coordinator(data_dir)
    coordinator.register(DataRequest(
        tickers   = tuple(_COORDINATOR_TICKERS.keys()),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "market_data",
    ))
    results = coordinator.fulfill()
    df = results.get("market_data")

    prices: dict[str, dict] = {}
    baseline_date = (date.today() - timedelta(days=1)).isoformat()  # fallback

    if df is not None:
        for ticker in _COORDINATOR_TICKERS:
            if ticker not in df.columns:
                continue
            col = df[ticker].dropna()
            if len(col) < 2:
                continue
            prices[ticker] = {
                "close":     float(col.iloc[-1]),
                "prev":      float(col.iloc[-2]),
                "last_date": col.index[-1].strftime("%Y-%m-%d"),
            }
            # baseline_date = the date of the second-to-last row (the prev close).
            # Use SPY as representative; all tickers share the same trading calendar.
            if ticker == "SPY":
                baseline_date = col.index[-2].strftime("%Y-%m-%d")

    return prices, baseline_date


# ---------------------------------------------------------------------------
# Intraday — live current price via single yf.download() batch call
# ---------------------------------------------------------------------------

def _fetch_intraday_current(tickers: list[str]) -> dict[str, float]:
    """
    Fetch current (latest intraday) price for a batch of tickers.

    Uses period="1d" which returns today's bars up to the last complete minute.
    Takes the final row's Close as "current price". Returns {} on failure.

    Not cached — intraday prices are too ephemeral for parquet storage.
    """
    try:
        raw = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        if raw.empty:
            return {}
        close_col = raw["Close"] if "Close" in raw.columns else raw
        if hasattr(close_col, "columns"):
            # Multi-ticker: columns are ticker names
            last = close_col.iloc[-1]
            return {t: float(last[t]) for t in tickers if t in last.index and not _is_nan(last[t])}
        else:
            # Single ticker (shouldn't happen here but guard anyway)
            val = float(close_col.iloc[-1])
            return {tickers[0]: val} if tickers else {}
    except Exception as exc:
        logger.warning(f"Intraday batch fetch failed: {exc}")
        return {}


def _is_nan(v) -> bool:
    try:
        return v != v  # NaN check without importing math
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------

def fetch_snapshot(out_path: str = _DEFAULT_OUT,
                   data_dir: str = "./data",
                   intraday: bool = False) -> dict:
    """
    Fetch a full market snapshot and write it to out_path.

    Parameters
    ----------
    out_path  : destination JSON file
    data_dir  : ShockArb data directory (coordinator cache lives here)
    intraday  : if True, use cached prev_close + live current price

    Returns the snapshot dict.

    Example
    -------
        snap = fetch_snapshot("data/snap.json", intraday=True)
    """
    mode = "intraday" if intraday else "daily"
    logger.info(f"Fetching market snapshot ({len(ALL_TICKERS)} tickers, mode={mode})…")

    # --- Step 1: get prev_close (and daily close) from coordinator cache ---
    daily_prices, baseline_date = _fetch_daily(data_dir)

    # --- Step 2: optionally overlay live current prices ---
    if intraday:
        non_vix = [t for t in _COORDINATOR_TICKERS if not t.startswith("^") or t == "^FTSE"
                   or t in _OVERSEAS]
        # Fetch live for all coordinator tickers; VIX handled separately below
        live = _fetch_intraday_current(list(_COORDINATOR_TICKERS.keys()))
        logger.info(f"  Intraday live prices: {len(live)} tickers fetched")
    else:
        live = {}

    # --- Step 3: assemble ticker rows ---
    data = []
    for ticker, (label, group) in ALL_TICKERS.items():
        if ticker == _VIX_TICKER:
            live_vix = live.get(_VIX_TICKER)  # will be None (VIX not in coordinator set)
            data.append(_fetch_vix(current_price=None))  # VIX always from yf.Ticker
            continue

        cached = daily_prices.get(ticker)
        if cached is None:
            logger.warning(f"{ticker}: not in coordinator result — skipped")
            data.append(_row_error(ticker, label, group))
            continue

        if intraday and ticker in live:
            close_val = live[ticker]
        else:
            close_val = cached["close"]

        data.append(_row_ok(
            ticker, label, group,
            close     = close_val,
            prev      = cached["prev"],
            last_date = cached["last_date"],
        ))

    snapshot = {
        "fetched_at":       datetime.now(timezone.utc).isoformat(),
        "fetched_at_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "baseline_date":    baseline_date,
        "mode":             mode,
        "tickers":          data,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    logger.success(f"Snapshot saved: {out_path}  ({len(data)} tickers, baseline {baseline_date})")
    return snapshot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch market snapshot for ShockArb market report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", "-o", default=_DEFAULT_OUT,
        help=f"Output path (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--intraday", action="store_true",
        help="Fetch live intraday prices vs prev close (default: daily close vs prev close)",
    )
    args = parser.parse_args()

    data_dir = os.environ.get("SHOCK_ARB_DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))
    fetch_snapshot(args.out, data_dir=data_dir, intraday=args.intraday)
