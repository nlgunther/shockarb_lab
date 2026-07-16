"""
stockfit.features — Extract a per-ticker feature dict from ShockArb pipeline outputs.

Pure module: reads three file paths, returns a list of named feature dicts.
No network calls, no side effects.

Feature spec (single source of truth — rules.py and report.py both consume this)
----------------------------------------------------------------------------------
  ticker            str   — ticker symbol
  r_squared         float — R² of the factor model fit (0–1)
  confidence_delta  float — primary ShockArb signal (delta_rel × r²)
  delta_rel         float — return unexplained by macro factors
  actual_return     float — stock’s raw return over the window
  expected_rel      float — model-predicted return (relative)
  residual_vol      float — volatility of the residual series (noise floor)
  price             float | None — last close price
  analyst_target    float | None — consensus analyst price target
  analyst_upside    float | None — (target − price) / price  (signed)
  fwd_pe            float | None — forward P/E
  ttm_eps           float | None — trailing twelve months EPS
  fwd_eps           float | None — forward EPS estimate
  ex_div            str   | None — next ex-dividend date
  div_amt           float | None — dividend amount
  next_earnings     str   | None — next earnings date ('' if blank)
  news_headlines    list[str]    — up to 3 news headlines for this ticker
  target_below_price bool        — True when analyst_target < price (data quality flag)
  earnings_imminent bool         — True when next_earnings is within earnings_window days
  rvol              float | None — relative volume: latest cached volume / trailing
                                    average volume (only populated when compute_rvol=True)
  rvol_window       int   | None — trailing window size (days) used for rvol
  intraday_price    float | None — live current price (only populated when
                                    compute_intraday=True)
  intraday_chg_pct  float | None — (intraday_price - price) / price; change vs
                                    the cached close used for `price`
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# All paths centralised in paths.py. See docs/PATHS.md for design rationale.
from paths import DATA, LIVE_ALPHA_US, FUNDAMENTALS, NEWS

# Resolve project root so `shockarb.store` is importable regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# RVOL window bounds (see docs/PATHS.md / RVOL design notes).
RVOL_MAX_WINDOW = 20
RVOL_MIN_WINDOW = 5


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _load_scores(path: str) -> list[dict[str, Any]]:
    """
    Load live_alpha_us.csv into a list of dicts with numeric fields cast to float.

    Expected columns (shockarb score output):
      Ticker, actual_return, expected_rel, expected_abs, delta_rel, delta_abs,
      r_squared, residual_vol, confidence_delta
    """
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            entry: dict[str, Any] = {}
            # DictReader fills missing trailing fields with None on short
            # (truncated) rows; treat those as empty strings rather than
            # crashing on .strip().
            raw = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            ticker = (
                raw.get("Ticker") or raw.get("ticker") or raw.get("")
            )
            if not ticker:
                continue
            entry["ticker"] = ticker.upper()
            for col in ("actual_return", "expected_rel", "expected_abs",
                        "delta_rel", "delta_abs", "r_squared",
                        "residual_vol", "confidence_delta"):
                try:
                    entry[col] = float(raw.get(col, "nan"))
                except (ValueError, TypeError):
                    entry[col] = float("nan")
            rows.append(entry)
    except FileNotFoundError:
        pass
    return rows


def _load_fundamentals(path: str) -> dict[str, dict[str, Any]]:
    """
    Load fundamentals.csv into {ticker: {field: value}} dict.

    Expected columns:
      Ticker, Price, Fwd P/E, TTM EPS, Fwd EPS, Next Earnings,
      Est. EPS, Ex-Div, Div Amt, Analyst Tgt
    """
    result: dict[str, dict[str, Any]] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            # See _load_scores: DictReader fills missing trailing fields
            # with None on short (truncated) rows.
            raw = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            ticker = raw.get("Ticker", "").upper()
            if not ticker:
                continue

            def _f(key: str) -> float | None:
                v = raw.get(key, "")
                try:
                    return float(v.replace("$", "").replace(",", "")) if v and v != "—" else None
                except (ValueError, TypeError):
                    return None

            result[ticker] = {
                "price":         _f("Price"),
                "fwd_pe":        _f("Fwd P/E"),
                "ttm_eps":       _f("TTM EPS"),
                "fwd_eps":       _f("Fwd EPS"),
                "analyst_target": _f("Analyst Tgt"),
                "div_amt":       _f("Div Amt"),
                "next_earnings": raw.get("Next Earnings", "").strip("—").strip() or None,
                "ex_div":        raw.get("Ex-Div", "").strip("—").strip() or None,
            }
    except FileNotFoundError:
        pass
    return result


def _load_news(path: str) -> dict[str, list[str]]:
    """
    Parse news.txt into {ticker: [headline, ...]} dict.
    Same separator format as catalyst_feed.txt — 87-dash blocks.
    """
    result: dict[str, list[str]] = {}
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks  = content.split("---" * 29)   # 87-dash separator
        for chunk in chunks:
            file_lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
            if not file_lines or not file_lines[0].startswith("["):
                continue
            ticker = file_lines[0].split("]")[0].replace("[", "").strip().upper()
            headlines = []
            for i, line in enumerate(file_lines):
                if line.startswith(">") and i + 1 < len(file_lines):
                    nxt = file_lines[i + 1]
                    if not nxt.startswith(">") and not nxt.startswith("["):
                        headlines.append(nxt)
            # Keep in sync with news_scanner.py::_MAX_HEADLINES — no shared
            # constant module exists between the two files.
            result[ticker] = headlines[:5]
    except FileNotFoundError:
        pass
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _earnings_imminent(next_earnings: str | None, window_days: int) -> bool:
    """Return True if next_earnings is a valid date within window_days of today."""
    if not next_earnings:
        return False
    try:
        earn_date = datetime.strptime(next_earnings, "%Y-%m-%d").date()
        return (earn_date - date.today()).days <= window_days
    except ValueError:
        return False


def _compute_rvol(
    ticker:     str,
    store:      Any,
    max_window: int = RVOL_MAX_WINDOW,
    min_window: int = RVOL_MIN_WINDOW,
) -> tuple[float | None, int | None]:
    """
    Compute relative volume for *ticker* from cached daily OHLCV.

    RVOL = most recent cached day's volume / trailing average volume.
    Uses a dynamic window: as many trailing days as are cached, capped at
    max_window and floored at min_window. Reads only the local parquet
    cache (no network calls) — returns (None, None) if fewer than
    min_window + 1 days are cached.

    Example
        rvol, window = _compute_rvol("ETN", store)
        # rvol=2.3, window=10  → today's volume is 2.3x the 10-day average
    """
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=max_window * 3)).isoformat()

    ohlcv = store.fetch_daily_ohlcv([ticker], start, end)
    if ohlcv.empty:
        return None, None

    # The cached parquet schema has used both "Volume" and "volume" across
    # store.py revisions; accept either rather than assuming one.
    col = next(
        (c for c in ((f, ticker) for f in ("Volume", "volume")) if c in ohlcv.columns),
        None,
    )
    if col is None:
        return None, None

    volumes = ohlcv[col].dropna()
    if len(volumes) < min_window + 1:
        return None, None

    today_vol = volumes.iloc[-1]
    window    = min(max_window, len(volumes) - 1)
    trailing  = volumes.iloc[-(window + 1):-1]
    avg       = trailing.mean()
    if avg <= 0:
        return None, None

    return today_vol / avg, window


def _fetch_intraday_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch current (latest intraday) price for a batch of tickers via yfinance.

    Single batch call, period="1d" — returns today's bars up to the last
    complete minute and takes the final row's Close as "current price".
    Not cached (intraday prices are too ephemeral). Returns {} on failure
    or if no tickers are given.

    Example
        prices = _fetch_intraday_prices(["ETN", "HON"])
        # {"ETN": 410.20, "HON": 225.10}
    """
    if not tickers:
        return {}

    import yfinance as yf

    try:
        raw = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        if raw.empty:
            return {}
        close_col = raw["Close"] if "Close" in raw.columns else raw
        if hasattr(close_col, "columns"):
            last = close_col.iloc[-1]
            return {
                t: float(last[t]) for t in tickers
                if t in last.index and last[t] == last[t]   # NaN check
            }
        val = float(close_col.iloc[-1])
        return {tickers[0]: val} if val == val else {}      # NaN check
    except Exception:
        return {}


def extract_all(
    scores_path:       Path = LIVE_ALPHA_US,
    fundamentals_path: Path = FUNDAMENTALS,
    news_path:         Path = NEWS,
    earnings_window:   int  = 14,
    compute_rvol:      bool = False,
    compute_intraday:  bool = False,
) -> list[dict[str, Any]]:
    """
    Extract per-ticker feature dicts from the three pipeline output files.

    Parameters
    ----------
    scores_path       : path to live_alpha_us.csv
    fundamentals_path : path to fundamentals.csv
    news_path         : path to news.txt
    earnings_window   : days-out threshold for earnings_imminent flag (default 14)
    compute_rvol      : if True, populate "rvol"/"rvol_window" from the local
                        DataStore parquet cache (data/prices/daily/). No network
                        calls — tickers without enough cached history get
                        rvol=None. Default False (no behavior change).
    compute_intraday  : if True, fetch live current prices for all tickers in a
                        single batch yfinance call and populate
                        "intraday_price"/"intraday_chg_pct". Network call —
                        tickers the fetch fails for get both fields as None.
                        Default False (no behavior change).

    Returns
    -------
    list of feature dicts, one per ticker, sorted by confidence_delta descending.

    Example
    -------
        from stockfit.features import extract_all
        candidates = extract_all()
        for c in candidates:
            print(c["ticker"], c["confidence_delta"], c["analyst_upside"])
    """
    scores       = _load_scores(scores_path)
    fundamentals = _load_fundamentals(fundamentals_path)
    news         = _load_news(news_path)

    store = None
    if compute_rvol:
        from shockarb.store import DataStore
        store = DataStore(DATA)

    intraday_prices: dict[str, float] = {}
    if compute_intraday:
        intraday_prices = _fetch_intraday_prices([row["ticker"] for row in scores])

    results = []
    for row in scores:
        ticker = row["ticker"]
        fund   = fundamentals.get(ticker, {})

        price  = fund.get("price")
        target = fund.get("analyst_target")

        analyst_upside = None
        if price and target and price > 0:
            analyst_upside = (target - price) / price

        target_below_price = bool(
            price is not None and target is not None and target < price
        )

        earnings_imminent = _earnings_imminent(fund.get("next_earnings"), earnings_window)

        rvol, rvol_window = (None, None)
        if compute_rvol:
            rvol, rvol_window = _compute_rvol(ticker, store)

        intraday_price = intraday_prices.get(ticker)
        intraday_chg_pct = None
        if intraday_price is not None and price and price > 0:
            intraday_chg_pct = (intraday_price - price) / price

        results.append({
            # Signal features
            "ticker":           ticker,
            "r_squared":        row.get("r_squared", float("nan")),
            "confidence_delta": row.get("confidence_delta", float("nan")),
            "delta_rel":        row.get("delta_rel", float("nan")),
            "actual_return":    row.get("actual_return", float("nan")),
            "expected_rel":     row.get("expected_rel", float("nan")),
            "residual_vol":     row.get("residual_vol", float("nan")),
            # Fundamentals
            "price":            price,
            "analyst_target":   target,
            "analyst_upside":   analyst_upside,
            "fwd_pe":           fund.get("fwd_pe"),
            "ttm_eps":          fund.get("ttm_eps"),
            "fwd_eps":          fund.get("fwd_eps"),
            "ex_div":           fund.get("ex_div"),
            "div_amt":          fund.get("div_amt"),
            "next_earnings":    fund.get("next_earnings"),
            # Catalyst features
            "news_headlines":   news.get(ticker, []),
            # Derived flags
            "target_below_price": target_below_price,
            "earnings_imminent":  earnings_imminent,
            # Volume context (informational only — see RVOL design notes)
            "rvol":               rvol,
            "rvol_window":        rvol_window,
            # Live-quote context (informational only — see intraday design notes)
            "intraday_price":     intraday_price,
            "intraday_chg_pct":   intraday_chg_pct,
        })

    results.sort(key=lambda d: d.get("confidence_delta") or float("-inf"), reverse=True)
    return results
