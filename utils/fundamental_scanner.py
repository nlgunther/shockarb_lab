"""
ShockArb Fundamental Scanner.

Fetches key fundamental data for a list of tickers via yfinance and prints
a compact summary table. Designed as a post-scan overlay for news_scanner.py.

Public API
----------
    from fundamental_scanner import fetch_fundamentals, print_fundamentals

    rows = fetch_fundamentals(["BLK", "TXN", "PH"])
    print_fundamentals(rows)

Data pulled per ticker
----------------------
    Price           Last closing price
    Fwd P/E         Forward price/earnings ratio
    TTM EPS         Trailing twelve-month EPS (GAAP)
    Fwd EPS         Forward EPS consensus estimate
    Next Earnings   Expected earnings announcement date
    Est. EPS        Consensus EPS estimate for next quarter
    Ex-Div          Next ex-dividend date
    Div Amt         Next dividend amount
    Target          Mean analyst 12-month price target
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

# Default cache location — mirrors the project's data directory convention
_DEFAULT_CACHE = Path(os.environ.get("SHOCK_ARB_DATA_DIR", "./data")) / "fundamentals_cache.json"

# Fields pulled from yf.Ticker.info — mapped to display names
_INFO_FIELDS: dict[str, str] = {
    "currentPrice":       "Price",
    "forwardPE":          "Fwd P/E",
    "trailingEps":        "TTM EPS",
    "forwardEps":         "Fwd EPS",
    "targetMeanPrice":    "Target",
}


def _safe_get(d: dict, key: str, fmt: str = "") -> str:
    """Return formatted value from dict, or '—' if missing/None."""
    val = d.get(key)
    if val is None or val != val:  # catches None and NaN
        return "—"
    try:
        return format(val, fmt) if fmt else str(val)
    except (ValueError, TypeError):
        return str(val)


def _next_earnings(ticker_obj: yf.Ticker) -> tuple[str, str]:
    """
    Return (date_str, est_eps_str) for the next earnings event.

    earnings_dates is a DataFrame indexed by date (descending), with columns
    'EPS Estimate' and 'Reported EPS'. Future dates have Reported EPS = NaN.
    """
    try:
        df = ticker_obj.earnings_dates
        if df is None or df.empty:
            return "—", "—"
        # Future rows have no Reported EPS
        future = df[df["Reported EPS"].isna()]
        if future.empty:
            return "—", "—"
        # earnings_dates index is timezone-aware — normalise for display
        next_date = future.index.max()
        date_str  = pd.Timestamp(next_date).strftime("%Y-%m-%d")
        est       = future.loc[next_date, "EPS Estimate"]
        est_str   = f"${est:.2f}" if pd.notna(est) else "—"
        return date_str, est_str
    except Exception:
        return "—", "—"


def _next_dividend(info: dict) -> tuple[str, str]:
    """Return (ex_date_str, amount_str) from ticker info dict."""
    ex_ts = info.get("exDividendDate")
    amt   = info.get("dividendRate")

    ex_str  = datetime.fromtimestamp(ex_ts).strftime("%Y-%m-%d") if ex_ts else "—"
    amt_str = f"${amt:.2f}" if amt else "—"
    return ex_str, amt_str


def _load_cache(cache_path: Path) -> dict:
    """Load the fundamentals cache from disk. Returns {} if absent or corrupt."""
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
    except Exception:
        pass
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Write the fundamentals cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def fetch_fundamentals(
    tickers: list[str],
    cache_path: Path | str = _DEFAULT_CACHE,
) -> list[dict]:
    """
    Fetch fundamental data for each ticker, write to cache, and return row dicts.

    Each dict contains display-ready strings suitable for tabular printing.
    Missing fields are represented as '—' rather than raising. Results are
    written to ``cache_path`` (default: data/fundamentals_cache.json) keyed by
    ticker so that ``load_cached_fundamentals`` can serve them without a
    network call.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to fetch.
    cache_path : Path or str, optional
        Where to persist the cache. Defaults to ``_DEFAULT_CACHE``.

    Returns
    -------
    list of dict
        One dict per ticker with keys matching the table columns.
    """
    cache_path = Path(cache_path)
    cache = _load_cache(cache_path)
    rows = []
    for symbol in tickers:
        try:
            t    = yf.Ticker(symbol)
            info = t.info or {}

            earn_date, earn_est = _next_earnings(t)
            ex_date, div_amt    = _next_dividend(info)

            row = {
                "Ticker":        symbol,
                "Price":         _safe_get(info, "currentPrice",    ".2f"),
                "Fwd P/E":       _safe_get(info, "forwardPE",       ".1f"),
                "TTM EPS":       _safe_get(info, "trailingEps",     ".2f"),
                "Fwd EPS":       _safe_get(info, "forwardEps",      ".2f"),
                "Next Earnings": earn_date,
                "Est. EPS":      earn_est,
                "Ex-Div":        ex_date,
                "Div Amt":       div_amt,
                "Analyst Tgt":   _safe_get(info, "targetMeanPrice", ".2f"),
            }
            rows.append(row)
            cache[symbol] = {**row, "_cached_at": datetime.now().isoformat(timespec="seconds")}
            logger.debug(f"[Fundamentals] {symbol} ok")
        except Exception as exc:
            logger.warning(f"[Fundamentals] {symbol} failed: {exc}")
            rows.append({"Ticker": symbol, **{k: "ERR" for k in (
                "Price", "Fwd P/E", "TTM EPS", "Fwd EPS",
                "Next Earnings", "Est. EPS", "Ex-Div", "Div Amt", "Analyst Tgt",
            )}})
    _save_cache(cache, cache_path)
    return rows


def load_cached_fundamentals(
    tickers: list[str],
    cache_path: Path | str = _DEFAULT_CACHE,
) -> list[dict]:
    """
    Return fundamentals from the on-disk cache without any network calls.

    Tickers absent from the cache are included as rows of '—' values with a
    note in the Ticker field so they are visible in the table.

    Parameters
    ----------
    tickers : list of str
    cache_path : Path or str, optional

    Returns
    -------
    list of dict — same shape as fetch_fundamentals output, plus '_cached_at'.
    """
    cache = _load_cache(Path(cache_path))
    rows = []
    for symbol in tickers:
        if symbol in cache:
            row = dict(cache[symbol])   # includes _cached_at
            rows.append(row)
        else:
            rows.append({"Ticker": symbol, **{k: "—" for k in (
                "Price", "Fwd P/E", "TTM EPS", "Fwd EPS",
                "Next Earnings", "Est. EPS", "Ex-Div", "Div Amt", "Analyst Tgt",
            )}, "_cached_at": None})
    return rows


def print_fundamentals(rows: list[dict]) -> None:
    """
    Print a fixed-width fundamental summary table to stdout.

    Parameters
    ----------
    rows : list of dict
        Output of fetch_fundamentals().

    Example
    -------
        print_fundamentals(fetch_fundamentals(["BLK", "TXN", "PH"]))
    """
    if not rows:
        return

    # Column widths
    cols = [
        ("Ticker",        6),
        ("Price",         8),
        ("Fwd P/E",       7),
        ("TTM EPS",       8),
        ("Fwd EPS",       8),
        ("Next Earnings", 14),
        ("Est. EPS",      9),
        ("Ex-Div",        10),
        ("Div Amt",       8),
        ("Analyst Tgt",   11),
    ]

    sep   = "  "
    width = sum(w for _, w in cols) + len(sep) * (len(cols) - 1)

    # Show cache age if all rows came from cache (have _cached_at)
    cached_at_vals = [r.get("_cached_at") for r in rows if r.get("_cached_at")]
    cache_note = ""
    if cached_at_vals:
        oldest = min(cached_at_vals)
        cache_note = f"  [cached — oldest entry: {oldest}]"

    print(f"\n{'='*width}")
    print(f"  FUNDAMENTAL OVERVIEW{cache_note}")
    print(f"{'='*width}")
    header = sep.join(f"{name:<{w}}" for name, w in cols)
    print(f"  {header}")
    print(f"  {'-'*width}")

    for row in rows:
        line = sep.join(
            f"{str(row.get(name, '—')):<{w}}"
            for name, w in cols
        )
        print(f"  {line}")

    print(f"{'='*width}\n")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and display fundamental data for one or more tickers.",
        epilog=(
            "Examples:\n"
            "  python utils/fundamental_scanner.py TXN CPRT PH\n"
            "  python utils/fundamental_scanner.py TXN --cached\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbol(s)")
    parser.add_argument(
        "--cached", action="store_true",
        help="Display last-cached values without downloading",
    )
    args = parser.parse_args()

    if args.cached:
        print_fundamentals(load_cached_fundamentals(args.tickers))
    else:
        print_fundamentals(fetch_fundamentals(args.tickers))
