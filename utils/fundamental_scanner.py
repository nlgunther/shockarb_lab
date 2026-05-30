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

from datetime import datetime

import pandas as pd
import yfinance as yf
from loguru import logger


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


def fetch_fundamentals(tickers: list[str]) -> list[dict]:
    """
    Fetch fundamental data for each ticker and return a list of row dicts.

    Each dict contains display-ready strings suitable for tabular printing.
    Missing fields are represented as '—' rather than raising.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to fetch.

    Returns
    -------
    list of dict
        One dict per ticker with keys matching the table columns below.

    Example
    -------
        rows = fetch_fundamentals(["BLK", "TXN"])
        print_fundamentals(rows)
    """
    rows = []
    for symbol in tickers:
        try:
            t    = yf.Ticker(symbol)
            info = t.info or {}

            earn_date, earn_est = _next_earnings(t)
            ex_date, div_amt    = _next_dividend(info)

            rows.append({
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
            })
            logger.debug(f"[Fundamentals] {symbol} ok")
        except Exception as exc:
            logger.warning(f"[Fundamentals] {symbol} failed: {exc}")
            rows.append({"Ticker": symbol, **{k: "ERR" for k in (
                "Price", "Fwd P/E", "TTM EPS", "Fwd EPS",
                "Next Earnings", "Est. EPS", "Ex-Div", "Div Amt", "Analyst Tgt",
            )}})
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

    print(f"\n{'='*width}")
    print("  FUNDAMENTAL OVERVIEW")
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
